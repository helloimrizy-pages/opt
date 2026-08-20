"""Adam optimizer-state instrumentation.

Everything here is *diagnostic*.  No adaptive policy, no gradient-similarity
decay, no learned rule: only exact clone / inspect / manipulate / restore of the
three pieces of state ``torch.optim.Adam`` keeps per parameter --

    ``step``         bias-correction counter
    ``exp_avg``      first-moment EMA   (m)
    ``exp_avg_sq``   second-moment EMA  (v)

Interventions are keyed by parameter *position* in ``optimizer.param_groups`` so
a snapshot taken from one model can be loaded into a bitwise-identical clone.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

INTERVENTIONS: Tuple[str, ...] = (
    "CARRY_ALL",
    "RESET_M_KEEP_V_STEP",
    "RESET_V_KEEP_M_STEP",
    "RESET_MV_KEEP_STEP",
    "RESET_STEP_ONLY",
    "FRESH_ADAM",
)

INTERVENTION_DOC = {
    "CARRY_ALL": "keep model params, exp_avg, exp_avg_sq and step",
    "RESET_M_KEEP_V_STEP": "exp_avg <- 0; keep exp_avg_sq and step",
    "RESET_V_KEEP_M_STEP": "exp_avg_sq <- 0; keep exp_avg and step",
    "RESET_MV_KEEP_STEP": "exp_avg <- 0, exp_avg_sq <- 0; keep step",
    "RESET_STEP_ONLY": "keep exp_avg and exp_avg_sq; step <- 0",
    "FRESH_ADAM": "recreate Adam: exp_avg = exp_avg_sq = 0, step = 0",
}


# --------------------------------------------------------------------------- #
# snapshot / restore
# --------------------------------------------------------------------------- #
@dataclass
class AdamSnapshot:
    """Position-keyed deep copy of Adam state plus the hyper-parameters."""
    entries: List[Optional[Dict[str, torch.Tensor]]]
    hyper: Dict[str, object]

    def clone(self) -> "AdamSnapshot":
        entries: List[Optional[Dict[str, torch.Tensor]]] = []
        for e in self.entries:
            entries.append(None if e is None
                           else {k: v.clone() for k, v in e.items()})
        return AdamSnapshot(entries=entries, hyper=dict(self.hyper))

    @property
    def n_params(self) -> int:
        return len(self.entries)

    @property
    def initialised(self) -> bool:
        return any(e is not None for e in self.entries)


def _as_step_tensor(value, reference: torch.Tensor) -> torch.Tensor:
    """Clone Adam's step counter, preserving its device.

    ``torch.optim.Adam`` keeps ``step`` on the CPU unless ``capturable`` or
    ``fused`` is set, even when the parameter lives on an accelerator.  The
    snapshot must preserve that so a restored branch is byte-for-byte the same
    optimizer configuration as the trajectory it was cloned from.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.tensor(float(value), dtype=torch.float32)


def iter_params(optimizer: torch.optim.Optimizer):
    for group in optimizer.param_groups:
        for p in group["params"]:
            yield group, p


def snapshot_adam(optimizer: torch.optim.Optimizer) -> AdamSnapshot:
    entries: List[Optional[Dict[str, torch.Tensor]]] = []
    for _, p in iter_params(optimizer):
        st = optimizer.state.get(p, None)
        if not st:
            entries.append(None)
            continue
        entries.append({
            "step": _as_step_tensor(st["step"], st["exp_avg"]),
            "exp_avg": st["exp_avg"].detach().clone(),
            "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
        })
    g0 = optimizer.param_groups[0]
    hyper = {
        "lr": g0["lr"], "betas": tuple(g0["betas"]), "eps": g0["eps"],
        "weight_decay": g0["weight_decay"], "amsgrad": bool(g0.get("amsgrad", False)),
    }
    return AdamSnapshot(entries=entries, hyper=hyper)


def restore_adam(optimizer: torch.optim.Optimizer, snap: AdamSnapshot) -> None:
    """Load ``snap`` into ``optimizer`` positionally (params may differ objects)."""
    params = [p for _, p in iter_params(optimizer)]
    if len(params) != snap.n_params:
        raise ValueError(f"param count mismatch: {len(params)} vs {snap.n_params}")
    optimizer.state = type(optimizer.state)()
    for p, entry in zip(params, snap.entries):
        if entry is None:
            continue
        optimizer.state[p] = {
            "step": entry["step"].detach().clone(),   # keep Adam's own device
            "exp_avg": entry["exp_avg"].detach().clone().to(p.device),
            "exp_avg_sq": entry["exp_avg_sq"].detach().clone().to(p.device),
        }


# --------------------------------------------------------------------------- #
# interventions
# --------------------------------------------------------------------------- #
def transform_snapshot(snap: AdamSnapshot, intervention: str) -> AdamSnapshot:
    """Return a new snapshot with ``intervention`` applied. Never touches weights."""
    if intervention not in INTERVENTIONS:
        raise ValueError(f"unknown intervention {intervention!r}")
    out = snap.clone()
    if intervention == "CARRY_ALL":
        return out
    for entry in out.entries:
        if entry is None:
            continue
        if intervention in ("RESET_M_KEEP_V_STEP", "RESET_MV_KEEP_STEP", "FRESH_ADAM"):
            entry["exp_avg"].zero_()
        if intervention in ("RESET_V_KEEP_M_STEP", "RESET_MV_KEEP_STEP", "FRESH_ADAM"):
            entry["exp_avg_sq"].zero_()
        if intervention in ("RESET_STEP_ONLY", "FRESH_ADAM"):
            entry["step"].zero_()
    return out


def build_branch_optimizer(params: Sequence[torch.nn.Parameter],
                           snap: AdamSnapshot, intervention: str,
                           ) -> torch.optim.Adam:
    """Create the branch optimizer for ``intervention`` over ``params``.

    ``FRESH_ADAM`` is built by genuinely re-instantiating ``torch.optim.Adam``
    with no state entries at all, which is the strongest form of the control;
    the unit tests check it is numerically identical to zeroing m, v and step.
    """
    hyper = snap.hyper
    opt = torch.optim.Adam(list(params), lr=hyper["lr"], betas=tuple(hyper["betas"]),
                           eps=hyper["eps"], weight_decay=hyper["weight_decay"],
                           amsgrad=bool(hyper["amsgrad"]))
    if intervention == "FRESH_ADAM":
        return opt                       # empty state == fresh Adam
    restore_adam(opt, transform_snapshot(snap, intervention))
    return opt


# --------------------------------------------------------------------------- #
# flattening helpers used by the boundary diagnostics
# --------------------------------------------------------------------------- #
def flat_state(snap: AdamSnapshot, key: str, device=None) -> torch.Tensor:
    """Concatenate one state field over all parameters in optimizer order."""
    parts = []
    for entry in snap.entries:
        if entry is None:
            continue
        t = entry[key].detach().reshape(-1)
        parts.append(t if device is None else t.to(device))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def flat_grads(optimizer: torch.optim.Optimizer) -> torch.Tensor:
    parts = []
    for _, p in iter_params(optimizer):
        g = p.grad
        parts.append(torch.zeros_like(p).reshape(-1) if g is None
                     else g.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(0)


def flat_params(optimizer: torch.optim.Optimizer) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in iter_params(optimizer)])


def steps_of(snap: AdamSnapshot) -> List[float]:
    return [float(e["step"].item()) for e in snap.entries if e is not None]


def implied_adam_update(m_prev: torch.Tensor, v_prev: torch.Tensor,
                        step_prev: torch.Tensor, grad: torch.Tensor,
                        lr: float, beta1: float, beta2: float, eps: float
                        ) -> torch.Tensor:
    """The parameter delta Adam *would* apply, computed without mutating state.

    Mirrors ``torch.optim.Adam`` single-tensor maths for ``weight_decay = 0``.
    """
    t = step_prev + 1.0
    m = beta1 * m_prev + (1.0 - beta1) * grad
    v = beta2 * v_prev + (1.0 - beta2) * grad * grad
    bc1 = 1.0 - torch.pow(torch.as_tensor(beta1, dtype=t.dtype, device=t.device), t)
    bc2 = 1.0 - torch.pow(torch.as_tensor(beta2, dtype=t.dtype, device=t.device), t)
    step_size = lr / bc1
    denom = (v.sqrt() / bc2.sqrt()) + eps
    return -step_size * m / denom


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((a @ b / (na * nb)).item())


def live_state_stats(optimizer: torch.optim.Optimizer, grad: torch.Tensor
                     ) -> Dict[str, float]:
    """Read-only ``|m|``, ``mean sqrt(v)``, ``step`` and ``cos(m, g)``.

    Unlike ``snapshot_adam`` this clones nothing and issues a single host
    synchronisation, so it is cheap enough to call on every instrumented batch.
    """
    ms, vs, step = [], [], 0.0
    for _, p in iter_params(optimizer):
        st = optimizer.state.get(p, None)
        if not st:
            continue
        ms.append(st["exp_avg"].detach().reshape(-1))
        vs.append(st["exp_avg_sq"].detach().reshape(-1))
        step = float(st["step"]) if not isinstance(st["step"], torch.Tensor) \
            else float(st["step"].item())
    if not ms:
        return {"cos_m_g": float("nan"), "m_norm": 0.0, "sqrt_v_mean": 0.0,
                "adam_step": 0.0}
    m = torch.cat(ms)
    v = torch.cat(vs)
    m_norm = m.norm()
    g_norm = grad.norm()
    denom = m_norm * g_norm
    cos = (m @ grad) / denom if float(denom) > 0 else torch.full((), float("nan"),
                                                                device=m.device)
    packed = torch.stack([m_norm, v.sqrt().mean(), cos.reshape(())]).to("cpu")
    return {"m_norm": float(packed[0]), "sqrt_v_mean": float(packed[1]),
            "cos_m_g": float(packed[2]), "adam_step": step}


def state_summary(snap: AdamSnapshot) -> Dict[str, float]:
    m = flat_state(snap, "exp_avg")
    v = flat_state(snap, "exp_avg_sq")
    steps = steps_of(snap)
    sqrt_v = v.sqrt()
    return {
        "m_norm": float(m.norm().item()) if m.numel() else 0.0,
        "m_abs_mean": float(m.abs().mean().item()) if m.numel() else 0.0,
        "v_norm": float(v.norm().item()) if v.numel() else 0.0,
        "sqrt_v_mean": float(sqrt_v.mean().item()) if v.numel() else 0.0,
        "sqrt_v_median": float(sqrt_v.median().item()) if v.numel() else 0.0,
        "step_min": min(steps) if steps else 0.0,
        "step_max": max(steps) if steps else 0.0,
        "n_state_entries": float(sum(1 for e in snap.entries if e is not None)),
    }
