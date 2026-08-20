"""Optimizer mechanics recorded at a boundary, before the first target update.

All quantities are computed from the entropy loss only.  Target labels appear
solely in the accuracy fields, which are never fed back into anything.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from . import adam_state as A
from .tent_core import softmax_entropy


@torch.enable_grad()
def boundary_diagnostics(model: nn.Module, optimizer: torch.optim.Optimizer,
                         snap: A.AdamSnapshot, x: torch.Tensor, y: torch.Tensor,
                         ) -> Dict[str, float]:
    """Measure the carried state against the first gradient of the new domain.

    ``model``/``optimizer`` must hold the boundary checkpoint.  The optimizer is
    used only to reach ``.grad`` buffers; ``optimizer.step`` is never called and
    the gradients are cleared before returning, so the caller's checkpoint is
    unchanged.
    """
    device = x.device
    for _, p in A.iter_params(optimizer):
        p.grad = None

    outputs = model(x)
    with torch.no_grad():
        ent = softmax_entropy(outputs)
        acc = float((outputs.argmax(1) == y).float().mean().item())
        mean_ent = float(ent.mean().item())
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()

    g = A.flat_grads(optimizer).to(device)
    m_prev = A.flat_state(snap, "exp_avg", device=device)
    v_prev = A.flat_state(snap, "exp_avg_sq", device=device)
    steps = A.steps_of(snap)
    step_prev = torch.tensor(steps[0] if steps else 0.0, dtype=torch.float32, device=device)

    lr = float(snap.hyper["lr"])
    beta1, beta2 = (float(b) for b in snap.hyper["betas"])
    eps = float(snap.hyper["eps"])

    out: Dict[str, float] = {
        "boundary_entropy_loss": float(loss.detach().item()),
        "boundary_mean_pred_entropy": mean_ent,
        "boundary_pre_update_accuracy": acc,
        "grad_norm": float(g.norm().item()),
        "cos_m_g": A.cosine(m_prev, g),
        "cos_m_neg_g": A.cosine(m_prev, -g),
        "step_prev": float(step_prev.item()),
    }
    out.update(A.state_summary(snap))
    out["m_over_g_norm"] = (out["m_norm"] / out["grad_norm"]) if out["grad_norm"] > 0 else float("nan")

    zeros = torch.zeros_like(g)
    variants = {
        "CARRY_ALL": (m_prev, v_prev, step_prev),
        "RESET_M_KEEP_V_STEP": (zeros, v_prev, step_prev),
        "RESET_V_KEEP_M_STEP": (m_prev, zeros, step_prev),
        "RESET_MV_KEEP_STEP": (zeros, zeros, step_prev),
        "RESET_STEP_ONLY": (m_prev, v_prev, torch.zeros_like(step_prev)),
        "FRESH_ADAM": (zeros, zeros, torch.zeros_like(step_prev)),
    }
    updates: Dict[str, torch.Tensor] = {}
    for name, (mm, vv, ss) in variants.items():
        u = A.implied_adam_update(mm, vv, ss, g, lr, beta1, beta2, eps)
        updates[name] = u
        out[f"update_cos_neg_g[{name}]"] = A.cosine(u, -g)
        out[f"update_norm[{name}]"] = float(u.norm().item())
        out[f"update_abs_mean[{name}]"] = float(u.abs().mean().item())
        out[f"update_abs_median[{name}]"] = float(u.abs().median().item())
    for name in variants:
        if name != "CARRY_ALL":
            out[f"update_cos_vs_carry[{name}]"] = A.cosine(updates[name], updates["CARRY_ALL"])

    for _, p in A.iter_params(optimizer):
        p.grad = None
    return out


@torch.enable_grad()
def gradient_alignment_trace(model: nn.Module, optimizer: torch.optim.Optimizer,
                             ) -> Dict[str, float]:
    """cos(m_current, g_current) using the optimizer's live state and grads."""
    snap = A.snapshot_adam(optimizer)
    g = A.flat_grads(optimizer)
    m = A.flat_state(snap, "exp_avg", device=g.device)
    return {
        "cos_m_g": A.cosine(m, g),
        "grad_norm": float(g.norm().item()),
        "m_norm": float(m.norm().item()) if m.numel() else 0.0,
    }


def weights_fingerprint(model: nn.Module) -> str:
    """Order-stable hash of every entry of ``state_dict``, for matched-branch checks.

    Every tensor is flattened and concatenated *on the device* so the whole
    check costs one host transfer rather than one per tensor; on an accelerator
    the per-tensor version is dominated by synchronisation latency.
    """
    import hashlib
    h = hashlib.sha256()
    parts = []
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        t = t.detach().reshape(-1)
        if t.dtype == torch.float32:
            parts.append(t)                       # batched into one transfer
        else:                                     # rare (e.g. int counters)
            h.update(t.to("cpu").contiguous().numpy().tobytes())
    if parts:
        flat = torch.cat(parts).to("cpu").contiguous()
        h.update(flat.numpy().tobytes())
    return h.hexdigest()
