#!/usr/bin/env python3
"""Section 8 - toy mechanistic sanity test for the state instrumentation.

A deterministic 2-D quadratic whose optimum moves abruptly at the boundary so
that the required gradient direction reverses.  Adam builds first/second-moment
state in phase A; at the boundary the *same* parameter vector is cloned into
CARRY_ALL, RESET_M_KEEP_V_STEP and FRESH_ADAM branches.

This is an implementation/mechanism check only.  It is NOT evidence that the
CIFAR-10-C phenomenon exists and is never mixed into the Stage 1 verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optstate import adam_state as A                      # noqa: E402
from optstate.env import environment_record, set_seed, write_json  # noqa: E402

# phase A optimum, phase B optimum: the gradient direction required in x flips
OPT_A = torch.tensor([3.0, 0.0])
OPT_B = torch.tensor([-3.0, 0.0])
CURV = torch.tensor([1.0, 0.25])


def loss_of(theta: torch.Tensor, centre: torch.Tensor) -> torch.Tensor:
    return (0.5 * CURV * (theta - centre) ** 2).sum()


def run_phase(theta: torch.Tensor, opt: torch.optim.Adam, centre: torch.Tensor,
              n_steps: int, phase: str, branch: str, rows: list, start_step: int = 0):
    for i in range(n_steps):
        snap = A.snapshot_adam(opt)
        opt.zero_grad(set_to_none=True)
        loss = loss_of(theta, centre)
        loss.backward()
        g = theta.grad.detach().clone()
        m_prev = A.flat_state(snap, "exp_avg") if snap.initialised else torch.zeros_like(g)
        v_prev = A.flat_state(snap, "exp_avg_sq") if snap.initialised else torch.zeros_like(g)
        step_prev = torch.tensor(A.steps_of(snap)[0] if snap.initialised else 0.0)
        predicted = A.implied_adam_update(
            m_prev, v_prev, step_prev, g.reshape(-1),
            opt.param_groups[0]["lr"], *opt.param_groups[0]["betas"],
            opt.param_groups[0]["eps"])
        before = theta.detach().clone()
        opt.step()
        actual = (theta.detach() - before).reshape(-1)
        rows.append({
            "branch": branch, "phase": phase, "step": start_step + i,
            "theta0": float(before[0]), "theta1": float(before[1]),
            "loss": float(loss.detach()),
            "g0": float(g[0]), "g1": float(g[1]), "g_norm": float(g.norm()),
            "m0": float(m_prev[0]), "m1": float(m_prev[1]), "m_norm": float(m_prev.norm()),
            "v0": float(v_prev[0]), "v1": float(v_prev[1]),
            "sqrt_v0": float(v_prev[0].sqrt()), "sqrt_v1": float(v_prev[1].sqrt()),
            "adam_step_count": float(step_prev),
            "cos_m_g": A.cosine(m_prev, g.reshape(-1)),
            "update0": float(actual[0]), "update1": float(actual[1]),
            "update_norm": float(actual.norm()),
            "predicted_update_norm": float(predicted.norm()),
            "predicted_matches_actual": bool(torch.allclose(predicted, actual, atol=1e-6)),
            "update_cos_neg_g": A.cosine(actual, -g.reshape(-1)),
        })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1/raw/toy"))
    ap.add_argument("--phase-a-steps", type=int, default=60)
    ap.add_argument("--phase-b-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-1)
    args = ap.parse_args()

    set_seed(0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list = []

    theta = torch.tensor([-1.0, 1.0], requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    run_phase(theta, opt, OPT_A, args.phase_a_steps, "A", "warmup", rows)

    boundary_theta = theta.detach().clone()
    snap = A.snapshot_adam(opt)
    boundary = {
        "theta": boundary_theta.tolist(),
        "state": A.state_summary(snap),
        "m": A.flat_state(snap, "exp_avg").tolist(),
        "exp_avg_sq": A.flat_state(snap, "exp_avg_sq").tolist(),
        "step": A.steps_of(snap),
    }

    # the gradient the new phase demands, measured at the boundary point
    probe = boundary_theta.clone().requires_grad_(True)
    loss_of(probe, OPT_B).backward()
    g_b = probe.grad.detach().reshape(-1)
    boundary["g_first_B"] = g_b.tolist()
    boundary["cos_m_g_first_B"] = A.cosine(A.flat_state(snap, "exp_avg"), g_b)

    branch_summary = {}
    for name in A.INTERVENTIONS:
        th = boundary_theta.clone().requires_grad_(True)
        bopt = A.build_branch_optimizer([th], snap, name)
        assert torch.equal(th.detach(), boundary_theta), "branch weights must match"
        run_phase(th, bopt, OPT_B, args.phase_b_steps, "B", name, rows)
        branch_summary[name] = {
            "final_theta": th.detach().tolist(),
            "final_loss": float(loss_of(th.detach(), OPT_B)),
            "loss_after_1": [r["loss"] for r in rows if r["branch"] == name][1],
            "loss_after_5": [r["loss"] for r in rows if r["branch"] == name][5],
            "loss_after_10": [r["loss"] for r in rows if r["branch"] == name][10],
        }

    csv_path = out_dir / "toy_trace.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    all_pred_ok = all(r["predicted_matches_actual"] for r in rows)
    payload = {
        "kind": "toy_mechanistic_sanity_check",
        "is_evidence_for_cv_phenomenon": False,
        "note": "Implementation/mechanism check only.",
        "config": {"lr": args.lr, "phase_a_steps": args.phase_a_steps,
                   "phase_b_steps": args.phase_b_steps,
                   "optimum_A": OPT_A.tolist(), "optimum_B": OPT_B.tolist(),
                   "curvature": CURV.tolist()},
        "boundary": boundary,
        "branches": branch_summary,
        "implied_update_predictor_matches_real_adam_everywhere": all_pred_ok,
        "environment": environment_record([0]),
    }
    write_json(out_dir / "toy_summary.json", payload)

    print(json.dumps({
        "boundary_cos_m_g_first_B": boundary["cos_m_g_first_B"],
        "boundary_step": boundary["step"],
        "predictor_exact": all_pred_ok,
        "loss_after_1": {k: v["loss_after_1"] for k, v in branch_summary.items()},
        "loss_after_10": {k: v["loss_after_10"] for k, v in branch_summary.items()},
        "final_loss": {k: v["final_loss"] for k, v in branch_summary.items()},
    }, indent=2))
    print(f"\nwrote {csv_path}\nwrote {out_dir / 'toy_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
