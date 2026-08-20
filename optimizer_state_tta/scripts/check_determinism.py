#!/usr/bin/env python3
"""Empirical run-to-run reproducibility check for the active backend.

PyTorch gives no deterministic-algorithm guarantee on MPS, so rather than
claiming determinism this script measures it: the same short chain is run twice
in the same process and the two trajectories are compared exactly.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optstate import adam_state as A                       # noqa: E402
from optstate import model as M                            # noqa: E402
from optstate import tent_core as T                        # noqa: E402
from optstate.data import Cifar10CStore, DomainSpec, DomainStream  # noqa: E402
from optstate.diagnostics import weights_fingerprint       # noqa: E402
from optstate.env import (enable_determinism, environment_record,  # noqa: E402
                          select_device, set_seed, write_json)


def one_run(arch, ckpt, device, store, corruption, seed, n, bs, lr):
    set_seed(seed)
    model = M.configure_tent_model(M.load_source_model(arch, ckpt, device))
    params, _ = M.collect_bn_params(model)
    opt = M.make_adam(params, lr, 0.9, 0.999, 0.0)
    stream = DomainStream(store, DomainSpec(corruption, 5, seed), bs, device)
    recs = T.run_domain(model, opt, stream, 0, n)
    return {
        "correct": [r.n_correct for r in recs],
        "loss": [r.entropy_loss for r in recs],
        "fingerprint": weights_fingerprint(model),
        "m_norm": float(A.flat_state(A.snapshot_adam(opt), "exp_avg").norm().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--arch", default="Standard")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "data/ckpt"))
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1/raw/determinism.json"))
    args = ap.parse_args()

    device = select_device()
    det = enable_determinism(False)
    store = Cifar10CStore(args.data_dir, 5)
    a = one_run(args.arch, args.ckpt_dir, device, store, args.corruption, 0,
                args.n, args.batch_size, 1e-3)
    b = one_run(args.arch, args.ckpt_dir, device, store, args.corruption, 0,
                args.n, args.batch_size, 1e-3)

    same_acc = a["correct"] == b["correct"]
    same_fp = a["fingerprint"] == b["fingerprint"]
    max_loss_diff = max(abs(x - y) for x, y in zip(a["loss"], b["loss"]))
    payload = {
        "device": str(device), "determinism_settings": det,
        "n_batches": args.n, "batch_size": args.batch_size,
        "identical_per_batch_correct_counts": same_acc,
        "identical_final_weight_fingerprint": same_fp,
        "max_abs_entropy_loss_difference": max_loss_diff,
        "m_norm_run_a": a["m_norm"], "m_norm_run_b": b["m_norm"],
        "correct_a": a["correct"], "correct_b": b["correct"],
        "interpretation": (
            "Bitwise run-to-run reproducibility on this backend."
            if same_acc and same_fp else
            "Run-to-run reproducibility is NOT bitwise on this backend; matched "
            "branches remain valid because they are constructed from one cloned "
            "checkpoint inside a single process and consume identical batches."),
        "environment": environment_record([0]),
    }
    write_json(Path(args.out), payload)
    print(f"identical accuracy trajectory : {same_acc}")
    print(f"identical weight fingerprint  : {same_fp}")
    print(f"max |entropy loss| difference : {max_loss_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
