#!/usr/bin/env python3
"""Section 5 - reproduce the frozen source model and standard online Tent.

Protocol is the official Tent CIFAR-10-C example (``DequanWang/tent/cifar10c.py``):
severity 5, the 15 standard corruptions in the conventional order, batch size
200, unshuffled samples, and a full model+optimizer reset between corruption
types (``model.reset()``), which makes each corruption an independent online
episode.  A continual variant with no reset between corruptions is also recorded
because that is the regime the Stage 1 boundary experiment operates in.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optstate import adam_state as A                       # noqa: E402
from optstate import model as M                            # noqa: E402
from optstate import tent_core as T                        # noqa: E402
from optstate.data import CONVENTIONAL_ORDER, Cifar10CStore, DomainSpec, DomainStream  # noqa: E402
from optstate.env import (enable_determinism, environment_record,  # noqa: E402
                          select_device, set_seed, write_json)

REFERENCE = {
    "Standard": {"source": 43.5, "norm": 20.4, "tent": 18.6,
                 "arch": "WideResNet-28-10"},
    "Hendrycks2020AugMix_WRN": {"source": 18.3, "norm": 14.5, "tent": 12.1,
                                "arch": "WideResNet-40-2 (AugMix)"},
}


def eval_source(model, store, device, batch_size, severity, rows, arch):
    model.eval()
    per_corr = {}
    for corruption in CONVENTIONAL_ORDER:
        stream = DomainStream(store, DomainSpec(corruption, severity, 0), batch_size, device)
        recs = [T.source_eval_step(model, x, y, k)
                for k, x, y in stream.batches(0, stream.n_batches)]
        n = sum(r.n for r in recs); c = sum(r.n_correct for r in recs)
        err = 100.0 * (1.0 - c / n)
        per_corr[corruption] = err
        rows.append({"arch": arch, "method": "source", "corruption": corruption,
                     "severity": severity, "n": n, "n_correct": c, "error_pct": err})
        print(f"  source  {corruption:<20s} err {err:6.2f}%", flush=True)
    return per_corr


def eval_norm(base_model, store, device, batch_size, severity, rows, arch):
    """Test-time normalisation: batch statistics, no parameter updates."""
    model = M.configure_tent_model(copy.deepcopy(base_model))
    per_corr = {}
    for corruption in CONVENTIONAL_ORDER:
        stream = DomainStream(store, DomainSpec(corruption, severity, 0), batch_size, device)
        n = c = 0
        with torch.no_grad():
            for k, x, y in stream.batches(0, stream.n_batches):
                out = model(x)
                c += int((out.argmax(1) == y).sum().item()); n += int(y.numel())
        err = 100.0 * (1.0 - c / n)
        per_corr[corruption] = err
        rows.append({"arch": arch, "method": "norm", "corruption": corruption,
                     "severity": severity, "n": n, "n_correct": c, "error_pct": err})
        print(f"  norm    {corruption:<20s} err {err:6.2f}%", flush=True)
    return per_corr


def eval_tent(base_model, store, device, batch_size, severity, rows, arch,
              lr, beta1, beta2, wd, reset_between_corruptions, method_name):
    model = M.configure_tent_model(copy.deepcopy(base_model))
    params, names = M.collect_bn_params(model)
    opt = M.make_adam(params, lr, beta1, beta2, wd)
    model_state = copy.deepcopy(model.state_dict())
    opt_state = copy.deepcopy(opt.state_dict())
    per_corr = {}
    for corruption in CONVENTIONAL_ORDER:
        if reset_between_corruptions:
            model.load_state_dict(model_state)
            opt.load_state_dict(opt_state)
        stream = DomainStream(store, DomainSpec(corruption, severity, 0), batch_size, device)
        recs = T.run_domain(model, opt, stream, 0, stream.n_batches)
        n = sum(r.n for r in recs); c = sum(r.n_correct for r in recs)
        err = 100.0 * (1.0 - c / n)
        per_corr[corruption] = err
        rows.append({"arch": arch, "method": method_name, "corruption": corruption,
                     "severity": severity, "n": n, "n_correct": c, "error_pct": err})
        print(f"  {method_name:<7s} {corruption:<20s} err {err:6.2f}%", flush=True)
    return per_corr, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "data/ckpt"))
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1/raw/baseline"))
    ap.add_argument("--archs", nargs="+", default=["Standard", "Hendrycks2020AugMix_WRN"])
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = select_device(args.device)
    det = enable_determinism(False)
    set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    store = Cifar10CStore(args.data_dir, args.severity)

    rows: list = []
    summary: dict = {}
    for arch in args.archs:
        print(f"\n=== {arch} ({REFERENCE.get(arch, {}).get('arch', '?')}) ===", flush=True)
        t0 = time.time()
        base = M.load_source_model(arch, args.ckpt_dir, device)
        cfg_probe = M.check_tent_config(M.configure_tent_model(copy.deepcopy(base)))
        src = eval_source(copy.deepcopy(base), store, device, args.batch_size,
                          args.severity, rows, arch)
        nrm = eval_norm(base, store, device, args.batch_size, args.severity, rows, arch)
        tnt, names = eval_tent(base, store, device, args.batch_size, args.severity, rows,
                               arch, args.lr, args.beta1, args.beta2, args.wd,
                               True, "tent")
        cont, _ = eval_tent(base, store, device, args.batch_size, args.severity, rows,
                            arch, args.lr, args.beta1, args.beta2, args.wd,
                            False, "tent_continual")

        def mean(d): return sum(d.values()) / len(d)
        ref = REFERENCE.get(arch, {})
        entry = {
            "arch": arch,
            "arch_description": ref.get("arch"),
            "n_params": sum(p.numel() for p in base.parameters()),
            "tent_config_probe": cfg_probe,
            "n_adapted_tensors": len(names),
            "per_corruption_error_pct": {"source": src, "norm": nrm, "tent": tnt,
                                         "tent_continual": cont},
            "mean_error_pct": {"source": mean(src), "norm": mean(nrm), "tent": mean(tnt),
                               "tent_continual": mean(cont)},
            "official_reference_error_pct": {k: v for k, v in ref.items() if k != "arch"},
            "abs_deviation_pp": {
                k: abs(mean(d) - ref[k]) for k, d in
                (("source", src), ("norm", nrm), ("tent", tnt)) if k in ref
            },
            "runtime_s": time.time() - t0,
        }
        entry["within_2pp_of_reference"] = {
            k: bool(v <= 2.0) for k, v in entry["abs_deviation_pp"].items()
        }
        entry["baseline_valid"] = bool(all(entry["within_2pp_of_reference"].values()))
        summary[arch] = entry
        print(f"  MEAN source {mean(src):.2f}  norm {mean(nrm):.2f}  tent {mean(tnt):.2f} "
              f" tent_continual {mean(cont):.2f}  (ref {ref})", flush=True)

    csv_path = out_dir / "baseline_per_corruption.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    payload = {
        "protocol": "official Tent CIFAR-10-C example (cifar10c.py), severity 5, "
                    "conventional corruption order, batch 200, unshuffled, "
                    "model+optimizer reset between corruption types",
        "optimizer": {"method": "Adam", "lr": args.lr, "betas": [args.beta1, args.beta2],
                      "weight_decay": args.wd, "steps": 1},
        "batch_size": args.batch_size,
        "severity": args.severity,
        "determinism": det,
        "results": summary,
        "baseline_valid_primary_arch": summary.get("Standard", {}).get("baseline_valid"),
        "environment": environment_record([args.seed]),
    }
    write_json(out_dir / "baseline_summary.json", payload)
    print("\n" + json.dumps({k: v["mean_error_pct"] for k, v in summary.items()}, indent=2))
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
