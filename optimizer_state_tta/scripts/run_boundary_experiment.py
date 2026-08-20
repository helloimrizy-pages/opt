#!/usr/bin/env python3
"""ORACLE_BOUNDARY_DIAGNOSTIC - matched-branch optimizer-state experiment.

Sections 9-13.  One continual Tent chain per (corruption order, seed).  At every
boundary the exact model checkpoint and the exact Adam snapshot are cloned into
all six state interventions, so every branch starts the target window with
bitwise-identical model weights and consumes identical batches in identical
order.  Only optimizer state differs.

Corruption identity and boundary location are oracle information used for
analysis only; target labels are used for scoring only.

Modes
-----
primary   branch at each domain boundary (abrupt corruption-type shift)
control   branch at the mid-point of each domain into a matched pair:
          stationary (held-out second half of the SAME corruption) and
          shifted (first half of the NEXT corruption), both from one checkpoint
gradual   branch at severity boundaries 1->2->3->4->5 inside one corruption
sequence  SECONDARY: run one whole 15-domain chain applying a single fixed
          state policy at every boundary, to see what a naive always-on rule
          would do to total stream error.  This is descriptive only; it is not
          a method and nothing is tuned on it.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optstate import adam_state as A                        # noqa: E402
from optstate import metrics as MET                         # noqa: E402
from optstate import model as M                             # noqa: E402
from optstate import tent_core as T                         # noqa: E402
from optstate.data import (CONVENTIONAL_ORDER, Cifar10CStore, DomainSpec,   # noqa: E402
                           DomainStream, corruption_orders)
from optstate.diagnostics import boundary_diagnostics, weights_fingerprint  # noqa: E402
from optstate.env import (enable_determinism, environment_record,           # noqa: E402
                          select_device, set_seed, write_json)

ALIGN_FIRST_K = 10

SHORT = {
    "CARRY_ALL": "CARRY", "RESET_M_KEEP_V_STEP": "RESET_M",
    "RESET_V_KEEP_M_STEP": "RESET_V", "RESET_MV_KEEP_STEP": "RESET_MV",
    "RESET_STEP_ONLY": "RESET_STEP", "FRESH_ADAM": "FRESH",
}


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w")
        self.n = 0

    def write(self, obj: dict) -> None:
        self.fh.write(json.dumps(obj, default=float) + "\n")
        self.n += 1

    def close(self) -> None:
        self.fh.flush(); self.fh.close()


def free_mps() -> None:
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def emit_batches(w: JsonlWriter, base: dict, recs: List[T.BatchRecord]) -> None:
    for j, r in enumerate(recs):
        row = dict(base)
        row.update({
            "type": "batch", "batch_since_boundary": j, "stream_batch_index": r.batch_index,
            "n": r.n, "n_correct": r.n_correct, "accuracy": r.accuracy,
            "entropy_loss": r.entropy_loss, "mean_pred_entropy": r.mean_pred_entropy,
            "grad_norm": r.grad_norm,
        })
        row.update({k: v for k, v in r.extra.items()})
        w.write(row)


def branch_summary_row(base: dict, recs: List[T.BatchRecord],
                       carry_plateau: Optional[float]) -> dict:
    row = dict(base)
    row["type"] = "branch_summary"
    row.update(MET.window_metrics(recs))
    row.update(MET.collapse_indicator(recs))
    row["plateau_last10"] = MET.plateau_accuracy(recs)
    if carry_plateau is not None and carry_plateau == carry_plateau:
        row["recovery_batch"] = MET.recovery_batch(
            recs, carry_plateau - MET.RECOVERY_TOLERANCE)
        row["recovery_threshold"] = carry_plateau - MET.RECOVERY_TOLERANCE
    else:
        row["recovery_batch"] = MET.RECOVERY_CENSORED
        row["recovery_threshold"] = float("nan")
    return row


def run_branch_set(master_model, master_opt, snap, stream, start, length,
                   w: JsonlWriter, base: dict, advance_master: bool,
                   device) -> Dict[str, List[T.BatchRecord]]:
    """Run all six interventions on the same window.

    ``advance_master=True`` makes CARRY_ALL the master chain's own continuation
    (identical maths, zero extra compute); otherwise CARRY_ALL is run on a clone
    and the master is left untouched.
    """
    fingerprint = weights_fingerprint(master_model)
    out: Dict[str, List[T.BatchRecord]] = {}
    branch_fps: Dict[str, str] = {}

    for name in A.INTERVENTIONS:
        if name == "CARRY_ALL" and advance_master:
            continue
        branch = copy.deepcopy(master_model)
        bparams, _ = M.collect_bn_params(branch)
        bopt = A.build_branch_optimizer(bparams, snap, name)
        branch_fps[name] = weights_fingerprint(branch)
        recs = T.run_domain(branch, bopt, stream, start, length,
                            record_state_first_k=ALIGN_FIRST_K)
        out[name] = recs
        del branch, bopt, bparams

    if advance_master:
        branch_fps["CARRY_ALL"] = fingerprint
        out["CARRY_ALL"] = T.run_domain(master_model, master_opt, stream, start, length,
                                        record_state_first_k=ALIGN_FIRST_K)

    free_mps()
    assert all(fp == fingerprint for fp in branch_fps.values()), \
        "matched-branch violation: branches did not start from identical weights"

    carry_plateau = MET.plateau_accuracy(out["CARRY_ALL"])
    for name in A.INTERVENTIONS:
        b = dict(base); b["intervention"] = name
        emit_batches(w, b, out[name])
        w.write(branch_summary_row(b, out[name], carry_plateau))
    w.write({**base, "type": "matched_branch_check",
             "boundary_weight_fingerprint": fingerprint,
             "all_branches_identical": True,
             "n_interventions": len(A.INTERVENTIONS)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["primary", "control", "gradual", "sequence"],
                    default="primary")
    ap.add_argument("--sequence-policy", default="CARRY_ALL", choices=list(A.INTERVENTIONS))
    ap.add_argument("--order", default="conventional")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--severity", type=int, default=5)
    ap.add_argument("--domain-batches", type=int, default=50)
    ap.add_argument("--branch-batches", type=int, default=50)
    ap.add_argument("--max-domains", type=int, default=15)
    ap.add_argument("--gradual-corruption", default="gaussian_noise")
    ap.add_argument("--arch", default="Standard")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "data/ckpt"))
    ap.add_argument("--out", default=str(ROOT / "results/optimizer_state_stage1/raw/boundary"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = select_device(args.device)
    det = enable_determinism(False)
    set_seed(args.seed)

    orders = {o["name"]: o for o in corruption_orders()}
    if args.mode == "gradual":
        order_spec = {"name": "gradual", "perm_seed": None,
                      "order": [args.gradual_corruption] * 5}
        severities = [1, 2, 3, 4, 5]
    else:
        if args.order not in orders:
            raise SystemExit(f"unknown order {args.order}; have {list(orders)}")
        order_spec = orders[args.order]
        severities = [args.severity] * len(order_spec["order"])

    domains = order_spec["order"][: args.max_domains]
    severities = severities[: args.max_domains]

    tag = args.tag or (f"{args.mode}_{order_spec['name']}_seed{args.seed}"
                       f"_b1{args.beta1:g}_lr{args.lr:g}"
                       + (f"_{args.sequence_policy}" if args.mode == "sequence" else ""))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = JsonlWriter(out_dir / f"{tag}.jsonl")

    store = Cifar10CStore(args.data_dir, args.severity)
    model = M.configure_tent_model(M.load_source_model(args.arch, args.ckpt_dir, device))
    params, param_names = M.collect_bn_params(model)
    opt = M.make_adam(params, args.lr, args.beta1, args.beta2, args.wd)

    run_meta = {
        "run_tag": tag, "mode": args.mode, "order_name": order_spec["name"],
        "order": domains, "severities": severities,
        "order_perm_seed": order_spec["perm_seed"], "seed": args.seed,
        "beta1": args.beta1, "beta2": args.beta2, "lr": args.lr, "weight_decay": args.wd,
        "batch_size": args.batch_size, "arch": args.arch,
        "domain_batches": args.domain_batches, "branch_batches": args.branch_batches,
        "oracle_flag": "ORACLE_BOUNDARY_DIAGNOSTIC",
        "sequence_policy": args.sequence_policy if args.mode == "sequence" else None,
        "n_adapted_tensors": len(params),
        "n_adapted_scalars": int(sum(p.numel() for p in params)),
        "adapted_param_names": param_names,
        "device": str(device), "determinism": det,
    }
    jsonl.write({"type": "run_meta", **run_meta})

    # Compact per-row metadata: the heavy fields (full corruption order, adapted
    # parameter names, determinism dict) live once in the run_meta row above.
    row_meta = {k: run_meta[k] for k in (
        "run_tag", "mode", "order_name", "seed", "beta1", "beta2", "lr",
        "weight_decay", "batch_size", "arch", "domain_batches", "branch_batches",
        "oracle_flag", "sequence_policy")}

    t0 = time.time()
    n_boundaries = 0

    if args.mode == "sequence":
        streams = [DomainStream(store, DomainSpec(c, s, args.seed), args.batch_size, device)
                   for c, s in zip(domains, severities)]
        policy = args.sequence_policy
        for i, stream in enumerate(streams):
            dom_name = f"{domains[i]}:s{severities[i]}"
            if i > 0:
                snap = A.snapshot_adam(opt)
                if policy == "FRESH_ADAM":
                    params2, _ = M.collect_bn_params(model)
                    opt = A.build_branch_optimizer(params2, snap, "FRESH_ADAM")
                else:
                    A.restore_adam(opt, A.transform_snapshot(snap, policy))
            recs = T.run_domain(model, opt, stream, 0, args.domain_batches)
            emit_batches(jsonl, {"boundary_id": i - 1, "condition": "sequence",
                                 "domain_from": (f"{domains[i-1]}:s{severities[i-1]}"
                                                 if i > 0 else None),
                                 "domain_to": dom_name,
                                 "transition": dom_name,
                                 "intervention": policy, **row_meta}, recs)
            jsonl.write(branch_summary_row(
                {"boundary_id": i - 1, "condition": "sequence", "intervention": policy,
                 "domain_from": (f"{domains[i-1]}:s{severities[i-1]}" if i > 0 else None),
                 "domain_to": dom_name, "transition": dom_name, **row_meta},
                recs, None))
            n_boundaries += 1 if i > 0 else 0
            print(f"[{tag}] {policy} domain {i} {dom_name} "
                  f"acc={MET.window_accuracy(recs, len(recs)):.4f}", flush=True)

    elif args.mode in ("primary", "gradual"):
        streams = [DomainStream(store, DomainSpec(c, s, args.seed), args.batch_size, device)
                   for c, s in zip(domains, severities)]
        for i, stream in enumerate(streams):
            dom_name = f"{domains[i]}:s{severities[i]}"
            if i == 0:
                recs = T.run_domain(model, opt, stream, 0, args.domain_batches)
                emit_batches(jsonl, {"boundary_id": -1, "condition": "warmup",
                                     "domain_to": dom_name, "domain_from": None,
                                     "intervention": "CARRY_ALL", **row_meta}, recs)
                print(f"[{tag}] warmup {dom_name} acc={MET.window_accuracy(recs, len(recs)):.4f}",
                      flush=True)
                continue

            bid = i - 1
            snap = A.snapshot_adam(opt)
            base = {"boundary_id": bid, "condition": "shift",
                    "domain_from": f"{domains[i-1]}:s{severities[i-1]}",
                    "domain_to": dom_name,
                    "transition": f"{domains[i-1]}:s{severities[i-1]}->{dom_name}",
                    **row_meta}
            x0, y0 = stream.batch(0)
            diag = boundary_diagnostics(model, opt, snap, x0, y0)
            jsonl.write({**base, "type": "boundary_diagnostic", **diag})
            del x0, y0
            out = run_branch_set(model, opt, snap, stream, 0, args.branch_batches,
                                 jsonl, base, advance_master=True, device=device)
            n_boundaries += 1
            kw = min(MET.PRIMARY_WINDOW, args.branch_batches)
            e10 = {k: MET.window_accuracy(v, kw) for k, v in out.items()}
            print(f"[{tag}] boundary {bid} {base['transition']} cos_m_g={diag['cos_m_g']:+.3f} "
                  f"carry{kw}={e10['CARRY_ALL']:.4f} | "
                  + "  ".join(f"{SHORT[k]}:{100*(e10[k]-e10['CARRY_ALL']):+.2f}"
                              for k in A.INTERVENTIONS if k != "CARRY_ALL"),
                  flush=True)
            free_mps()

    else:  # control: matched stationary / shifted pseudo-boundary
        half = args.domain_batches // 2
        streams = [DomainStream(store, DomainSpec(c, s, args.seed), args.batch_size, device)
                   for c, s in zip(domains, severities)]
        for i, stream in enumerate(streams):
            dom_name = f"{domains[i]}:s{severities[i]}"
            recs = T.run_domain(model, opt, stream, 0, half)
            emit_batches(jsonl, {"boundary_id": i, "condition": "pre_pseudo_boundary",
                                 "domain_from": None, "domain_to": dom_name,
                                 "intervention": "CARRY_ALL", **row_meta}, recs)

            snap = A.snapshot_adam(opt)

            if i + 1 < len(streams):
                nxt = streams[i + 1]
                nxt_name = f"{domains[i+1]}:s{severities[i+1]}"
                sbase = {"boundary_id": i, "condition": "shifted",
                         "domain_from": dom_name, "domain_to": nxt_name,
                         "transition": f"{dom_name}->{nxt_name}", **row_meta}
                x0, y0 = nxt.batch(0)
                jsonl.write({**sbase, "type": "boundary_diagnostic",
                             **boundary_diagnostics(model, opt, snap, x0, y0)})
                del x0, y0
                run_branch_set(model, opt, snap, nxt, 0, half, jsonl, sbase,
                               advance_master=False, device=device)
                free_mps()

            cbase = {"boundary_id": i, "condition": "stationary",
                     "domain_from": dom_name, "domain_to": dom_name,
                     "transition": f"{dom_name}->{dom_name}", **row_meta}
            xh, yh = stream.batch(half)
            jsonl.write({**cbase, "type": "boundary_diagnostic",
                         **boundary_diagnostics(model, opt, snap, xh, yh)})
            del xh, yh
            run_branch_set(model, opt, snap, stream, half, half, jsonl, cbase,
                           advance_master=True, device=device)
            n_boundaries += 1
            print(f"[{tag}] pseudo-boundary {i} {dom_name} done", flush=True)
            free_mps()

    elapsed = time.time() - t0
    jsonl.write({"type": "run_complete", "run_tag": tag, "n_boundaries": n_boundaries,
                 "elapsed_s": elapsed, "n_rows": jsonl.n})
    jsonl.close()
    write_json(out_dir / f"{tag}.meta.json",
               {**run_meta, "n_boundaries": n_boundaries, "elapsed_s": elapsed,
                "environment": environment_record([args.seed])})
    print(f"[{tag}] done: {n_boundaries} boundaries in {elapsed/60:.1f} min -> "
          f"{out_dir / (tag + '.jsonl')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
