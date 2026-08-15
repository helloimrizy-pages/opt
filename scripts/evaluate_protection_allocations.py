#!/usr/bin/env python3
"""Evaluate frozen Stage 2B allocations on a held-out split (CUDA production)."""
from __future__ import annotations

import os

# Strict determinism requires this before the first cuBLAS call, so it is set
# before torch can be imported by any dependency below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import EXPECTED_MODEL_REVISION, array_sha256
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.heldout_splits import load_heldout_split
from expert_analysis.io_utils import atomic_write_json, package_versions, read_json
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_allocations import load_frozen_registry
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    evaluate_allocation_records,
    evaluation_run_fingerprint,
    run_repeated_baseline_check,
    verify_layout_against_memory_shapes,
)
from expert_analysis.protection_reporting import phase_records
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS, STAGE2B_DOMAINS
from expert_analysis.stage2b_preflight import verify_frozen_upstream_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--determinism-warn-only",
        action="store_true",
        help=(
            "Permit warn-only deterministic algorithms. The bitwise repeated-"
            "baseline gate still runs and still stops the run on any mismatch."
        ),
    )
    args = parser.parse_args()

    verify_frozen_upstream_decisions(args.results_root)
    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)

    allocations_dir = args.results_dir / "allocations"
    registry = load_frozen_registry(allocations_dir)
    print(f"Frozen registry verified: {registry['registry_sha256'][:16]}...")

    phase_dir = args.results_dir / args.phase
    losses_dir = phase_dir / "losses"
    final_losses_dir = args.results_dir / "final" / "losses"
    if args.phase == "development":
        if final_losses_dir.exists() and any(final_losses_dir.iterdir()):
            raise RuntimeError(
                "The final evaluation directory already contains data; the final "
                "split must remain untouched during development"
            )
    else:
        decision_path = args.results_dir / "stage2b_decision.json"
        if not decision_path.exists():
            raise RuntimeError("Final evaluation requires stage2b_decision.json")
        decision = read_json(decision_path)
        if decision.get("decision") != "FULL_EVALUATION_GO":
            raise RuntimeError(
                f"Final evaluation is blocked: development decision is "
                f"{decision.get('decision')!r}"
            )

    references, competitors = phase_records(registry, allocations_dir, args.phase)
    competitors.sort(
        key=lambda r: (r["regime"], r["budget_fraction"], r["method_kind"], r["method"])
    )
    records = references + competitors
    print(
        f"Phase {args.phase}: {len(references)} reference points and "
        f"{len(competitors)} matched-budget allocations."
    )

    splits = {
        domain: load_heldout_split(args.results_dir / "splits", args.phase, domain)
        for domain in STAGE2B_DOMAINS
    }
    split_hashes = {
        domain: array_sha256(item.input_ids) for domain, item in splits.items()
    }

    runtime = resolve_runtime(args.device, args.dtype)
    print(f"Loading {args.model} on {runtime.description}...", flush=True)
    bundle = load_model_and_tokenizer(
        checkpoint=args.model,
        runtime=runtime,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    if bundle.resolved_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            f"Resolved model revision {bundle.resolved_revision} does not match the "
            f"pinned {EXPECTED_MODEL_REVISION}"
        )
    layer_specs = discover_moe_layers(bundle.model)
    if len(layer_specs) != NUM_MOE_LAYERS:
        raise RuntimeError("Unexpected MoE layer count")
    manager = MixedPrecisionExpertManager(bundle, layer_specs)
    with np.load(
        args.results_dir / "calibration" / "memory_matrix.npz", allow_pickle=False
    ) as memory_arrays:
        expected_shapes = [
            [tuple(shape) for shape in memory_arrays["tensor_shapes"].tolist()]
        ] * NUM_MOE_LAYERS
    verify_layout_against_memory_shapes(manager, expected_shapes)
    manager.verify_clean()
    print("Clean BF16 expert state snapshot and fingerprints established.")

    run_fingerprint = evaluation_run_fingerprint(
        bundle, registry, split_hashes, args.phase, args.batch_size, determinism
    )
    print(f"Run fingerprint: {run_fingerprint}")

    reproduction = run_repeated_baseline_check(bundle, splits, args.batch_size)
    atomic_write_json(phase_dir / "baseline_reproduction.json", reproduction)
    print("Repeated clean-baseline evaluation is bitwise reproducible.")

    run_config = {
        "stage": "stage2b_robust_specialist_preservation",
        "phase": args.phase,
        "model": args.model,
        "requested_model_revision": args.model_revision,
        "resolved_model_revision": bundle.resolved_revision,
        "device": str(runtime.device),
        "device_description": runtime.description,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic_settings": determinism,
        "registry_sha256": registry["registry_sha256"],
        "split_input_hashes": split_hashes,
        "run_fingerprint": run_fingerprint,
        "num_records": len(records),
        "package_versions": package_versions(),
        "expert_state_clean_sha256": manager.expert_state_sha256(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(phase_dir / "run_config.json", run_config)

    diagnostics = evaluate_allocation_records(
        bundle,
        manager,
        records,
        splits,
        split_hashes,
        losses_dir,
        run_fingerprint,
        args.batch_size,
        resume=args.resume,
    )
    manager.verify_clean()
    atomic_write_json(phase_dir / "evaluation_diagnostics.json", diagnostics)
    print(f"Phase {args.phase} evaluation complete: {losses_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
