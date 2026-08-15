#!/usr/bin/env python3
"""Stage 2C seed-45 development evaluation at the 20% budget (CUDA production).

Evaluates the frozen Fragility-Robust allocations against every frozen Stage
2B comparator on the new seed-45 development split, applies the five
preregistered gates, and writes FINAL_CONFIRMATION_GO or
FRAGILITY_ROBUST_NO_GO. Seed 44 is never touched here.
"""
from __future__ import annotations

import os

# Strict determinism requires this before the first cuBLAS call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import EXPECTED_MODEL_REVISION, array_sha256
from expert_analysis.fragility import load_frozen_fragility
from expert_analysis.fragility_evaluation import (
    load_stage2c_development_split,
    stage2c_phase_records,
    stage2c_run_fingerprint,
    verify_preregistration_unchanged,
)
from expert_analysis.fragility_optimization import load_frozen_stage2c_registry
from expert_analysis.fragility_reporting import (
    analyze_stage2c_phase,
    create_stage2c_figures,
    phase_dir_name,
    write_stage2c_development_decision,
    write_stage2c_phase_outputs,
    write_stage2c_summary,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions, read_json
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    evaluate_allocation_records,
    run_repeated_baseline_check,
    verify_layout_against_memory_shapes,
)
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS, STAGE2B_DOMAINS
from expert_analysis.stage2c_preflight import (
    verify_seed44_untouched,
    verify_stage2c_upstream,
)


def git_state() -> dict[str, str | bool]:
    def run(*command: str) -> str:
        return subprocess.run(
            command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {"git_commit": commit, "git_dirty": dirty}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2b-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
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
    parser.add_argument("--determinism-warn-only", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    verify_stage2c_upstream(args.results_root)
    verify_seed44_untouched(args.results_root, args.results_dir)
    prereg_sha = verify_preregistration_unchanged(args.results_dir)
    print(f"Preregistration unchanged: {prereg_sha[:16]}...")
    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)

    allocations_dir = args.results_dir / "allocations"
    stage2b_allocations_dir = args.stage2b_dir / "allocations"
    registry = load_frozen_stage2c_registry(allocations_dir, stage2b_allocations_dir)
    print(f"Frozen Stage 2C registry verified: {registry['registry_sha256'][:16]}...")
    fragility_record = load_frozen_fragility(args.results_dir / "calibration")
    if registry["fragility_sha256"] != fragility_record["fragility_sha256"]:
        raise RuntimeError("Registry and frozen fragility record disagree")
    preregistration = read_json(args.results_dir / "stage2c_preregistration.json")
    if preregistration["allocation_registry_sha256"] != registry["registry_sha256"]:
        raise RuntimeError("Preregistration references a different allocation registry")

    splits = {
        domain: load_stage2c_development_split(args.results_dir / "splits", domain)
        for domain in STAGE2B_DOMAINS
    }
    split_hashes = {
        domain: array_sha256(item.input_ids) for domain, item in splits.items()
    }
    if preregistration["development_split_input_hashes"] != split_hashes:
        raise RuntimeError("Seed-45 split hashes do not match the preregistration")
    print("Seed-45 development split verified against the preregistration.")

    references, competitors = stage2c_phase_records(
        registry, allocations_dir, stage2b_allocations_dir, "development"
    )
    records = references + competitors
    print(
        f"Development: {len(references)} reference points and "
        f"{len(competitors)} matched-budget allocations at the 20% budget."
    )

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
        args.stage2b_dir / "calibration" / "memory_matrix.npz", allow_pickle=False
    ) as memory_arrays:
        expected_shapes = [
            [tuple(shape) for shape in memory_arrays["tensor_shapes"].tolist()]
        ] * NUM_MOE_LAYERS
    verify_layout_against_memory_shapes(manager, expected_shapes)
    manager.verify_clean()
    print("Clean BF16 expert state snapshot and fingerprints established.")

    run_fingerprint = stage2c_run_fingerprint(
        bundle, registry, prereg_sha, split_hashes, "development",
        args.batch_size, determinism,
    )
    print(f"Run fingerprint: {run_fingerprint}")

    phase_dir = args.results_dir / phase_dir_name("development")
    losses_dir = phase_dir / "losses"
    reproduction = run_repeated_baseline_check(bundle, splits, args.batch_size)
    atomic_write_json(phase_dir / "baseline_reproduction.json", reproduction)
    print("Repeated clean-baseline evaluation is bitwise reproducible.")

    run_config = {
        "stage": "stage2c_fragility_robust_preservation",
        "phase": "development",
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
        "fragility_sha256": fragility_record["fragility_sha256"],
        "preregistration_sha256": prereg_sha,
        "split_input_hashes": split_hashes,
        "run_fingerprint": run_fingerprint,
        "num_records": len(records),
        "package_versions": package_versions(),
        "expert_state_clean_sha256": manager.expert_state_sha256(),
        **git_state(),
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
    print(f"Development evaluation complete: {losses_dir}")

    analysis = analyze_stage2c_phase(
        "development",
        args.results_dir,
        stage2b_allocations_dir,
        run_fingerprint,
    )
    paths = write_stage2c_phase_outputs(analysis, phase_dir)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    decision_path = write_stage2c_development_decision(
        analysis, args.results_dir, prereg_sha
    )
    decision = analysis["development_decision"]["decision"]
    print(f"Stage 2C development decision: {decision} ({decision_path})")
    for regime, gates in analysis["development_gates"].items():
        summary = ", ".join(
            f"{key}={'PASS' if gates[key]['passed'] else 'FAIL'}"
            for key in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e")
        )
        print(f"  {regime}: {summary}")
    if not args.skip_figures:
        figures = create_stage2c_figures(
            args.results_dir, stage2b_allocations_dir,
            args.results_dir / "figures", analysis,
        )
        print(f"Created {len(figures)} figure files.")
    write_stage2c_summary(args.results_dir)
    if decision != "FINAL_CONFIRMATION_GO":
        print(
            "FRAGILITY_ROBUST_NO_GO: stop here. Preserve all results; do not "
            "modify the objective, budgets, or fragility weighting; do not "
            "inspect seed 44. This is the final optimization hypothesis on "
            "this branch."
        )
    else:
        authorized = analysis["development_decision"]["authorized_regimes"]
        print(
            f"FINAL_CONFIRMATION_GO: regimes {authorized} are authorized for the "
            "seed-44 final confirmation. Run the independent audit before "
            "scripts/run_fragility_robust_final.py."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
