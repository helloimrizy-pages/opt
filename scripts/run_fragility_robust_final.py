#!/usr/bin/env python3
"""Stage 2C seed-44 final confirmation (CUDA production; requires GO).

Runs only after a FINAL_CONFIRMATION_GO development decision and a passing
independent audit. Evaluates the frozen allocations of the authorized
regime(s) at all four budgets on the untouched seed-44 final split. MILPs are
never resolved and allocations are never altered here.
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
    stage2c_phase_records,
    stage2c_run_fingerprint,
    verify_preregistration_unchanged,
)
from expert_analysis.fragility_optimization import load_frozen_stage2c_registry
from expert_analysis.fragility_reporting import (
    analyze_stage2c_phase,
    create_stage2c_figures,
    phase_dir_name,
    write_stage2c_final_decision,
    write_stage2c_final_summary,
    write_stage2c_phase_outputs,
    write_stage2c_summary,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.heldout_splits import load_heldout_split
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
    parser.add_argument(
        "--allow-git-mismatch",
        action="store_true",
        help=(
            "Proceed although the current git state differs from the development "
            "run. Only formatting-level changes that provably do not alter "
            "numerical results are permitted, and the mismatch is logged."
        ),
    )
    args = parser.parse_args()

    verify_stage2c_upstream(args.results_root)
    prereg_sha = verify_preregistration_unchanged(args.results_dir)
    print(f"Preregistration unchanged: {prereg_sha[:16]}...")

    decision_path = args.results_dir / "stage2c_decision.json"
    if not decision_path.is_file():
        raise RuntimeError("Final evaluation requires stage2c_decision.json")
    decision = read_json(decision_path)
    if decision.get("decision") != "FINAL_CONFIRMATION_GO":
        raise RuntimeError(
            f"Final evaluation is blocked: development decision is "
            f"{decision.get('decision')!r}"
        )
    if decision.get("preregistration_sha256") != prereg_sha:
        raise RuntimeError(
            "The development decision was made under a different preregistration"
        )
    authorized_regimes = list(
        decision["development_decision"]["authorized_regimes"]
    )
    if not authorized_regimes:
        raise RuntimeError("No regime is authorized for final evaluation")
    print(f"Authorized regimes: {authorized_regimes}")

    audit_path = args.results_dir / "audits" / "independent_audit.json"
    if not audit_path.is_file():
        raise RuntimeError(
            "The independent development audit has not run; execute "
            "scripts/audit_fragility_robust_preservation.py first"
        )
    audit = read_json(audit_path)
    if audit.get("passed") is not True or not audit.get("development_results_audited"):
        raise RuntimeError(
            "The independent development audit did not pass; final evaluation "
            "is blocked"
        )
    print("Independent development audit verified as passing.")

    # Seed-44 inputs must be unchanged; existing Stage 2C final outputs are
    # only acceptable as resumable checkpoints of this authorized evaluation.
    verify_seed44_untouched(
        args.results_root, args.results_dir, allow_authorized_final=True
    )
    phase_dir = args.results_dir / phase_dir_name("final")
    losses_dir = phase_dir / "losses"
    if losses_dir.exists() and any(losses_dir.iterdir()) and not args.resume:
        raise RuntimeError(
            "Seed-44 outputs already exist and --no-resume was requested; "
            "refusing to overwrite final evidence"
        )

    development_config = read_json(
        args.results_dir / phase_dir_name("development") / "run_config.json"
    )
    current_git = git_state()
    git_mismatch = (
        development_config.get("git_commit") != current_git["git_commit"]
        or development_config.get("git_dirty") != current_git["git_dirty"]
    )
    if git_mismatch and not args.allow_git_mismatch:
        raise RuntimeError(
            "The repository state changed after the seed-45 development run "
            f"(development {development_config.get('git_commit')!r} dirty="
            f"{development_config.get('git_dirty')}, current "
            f"{current_git['git_commit']!r} dirty={current_git['git_dirty']}). "
            "Scientific-code changes invalidate seed-44 confirmation unless "
            "separately preregistered; pass --allow-git-mismatch only for "
            "logged formatting-level changes."
        )

    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)

    allocations_dir = args.results_dir / "allocations"
    stage2b_allocations_dir = args.stage2b_dir / "allocations"
    registry = load_frozen_stage2c_registry(allocations_dir, stage2b_allocations_dir)
    if registry["registry_sha256"] != decision["registry_sha256"]:
        raise RuntimeError("The allocation registry changed after the development run")
    fragility_record = load_frozen_fragility(args.results_dir / "calibration")
    if registry["fragility_sha256"] != fragility_record["fragility_sha256"]:
        raise RuntimeError("Registry and frozen fragility record disagree")

    splits = {
        domain: load_heldout_split(args.stage2b_dir / "splits", "final", domain)
        for domain in STAGE2B_DOMAINS
    }
    split_hashes = {
        domain: array_sha256(item.input_ids) for domain, item in splits.items()
    }
    print("Untouched seed-44 final split loaded and hash-verified.")

    references, competitors = stage2c_phase_records(
        registry, allocations_dir, stage2b_allocations_dir, "final",
        authorized_regimes,
    )
    records = references + competitors
    print(
        f"Final: {len(references)} reference points and {len(competitors)} "
        f"matched-budget allocations across all budgets of {authorized_regimes}."
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

    run_fingerprint = stage2c_run_fingerprint(
        bundle, registry, prereg_sha, split_hashes, "final",
        args.batch_size, determinism,
    )
    print(f"Run fingerprint: {run_fingerprint}")
    reproduction = run_repeated_baseline_check(bundle, splits, args.batch_size)
    atomic_write_json(phase_dir / "baseline_reproduction.json", reproduction)
    print("Repeated clean-baseline evaluation is bitwise reproducible.")

    run_config = {
        "stage": "stage2c_fragility_robust_preservation",
        "phase": "final",
        "authorized_regimes": authorized_regimes,
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
        "development_git_commit": development_config.get("git_commit"),
        "git_mismatch_allowed": bool(git_mismatch and args.allow_git_mismatch),
        **current_git,
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
    print(f"Final evaluation complete: {losses_dir}")

    analysis = analyze_stage2c_phase(
        "final",
        args.results_dir,
        stage2b_allocations_dir,
        run_fingerprint,
        authorized_regimes=authorized_regimes,
    )
    paths = write_stage2c_phase_outputs(analysis, phase_dir)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    write_stage2c_final_decision(analysis, args.results_dir)
    print(f"Final decision: {analysis['final_decision']['decision']}")
    write_stage2c_final_summary(args.results_dir, analysis)
    if not args.skip_figures:
        figures = create_stage2c_figures(
            args.results_dir, stage2b_allocations_dir,
            args.results_dir / "figures", analysis,
        )
        print(f"Created {len(figures)} figure files.")
    write_stage2c_summary(args.results_dir)
    print(
        "Run scripts/audit_fragility_robust_preservation.py to independently "
        "audit the final results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
