#!/usr/bin/env python3
"""Stage 3 additivity gate on frozen calibration probes (CUDA production).

Evaluates every frozen 20%-budget allocation of both regimes on the frozen
calibration subsets, compares the measured domain delta NLL with the additive
prediction from the frozen damage matrix, and applies the two preregistered
additivity gates. Only passing regimes are authorized for the seed-46
development evaluation; if no regime passes, the stage decision is
MEASURED_DAMAGE_NO_GO and neither seed 46 nor seed 44 is ever evaluated.
"""
from __future__ import annotations

import os

# Strict determinism requires this before the first cuBLAS call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import (
    EXPECTED_MODEL_REVISION,
    array_sha256,
    load_controlled_source,
)
from expert_analysis.fragility import (
    calibration_subset_inputs,
    load_frozen_stage2b_scores,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions
from expert_analysis.measured_damage import load_frozen_damage
from expert_analysis.measured_damage_evaluation import (
    stage3_phase_dir_name,
    stage3_phase_records,
    stage3_run_fingerprint,
    verify_stage3_preregistration_unchanged,
)
from expert_analysis.measured_damage_optimization import load_frozen_stage3_registry
from expert_analysis.measured_damage_reporting import (
    analyze_stage3_additivity,
    write_additivity_outputs,
    write_stage3_additivity_decision,
    write_stage3_summary,
)
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    evaluate_allocation_records,
    run_repeated_baseline_check,
    verify_layout_against_memory_shapes,
)
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS, STAGE2B_DOMAINS
from expert_analysis.stage3_preflight import (
    verify_seed44_untouched_stage3,
    verify_stage3_upstream,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--stage2b-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--stage2c-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/measured_damage_preservation"),
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
    args = parser.parse_args()

    verify_stage3_upstream(args.results_root)
    verify_seed44_untouched_stage3(args.results_root, args.results_dir)
    prereg_sha = verify_stage3_preregistration_unchanged(args.results_dir)
    print(f"Preregistration unchanged: {prereg_sha[:16]}...")
    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)

    allocations_dir = args.results_dir / "allocations"
    stage2b_allocations_dir = args.stage2b_dir / "allocations"
    stage2c_allocations_dir = args.stage2c_dir / "allocations"
    registry = load_frozen_stage3_registry(
        allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir
    )
    print(f"Frozen Stage 3 registry verified: {registry['registry_sha256'][:16]}...")
    damage_record, _ = load_frozen_damage(args.results_dir / "damage")
    if registry["damage_sha256"] != damage_record["damage_sha256"]:
        raise RuntimeError("Registry and frozen damage matrix disagree")

    scores = load_frozen_stage2b_scores(args.stage2b_dir)
    source = load_controlled_source(args.source_dir)
    subsets = calibration_subset_inputs(source, scores)
    subset_hashes = {
        domain: array_sha256(subsets[domain].input_ids) for domain in STAGE2B_DOMAINS
    }
    for domain in STAGE2B_DOMAINS:
        recorded = damage_record["calibration_subset_hashes"][domain][
            "input_ids_sha256"
        ]
        if subset_hashes[domain] != recorded:
            raise RuntimeError(
                f"Calibration subset for {domain} does not match the frozen "
                "damage matrix"
            )
    print("Calibration subsets verified against the frozen damage matrix.")

    _, probes = stage3_phase_records(
        registry, allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir,
        "additivity",
    )
    print(f"Additivity: {len(probes)} frozen 20%-budget probe allocations.")

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

    run_fingerprint = stage3_run_fingerprint(
        bundle, registry, prereg_sha, subset_hashes, "additivity",
        args.batch_size, determinism,
    )
    print(f"Run fingerprint: {run_fingerprint}")

    phase_dir = args.results_dir / stage3_phase_dir_name("additivity")
    losses_dir = phase_dir / "losses"
    reproduction = run_repeated_baseline_check(bundle, subsets, args.batch_size)
    atomic_write_json(phase_dir / "baseline_reproduction.json", reproduction)
    print("Repeated clean-baseline evaluation is bitwise reproducible.")

    atomic_write_json(
        phase_dir / "run_config.json",
        {
            "stage": "stage3_measured_damage_preservation",
            "phase": "additivity",
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
            "damage_sha256": damage_record["damage_sha256"],
            "preregistration_sha256": prereg_sha,
            "calibration_subset_hashes": subset_hashes,
            "run_fingerprint": run_fingerprint,
            "num_probes": len(probes),
            "package_versions": package_versions(),
            "expert_state_clean_sha256": manager.expert_state_sha256(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    diagnostics = evaluate_allocation_records(
        bundle,
        manager,
        probes,
        subsets,
        subset_hashes,
        losses_dir,
        run_fingerprint,
        args.batch_size,
        resume=args.resume,
    )
    manager.verify_clean()
    atomic_write_json(phase_dir / "evaluation_diagnostics.json", diagnostics)
    print(f"Probe evaluation complete: {losses_dir}")

    report = analyze_stage3_additivity(
        args.results_dir,
        stage2b_allocations_dir,
        stage2c_allocations_dir,
        run_fingerprint,
    )
    report_path = write_additivity_outputs(report, args.results_dir)
    print(f"Additivity report: {report_path}")
    decision = report["additivity_decision"]
    for regime, gates in report["gates_by_regime"].items():
        print(
            f"  {regime}: gate_add_1="
            f"{'PASS' if gates['gate_add_1']['passed'] else 'FAIL'}, "
            f"gate_add_2={'PASS' if gates['gate_add_2']['passed'] else 'FAIL'}"
        )
    print(
        f"Additivity decision: {decision['decision']} "
        f"(authorized regimes: {decision['authorized_regimes']})"
    )
    if not decision["authorized_regimes"]:
        decision_path = write_stage3_additivity_decision(
            report, args.results_dir, prereg_sha
        )
        print(
            f"MEASURED_DAMAGE_NO_GO written to {decision_path}: the additive "
            "damage model failed on calibration probes. Stop here; preserve "
            "the negative result; never evaluate seed 46 or seed 44 under "
            "this preregistration."
        )
    else:
        print(
            "Run scripts/audit_measured_damage_preservation.py, then "
            "scripts/run_measured_damage_development.py for the authorized "
            "regime(s)."
        )
    write_stage3_summary(args.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
