#!/usr/bin/env python3
"""Run one Stage 3D sweep and append one JSONL record per evaluated state.

Sweep A, 36 runs. Twenty random protection sets at 4-bit base, ten of the same
sets at 3-bit base, plus the most-routed, least-routed, and no-protection sets
in each regime, all at the 20% Stage 2C byte budget.

Sweep B, 16 runs. Each layer's 64 experts at 4 bits with every other parameter
at BF16.

Sweep C, 1 run. Every expert at 4 bits with every MoE router weight also at
4 bits. The routers-at-BF16 comparison point is Sweep A's
``a_4to8_no_protection`` run, which is the same expert assignment, so it is not
re-evaluated.

Records are appended and fsynced one at a time, so a crash loses at most the
run in flight. Per-domain losses are also checkpointed, so a resumed run does
not recompute a domain it already finished.
"""
from __future__ import annotations

import os

# Strict determinism requires this before the first cuBLAS call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import EXPECTED_MODEL_REVISION
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions, read_json
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    run_repeated_baseline_check,
    verify_layout_against_memory_shapes,
)
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS, STAGE2B_DOMAINS
from expert_analysis.stage3d_diagnostics import (
    STAGE3D_RESULTS_DIRNAME,
    STAGE3D_STAGE,
    ReversibleRouterQuantization,
    append_run_record,
    build_run_record,
    calibration_routing_counts,
    completed_run_ids,
    domain_loss,
    evaluate_all_domains,
    evaluation_split_hashes,
    git_commit,
    load_evaluation_set,
    load_frozen_memory_matrix,
    peak_memory_bytes,
    reset_peak_memory,
    router_memory_accounting,
    run_config_fingerprint,
    summarize_run_losses,
    sweep_a_protection_sets,
    sweep_b_protection_sets,
    sweep_c_protection_sets,
    sweep_jsonl_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=("a", "b", "c"), required=True)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results") / STAGE3D_RESULTS_DIRNAME
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
    parser.add_argument(
        "--require-harness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse to run unless the correctness harness passed all four checks.",
    )
    args = parser.parse_args()

    prereg_path = REPOSITORY_ROOT / "prereg" / "stage3d.md"
    if not prereg_path.is_file():
        raise RuntimeError(
            f"{prereg_path} is missing; the preregistration must exist before "
            "any sweep runs"
        )

    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)

    registry = read_json(args.results_dir / "allocations" / "allocation_registry.json")
    examples = load_evaluation_set(args.stage2b_dir, args.stage2c_dir)
    split_hashes = evaluation_split_hashes(examples)
    for domain, digest in split_hashes.items():
        if registry["evaluation_set_input_hashes"][domain] != digest:
            raise RuntimeError(
                f"The evaluation set for {domain} does not match the frozen registry"
            )

    memory = load_frozen_memory_matrix(args.stage2b_dir)
    if args.sweep == "a":
        protection_sets = sweep_a_protection_sets(
            memory, calibration_routing_counts(args.stage2b_dir)
        )
    elif args.sweep == "b":
        protection_sets = sweep_b_protection_sets()
    else:
        protection_sets = sweep_c_protection_sets()
    frozen = {entry["run_id"]: entry for entry in registry["entries"]}
    for protection in protection_sets:
        entry = frozen.get(protection.run_id)
        if entry is None:
            raise RuntimeError(f"{protection.run_id} is not in the frozen registry")
        if entry["bits_matrix_sha256"] != protection.bits_sha256:
            raise RuntimeError(
                f"{protection.run_id} does not reproduce its frozen bit assignment"
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
            f"Resolved model revision {bundle.resolved_revision} does not match "
            f"the pinned {EXPECTED_MODEL_REVISION}"
        )
    layer_specs = discover_moe_layers(bundle.model)
    if len(layer_specs) != NUM_MOE_LAYERS:
        raise RuntimeError(f"Discovered {len(layer_specs)} MoE layers, expected 16")
    manager = MixedPrecisionExpertManager(bundle, layer_specs)
    verify_layout_against_memory_shapes(
        manager, [memory.tensor_shapes[0]] * NUM_MOE_LAYERS
    )
    manager.verify_clean()

    config_sha256 = run_config_fingerprint(
        args.model,
        bundle.resolved_revision,
        str(runtime.dtype).replace("torch.", ""),
        args.batch_size,
        memory.group_size,
        split_hashes,
        determinism,
    )
    harness_path = args.results_dir / "harness" / "harness_report.json"
    if args.require_harness:
        if not harness_path.is_file():
            raise RuntimeError(
                f"{harness_path} is missing; run scripts/run_stage3d_harness.py first"
            )
        harness = read_json(harness_path)
        failed = [
            key
            for key in (
                "check_2_repeated_baseline",
                "check_3_restore",
                "check_4_stage1_reproduction",
            )
            if not harness.get(key, {}).get("passed")
        ]
        if failed:
            raise RuntimeError(f"The correctness harness did not pass: {failed}")
        if harness.get("config_sha256") != config_sha256:
            raise RuntimeError(
                "The harness ran under a different configuration than this sweep"
            )

    baseline_path = args.results_dir / "baseline" / "bf16_baseline.json"
    baseline_losses_dir = args.results_dir / "baseline" / "losses"
    baseline_metadata = {"config_sha256": config_sha256, "state": "bf16_baseline"}
    if baseline_path.is_file() and args.resume:
        baseline = read_json(baseline_path)
        if baseline["config_sha256"] != config_sha256:
            raise RuntimeError(
                "The stored BF16 baseline was measured under a different "
                "configuration; delete it or fix the configuration"
            )
        baseline_loss = baseline["loss_by_domain"]
        print(f"Reusing the BF16 baseline from {baseline_path}")
    else:
        repeat = run_repeated_baseline_check(bundle, examples, args.batch_size)
        statistics, seconds = evaluate_all_domains(
            bundle,
            examples,
            args.batch_size,
            baseline_losses_dir,
            baseline_metadata,
            resume=args.resume,
        )
        baseline_loss = {
            domain: domain_loss(value) for domain, value in statistics.items()
        }
        atomic_write_json(
            baseline_path,
            {
                "stage": STAGE3D_STAGE,
                "config_sha256": config_sha256,
                "loss_by_domain": baseline_loss,
                "tokens_by_domain": {
                    domain: int(value.token_counts.sum())
                    for domain, value in statistics.items()
                },
                "repeated_evaluation_bitwise_identical": repeat["domains"],
                "evaluation_seconds": seconds,
                **git_commit(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    for domain in STAGE2B_DOMAINS:
        print(f"  BF16 {domain:10s} {baseline_loss[domain]:.10f}")

    jsonl_path = sweep_jsonl_path(args.results_dir, args.sweep)
    already_done = completed_run_ids(jsonl_path, config_sha256) if args.resume else set()
    pending = [item for item in protection_sets if item.run_id not in already_done]
    print(
        f"\nSweep {args.sweep.upper()}: {len(protection_sets)} runs, "
        f"{len(already_done)} already recorded, {len(pending)} to evaluate."
    )

    run_config = {
        "stage": STAGE3D_STAGE,
        "sweep": args.sweep,
        "model": args.model,
        "resolved_model_revision": bundle.resolved_revision,
        "device": str(runtime.device),
        "device_description": runtime.description,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "group_size": memory.group_size,
        "deterministic_settings": determinism,
        "config_sha256": config_sha256,
        "registry_sha256": registry["registry_sha256"],
        "evaluation_split_input_hashes": split_hashes,
        "total_runs": len(protection_sets),
        **git_commit(),
        "package_versions": package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.results_dir / f"run_config_sweep_{args.sweep}.json", run_config)

    for index, protection in enumerate(pending, start=1):
        print(
            f"\n[{index}/{len(pending)}] {protection.run_id}: {protection.description}",
            flush=True,
        )
        manager.verify_clean()
        reset_peak_memory(runtime.device.type)
        started = time.monotonic()
        bits = np.asarray(protection.bits, dtype=np.int64)
        is_clean = bool(np.all(bits == 16)) and protection.router_bits is None
        application: dict[str, Any] = {"applied": False}
        router_diagnostics: dict[str, Any] | None = None
        router_memory: dict[str, Any] | None = None
        quantization_seconds = 0.0
        restoration_seconds = 0.0
        try:
            if not np.all(bits == 16):
                quantization_started = time.monotonic()
                application = manager.apply_allocation(
                    bits, int(protection.bits_sha256[:8], 16)
                )
                quantization_seconds = time.monotonic() - quantization_started
            if protection.router_bits is not None:
                router_memory = router_memory_accounting(
                    layer_specs,
                    bundle.model,
                    bits,
                    memory,
                    router_bits=protection.router_bits,
                    group_size=memory.group_size,
                )
                router_context = ReversibleRouterQuantization(
                    layer_specs,
                    bundle.model,
                    bits=protection.router_bits,
                    group_size=memory.group_size,
                )
                with router_context:
                    statistics, evaluation_seconds = evaluate_all_domains(
                        bundle,
                        examples,
                        args.batch_size,
                        args.results_dir / "losses" / protection.run_id,
                        {
                            "config_sha256": config_sha256,
                            "bits_matrix_sha256": protection.bits_sha256,
                            "router_bits": protection.router_bits,
                        },
                        resume=args.resume,
                    )
                router_diagnostics = router_context.diagnostics()
            else:
                statistics, evaluation_seconds = evaluate_all_domains(
                    bundle,
                    examples,
                    args.batch_size,
                    args.results_dir / "losses" / protection.run_id,
                    {
                        "config_sha256": config_sha256,
                        "bits_matrix_sha256": protection.bits_sha256,
                        "router_bits": None,
                    },
                    resume=args.resume,
                )
        finally:
            restoration_started = time.monotonic()
            if is_clean:
                manager.verify_clean()
            else:
                manager.restore_clean()
            restoration_seconds = time.monotonic() - restoration_started

        loss_by_domain = {
            domain: domain_loss(value) for domain, value in statistics.items()
        }
        summary = summarize_run_losses(loss_by_domain, baseline_loss)
        extra: dict[str, Any] = {"application": application}
        if router_diagnostics is not None:
            extra["router_quantization"] = router_diagnostics
        if router_memory is not None:
            extra["router_memory"] = router_memory
        record = build_run_record(
            protection,
            summary,
            {
                domain: int(value.token_counts.sum())
                for domain, value in statistics.items()
            },
            {domain: int(len(value.loss_sums)) for domain, value in statistics.items()},
            config_sha256,
            {
                "quantization_seconds": quantization_seconds,
                "evaluation_seconds": evaluation_seconds,
                "restoration_seconds": restoration_seconds,
                "wall_clock_seconds": time.monotonic() - started,
            },
            peak_memory_bytes(runtime.device.type),
            extra,
        )
        append_run_record(jsonl_path, record)
        print(
            f"    worst domain (relative) {summary['worst_domain_relative']:+.6f} "
            f"on {summary['worst_domain_relative_domain']}, "
            f"worst domain (raw) {summary['worst_domain_raw']:.6f} on "
            f"{summary['worst_domain_raw_domain']}, "
            f"{record['wall_clock_seconds']:.1f} s"
        )

    manager.verify_clean()
    print(f"\nSweep {args.sweep.upper()} complete: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
