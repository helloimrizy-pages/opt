#!/usr/bin/env python3
"""Stage 3D correctness harness: prove the measurement setup before any sweep.

Four checks, in order, all of which must pass before Sweep A runs.

1. Strict determinism, eval mode, fixed seed, fixed data order.
2. The BF16 baseline evaluated twice must agree bitwise. Reports the per-domain
   losses and token counts.
3. One expert quantized to 4 bits, evaluated, restored from a cached copy of
   its original tensors, and evaluated again. The restored losses must equal the
   baseline exactly. The model is never reloaded from disk between runs.
4. Two Stage 1 single-expert measurements reproduced on the frozen controlled
   100-example-per-domain set that Stage 1 used. Any disagreement stops the run
   and reports its size.

Also reports peak device memory and wall clock for one evaluation pass.
"""
from __future__ import annotations

import os

# Strict determinism requires this before the first cuBLAS call.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import (
    EXPECTED_MODEL_REVISION,
    array_sha256,
    load_controlled_source,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    verify_layout_against_memory_shapes,
)
from expert_analysis.quantization import ExpertWeightLayout, ReversibleExpertQuantization
from expert_analysis.specialist_preservation import (
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
)
from expert_analysis.stage3d_diagnostics import (
    STAGE3D_RESULTS_DIRNAME,
    STAGE3D_STAGE,
    domain_loss,
    evaluate_all_domains,
    evaluation_split_hashes,
    git_commit,
    load_evaluation_set,
    load_frozen_memory_matrix,
    peak_memory_bytes,
    reset_peak_memory,
    run_config_fingerprint,
)

# The two Stage 1 interventions with the largest measured 4-bit effect. Both
# are single experts quantized alone, which is exactly what check 4 repeats.
DEFAULT_STAGE1_EXPERTS = "11:27,12:43"
REQUIRED_DECIMAL_PLACES = 6
STAGE1_BIT_WIDTH = 4


def parse_experts(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        layer_text, _, expert_text = item.partition(":")
        pairs.append((int(layer_text), int(expert_text)))
    if len(pairs) < 2:
        raise ValueError("At least two Stage 1 experts are required for check 4")
    return pairs


def stage1_reference_rows(path: Path) -> dict[tuple[int, int, str], dict[str, float]]:
    reference: dict[tuple[int, int, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["bit_width"]) != STAGE1_BIT_WIDTH:
                continue
            key = (int(row["layer"]), int(row["expert_id"]), row["domain"])
            reference[key] = {
                "baseline_nll": float(row["baseline_nll"]),
                "quantized_nll": float(row["quantized_nll"]),
                "delta_nll": float(row["delta_nll"]),
                "examples": float(row["examples"]),
                "evaluated_tokens": float(row["evaluated_tokens"]),
            }
    if not reference:
        raise RuntimeError(f"No {STAGE1_BIT_WIDTH}-bit Stage 1 rows found in {path}")
    return reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--controlled-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--stage1-results",
        type=Path,
        default=Path("results/expert_quantization_pilot/quantization_pilot_results.csv"),
    )
    parser.add_argument("--stage1-experts", default=DEFAULT_STAGE1_EXPERTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--determinism-warn-only",
        action="store_true",
        help=(
            "Permit warn-only deterministic algorithms. Check 2 still requires "
            "bitwise repeated evaluation and still stops the run on a mismatch."
        ),
    )
    parser.add_argument(
        "--stage1-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum absolute NLL difference accepted in check 4. The default of "
            "zero requires exact agreement, which holds only on the pinned "
            "environment Stage 1 ran on."
        ),
    )
    args = parser.parse_args()

    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)
    stage1_experts = parse_experts(args.stage1_experts)
    stage1_reference = stage1_reference_rows(args.stage1_results)

    examples = load_evaluation_set(args.stage2b_dir, args.stage2c_dir)
    split_hashes = evaluation_split_hashes(examples)
    controlled = load_controlled_source(args.controlled_dir)

    runtime = resolve_runtime(args.device, args.dtype)
    print(f"Loading {args.model} on {runtime.description}...", flush=True)
    load_started = time.monotonic()
    bundle = load_model_and_tokenizer(
        checkpoint=args.model,
        runtime=runtime,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    load_seconds = time.monotonic() - load_started
    if bundle.resolved_revision != EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            f"Resolved model revision {bundle.resolved_revision} does not match "
            f"the pinned {EXPECTED_MODEL_REVISION}"
        )
    if bundle.model.training:
        raise RuntimeError("The model is in training mode; dropout would be active")
    layer_specs = discover_moe_layers(bundle.model)
    if len(layer_specs) != NUM_MOE_LAYERS:
        raise RuntimeError(f"Discovered {len(layer_specs)} MoE layers, expected 16")

    memory = load_frozen_memory_matrix(args.stage2b_dir)
    manager = MixedPrecisionExpertManager(bundle, layer_specs)
    verify_layout_against_memory_shapes(
        manager, [memory.tensor_shapes[0]] * NUM_MOE_LAYERS
    )
    manager.verify_clean()
    print("Clean BF16 expert snapshot established and verified.")

    config_sha256 = run_config_fingerprint(
        args.model,
        bundle.resolved_revision,
        str(runtime.dtype).replace("torch.", ""),
        args.batch_size,
        memory.group_size,
        split_hashes,
        determinism,
    )
    report: dict[str, Any] = {
        "stage": STAGE3D_STAGE,
        "check": "correctness_harness",
        "config_sha256": config_sha256,
        "model": args.model,
        "resolved_model_revision": bundle.resolved_revision,
        "device": str(runtime.device),
        "device_description": runtime.description,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model_in_eval_mode": not bundle.model.training,
        "deterministic_settings": determinism,
        "evaluation_split_input_hashes": split_hashes,
        "model_load_seconds": load_seconds,
        **git_commit(),
        "package_versions": package_versions(),
    }

    # ---- Check 2: repeated BF16 baseline ---------------------------------
    print("\n[check 2] repeated BF16 baseline", flush=True)
    reset_peak_memory(runtime.device.type)
    pass_started = time.monotonic()
    baseline_statistics, evaluation_seconds = evaluate_all_domains(
        bundle, examples, args.batch_size
    )
    single_pass_seconds = time.monotonic() - pass_started
    single_pass_peak = peak_memory_bytes(runtime.device.type)
    second_statistics, _ = evaluate_all_domains(bundle, examples, args.batch_size)
    baseline_loss = {
        domain: domain_loss(statistics)
        for domain, statistics in baseline_statistics.items()
    }
    second_loss = {
        domain: domain_loss(statistics)
        for domain, statistics in second_statistics.items()
    }
    # A value agrees to N decimal places when it rounds to the same digits, so
    # the difference must be under half a unit in the last kept place.
    required_difference = 0.5 * 10.0**-REQUIRED_DECIMAL_PLACES
    bitwise = {}
    for domain in STAGE2B_DOMAINS:
        difference = abs(baseline_loss[domain] - second_loss[domain])
        bitwise[domain] = {
            "bitwise_identical": bool(
                np.array_equal(
                    baseline_statistics[domain].loss_sums,
                    second_statistics[domain].loss_sums,
                )
                and np.array_equal(
                    baseline_statistics[domain].token_counts,
                    second_statistics[domain].token_counts,
                )
            ),
            "absolute_loss_difference": float(difference),
        }
        if difference >= required_difference:
            raise RuntimeError(
                f"Repeated baseline for {domain} differs by {difference:.3e}, "
                f"which does not agree to {REQUIRED_DECIMAL_PLACES} decimal "
                "places. Find the nondeterminism before continuing."
            )
    not_bitwise = [d for d in STAGE2B_DOMAINS if not bitwise[d]["bitwise_identical"]]
    if not_bitwise:
        raise RuntimeError(
            "Repeated clean-baseline evaluation is not bitwise reproducible for: "
            + ", ".join(not_bitwise)
            + "; strict determinism is not achieved. Stopping as preregistered."
        )
    report["check_2_repeated_baseline"] = {
        "passed": True,
        "bitwise_identical_by_domain": bitwise,
        "loss_by_domain": baseline_loss,
        "repeat_loss_by_domain": second_loss,
        "required_decimal_places": REQUIRED_DECIMAL_PLACES,
        "required_absolute_difference": required_difference,
        "tokens_by_domain": {
            domain: int(statistics.token_counts.sum())
            for domain, statistics in baseline_statistics.items()
        },
        "examples_by_domain": {
            domain: int(len(statistics.loss_sums))
            for domain, statistics in baseline_statistics.items()
        },
        "single_pass_seconds": single_pass_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_memory_bytes": single_pass_peak,
    }
    for domain in STAGE2B_DOMAINS:
        statistics = baseline_statistics[domain]
        print(
            f"  {domain:10s} loss={baseline_loss[domain]:.10f} "
            f"tokens={int(statistics.token_counts.sum())} "
            f"examples={len(statistics.loss_sums)}"
        )
    print(
        f"  one evaluation pass: {single_pass_seconds:.1f} s"
        + (
            f", peak device memory {single_pass_peak / 2**30:.2f} GiB"
            if single_pass_peak
            else ""
        )
    )

    # ---- Check 3: quantize, restore, re-evaluate --------------------------
    print("\n[check 3] quantize layer 0 expert 0 to 4 bits, restore, re-evaluate")
    layout = ExpertWeightLayout(layer_specs[0])
    with ReversibleExpertQuantization(layout, 0, bits=4) as quantization:
        quantized_statistics, _ = evaluate_all_domains(bundle, examples, args.batch_size)
        quantization_diagnostics = quantization.diagnostics()
    if not quantization.restoration_verified:
        raise RuntimeError("The reversible quantization context did not verify restore")
    restored_statistics, _ = evaluate_all_domains(bundle, examples, args.batch_size)
    mismatched = [
        domain
        for domain in STAGE2B_DOMAINS
        if not np.array_equal(
            restored_statistics[domain].loss_sums, baseline_statistics[domain].loss_sums
        )
    ]
    if mismatched:
        raise RuntimeError(
            "Restored losses differ from the BF16 baseline for: "
            + ", ".join(mismatched)
            + "; the restore path is not clean"
        )
    manager.verify_clean()
    report["check_3_restore"] = {
        "passed": True,
        "layer": layer_specs[0].model_layer_index,
        "expert": 0,
        "bits": 4,
        "quantized_loss_by_domain": {
            domain: domain_loss(statistics)
            for domain, statistics in quantized_statistics.items()
        },
        "restored_loss_by_domain": {
            domain: domain_loss(statistics)
            for domain, statistics in restored_statistics.items()
        },
        "restored_matches_baseline_bitwise": True,
        "model_reloaded_from_disk": False,
        "quantization_diagnostics": quantization_diagnostics,
    }
    for domain in STAGE2B_DOMAINS:
        delta = (
            report["check_3_restore"]["quantized_loss_by_domain"][domain]
            - baseline_loss[domain]
        )
        print(f"  {domain:10s} delta while quantized = {delta:+.10f}, restored exactly")

    # ---- Check 4: reproduce two Stage 1 measurements ----------------------
    print("\n[check 4] reproduce two Stage 1 single-expert measurements")
    controlled_examples = {
        domain: controlled.prepared[domain] for domain in STAGE2B_DOMAINS
    }
    controlled_baseline, _ = evaluate_all_domains(
        bundle, controlled_examples, args.batch_size
    )
    controlled_baseline_loss = {
        domain: domain_loss(statistics)
        for domain, statistics in controlled_baseline.items()
    }
    stage1_rows: list[dict[str, Any]] = []
    largest_difference = 0.0
    for layer_index, expert_id in stage1_experts:
        spec = next(
            (item for item in layer_specs if item.model_layer_index == layer_index),
            None,
        )
        if spec is None:
            raise RuntimeError(f"No MoE layer with model index {layer_index}")
        expert_layout = ExpertWeightLayout(spec)
        with ReversibleExpertQuantization(
            expert_layout, expert_id, bits=STAGE1_BIT_WIDTH
        ):
            quantized, _ = evaluate_all_domains(
                bundle, controlled_examples, args.batch_size
            )
        for domain in STAGE2B_DOMAINS:
            key = (layer_index, expert_id, domain)
            if key not in stage1_reference:
                raise RuntimeError(f"Stage 1 has no row for {key}")
            expected = stage1_reference[key]
            observed_baseline = controlled_baseline_loss[domain]
            observed_quantized = domain_loss(quantized[domain])
            row = {
                "layer": layer_index,
                "expert": expert_id,
                "domain": domain,
                "stage1_baseline_nll": expected["baseline_nll"],
                "observed_baseline_nll": observed_baseline,
                "baseline_difference": observed_baseline - expected["baseline_nll"],
                "stage1_quantized_nll": expected["quantized_nll"],
                "observed_quantized_nll": observed_quantized,
                "quantized_difference": observed_quantized - expected["quantized_nll"],
                "stage1_delta_nll": expected["delta_nll"],
                "observed_delta_nll": observed_quantized - observed_baseline,
                "delta_difference": (observed_quantized - observed_baseline)
                - expected["delta_nll"],
            }
            largest_difference = max(
                largest_difference,
                abs(row["baseline_difference"]),
                abs(row["quantized_difference"]),
            )
            stage1_rows.append(row)
            print(
                f"  L{layer_index} E{expert_id} {domain:10s} "
                f"baseline {row['baseline_difference']:+.3e} "
                f"quantized {row['quantized_difference']:+.3e}"
            )
    agrees = largest_difference <= args.stage1_tolerance
    report["check_4_stage1_reproduction"] = {
        "passed": agrees,
        "bit_width": STAGE1_BIT_WIDTH,
        "experts": [{"layer": l, "expert": e} for l, e in stage1_experts],
        "evaluated_on": "frozen controlled 100-example-per-domain set",
        "controlled_input_hashes": {
            domain: array_sha256(controlled.prepared[domain].input_ids)
            for domain in STAGE2B_DOMAINS
        },
        "tolerance": args.stage1_tolerance,
        "largest_absolute_difference": largest_difference,
        "rows": stage1_rows,
    }
    manager.verify_clean()

    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    harness_path = args.results_dir / "harness" / "harness_report.json"
    atomic_write_json(harness_path, report)
    print(f"\nHarness report: {harness_path}")

    if not agrees:
        raise SystemExit(
            "\nSTOP. Check 4 does not agree with Stage 1. Largest absolute NLL "
            f"difference: {largest_difference:.6e} (tolerance "
            f"{args.stage1_tolerance:.6e}). The full per-domain table is in the "
            "harness report. Report this size before running any sweep."
        )
    print("All four checks passed. Sweep A may run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
