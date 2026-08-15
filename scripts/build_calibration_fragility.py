#!/usr/bin/env python3
"""Measure and freeze Stage 2C calibration domain fragility (CUDA production).

Evaluates the clean BF16 model and the uniform expert-only 4-bit and 3-bit
QDQ models on the frozen Stage 2B 25-example/domain calibration subset, runs
every calibration baseline twice with a bitwise reproduction requirement, and
freezes ``calibration_fragility.json`` before any allocation is solved.
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
    canonical_sha256,
    load_controlled_source,
)
from expert_analysis.fragility import (
    STAGE2C_REGIMES,
    build_calibration_fragility_record,
    calibration_subset_inputs,
    compute_regime_fragility,
    load_frozen_stage2b_scores,
    mean_calibration_nll,
    save_calibration_fragility,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions
from expert_analysis.masking import evaluate_next_token_loss
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    verify_layout_against_memory_shapes,
)
from expert_analysis.protection_optimization import uniform_bits_matrix
from expert_analysis.quantization import load_or_compute_loss_checkpoint
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS, STAGE2B_DOMAINS
from expert_analysis.stage2c_preflight import (
    verify_seed44_untouched,
    verify_stage2c_upstream,
)

CALIBRATION_STATES = (
    ("bf16", 16),
    ("uniform4", 4),
    ("uniform3", 3),
)
QDQ_CONFIG = {
    "quantizer": "stage1_symmetric_groupwise_qdq",
    "granularity": "group-wise along the input-feature dimension",
    "symmetric": True,
    "group_size": 128,
    "scale_dtype": "float16",
    "quantized_parameters": "expert FFN weights only (fused gate_up_proj and down_proj)",
    "kept_bf16": (
        "routers, attention, embeddings, normalization, lm_head, and all other "
        "non-expert parameters"
    ),
}


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
    args = parser.parse_args()

    verify_stage2c_upstream(args.results_root)
    verify_seed44_untouched(args.results_root, args.results_dir)
    print("Frozen prior state verified; seed 44 untouched.")
    determinism = configure_strict_determinism(
        args.device, warn_only=args.determinism_warn_only
    )
    set_reproducible_seed(args.seed, deterministic=False)

    scores = load_frozen_stage2b_scores(args.stage2b_dir)
    print(
        "Frozen Stage 2B scores reloaded and hash-verified: "
        f"{scores.calibration_fingerprint[:16]}..."
    )
    source = load_controlled_source(args.source_dir)
    subsets = calibration_subset_inputs(source, scores)
    subset_hashes = {
        domain: {
            "calibration_indices_into_frozen_set": (
                subsets[domain].metadata["calibration_indices_into_frozen_set"]
            ),
            "calibration_input_row_sha256": (
                subsets[domain].metadata["calibration_input_row_sha256"]
            ),
            "input_ids_sha256": array_sha256(subsets[domain].input_ids),
        }
        for domain in STAGE2B_DOMAINS
    }
    print("Calibration subsets verified against the frozen Stage 2B row hashes.")

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
    print("Clean BF16 expert snapshot established.")

    calibration_fingerprint = canonical_sha256(
        {
            "stage": "stage2c_calibration_fragility",
            "model": args.model,
            "resolved_model_revision": bundle.resolved_revision,
            "dtype": str(runtime.dtype).replace("torch.", ""),
            "batch_size": args.batch_size,
            "subset_hashes": subset_hashes,
            "qdq_config": QDQ_CONFIG,
            "deterministic_settings": {
                key: value
                for key, value in determinism.items()
                if key != "torch_version"
            },
        }
    )
    print(f"Calibration run fingerprint: {calibration_fingerprint}")

    losses_root = args.results_dir / "calibration" / "losses"
    nll_by_state: dict[str, dict[str, float]] = {}
    reproduction: dict[str, dict[str, bool]] = {}
    for state_name, bits in CALIBRATION_STATES:
        manager.verify_clean()
        if bits != 16:
            manager.apply_allocation(
                uniform_bits_matrix(bits), verification_seed=args.seed
            )
        try:
            nll_by_state[state_name] = {}
            reproduction[state_name] = {}
            for domain in STAGE2B_DOMAINS:
                examples = subsets[domain]
                path = losses_root / state_name / f"{domain}.npz"
                expected_metadata = {
                    "run_fingerprint": calibration_fingerprint,
                    "state": state_name,
                    "bits": bits,
                    "domain": domain,
                    "resolved_model_revision": bundle.resolved_revision,
                    "input_ids_sha256": array_sha256(examples.input_ids),
                }

                def compute(examples=examples):
                    first, _ = evaluate_next_token_loss(
                        bundle, examples, batch_size=args.batch_size
                    )
                    second, _ = evaluate_next_token_loss(
                        bundle, examples, batch_size=args.batch_size
                    )
                    identical = bool(
                        np.array_equal(first.loss_sums, second.loss_sums)
                        and np.array_equal(first.token_counts, second.token_counts)
                    )
                    if not identical:
                        raise RuntimeError(
                            "Repeated calibration evaluation is not bitwise "
                            f"reproducible for {state_name}/{examples.domain}; "
                            "stopping as preregistered"
                        )
                    return first, {"reproduced_bitwise": True}

                checkpoint = load_or_compute_loss_checkpoint(
                    path,
                    expected_metadata,
                    examples.measurement_mask.sum(axis=1).astype(np.uint32),
                    compute,
                    resume=args.resume,
                )
                nll_by_state[state_name][domain] = mean_calibration_nll(
                    checkpoint.statistics
                )
                reproduction[state_name][domain] = bool(
                    checkpoint.metadata.get("diagnostics", {}).get(
                        "reproduced_bitwise", True
                    )
                )
                print(
                    f"[{state_name}/{domain}] calibration NLL "
                    f"{nll_by_state[state_name][domain]:.6f} "
                    f"(resumed={checkpoint.resumed})"
                )
        finally:
            if bits != 16:
                manager.restore_clean()
            else:
                manager.verify_clean()
    manager.verify_clean()

    regime_results = {
        regime: compute_regime_fragility(
            nll_by_state["bf16"],
            nll_by_state[f"uniform{base_bits}"],
            base_bits,
        )
        for regime, base_bits in STAGE2C_REGIMES.items()
    }
    record = build_calibration_fragility_record(
        regime_results=regime_results,
        calibration_subset_hashes=subset_hashes,
        model_info={
            "model": args.model,
            "requested_model_revision": args.model_revision,
            "resolved_model_revision": bundle.resolved_revision,
            "dtype": str(runtime.dtype).replace("torch.", ""),
            "device": str(runtime.device),
            "device_description": runtime.description,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "run_fingerprint": calibration_fingerprint,
        },
        qdq_config=QDQ_CONFIG,
        environment={
            "package_versions": package_versions(),
            "deterministic_settings": determinism,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        reproduction={
            "requirement": "every calibration state evaluated twice, bitwise equal",
            "states": reproduction,
            "all_reproduced": bool(
                all(all(v.values()) for v in reproduction.values())
            ),
        },
    )
    json_path, csv_path = save_calibration_fragility(
        record, args.results_dir / "calibration"
    )
    atomic_write_json(
        args.results_dir / "calibration" / "calibration_run_config.json",
        {
            "run_fingerprint": calibration_fingerprint,
            "deterministic_settings": determinism,
            "package_versions": package_versions(),
            "expert_state_clean_sha256": manager.expert_state_sha256(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Froze calibration fragility: {json_path}")
    print(f"Fragility table: {csv_path}")
    for regime in STAGE2C_REGIMES:
        entry = record["regimes"][regime]
        print(f"[{regime}] regime_valid={entry['regime_valid']}")
        for domain in STAGE2B_DOMAINS:
            values = entry["domains"][domain]
            print(
                f"  {domain}: relative fragility {values['relative_delta']:+.6f}, "
                f"normalized {values['normalized_fragility']}"
            )
    print(f"Fragility SHA-256: {record['fragility_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
