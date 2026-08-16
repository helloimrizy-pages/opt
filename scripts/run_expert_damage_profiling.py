#!/usr/bin/env python3
"""Measure and freeze the Stage 3 per-expert damage matrix (CUDA production).

For every MoE layer, expert, and bit width in {3, 4, 8}, quantizes ONLY that
expert with the audited Stage-1 QDQ, evaluates all four frozen 25-example
calibration subsets twice with a bitwise reproduction requirement, restores
the expert exactly, and checkpoints per (bit width, layer) chunk. Also
evaluates the clean BF16 model and the uniform 8/4/3-bit expert-only QDQ
states, verifies the BF16 and uniform-4/3 values against the frozen Stage 2C
calibration fragility record, and freezes ``damage/damage_matrix.json``.

Nothing is estimated: every value is a measured ground-truth calibration loss
difference. Expected wall clock on one NVIDIA A40 at batch size 1 is roughly
6-12 hours; the run is resumable per (bit width, layer) chunk.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import (
    EXPECTED_MODEL_REVISION,
    array_sha256,
    canonical_sha256,
    file_sha256,
    load_controlled_source,
)
from expert_analysis.fragility import (
    calibration_subset_inputs,
    load_frozen_fragility,
    load_frozen_stage2b_scores,
    mean_calibration_nll,
)
from expert_analysis.hardware import resolve_runtime, set_reproducible_seed
from expert_analysis.io_utils import atomic_write_json, package_versions
from expert_analysis.masking import evaluate_next_token_loss
from expert_analysis.measured_damage import (
    DamageChunk,
    STAGE3_PROFILE_BITS,
    STAGE3_STAGE,
    assemble_damage_arrays,
    build_damage_record,
    damage_chunk_path,
    load_damage_chunk,
    load_frozen_damage,
    save_damage_chunk,
    save_damage_matrix,
)
from expert_analysis.modeling import discover_moe_layers, load_model_and_tokenizer
from expert_analysis.protection_evaluation import (
    MixedPrecisionExpertManager,
    configure_strict_determinism,
    verify_layout_against_memory_shapes,
)
from expert_analysis.protection_optimization import uniform_bits_matrix
from expert_analysis.quantization import (
    ExpertWeightLayout,
    ReversibleExpertQuantization,
    load_or_compute_loss_checkpoint,
)
from expert_analysis.specialist_preservation import (
    CALIBRATION_EXAMPLES_PER_DOMAIN,
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
)
from expert_analysis.stage3_preflight import (
    verify_seed44_untouched_stage3,
    verify_stage3_upstream,
)

REFERENCE_STATES = (
    ("bf16", 16),
    ("uniform8", 8),
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
# Maximum allowed difference between this run's BF16/uniform calibration NLL
# and the frozen Stage 2C values. A same-environment rerun reproduces them
# exactly; anything beyond rounding noise means a different numeric stack.
DEFAULT_BASELINE_TOLERANCE = 5e-6


def evaluate_twice_bitwise(bundle, examples, batch_size):
    first, _ = evaluate_next_token_loss(bundle, examples, batch_size=batch_size)
    second, _ = evaluate_next_token_loss(bundle, examples, batch_size=batch_size)
    identical = bool(
        np.array_equal(first.loss_sums, second.loss_sums)
        and np.array_equal(first.token_counts, second.token_counts)
    )
    if not identical:
        raise RuntimeError(
            f"Repeated evaluation is not bitwise reproducible for "
            f"{examples.domain}; stopping as preregistered"
        )
    return first


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
    parser.add_argument(
        "--baseline-tolerance", type=float, default=DEFAULT_BASELINE_TOLERANCE
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Restrict profiling to these layers (smoke testing only; the "
        "damage matrix is assembled and frozen only when every chunk exists).",
    )
    args = parser.parse_args()

    verify_stage3_upstream(args.results_root)
    verify_seed44_untouched_stage3(args.results_root, args.results_dir)
    print("Frozen prior state verified; seed 44 untouched.")
    damage_json = args.results_dir / "damage" / "damage_matrix.json"
    if damage_json.is_file():
        record, _ = load_frozen_damage(args.results_dir / "damage")
        print(
            "Damage matrix already frozen; verifying it instead of recomputing. "
            f"Damage SHA-256: {record['damage_sha256']}"
        )
        return 0
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
    fragility_record = load_frozen_fragility(args.stage2c_dir / "calibration")
    print("Frozen Stage 2C fragility record loaded for the cross-run drift check.")

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
    layouts = {
        spec.model_layer_index: ExpertWeightLayout(spec) for spec in layer_specs
    }
    print("Clean BF16 expert snapshot established.")

    profiling_fingerprint = canonical_sha256(
        {
            "stage": "stage3_damage_profiling",
            "model": args.model,
            "resolved_model_revision": bundle.resolved_revision,
            "dtype": str(runtime.dtype).replace("torch.", ""),
            "batch_size": args.batch_size,
            "subset_hashes": subset_hashes,
            "qdq_config": QDQ_CONFIG,
            "profile_bits": list(STAGE3_PROFILE_BITS),
            "deterministic_settings": {
                key: value
                for key, value in determinism.items()
                if key != "torch_version"
            },
        }
    )
    print(f"Profiling fingerprint: {profiling_fingerprint}")

    damage_dir = args.results_dir / "damage"
    losses_root = damage_dir / "losses"
    token_counts = {
        domain: subsets[domain].measurement_mask.sum(axis=1).astype(np.uint32)
        for domain in STAGE2B_DOMAINS
    }

    # Reference states: BF16 plus uniform 8/4/3-bit expert-only QDQ.
    nll_by_state: dict[str, dict[str, float]] = {}
    for state_name, bits in REFERENCE_STATES:
        manager.verify_clean()
        if bits != 16:
            manager.apply_allocation(
                uniform_bits_matrix(bits), verification_seed=args.seed
            )
        try:
            nll_by_state[state_name] = {}
            for domain in STAGE2B_DOMAINS:
                examples = subsets[domain]
                path = losses_root / state_name / f"{domain}.npz"
                expected_metadata = {
                    "run_fingerprint": profiling_fingerprint,
                    "state": state_name,
                    "bits": bits,
                    "domain": domain,
                    "resolved_model_revision": bundle.resolved_revision,
                    "input_ids_sha256": array_sha256(examples.input_ids),
                }

                def compute(examples=examples):
                    statistics = evaluate_twice_bitwise(
                        bundle, examples, args.batch_size
                    )
                    return statistics, {"reproduced_bitwise": True}

                checkpoint = load_or_compute_loss_checkpoint(
                    path,
                    expected_metadata,
                    token_counts[domain],
                    compute,
                    resume=args.resume,
                )
                nll_by_state[state_name][domain] = mean_calibration_nll(
                    checkpoint.statistics
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

    # Cross-run drift check against the frozen Stage 2C calibration values.
    drift: dict[str, dict[str, float]] = {}
    frozen_bf16 = {
        domain: fragility_record["regimes"]["4to8"]["domains"][domain]["bf16_nll"]
        for domain in STAGE2B_DOMAINS
    }
    frozen_by_state = {
        "bf16": frozen_bf16,
        "uniform4": {
            domain: fragility_record["regimes"]["4to8"]["domains"][domain]["base_nll"]
            for domain in STAGE2B_DOMAINS
        },
        "uniform3": {
            domain: fragility_record["regimes"]["3to8"]["domains"][domain]["base_nll"]
            for domain in STAGE2B_DOMAINS
        },
    }
    for state_name, frozen_values in frozen_by_state.items():
        drift[state_name] = {}
        for domain in STAGE2B_DOMAINS:
            difference = abs(nll_by_state[state_name][domain] - frozen_values[domain])
            drift[state_name][domain] = difference
            if difference > args.baseline_tolerance:
                raise RuntimeError(
                    f"{state_name}/{domain} calibration NLL differs from the "
                    f"frozen Stage 2C value by {difference:.3e} (tolerance "
                    f"{args.baseline_tolerance:.1e}); the numeric environment "
                    "does not reproduce the frozen calibration state"
                )
    print("BF16 and uniform-4/3 values match the frozen Stage 2C record.")

    # Per-expert damage chunks.
    layers = sorted(args.layers) if args.layers else list(range(NUM_MOE_LAYERS))
    if any(layer < 0 or layer >= NUM_MOE_LAYERS for layer in layers):
        raise ValueError("Requested layers are outside the model's MoE range")
    specs_by_layer = {spec.model_layer_index: spec for spec in layer_specs}
    chunk_metadata_base = {
        "run_fingerprint": profiling_fingerprint,
        "resolved_model_revision": bundle.resolved_revision,
        "group_size": QDQ_CONFIG["group_size"],
        "reproduced_bitwise": True,
    }
    total_chunks = len(STAGE3_PROFILE_BITS) * len(layers)
    completed = 0
    for bits in STAGE3_PROFILE_BITS:
        for layer in layers:
            completed += 1
            path = damage_chunk_path(damage_dir, bits, layer)
            expected = {**chunk_metadata_base, "layer": layer, "bits": bits}
            if args.resume:
                try:
                    load_damage_chunk(
                        path, expected, CALIBRATION_EXAMPLES_PER_DOMAIN
                    )
                    print(
                        f"[{completed}/{total_chunks}] resume complete: "
                        f"bits{bits}/layer{layer:02d}",
                        flush=True,
                    )
                    continue
                except FileNotFoundError:
                    pass
            started = time.monotonic()
            manager.verify_clean()
            layout = layouts[layer]
            loss_sums = np.zeros(
                (NUM_EXPERTS, len(STAGE2B_DOMAINS), CALIBRATION_EXAMPLES_PER_DOMAIN),
                dtype=np.float64,
            )
            counts = np.zeros_like(loss_sums, dtype=np.uint32)
            for expert_id in range(NUM_EXPERTS):
                context = ReversibleExpertQuantization(
                    layout,
                    expert_id,
                    bits,
                    QDQ_CONFIG["group_size"],
                    verify_unrelated_experts=False,
                )
                with context:
                    for domain_index, domain in enumerate(STAGE2B_DOMAINS):
                        statistics = evaluate_twice_bitwise(
                            bundle, subsets[domain], args.batch_size
                        )
                        if not np.array_equal(
                            statistics.token_counts, token_counts[domain]
                        ):
                            raise RuntimeError(
                                f"Token geometry changed for {domain} at "
                                f"bits{bits}/L{layer}/E{expert_id}"
                            )
                        loss_sums[expert_id, domain_index] = statistics.loss_sums
                        counts[expert_id, domain_index] = statistics.token_counts
                if not context.restoration_verified:
                    raise RuntimeError(
                        f"Expert bits{bits}/L{layer}/E{expert_id} was not restored"
                    )
            manager.verify_clean()
            spec = specs_by_layer[layer]
            chunk = DamageChunk(
                layer=layer,
                bits=bits,
                loss_sums=loss_sums,
                token_counts=counts,
                metadata={},
            )
            save_damage_chunk(
                path, chunk, expected, CALIBRATION_EXAMPLES_PER_DOMAIN
            )
            print(
                f"[{completed}/{total_chunks}] saved bits{bits}/layer"
                f"{spec.model_layer_index:02d} in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    # Assemble and freeze the damage matrix only when the full grid exists.
    missing = [
        (bits, layer)
        for bits in STAGE3_PROFILE_BITS
        for layer in range(NUM_MOE_LAYERS)
        if not damage_chunk_path(damage_dir, bits, layer).is_file()
    ]
    if missing:
        print(
            f"{len(missing)} chunks are still missing; rerun without --layers "
            "to complete the grid before the damage matrix can be frozen."
        )
        return 0

    arrays = assemble_damage_arrays(
        damage_dir,
        chunk_metadata_base,
        nll_by_state["bf16"],
        CALIBRATION_EXAMPLES_PER_DOMAIN,
    )
    chunk_hashes = {
        f"chunks/bits{bits}/layer_{layer:02d}.npz": file_sha256(
            damage_chunk_path(damage_dir, bits, layer)
        )
        for bits in STAGE3_PROFILE_BITS
        for layer in range(NUM_MOE_LAYERS)
    }
    record = build_damage_record(
        arrays=arrays,
        uniform_nll={
            state: nll_by_state[state]
            for state in ("uniform8", "uniform4", "uniform3")
        },
        frozen_reference_drift={
            "stage2c_fragility_sha256": fragility_record["fragility_sha256"],
            "tolerance": args.baseline_tolerance,
            "absolute_difference_by_state_domain": drift,
        },
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
            "run_fingerprint": profiling_fingerprint,
        },
        qdq_config=QDQ_CONFIG,
        environment={
            "package_versions": package_versions(),
            "deterministic_settings": determinism,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        reproduction={
            "requirement": (
                "every reference state and every single-expert state evaluated "
                "twice, bitwise equal"
            ),
            "all_reproduced": True,
        },
        chunk_hashes=chunk_hashes,
        examples_per_domain=CALIBRATION_EXAMPLES_PER_DOMAIN,
    )
    json_path, npz_path = save_damage_matrix(record, arrays, damage_dir)
    atomic_write_json(
        damage_dir / "profiling_run_config.json",
        {
            "stage": STAGE3_STAGE,
            "run_fingerprint": profiling_fingerprint,
            "deterministic_settings": determinism,
            "package_versions": package_versions(),
            "expert_state_clean_sha256": manager.expert_state_sha256(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Froze damage matrix: {json_path}")
    print(f"Damage arrays: {npz_path}")
    for bits_key, summary in record["summary"].items():
        totals = summary["total_delta_nll_by_domain"]
        print(
            f"[{bits_key}] total delta NLL: "
            + ", ".join(
                f"{domain} {totals[domain]:+.6f}" for domain in STAGE2B_DOMAINS
            )
        )
    print(f"Damage SHA-256: {record['damage_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
