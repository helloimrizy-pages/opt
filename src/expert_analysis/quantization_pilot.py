from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .balanced import (
    BALANCED_DOMAINS,
    EXPECTED_DATASET_REVISIONS,
    EXPECTED_EXAMPLES,
    EXPECTED_MEASURED_POSITIONS,
    EXPECTED_MODEL,
    EXPECTED_MODEL_REVISION,
    canonical_sha256,
    file_sha256,
)
from .io_utils import atomic_save_npz, atomic_write_json, read_json, write_csv
from .masking import LossStatistics


PILOT_SCHEMA_VERSION = 1
PILOT_SELECTION_VERSION = "balanced_frozen_top2_functional_margin_v1"
PILOT_ANALYSIS_VERSION = "expert_qdq_bootstrap_v1"
DEFAULT_PRIMARY_BITS = 4
DEFAULT_FALLBACK_BITS = 3
DEFAULT_GROUP_SIZE = 128
DEFAULT_BOOTSTRAP_REPLICATES = 1000
DEFAULT_SEED = 42


def build_pilot_preregistration(
    balanced_preregistration_path: Path,
    matched_controls_path: Path,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    primary_bits: int = DEFAULT_PRIMARY_BITS,
    fallback_bits: int = DEFAULT_FALLBACK_BITS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Select the frozen top two specialists/domain without reading outcomes."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    if primary_bits != 4 or fallback_bits != 3:
        raise ValueError("The Stage-1 pilot is preregistered for 4-bit then 3-bit")
    if bootstrap_replicates != 1000 or seed != 42:
        raise ValueError("The Stage-1 pilot requires 1,000 replicates and seed 42")
    frozen = read_json(balanced_preregistration_path)
    _validate_balanced_preregistration(frozen)
    _validate_matched_control_csv(frozen, matched_controls_path)
    controls_by_specialist = {
        (
            pair["target_domain"],
            int(pair["specialized"]["layer"]),
            int(pair["specialized"]["expert_id"]),
        ): pair
        for pair in frozen["matched_controls"]
    }

    selected_pairs: list[dict[str, Any]] = []
    for domain in BALANCED_DOMAINS:
        eligible = sorted(
            (
                row
                for row in frozen["selected_experts"]
                if row["target_domain"] == domain
            ),
            key=lambda row: (
                -float(row["specialization_margin"]),
                float(row["target_rank"]),
                int(row["layer"]),
                int(row["expert_id"]),
            ),
        )
        if len(eligible) != 3:
            raise RuntimeError(f"Frozen balanced panel does not have three {domain} specialists")
        for pilot_rank, specialist in enumerate(eligible[:2], 1):
            key = (domain, int(specialist["layer"]), int(specialist["expert_id"]))
            pair = controls_by_specialist.get(key)
            if pair is None:
                raise RuntimeError(f"No frozen matched control exists for {key}")
            if not pair["same_layer"] or pair["fallback_used"]:
                raise RuntimeError(f"Pilot pair {pair['pair_id']} is not a strict same-layer match")
            if pair["specialized"] != specialist:
                raise RuntimeError(f"Frozen specialist record mismatch for {pair['pair_id']}")
            selected_pairs.append(
                {
                    "pair_id": pair["pair_id"],
                    "target_domain": domain,
                    "pilot_functional_specialization_rank": pilot_rank,
                    "selection_score_name": "frozen_functional_specialization_margin",
                    "selection_score": float(specialist["specialization_margin"]),
                    "specialist": dict(specialist),
                    "matched_control": dict(pair["control"]),
                    "matching": {
                        key: pair[key]
                        for key in (
                            "same_layer",
                            "control_reused",
                            "target_routing_frequency_absolute_difference",
                            "target_routing_coverage_absolute_difference",
                            "control_to_specialist_positive_margin_ratio",
                            "matching_tier_specialization_cap_ratio",
                            "matching_cost",
                            "fallback_used",
                            "matching_rationale",
                        )
                    },
                }
            )

    input_hashes = frozen["source"]["input_files_sha256"]
    deterministic = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "selection_version": PILOT_SELECTION_VERSION,
        "source": {
            "model": EXPECTED_MODEL,
            "model_revision": EXPECTED_MODEL_REVISION,
            "dataset_revisions": EXPECTED_DATASET_REVISIONS,
            "balanced_selection_input_fingerprint": frozen[
                "selection_input_fingerprint"
            ],
            "balanced_preregistration_fingerprint": frozen[
                "preregistration_fingerprint"
            ],
            "balanced_preregistration_sha256": file_sha256(
                balanced_preregistration_path
            ),
            "balanced_matched_controls_sha256": file_sha256(matched_controls_path),
            "controlled_input_file_sha256": {
                domain: input_hashes[f"controlled_inputs/{domain}.npz"]
                for domain in BALANCED_DOMAINS
            },
            "examples_per_domain": EXPECTED_EXAMPLES,
            "measured_positions_per_example": EXPECTED_MEASURED_POSITIONS,
            "measured_positions_per_domain": (
                EXPECTED_EXAMPLES * EXPECTED_MEASURED_POSITIONS
            ),
        },
        "selection": {
            "specialists_per_domain": 2,
            "specialist_score": (
                "frozen specialization_margin = I_functional(target) - "
                "max I_functional(non-target)"
            ),
            "ordering": [
                "specialization margin descending",
                "target functional rank ascending",
                "layer ascending",
                "expert ID ascending",
            ],
            "controls": "already preregistered one-to-one routing-matched controls",
            "masking_effect_sizes_used": False,
            "quantization_results_used": False,
            "intervention_results_read_by_selection": False,
        },
        "quantization_preregistration": {
            "method": "deterministic symmetric group-wise weight-only fake quantization/QDQ",
            "scope": "one expert FFN at a time; all non-expert parameters remain BF16",
            "grouping_axis": "input-feature dimension (last axis of expert matrix)",
            "group_size": group_size,
            "scale_storage_dtype": "float16",
            "primary_bits": primary_bits,
            "fallback_bits": fallback_bits,
            "qmax": "2^(bits-1)-1",
            "zero_group_behavior": "scale and dequantized values remain exactly zero",
        },
        "analysis_preregistration": {
            "version": PILOT_ANALYSIS_VERSION,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "primary_contrast": (
                "target-domain delta NLL minus arithmetic mean of three "
                "non-target-domain delta NLL values"
            ),
            "resampling": (
                "examples independently resampled within domain; shared indices across "
                "interventions preserve specialist-control pairing"
            ),
            "gates": {
                "A": "overall specialist contrast > 0 with 95% CI excluding zero",
                "B": "at least 3 of 4 domain aggregate specialist contrasts > 0",
                "C": "overall specialist-minus-control contrast > 0; CI exclusion reported",
                "D": (
                    "median absolute domain-level delta NLL exceeds the clear-noise threshold"
                ),
            },
            "numerical_noise_rule": {
                "minimum_noise_floor_nll": 1e-7,
                "clear_signal_multiplier": 10.0,
                "threshold": (
                    "max(1e-7, 10 * maximum per-token BF16 baseline reproduction error)"
                ),
            },
            "fallback_rule": {
                "trigger_only_when_gate_d_fails": True,
                "overall_specialist_contrast_within_noise_threshold": True,
                "overall_specialist_control_difference_within_noise_threshold": True,
                "no_domain_aggregate_more_negative_than_noise_threshold": True,
                "rationale": (
                    "3-bit is run only when 4-bit failure is consistent with an "
                    "unmeasurably small perturbation, never merely an unattractive result"
                ),
            },
            "risk_proxies": {
                "functional": "I_functional(expert, domain) * Q_weight(expert, bits)",
                "routing": "I_routing(expert, domain) * Q_weight(expert, bits)",
                "coefficient_fitting": False,
            },
        },
        "pairs": selected_pairs,
    }
    return {
        "status": "FROZEN_BEFORE_QUANTIZATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **deterministic,
        "pilot_panel_fingerprint": canonical_sha256(deterministic),
    }


def write_or_validate_pilot_preregistration(
    expected: Mapping[str, Any], output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        observed = read_json(output_path)
        validate_pilot_preregistration(observed)
        if observed["pilot_panel_fingerprint"] != expected["pilot_panel_fingerprint"]:
            raise RuntimeError("Existing quantization pilot panel differs from frozen inputs")
        return observed
    atomic_write_json(output_path, dict(expected))
    return dict(expected)


def validate_pilot_preregistration(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "FROZEN_BEFORE_QUANTIZATION":
        raise RuntimeError("Pilot panel does not declare a frozen pre-quantization state")
    deterministic = _pilot_deterministic_content(payload)
    if canonical_sha256(deterministic) != payload.get("pilot_panel_fingerprint"):
        raise RuntimeError("Frozen quantization pilot panel content was modified")
    pairs = payload.get("pairs", [])
    if len(pairs) != 8:
        raise RuntimeError("Quantization pilot requires eight specialist-control pairs")
    for domain in BALANCED_DOMAINS:
        domain_pairs = [row for row in pairs if row["target_domain"] == domain]
        if len(domain_pairs) != 2:
            raise RuntimeError(f"Quantization pilot is not balanced for {domain}")
        if sorted(row["pilot_functional_specialization_rank"] for row in domain_pairs) != [1, 2]:
            raise RuntimeError(f"Quantization pilot ranks are invalid for {domain}")
    identities = []
    for pair in pairs:
        identities.extend(
            [
                (
                    int(pair["specialist"]["layer"]),
                    int(pair["specialist"]["expert_id"]),
                ),
                (
                    int(pair["matched_control"]["layer"]),
                    int(pair["matched_control"]["expert_id"]),
                ),
            ]
        )
    if len(set(identities)) != 16:
        raise RuntimeError("Quantization pilot expert identities are not unique")


def pilot_intervention_panel(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_pilot_preregistration(payload)
    panel: list[dict[str, Any]] = []
    for pair in payload["pairs"]:
        specialist = pair["specialist"]
        control = pair["matched_control"]
        panel.append(
            {
                "intervention_id": f"specialist_{pair['pair_id']}",
                "pair_id": pair["pair_id"],
                "role": "specialist",
                "target_domain": pair["target_domain"],
                "layer": int(specialist["layer"]),
                "expert_id": int(specialist["expert_id"]),
                "baseline_record": specialist,
            }
        )
        panel.append(
            {
                "intervention_id": (
                    f"control_{pair['pair_id']}_L{control['layer']}_E"
                    f"{control['expert_id']}"
                ),
                "pair_id": pair["pair_id"],
                "role": "control",
                "target_domain": pair["target_domain"],
                "layer": int(control["layer"]),
                "expert_id": int(control["expert_id"]),
                "baseline_record": control,
            }
        )
    return panel


def analyze_quantization_pilot(
    preregistration: Mapping[str, Any],
    baselines: Mapping[str, LossStatistics],
    quantized: Mapping[tuple[int, int, int, str], LossStatistics],
    intervention_metadata: Mapping[tuple[int, int, int], Mapping[str, Any]],
    balanced_results: Mapping[str, Any],
    evaluated_bits: Sequence[int],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_SEED,
    baseline_reproduction_noise_nll: float = 0.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Analyze saved per-example losses without performing model inference."""

    if bootstrap_replicates != DEFAULT_BOOTSTRAP_REPLICATES:
        raise ValueError("The Stage-1 pilot requires exactly 1,000 bootstrap replicates")
    bits = tuple(int(value) for value in evaluated_bits)
    if not bits or len(set(bits)) != len(bits):
        raise ValueError("evaluated_bits must contain unique bit widths")
    panel = pilot_intervention_panel(preregistration)
    _validate_analysis_inputs(panel, baselines, quantized, intervention_metadata, bits)
    masking_contrasts = _load_frozen_masking_contrasts(
        balanced_results, preregistration
    )

    domain_names = list(BALANCED_DOMAINS)
    examples = EXPECTED_EXAMPLES
    delta = np.empty((len(bits), len(panel), len(domain_names), examples), dtype=np.float64)
    quantized_nll = np.empty_like(delta)
    baseline_nll = np.stack(
        [baselines[domain].per_token_nll for domain in domain_names]
    )
    token_counts = np.stack([baselines[domain].token_counts for domain in domain_names])
    for bit_index, bit_width in enumerate(bits):
        for intervention_index, intervention in enumerate(panel):
            for domain_index, domain in enumerate(domain_names):
                result = quantized[
                    (bit_width, intervention["layer"], intervention["expert_id"], domain)
                ]
                values = result.per_token_nll
                quantized_nll[bit_index, intervention_index, domain_index] = values
                delta[bit_index, intervention_index, domain_index] = (
                    values - baseline_nll[domain_index]
                )

    bootstrap_indices = _bootstrap_indices(
        bootstrap_replicates, examples, seed, domain_names
    )
    bootstrap_domain_means = np.empty(
        (len(bits), len(panel), len(domain_names), bootstrap_replicates),
        dtype=np.float64,
    )
    for domain_index, domain in enumerate(domain_names):
        indices = bootstrap_indices[domain]
        bootstrap_domain_means[:, :, domain_index, :] = np.transpose(
            delta[:, :, domain_index, :][:, :, indices].mean(axis=-1),
            (0, 1, 2),
        )

    domain_means = delta.mean(axis=-1)
    result_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    contrast_bootstrap = np.empty(
        (len(bits), len(panel), bootstrap_replicates), dtype=np.float64
    )

    for bit_index, bit_width in enumerate(bits):
        for intervention_index, intervention in enumerate(panel):
            record = intervention["baseline_record"]
            target_index = domain_names.index(intervention["target_domain"])
            other_indices = [
                index for index in range(len(domain_names)) if index != target_index
            ]
            boot_contrast = (
                bootstrap_domain_means[bit_index, intervention_index, target_index]
                - bootstrap_domain_means[
                    bit_index, intervention_index, other_indices
                ].mean(axis=0)
            )
            contrast_bootstrap[bit_index, intervention_index] = boot_contrast
            point_contrast = float(
                domain_means[bit_index, intervention_index, target_index]
                - domain_means[bit_index, intervention_index, other_indices].mean()
            )
            contrast_low, contrast_high = _ci(boot_contrast)
            target_boot = bootstrap_domain_means[
                bit_index, intervention_index, target_index
            ]
            other_boot = bootstrap_domain_means[
                bit_index, intervention_index, other_indices
            ].mean(axis=0)
            target_low, target_high = _ci(target_boot)
            other_low, other_high = _ci(other_boot)
            metadata = intervention_metadata[
                (bit_width, intervention["layer"], intervention["expert_id"])
            ]
            distortion = float(metadata["quantization_distortion"])
            memory = metadata["memory_accounting"]
            contrast_rows.append(
                {
                    "bit_width": bit_width,
                    "intervention_id": intervention["intervention_id"],
                    "pair_id": intervention["pair_id"],
                    "role": intervention["role"],
                    "target_domain": intervention["target_domain"],
                    "layer": intervention["layer"],
                    "expert_id": intervention["expert_id"],
                    "target_delta_nll": float(
                        domain_means[bit_index, intervention_index, target_index]
                    ),
                    "target_delta_nll_ci_low": target_low,
                    "target_delta_nll_ci_high": target_high,
                    "mean_non_target_delta_nll": float(
                        domain_means[
                            bit_index, intervention_index, other_indices
                        ].mean()
                    ),
                    "mean_non_target_delta_nll_ci_low": other_low,
                    "mean_non_target_delta_nll_ci_high": other_high,
                    "target_minus_mean_other_contrast": point_contrast,
                    "contrast_ci_low": contrast_low,
                    "contrast_ci_high": contrast_high,
                    "contrast_positive": point_contrast > 0,
                    "positive_contrast_ci_excludes_zero": contrast_low > 0,
                    "functional_specialization": float(record["specialization_margin"]),
                    "routing_specialization": float(
                        record["routing_specialization_margin"]
                    ),
                    "target_routing_frequency": float(record["target_routing_frequency"]),
                    "target_routing_coverage": float(record["target_routing_coverage"]),
                    "quantization_distortion": distortion,
                    "effective_bits_per_weight": memory["effective_bits_per_weight"],
                    "projected_bytes": memory["projected_bytes"],
                    "compression_ratio_vs_bf16": memory["compression_ratio_vs_bf16"],
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )

            for domain_index, domain in enumerate(domain_names):
                values = delta[bit_index, intervention_index, domain_index]
                boot = bootstrap_domain_means[
                    bit_index, intervention_index, domain_index
                ]
                low, high = _ci(boot)
                base_mean = float(baseline_nll[domain_index].mean())
                quant_mean = float(
                    quantized_nll[bit_index, intervention_index, domain_index].mean()
                )
                functional_importance = float(
                    record["normalized_contribution_by_domain"][domain]
                )
                routing_importance = float(record["routing_frequency_by_domain"][domain])
                result_rows.append(
                    {
                        "bit_width": bit_width,
                        "intervention_id": intervention["intervention_id"],
                        "pair_id": intervention["pair_id"],
                        "role": intervention["role"],
                        "target_domain": intervention["target_domain"],
                        "layer": intervention["layer"],
                        "expert_id": intervention["expert_id"],
                        "domain": domain,
                        "is_target_domain": domain == intervention["target_domain"],
                        "examples": examples,
                        "evaluated_tokens": int(token_counts[domain_index].sum()),
                        "baseline_nll": base_mean,
                        "quantized_nll": quant_mean,
                        "delta_nll": float(values.mean()),
                        "delta_nll_ci_low": low,
                        "delta_nll_ci_high": high,
                        "normalized_relative_delta_nll": (
                            float(values.mean()) / base_mean if base_mean > 0 else math.nan
                        ),
                        "relative_delta_percent": (
                            100.0 * float(values.mean()) / base_mean
                            if base_mean > 0
                            else math.nan
                        ),
                        "positive_delta_example_fraction": float(np.mean(values > 0)),
                        "routing_coverage": float(record["routing_coverage_by_domain"][domain]),
                        "routing_frequency": routing_importance,
                        "functional_importance": functional_importance,
                        "functional_rank": float(record["functional_rank_by_domain"][domain]),
                        "quantization_distortion": distortion,
                        "risk_functional": functional_importance * distortion,
                        "risk_routing": routing_importance * distortion,
                        "effective_bits_per_weight": memory["effective_bits_per_weight"],
                        "projected_bytes": memory["projected_bytes"],
                        "compression_ratio_vs_bf16": memory[
                            "compression_ratio_vs_bf16"
                        ],
                        "bootstrap_replicates": bootstrap_replicates,
                    }
                )
            for comparison_index in other_indices:
                pair_boot = (
                    bootstrap_domain_means[
                        bit_index, intervention_index, target_index
                    ]
                    - bootstrap_domain_means[
                        bit_index, intervention_index, comparison_index
                    ]
                )
                low, high = _ci(pair_boot)
                pairwise_rows.append(
                    {
                        "bit_width": bit_width,
                        "intervention_id": intervention["intervention_id"],
                        "pair_id": intervention["pair_id"],
                        "role": intervention["role"],
                        "target_domain": intervention["target_domain"],
                        "comparison_domain": domain_names[comparison_index],
                        "layer": intervention["layer"],
                        "expert_id": intervention["expert_id"],
                        "target_delta_nll": float(
                            domain_means[bit_index, intervention_index, target_index]
                        ),
                        "comparison_delta_nll": float(
                            domain_means[
                                bit_index, intervention_index, comparison_index
                            ]
                        ),
                        "target_minus_comparison_delta_nll": float(
                            domain_means[bit_index, intervention_index, target_index]
                            - domain_means[
                                bit_index, intervention_index, comparison_index
                            ]
                        ),
                        "contrast_ci_low": low,
                        "contrast_ci_high": high,
                        "positive_contrast_ci_excludes_zero": low > 0,
                        "bootstrap_replicates": bootstrap_replicates,
                    }
                )

    paired_rows = _paired_specialist_control_rows(
        preregistration,
        panel,
        bits,
        contrast_rows,
        contrast_bootstrap,
        bootstrap_replicates,
    )
    aggregate_rows, aggregate_bootstrap = _aggregate_rows(
        panel,
        bits,
        contrast_rows,
        contrast_bootstrap,
        bootstrap_replicates,
    )
    quantization_vs_masking, correlation_results = _correlation_rows(
        panel,
        bits,
        contrast_rows,
        contrast_bootstrap,
        masking_contrasts,
    )
    risk_correlations = _risk_correlations(
        bits, result_rows, bootstrap_domain_means
    )
    distortion_rows = _distortion_rows(panel, bits, intervention_metadata)
    gates = _evaluate_gates(
        bits,
        domain_means,
        panel,
        aggregate_rows,
        baseline_reproduction_noise_nll,
    )
    decision = _stage1_decision(
        gates,
        bits,
        primary_bits=int(preregistration["quantization_preregistration"]["primary_bits"]),
        fallback_bits=int(preregistration["quantization_preregistration"]["fallback_bits"]),
    )

    arrays = {
        "bit_widths": np.asarray(bits, dtype=np.int8),
        "intervention_ids": np.asarray(
            [row["intervention_id"] for row in panel], dtype=np.str_
        ),
        "pair_ids": np.asarray([row["pair_id"] for row in panel], dtype=np.str_),
        "roles": np.asarray([row["role"] for row in panel], dtype=np.str_),
        "target_domains": np.asarray(
            [row["target_domain"] for row in panel], dtype=np.str_
        ),
        "layers": np.asarray([row["layer"] for row in panel], dtype=np.int16),
        "expert_ids": np.asarray(
            [row["expert_id"] for row in panel], dtype=np.int16
        ),
        "domain_names": np.asarray(domain_names, dtype=np.str_),
        "baseline_per_example_nll": baseline_nll,
        "quantized_per_example_nll": quantized_nll,
        "per_example_loss_changes": delta,
        "token_counts": token_counts,
        "quantization_distortion": np.asarray(
            [
                [
                    float(
                        intervention_metadata[
                            (bit_width, row["layer"], row["expert_id"])
                        ]["quantization_distortion"]
                    )
                    for row in panel
                ]
                for bit_width in bits
            ],
            dtype=np.float64,
        ),
    }
    analysis = {
        "analysis_version": PILOT_ANALYSIS_VERSION,
        "evaluated_bits": list(bits),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
        "quantization_pilot_results": result_rows,
        "intervention_contrasts": contrast_rows,
        "quantization_pilot_pairwise": pairwise_rows,
        "specialist_vs_control": paired_rows,
        "aggregate_results": aggregate_rows,
        "aggregate_bootstrap": aggregate_bootstrap,
        "quantization_vs_masking": quantization_vs_masking,
        "correlation_results": correlation_results,
        "risk_proxy_correlations": risk_correlations,
        "quantization_distortion": distortion_rows,
        "gates_by_bit_width": gates,
        "stage1_decision": decision,
        "numerical_noise": {
            "baseline_reproduction_noise_nll": baseline_reproduction_noise_nll,
            "minimum_noise_floor_nll": 1e-7,
            "clear_signal_multiplier": 10.0,
        },
    }
    return analysis, arrays


def _pilot_deterministic_content(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "selection_version",
        "source",
        "selection",
        "quantization_preregistration",
        "analysis_preregistration",
        "pairs",
    )
    try:
        return {key: payload[key] for key in keys}
    except KeyError as exc:
        raise RuntimeError(f"Pilot preregistration is missing {exc.args[0]}") from exc


def _validate_balanced_preregistration(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "FROZEN_BEFORE_MASKING":
        raise RuntimeError("Balanced source panel is not frozen before masking")
    if payload.get("masking_outcomes_used_for_selection") is not False:
        raise RuntimeError("Balanced source panel does not exclude masking outcomes")
    deterministic = {
        "source_collection_fingerprint": payload["source"]["collection_fingerprint"],
        "selection_input_fingerprint": payload["selection_input_fingerprint"],
        "selection_algorithm_version": payload["selection_algorithm_version"],
        "control_algorithm_version": payload["control_algorithm_version"],
        "selection_algorithm": payload["selection_algorithm"],
        "domain_selection_tiers": payload["domain_selection_tiers"],
        "control_matching_tiers": payload["control_matching_tiers"],
        "ranked_candidate_pool": payload["ranked_candidate_pool"],
        "selected_experts": payload["selected_experts"],
        "matched_controls": payload["matched_controls"],
        "analysis_preregistration": payload["analysis_preregistration"],
    }
    if canonical_sha256(deterministic) != payload.get("preregistration_fingerprint"):
        raise RuntimeError("Balanced source preregistration content was modified")
    if len(payload["selected_experts"]) != 12 or len(payload["matched_controls"]) != 12:
        raise RuntimeError("Balanced source panel has an unexpected size")


def _validate_matched_control_csv(
    payload: Mapping[str, Any], matched_controls_path: Path
) -> None:
    with matched_controls_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(payload["matched_controls"]):
        raise RuntimeError("Frozen matched-control CSV has an unexpected row count")
    csv_pairs = {
        (
            row["pair_id"],
            int(row["specialized_layer"]),
            int(row["specialized_expert_id"]),
            int(row["control_layer"]),
            int(row["control_expert_id"]),
        )
        for row in rows
    }
    json_pairs = {
        (
            row["pair_id"],
            int(row["specialized"]["layer"]),
            int(row["specialized"]["expert_id"]),
            int(row["control"]["layer"]),
            int(row["control"]["expert_id"]),
        )
        for row in payload["matched_controls"]
    }
    if csv_pairs != json_pairs:
        raise RuntimeError("Frozen matched-control CSV and JSON identities disagree")


def _validate_analysis_inputs(
    panel: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, LossStatistics],
    quantized: Mapping[tuple[int, int, int, str], LossStatistics],
    metadata: Mapping[tuple[int, int, int], Mapping[str, Any]],
    bits: Sequence[int],
) -> None:
    if set(baselines) != set(BALANCED_DOMAINS):
        raise RuntimeError("Pilot baselines do not cover all four frozen domains")
    for domain in BALANCED_DOMAINS:
        baselines[domain].validate()
        if len(baselines[domain].loss_sums) != EXPECTED_EXAMPLES:
            raise RuntimeError(f"Pilot baseline has the wrong size for {domain}")
        if not np.all(baselines[domain].token_counts == EXPECTED_MEASURED_POSITIONS):
            raise RuntimeError(f"Pilot baseline has the wrong token budget for {domain}")
    for bit_width in bits:
        for intervention in panel:
            metadata_key = (
                bit_width,
                int(intervention["layer"]),
                int(intervention["expert_id"]),
            )
            if metadata_key not in metadata:
                raise RuntimeError(f"Missing expert QDQ metadata for {metadata_key}")
            if float(metadata[metadata_key]["quantization_distortion"]) < 0:
                raise RuntimeError(f"Negative expert QDQ distortion for {metadata_key}")
            for domain in BALANCED_DOMAINS:
                key = (*metadata_key, domain)
                if key not in quantized:
                    raise RuntimeError(f"Missing expert QDQ loss checkpoint for {key}")
                quantized[key].validate()
                if not np.array_equal(
                    quantized[key].token_counts, baselines[domain].token_counts
                ):
                    raise RuntimeError(f"Pilot token counts differ for {key}")


def _load_frozen_masking_contrasts(
    results: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> dict[tuple[str, int, int], float]:
    if results.get("integrity_validation", {}).get("passed") is not True:
        raise RuntimeError("Balanced masking results did not pass integrity validation")
    source_prereg = results.get("preregistration", {})
    expected = preregistration["source"]["balanced_preregistration_fingerprint"]
    if source_prereg.get("preregistration_fingerprint") != expected:
        raise RuntimeError("Balanced masking results use a different frozen panel")
    analysis = results.get("balanced_analysis", results)
    rows = results.get(
        "_pilot_raw_masking_contrasts",
        analysis.get("intervention_contrasts", []),
    )
    output: dict[tuple[str, int, int], float] = {}
    for row in rows:
        role = "specialist" if row["role"] == "specialized" else row["role"]
        output[(role, int(row["layer"]), int(row["expert_id"]))] = float(
            row["target_minus_mean_other_contrast"]
        )
    expected_keys = {
        (row["role"], int(row["layer"]), int(row["expert_id"]))
        for row in pilot_intervention_panel(preregistration)
    }
    missing = sorted(expected_keys - set(output))
    if missing:
        raise RuntimeError(f"Balanced masking contrasts are missing pilot experts: {missing}")
    return output


def _bootstrap_indices(
    replicates: int, examples: int, seed: int, domains: Sequence[str]
) -> dict[str, np.ndarray]:
    output = {}
    for domain in domains:
        derived = _derived_seed(seed, f"pilot-bootstrap-{domain}")
        rng = np.random.default_rng(derived)
        output[domain] = rng.integers(
            0, examples, size=(replicates, examples), dtype=np.int64
        )
    return output


def _paired_specialist_control_rows(
    preregistration: Mapping[str, Any],
    panel: Sequence[Mapping[str, Any]],
    bits: Sequence[int],
    contrast_rows: Sequence[Mapping[str, Any]],
    contrast_bootstrap: np.ndarray,
    replicates: int,
) -> list[dict[str, Any]]:
    panel_index = {row["intervention_id"]: index for index, row in enumerate(panel)}
    contrasts = {
        (int(row["bit_width"]), row["intervention_id"]): row
        for row in contrast_rows
    }
    output = []
    for bit_index, bit_width in enumerate(bits):
        for pair in preregistration["pairs"]:
            specialist_id = f"specialist_{pair['pair_id']}"
            control = pair["matched_control"]
            control_id = (
                f"control_{pair['pair_id']}_L{control['layer']}_E{control['expert_id']}"
            )
            specialist_row = contrasts[(bit_width, specialist_id)]
            control_row = contrasts[(bit_width, control_id)]
            difference_boot = (
                contrast_bootstrap[bit_index, panel_index[specialist_id]]
                - contrast_bootstrap[bit_index, panel_index[control_id]]
            )
            low, high = _ci(difference_boot)
            difference = float(
                specialist_row["target_minus_mean_other_contrast"]
                - control_row["target_minus_mean_other_contrast"]
            )
            output.append(
                {
                    "bit_width": bit_width,
                    "pair_id": pair["pair_id"],
                    "target_domain": pair["target_domain"],
                    "specialist_layer": pair["specialist"]["layer"],
                    "specialist_expert_id": pair["specialist"]["expert_id"],
                    "control_layer": control["layer"],
                    "control_expert_id": control["expert_id"],
                    "specialist_contrast": specialist_row[
                        "target_minus_mean_other_contrast"
                    ],
                    "specialist_contrast_ci_low": specialist_row["contrast_ci_low"],
                    "specialist_contrast_ci_high": specialist_row["contrast_ci_high"],
                    "control_contrast": control_row[
                        "target_minus_mean_other_contrast"
                    ],
                    "control_contrast_ci_low": control_row["contrast_ci_low"],
                    "control_contrast_ci_high": control_row["contrast_ci_high"],
                    "specialist_minus_control_difference": difference,
                    "difference_ci_low": low,
                    "difference_ci_high": high,
                    "difference_positive": difference > 0,
                    "positive_difference_ci_excludes_zero": low > 0,
                    "bootstrap_replicates": replicates,
                }
            )
    return output


def _aggregate_rows(
    panel: Sequence[Mapping[str, Any]],
    bits: Sequence[int],
    contrast_rows: Sequence[Mapping[str, Any]],
    contrast_bootstrap: np.ndarray,
    replicates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_by_key = {
        (int(row["bit_width"]), row["intervention_id"]): row
        for row in contrast_rows
    }
    panel_index = {row["intervention_id"]: index for index, row in enumerate(panel)}
    output = []
    saved_bootstrap: dict[str, Any] = {}
    scopes: list[tuple[str, str]] = [
        ("domain", domain) for domain in BALANCED_DOMAINS
    ] + [("overall", "all")]
    for bit_index, bit_width in enumerate(bits):
        for scope, target_domain in scopes:
            specialists = [
                row
                for row in panel
                if row["role"] == "specialist"
                and (scope == "overall" or row["target_domain"] == target_domain)
            ]
            controls = [
                row
                for row in panel
                if row["role"] == "control"
                and (scope == "overall" or row["target_domain"] == target_domain)
            ]
            specialists.sort(key=lambda row: row["pair_id"])
            controls.sort(key=lambda row: row["pair_id"])
            if [row["pair_id"] for row in specialists] != [
                row["pair_id"] for row in controls
            ]:
                raise RuntimeError("Aggregate specialist/control pairing is inconsistent")
            specialist_values = np.asarray(
                [
                    row_by_key[(bit_width, row["intervention_id"])][
                        "target_minus_mean_other_contrast"
                    ]
                    for row in specialists
                ],
                dtype=np.float64,
            )
            control_values = np.asarray(
                [
                    row_by_key[(bit_width, row["intervention_id"])][
                        "target_minus_mean_other_contrast"
                    ]
                    for row in controls
                ],
                dtype=np.float64,
            )
            specialist_boot = np.stack(
                [
                    contrast_bootstrap[bit_index, panel_index[row["intervention_id"]]]
                    for row in specialists
                ]
            )
            control_boot = np.stack(
                [
                    contrast_bootstrap[bit_index, panel_index[row["intervention_id"]]]
                    for row in controls
                ]
            )
            mean_specialist_boot = specialist_boot.mean(axis=0)
            mean_control_boot = control_boot.mean(axis=0)
            mean_difference_boot = (specialist_boot - control_boot).mean(axis=0)
            median_specialist_boot = np.median(specialist_boot, axis=0)
            specialist_low, specialist_high = _ci(mean_specialist_boot)
            control_low, control_high = _ci(mean_control_boot)
            difference_low, difference_high = _ci(mean_difference_boot)
            median_low, median_high = _ci(median_specialist_boot)
            output.append(
                {
                    "bit_width": bit_width,
                    "scope": scope,
                    "target_domain": target_domain,
                    "num_specialists": len(specialists),
                    "mean_specialist_contrast": float(specialist_values.mean()),
                    "mean_specialist_contrast_ci_low": specialist_low,
                    "mean_specialist_contrast_ci_high": specialist_high,
                    "median_specialist_contrast": float(np.median(specialist_values)),
                    "median_specialist_contrast_ci_low": median_low,
                    "median_specialist_contrast_ci_high": median_high,
                    "specialist_positive_proportion": float(
                        np.mean(specialist_values > 0)
                    ),
                    "mean_control_contrast": float(control_values.mean()),
                    "mean_control_contrast_ci_low": control_low,
                    "mean_control_contrast_ci_high": control_high,
                    "mean_specialist_minus_control_difference": float(
                        np.mean(specialist_values - control_values)
                    ),
                    "mean_difference_ci_low": difference_low,
                    "mean_difference_ci_high": difference_high,
                    "pair_difference_positive_proportion": float(
                        np.mean((specialist_values - control_values) > 0)
                    ),
                    "bootstrap_replicates": replicates,
                }
            )
            saved_bootstrap[f"bit_{bit_width}_{target_domain}"] = {
                "mean_specialist_contrast": mean_specialist_boot,
                "mean_control_contrast": mean_control_boot,
                "mean_specialist_minus_control_difference": mean_difference_boot,
            }
    return output, saved_bootstrap


def _correlation_rows(
    panel: Sequence[Mapping[str, Any]],
    bits: Sequence[int],
    contrast_rows: Sequence[Mapping[str, Any]],
    contrast_bootstrap: np.ndarray,
    masking_contrasts: Mapping[tuple[str, int, int], float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contrast_lookup = {
        (int(row["bit_width"]), row["intervention_id"]): row
        for row in contrast_rows
    }
    comparison_rows = []
    correlations = []
    predictors = (
        ("functional_specialization", "specialization_margin"),
        ("routing_specialization", "routing_specialization_margin"),
        ("target_routing_frequency", "target_routing_frequency"),
    )
    for bit_index, bit_width in enumerate(bits):
        quant_values = np.asarray(
            [
                contrast_lookup[(bit_width, row["intervention_id"])][
                    "target_minus_mean_other_contrast"
                ]
                for row in panel
            ],
            dtype=np.float64,
        )
        mask_values = np.asarray(
            [
                masking_contrasts[(row["role"], row["layer"], row["expert_id"])]
                for row in panel
            ],
            dtype=np.float64,
        )
        for row, masking, quantization in zip(
            panel, mask_values, quant_values, strict=True
        ):
            comparison_rows.append(
                {
                    "bit_width": bit_width,
                    "intervention_id": row["intervention_id"],
                    "pair_id": row["pair_id"],
                    "role": row["role"],
                    "target_domain": row["target_domain"],
                    "layer": row["layer"],
                    "expert_id": row["expert_id"],
                    "masking_causal_contrast": masking,
                    "quantization_causal_contrast": quantization,
                    "sign_agreement": bool(np.sign(masking) == np.sign(quantization)),
                    "functional_specialization": row["baseline_record"][
                        "specialization_margin"
                    ],
                    "routing_specialization": row["baseline_record"][
                        "routing_specialization_margin"
                    ],
                    "target_routing_frequency": row["baseline_record"][
                        "target_routing_frequency"
                    ],
                }
            )
        boot = contrast_bootstrap[bit_index]
        spearman_boot = np.asarray(
            [_safe_spearman(mask_values, boot[:, index]) for index in range(boot.shape[1])]
        )
        kendall_boot = np.asarray(
            [_safe_kendall(mask_values, boot[:, index]) for index in range(boot.shape[1])]
        )
        sign_boot = np.asarray(
            [
                np.mean(np.sign(mask_values) == np.sign(boot[:, index]))
                for index in range(boot.shape[1])
            ],
            dtype=np.float64,
        )
        spearman_low, spearman_high = _finite_ci(spearman_boot)
        kendall_low, kendall_high = _finite_ci(kendall_boot)
        sign_low, sign_high = _ci(sign_boot)
        correlations.append(
            {
                "bit_width": bit_width,
                "predictor": "masking_causal_contrast",
                "outcome": "quantization_causal_contrast",
                "num_interventions": len(panel),
                "spearman": _safe_spearman(mask_values, quant_values),
                "spearman_ci_low": spearman_low,
                "spearman_ci_high": spearman_high,
                "kendall_tau": _safe_kendall(mask_values, quant_values),
                "kendall_tau_ci_low": kendall_low,
                "kendall_tau_ci_high": kendall_high,
                "sign_agreement": float(
                    np.mean(np.sign(mask_values) == np.sign(quant_values))
                ),
                "sign_agreement_ci_low": sign_low,
                "sign_agreement_ci_high": sign_high,
            }
        )
        for predictor_name, record_key in predictors:
            predictor = np.asarray(
                [float(row["baseline_record"][record_key]) for row in panel]
            )
            predictor_boot = np.asarray(
                [
                    _safe_spearman(predictor, boot[:, index])
                    for index in range(boot.shape[1])
                ]
            )
            low, high = _finite_ci(predictor_boot)
            correlations.append(
                {
                    "bit_width": bit_width,
                    "predictor": predictor_name,
                    "outcome": "quantization_causal_contrast",
                    "num_interventions": len(panel),
                    "spearman": _safe_spearman(predictor, quant_values),
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "kendall_tau": None,
                    "kendall_tau_ci_low": None,
                    "kendall_tau_ci_high": None,
                    "sign_agreement": None,
                    "sign_agreement_ci_low": None,
                    "sign_agreement_ci_high": None,
                }
            )
    return comparison_rows, correlations


def _risk_correlations(
    bits: Sequence[int],
    result_rows: Sequence[Mapping[str, Any]],
    bootstrap_domain_means: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    for bit_index, bit_width in enumerate(bits):
        rows = [row for row in result_rows if int(row["bit_width"]) == bit_width]
        actual = np.asarray([float(row["delta_nll"]) for row in rows])
        boot = bootstrap_domain_means[bit_index].reshape(
            -1, bootstrap_domain_means.shape[-1]
        )
        for proxy_name in ("risk_functional", "risk_routing"):
            proxy = np.asarray([float(row[proxy_name]) for row in rows])
            correlations = np.asarray(
                [_safe_spearman(proxy, boot[:, index]) for index in range(boot.shape[1])]
            )
            low, high = _finite_ci(correlations)
            output.append(
                {
                    "bit_width": bit_width,
                    "predictor": proxy_name,
                    "outcome": "domain_level_quantization_delta_nll",
                    "num_expert_domain_observations": len(rows),
                    "spearman": _safe_spearman(proxy, actual),
                    "spearman_ci_low": low,
                    "spearman_ci_high": high,
                    "formula_tuned_on_pilot_outcomes": False,
                }
            )
    return output


def _distortion_rows(
    panel: Sequence[Mapping[str, Any]],
    bits: Sequence[int],
    metadata: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for bit_width in bits:
        for row in panel:
            values = metadata[(bit_width, row["layer"], row["expert_id"])]
            memory = values["memory_accounting"]
            output.append(
                {
                    "bit_width": bit_width,
                    "intervention_id": row["intervention_id"],
                    "pair_id": row["pair_id"],
                    "role": row["role"],
                    "target_domain": row["target_domain"],
                    "layer": row["layer"],
                    "expert_id": row["expert_id"],
                    "quantization_distortion": values["quantization_distortion"],
                    "weight_count": memory["weight_count"],
                    "number_of_groups": memory["number_of_groups"],
                    "quantized_weight_payload_bits": memory[
                        "quantized_weight_payload_bits"
                    ],
                    "weight_packing_padding_bits": memory[
                        "weight_packing_padding_bits"
                    ],
                    "scale_storage_bits": memory["scale_storage_bits"],
                    "projected_bytes": memory["projected_bytes"],
                    "raw_nominal_bit_width": memory["raw_nominal_bit_width"],
                    "effective_bits_per_weight": memory["effective_bits_per_weight"],
                    "compression_ratio_vs_bf16": memory[
                        "compression_ratio_vs_bf16"
                    ],
                    "original_expert_fingerprint": values[
                        "original_expert_fingerprint"
                    ],
                    "quantized_expert_fingerprint": values[
                        "quantized_expert_fingerprint"
                    ],
                    "exact_restoration_verified": values[
                        "exact_restoration_verified"
                    ],
                    "unrelated_experts_verified_unchanged": values[
                        "unrelated_experts_verified_unchanged"
                    ],
                    "projected_not_measured_runtime_memory": True,
                }
            )
    return output


def _evaluate_gates(
    bits: Sequence[int],
    domain_means: np.ndarray,
    panel: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    baseline_noise_nll: float,
) -> list[dict[str, Any]]:
    clear_threshold = max(1e-7, 10.0 * max(0.0, baseline_noise_nll))
    output = []
    for bit_index, bit_width in enumerate(bits):
        overall = next(
            row
            for row in aggregate_rows
            if int(row["bit_width"]) == bit_width and row["scope"] == "overall"
        )
        domains = [
            row
            for row in aggregate_rows
            if int(row["bit_width"]) == bit_width and row["scope"] == "domain"
        ]
        domains.sort(key=lambda row: BALANCED_DOMAINS.index(row["target_domain"]))
        positive_domains = [
            row["target_domain"]
            for row in domains
            if float(row["mean_specialist_contrast"]) > 0
        ]
        median_absolute_delta = float(np.median(np.abs(domain_means[bit_index])))
        gate_a = bool(
            overall["mean_specialist_contrast"] > 0
            and overall["mean_specialist_contrast_ci_low"] > 0
        )
        gate_b = len(positive_domains) >= 3
        gate_c = bool(overall["mean_specialist_minus_control_difference"] > 0)
        gate_d = median_absolute_delta > clear_threshold
        fallback_eligible = bool(
            not gate_d
            and abs(float(overall["mean_specialist_contrast"])) <= clear_threshold
            and abs(float(overall["mean_specialist_minus_control_difference"]))
            <= clear_threshold
            and all(
                float(row["mean_specialist_contrast"]) >= -clear_threshold
                for row in domains
            )
        )
        output.append(
            {
                "bit_width": bit_width,
                "gate_a": {
                    "passed": gate_a,
                    "overall_mean_specialist_contrast": overall[
                        "mean_specialist_contrast"
                    ],
                    "ci_low": overall["mean_specialist_contrast_ci_low"],
                    "ci_high": overall["mean_specialist_contrast_ci_high"],
                    "rationale": (
                        "requires a positive overall specialist contrast with a "
                        "95% interval strictly above zero"
                    ),
                },
                "gate_b": {
                    "passed": gate_b,
                    "positive_domain_count": len(positive_domains),
                    "positive_domains": positive_domains,
                    "domain_contrasts": {
                        row["target_domain"]: row["mean_specialist_contrast"]
                        for row in domains
                    },
                    "rationale": (
                        "requires positive aggregate specialist contrasts in 3 of 4 domains"
                    ),
                },
                "gate_c": {
                    "passed": gate_c,
                    "overall_mean_specialist_minus_control": overall[
                        "mean_specialist_minus_control_difference"
                    ],
                    "ci_low": overall["mean_difference_ci_low"],
                    "ci_high": overall["mean_difference_ci_high"],
                    "ci_excludes_zero": overall["mean_difference_ci_low"] > 0,
                    "rationale": "requires a positive overall specialist-control difference",
                },
                "gate_d": {
                    "passed": gate_d,
                    "statistic": "median absolute domain-level delta NLL",
                    "observed": median_absolute_delta,
                    "clear_noise_threshold": clear_threshold,
                    "baseline_reproduction_noise_nll": baseline_noise_nll,
                    "rationale": (
                        "requires perturbations above the preregistered numerical-noise rule"
                    ),
                },
                "all_required_gates_passed": bool(gate_a and gate_b and gate_c and gate_d),
                "three_bit_fallback_eligible_due_only_to_small_effects": fallback_eligible,
            }
        )
    return output


def _stage1_decision(
    gates: Sequence[Mapping[str, Any]],
    evaluated_bits: Sequence[int],
    primary_bits: int,
    fallback_bits: int,
) -> dict[str, Any]:
    passing = [
        int(row["bit_width"]) for row in gates if row["all_required_gates_passed"]
    ]
    if passing:
        chosen = primary_bits if primary_bits in passing else passing[0]
        return {
            "decision": "GO",
            "passing_bit_widths": passing,
            "selected_bit_width": chosen,
            "gates_by_bit_width": list(gates),
            "rationale": (
                f"{chosen}-bit satisfied all four preregistered Stage-1 gates; "
                "Stage 2 remains a separate future task"
            ),
        }
    primary = next((row for row in gates if row["bit_width"] == primary_bits), None)
    if (
        primary is not None
        and primary["three_bit_fallback_eligible_due_only_to_small_effects"]
        and fallback_bits not in evaluated_bits
    ):
        return {
            "decision": "PENDING_FALLBACK",
            "passing_bit_widths": [],
            "selected_bit_width": None,
            "gates_by_bit_width": list(gates),
            "rationale": (
                f"{primary_bits}-bit failed only the preregistered measurability rule; "
                f"run the {fallback_bits}-bit fallback before a final decision"
            ),
        }
    return {
        "decision": "NO_GO",
        "passing_bit_widths": [],
        "selected_bit_width": None,
        "gates_by_bit_width": list(gates),
        "rationale": (
            "No evaluated low-bit setting satisfied all four preregistered Stage-1 gates"
        ),
    }


def _derived_seed(seed: int, label: str) -> int:
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:16], 16) % (2**63 - 1)


def _ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _finite_ci(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    return _ci(finite)


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return math.nan
    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else math.nan


def _safe_kendall(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return math.nan
    value = float(kendalltau(first, second).statistic)
    return value if math.isfinite(value) else math.nan


def write_quantization_outputs(
    analysis: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    output_dir: Path,
) -> None:
    write_csv(
        output_dir / "quantization_pilot_results.csv",
        analysis["quantization_pilot_results"],
        quantization_result_fields(),
    )
    write_csv(
        output_dir / "quantization_pilot_pairwise.csv",
        analysis["quantization_pilot_pairwise"],
        pairwise_fields(),
    )
    write_csv(
        output_dir / "specialist_vs_control.csv",
        analysis["specialist_vs_control"],
        specialist_control_fields(),
    )
    write_csv(
        output_dir / "quantization_vs_masking.csv",
        analysis["quantization_vs_masking"],
        quantization_masking_fields(),
    )
    write_csv(
        output_dir / "quantization_distortion.csv",
        analysis["quantization_distortion"],
        distortion_fields(),
    )
    atomic_save_npz(output_dir / "per_example_quantization_losses.npz", **arrays)
    atomic_write_json(output_dir / "stage1_decision.json", analysis["stage1_decision"])


def write_quantization_summary(
    results: Mapping[str, Any], output_path: Path
) -> str:
    prereg = results["pilot_panel_preregistration"]
    analysis = results["quantization_analysis"]
    config = results["run_config"]
    lines = [
        "# OLMoE Stage-1 Quantization Sensitivity Pilot",
        "",
        "## Experimental Setup",
        "",
        f"- Model: `{config['model']}`",
        f"- Revision: `{config['model_revision']}`",
        f"- Runtime: {config.get('runtime_description', config['device'])}, {config['dtype']}",
        (
            f"- Frozen inputs: {EXPECTED_EXAMPLES} examples/domain, "
            f"{EXPECTED_MEASURED_POSITIONS} measured positions/example, "
            f"{EXPECTED_EXAMPLES * EXPECTED_MEASURED_POSITIONS:,} positions/domain"
        ),
        (
            "- Intervention: symmetric group-wise expert-only weight QDQ, one expert "
            f"at a time, group size {config['group_size']} along input features"
        ),
        "- Memory values are projected packed expert storage, not measured runtime memory.",
        "- Uncertainty: 1,000 deterministic bootstrap replicates from per-example losses.",
        "",
        "## Frozen Pilot Panel",
        "",
        (
            "The panel was selected only from the frozen functional-specialization "
            f"statistics and controls, with fingerprint `{prereg['pilot_panel_fingerprint']}`."
        ),
        "",
        "| Target | Specialist | Frozen functional margin | Rank | Matched control |",
        "|---|---|---:|---:|---|",
    ]
    for pair in prereg["pairs"]:
        specialist = pair["specialist"]
        control = pair["matched_control"]
        lines.append(
            f"| {pair['target_domain'].title()} | L{specialist['layer']}/E"
            f"{specialist['expert_id']} | {specialist['specialization_margin']:.6f} | "
            f"{pair['pilot_functional_specialization_rank']} | L{control['layer']}/E"
            f"{control['expert_id']} |"
        )

    lines.extend(["", "## Quantization Results", ""])
    for bit_width in analysis["evaluated_bits"]:
        lines.extend(
            [
                f"### {bit_width}-bit",
                "",
                (
                    "| Scope | Specialist contrast | 95% CI | Control contrast | "
                    "Specialist-control | 95% CI |"
                ),
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        rows = [
            row
            for row in analysis["aggregate_results"]
            if int(row["bit_width"]) == int(bit_width)
        ]
        rows.sort(
            key=lambda row: (
                row["scope"] == "overall",
                BALANCED_DOMAINS.index(row["target_domain"])
                if row["target_domain"] in BALANCED_DOMAINS
                else len(BALANCED_DOMAINS),
            )
        )
        for row in rows:
            label = (
                "All domains"
                if row["target_domain"] == "all"
                else row["target_domain"].title()
            )
            lines.append(
                f"| {label} | {row['mean_specialist_contrast']:+.8f} | "
                f"[{row['mean_specialist_contrast_ci_low']:+.8f}, "
                f"{row['mean_specialist_contrast_ci_high']:+.8f}] | "
                f"{row['mean_control_contrast']:+.8f} | "
                f"{row['mean_specialist_minus_control_difference']:+.8f} | "
                f"[{row['mean_difference_ci_low']:+.8f}, "
                f"{row['mean_difference_ci_high']:+.8f}] |"
            )
        gate = next(
            row
            for row in analysis["gates_by_bit_width"]
            if int(row["bit_width"]) == int(bit_width)
        )
        lines.extend(
            [
                "",
                "Gate results: "
                + ", ".join(
                    f"{name[-1].upper()}={'PASS' if gate[name]['passed'] else 'FAIL'}"
                    for name in ("gate_a", "gate_b", "gate_c", "gate_d")
                )
                + ".",
                "",
            ]
        )

    lines.extend(
        [
            "## Correlations",
            "",
            "| Bits | Predictor | Outcome | Spearman | Kendall tau | Sign agreement |",
            "|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in analysis["correlation_results"]:
        lines.append(
            f"| {row['bit_width']} | {row['predictor']} | {row['outcome']} | "
            f"{_format_optional(row['spearman'])} | "
            f"{_format_optional(row['kendall_tau'])} | "
            f"{_format_optional(row['sign_agreement'])} |"
        )
    for row in analysis["risk_proxy_correlations"]:
        lines.append(
            f"| {row['bit_width']} | {row['predictor']} | {row['outcome']} | "
            f"{_format_optional(row['spearman'])} | — | — |"
        )

    decision = analysis["stage1_decision"]
    lines.extend(
        [
            "",
            "## Stage-1 Decision",
            "",
            f"### {decision['decision']}",
            "",
            decision["rationale"] + ".",
            "",
            "This file does not authorize or implement Stage 2 or a mixed-precision optimizer.",
            "",
            "## Limitations",
            "",
            (
                "- Fake quantization measures numerical effects after QDQ but does not "
                "provide a low-bit kernel or measured runtime-memory savings."
            ),
            (
                "- Results condition on one checkpoint, the frozen controlled corpora, "
                "and 16 selected interventions."
            ),
            (
                "- Bootstrap intervals capture example sampling only; they do not cover "
                "checkpoint, dataset, prompt, or expert-selection uncertainty."
            ),
            (
                "- The pilot reuses the causal-validation inputs to test mechanism "
                "consistency, not held-out generalization."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def create_quantization_figures(
    results: Mapping[str, Any], output_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = results["quantization_analysis"]
    bits = analysis["evaluated_bits"]
    # If the preregistered fallback ran, plot the setting that determines the final
    # decision; all evaluated bit widths remain tabulated in the CSV/JSON outputs.
    primary = analysis["stage1_decision"].get("selected_bit_width") or bits[-1]
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = {
        "general": "#4C78A8",
        "math": "#F58518",
        "coding": "#54A24B",
        "reasoning": "#E45756",
    }

    rows = [
        row
        for row in analysis["intervention_contrasts"]
        if row["bit_width"] == primary and row["role"] == "specialist"
    ]
    labels = [f"L{row['layer']}/E{row['expert_id']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        [row["target_delta_nll"] for row in rows],
        width,
        label="Target",
        color=[colors[row["target_domain"]] for row in rows],
    )
    ax.bar(
        x + width / 2,
        [row["mean_non_target_delta_nll"] for row in rows],
        width,
        label="Mean non-target",
        color="#B9B9B9",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("ΔNLL (nats/token)")
    ax.set_title(f"Figure 1: Specialist target vs non-target ΔNLL ({primary}-bit)")
    ax.legend()
    paths.extend(_save_figure(fig, figure_dir / "figure_1_specialist_target_vs_nontarget", plt))

    rows = [
        row
        for row in analysis["specialist_vs_control"]
        if row["bit_width"] == primary
    ]
    labels = [
        f"{row['target_domain'][0].upper()} L{row['specialist_layer']}/E"
        f"{row['specialist_expert_id']}"
        for row in rows
    ]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        [row["specialist_contrast"] for row in rows],
        width,
        label="Specialist",
        color=[colors[row["target_domain"]] for row in rows],
    )
    ax.bar(
        x + width / 2,
        [row["control_contrast"] for row in rows],
        width,
        label="Matched control",
        color="#9D9D9D",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Target-minus-mean-other ΔNLL")
    ax.set_title(f"Figure 2: Specialist vs matched-control contrasts ({primary}-bit)")
    ax.legend()
    paths.extend(_save_figure(fig, figure_dir / "figure_2_specialist_vs_control", plt))

    rows = [
        row
        for row in analysis["quantization_vs_masking"]
        if row["bit_width"] == primary
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for domain in BALANCED_DOMAINS:
        subset = [row for row in rows if row["target_domain"] == domain]
        ax.scatter(
            [row["masking_causal_contrast"] for row in subset],
            [row["quantization_causal_contrast"] for row in subset],
            label=domain.title(),
            color=colors[domain],
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Frozen masking causal contrast")
    ax.set_ylabel("Quantization causal contrast")
    ax.set_title(f"Figure 3: Masking vs quantization specificity ({primary}-bit)")
    ax.legend()
    paths.extend(_save_figure(fig, figure_dir / "figure_3_masking_vs_quantization", plt))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for domain in BALANCED_DOMAINS:
        subset = [row for row in rows if row["target_domain"] == domain]
        ax.scatter(
            [row["functional_specialization"] for row in subset],
            [row["quantization_causal_contrast"] for row in subset],
            label=domain.title(),
            color=colors[domain],
            alpha=0.85,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Frozen functional specialization")
    ax.set_ylabel("Quantization causal contrast")
    ax.set_title(f"Figure 4: Functional vs quantization specificity ({primary}-bit)")
    ax.legend()
    paths.extend(_save_figure(fig, figure_dir / "figure_4_functional_vs_quantization", plt))

    domain_rows = [
        row
        for row in analysis["quantization_pilot_results"]
        if row["bit_width"] == primary
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, proxy, label in (
        (axes[0], "quantization_distortion", "Weight distortion"),
        (axes[1], "risk_functional", "Functional risk proxy"),
    ):
        for domain in BALANCED_DOMAINS:
            subset = [row for row in domain_rows if row["domain"] == domain]
            axis.scatter(
                [row[proxy] for row in subset],
                [row["delta_nll"] for row in subset],
                label=domain.title(),
                color=colors[domain],
                alpha=0.75,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel(label)
        axis.set_ylabel("Actual ΔNLL")
    axes[1].legend()
    fig.suptitle(f"Figure 5: Distortion/risk proxy vs actual ΔNLL ({primary}-bit)")
    paths.extend(_save_figure(fig, figure_dir / "figure_5_distortion_risk_vs_delta_nll", plt))
    return paths


def quantization_result_fields() -> list[str]:
    return [
        "bit_width", "intervention_id", "pair_id", "role", "target_domain",
        "layer", "expert_id", "domain", "is_target_domain", "examples",
        "evaluated_tokens", "baseline_nll", "quantized_nll", "delta_nll",
        "delta_nll_ci_low", "delta_nll_ci_high", "normalized_relative_delta_nll",
        "relative_delta_percent", "positive_delta_example_fraction",
        "routing_coverage", "routing_frequency", "functional_importance",
        "functional_rank", "quantization_distortion", "risk_functional",
        "risk_routing", "effective_bits_per_weight", "projected_bytes",
        "compression_ratio_vs_bf16", "bootstrap_replicates",
    ]


def pairwise_fields() -> list[str]:
    return [
        "bit_width", "intervention_id", "pair_id", "role", "target_domain",
        "comparison_domain", "layer", "expert_id", "target_delta_nll",
        "comparison_delta_nll", "target_minus_comparison_delta_nll",
        "contrast_ci_low", "contrast_ci_high", "positive_contrast_ci_excludes_zero",
        "bootstrap_replicates",
    ]


def specialist_control_fields() -> list[str]:
    return [
        "bit_width", "pair_id", "target_domain", "specialist_layer",
        "specialist_expert_id", "control_layer", "control_expert_id",
        "specialist_contrast", "specialist_contrast_ci_low",
        "specialist_contrast_ci_high", "control_contrast", "control_contrast_ci_low",
        "control_contrast_ci_high", "specialist_minus_control_difference",
        "difference_ci_low", "difference_ci_high", "difference_positive",
        "positive_difference_ci_excludes_zero", "bootstrap_replicates",
    ]


def quantization_masking_fields() -> list[str]:
    return [
        "bit_width", "intervention_id", "pair_id", "role", "target_domain",
        "layer", "expert_id", "masking_causal_contrast",
        "quantization_causal_contrast", "sign_agreement", "functional_specialization",
        "routing_specialization", "target_routing_frequency",
    ]


def distortion_fields() -> list[str]:
    return [
        "bit_width", "intervention_id", "pair_id", "role", "target_domain",
        "layer", "expert_id", "quantization_distortion", "weight_count",
        "number_of_groups", "quantized_weight_payload_bits", "scale_storage_bits",
        "weight_packing_padding_bits", "projected_bytes", "raw_nominal_bit_width",
        "effective_bits_per_weight",
        "compression_ratio_vs_bf16", "original_expert_fingerprint",
        "quantized_expert_fingerprint", "exact_restoration_verified",
        "unrelated_experts_verified_unchanged", "projected_not_measured_runtime_memory",
    ]


def _save_figure(figure: Any, base: Path, plt: Any) -> list[Path]:
    figure.tight_layout()
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    return paths


def _format_optional(value: Any) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):+.4f}"
