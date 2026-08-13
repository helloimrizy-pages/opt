from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .balanced import BALANCED_DOMAINS
from .io_utils import atomic_save_npz, atomic_write_json, write_csv
from .masking import LossStatistics
from .statistics import confidence_interval, safe_spearman


DOMAIN_COLORS = {
    "general": "#4C78A8",
    "math": "#F58518",
    "coding": "#54A24B",
    "reasoning": "#B279A2",
}


def intervention_panel(preregistration: Mapping[str, Any]) -> list[dict[str, Any]]:
    panel: list[dict[str, Any]] = []
    for pair in preregistration["matched_controls"]:
        specialist = pair["specialized"]
        panel.append(
            {
                "intervention_id": f"specialized_{pair['pair_id']}",
                "pair_id": pair["pair_id"],
                "role": "specialized",
                "target_domain": pair["target_domain"],
                "layer": specialist["layer"],
                "expert_id": specialist["expert_id"],
                "baseline_record": specialist,
            }
        )
    for pair in preregistration["matched_controls"]:
        control = pair["control"]
        panel.append(
            {
                "intervention_id": (
                    f"control_{pair['pair_id']}_L{control['layer']}_E{control['expert_id']}"
                ),
                "pair_id": pair["pair_id"],
                "role": "control",
                "target_domain": pair["target_domain"],
                "layer": control["layer"],
                "expert_id": control["expert_id"],
                "baseline_record": control,
            }
        )
    identities = {(row["layer"], row["expert_id"]) for row in panel}
    if len(identities) != len(panel):
        raise RuntimeError("The frozen panel contains a repeated intervention identity")
    return panel


def analyze_balanced_interventions(
    preregistration: Mapping[str, Any],
    baselines: Mapping[str, LossStatistics],
    masked: Mapping[tuple[int, int, str], LossStatistics],
    provenance: Mapping[tuple[int, int], Mapping[str, Any]],
    bootstrap_replicates: int = 1000,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if bootstrap_replicates < 1:
        raise ValueError("Balanced analysis requires at least one bootstrap replicate")
    panel = intervention_panel(preregistration)
    _validate_loss_inputs(panel, baselines, masked)
    deltas: dict[str, dict[str, np.ndarray]] = {}
    domain_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for intervention in panel:
        identifier = intervention["intervention_id"]
        identity = (intervention["layer"], intervention["expert_id"])
        record = intervention["baseline_record"]
        deltas[identifier] = {}
        row_by_domain: dict[str, dict[str, Any]] = {}
        for domain in BALANCED_DOMAINS:
            baseline = baselines[domain]
            result = masked[(identity[0], identity[1], domain)]
            values = result.per_token_nll - baseline.per_token_nll
            deltas[identifier][domain] = values
            bootstrap = _bootstrap_mean(
                values,
                bootstrap_replicates,
                _seed(seed, f"domain:{identifier}:{domain}"),
            )
            _, ci_low, ci_high = confidence_interval(bootstrap)
            route_counts = result.route_counts
            if route_counts is None:
                raise RuntimeError(f"Intervention {identifier}/{domain} has no route counts")
            total_tokens = int(result.token_counts.sum())
            row = {
                "intervention_id": identifier,
                "pair_id": intervention["pair_id"],
                "role": intervention["role"],
                "target_domain": intervention["target_domain"],
                "layer": identity[0],
                "expert_id": identity[1],
                "domain": domain,
                "is_target_domain": domain == intervention["target_domain"],
                "examples": len(values),
                "evaluated_tokens": total_tokens,
                "baseline_nll": float(
                    baseline.loss_sums.sum() / baseline.token_counts.sum()
                ),
                "masked_nll": float(result.loss_sums.sum() / result.token_counts.sum()),
                "delta_nll": float(values.mean()),
                "delta_nll_ci_low": ci_low,
                "delta_nll_ci_high": ci_high,
                "positive_delta_example_fraction": float(np.mean(values > 0)),
                "masked_routes": int(route_counts.sum()),
                "observed_routing_coverage": float(route_counts.sum() / total_tokens),
                "baseline_routing_coverage": float(
                    record["routing_coverage_by_domain"][domain]
                ),
                "baseline_routing_frequency": float(
                    record["routing_frequency_by_domain"][domain]
                ),
                "baseline_normalized_contribution": float(
                    record["normalized_contribution_by_domain"][domain]
                ),
                "baseline_functional_rank": float(
                    record["functional_rank_by_domain"][domain]
                ),
                "baseline_specialization_margin": float(
                    record["specialization_margin"]
                ),
                "baseline_routing_specialization_margin": float(
                    record["routing_specialization_margin"]
                ),
                "target_routing_frequency": float(record["target_routing_frequency"]),
                "target_routing_coverage": float(record["target_routing_coverage"]),
                "inference_source": provenance[identity]["source"],
                "bootstrap_replicates": bootstrap_replicates,
            }
            if not np.isclose(
                row["observed_routing_coverage"],
                row["baseline_routing_coverage"],
                rtol=0,
                atol=0,
            ):
                raise RuntimeError(
                    f"Observed routing coverage changed for {identifier}/{domain}"
                )
            domain_rows.append(row)
            row_by_domain[domain] = row

        target = intervention["target_domain"]
        others = [domain for domain in BALANCED_DOMAINS if domain != target]
        domain_bootstrap = _bootstrap_domain_means(
            deltas[identifier],
            bootstrap_replicates,
            _seed(seed, f"primary:{identifier}"),
        )
        target_index = BALANCED_DOMAINS.index(target)
        other_indices = [BALANCED_DOMAINS.index(domain) for domain in others]
        target_bootstrap = domain_bootstrap[:, target_index]
        mean_other_bootstrap = domain_bootstrap[:, other_indices].mean(axis=1)
        contrast_bootstrap = target_bootstrap - mean_other_bootstrap
        _, other_ci_low, other_ci_high = confidence_interval(mean_other_bootstrap)
        _, contrast_ci_low, contrast_ci_high = confidence_interval(contrast_bootstrap)
        target_delta = float(deltas[identifier][target].mean())
        mean_other_delta = float(
            np.mean([deltas[identifier][domain].mean() for domain in others])
        )
        summary = {
            "intervention_id": identifier,
            "pair_id": intervention["pair_id"],
            "role": intervention["role"],
            "target_domain": target,
            "layer": identity[0],
            "expert_id": identity[1],
            "target_delta_nll": target_delta,
            "target_delta_nll_ci_low": row_by_domain[target]["delta_nll_ci_low"],
            "target_delta_nll_ci_high": row_by_domain[target]["delta_nll_ci_high"],
            "mean_non_target_delta_nll": mean_other_delta,
            "mean_non_target_delta_nll_ci_low": other_ci_low,
            "mean_non_target_delta_nll_ci_high": other_ci_high,
            "target_minus_mean_other_contrast": target_delta - mean_other_delta,
            "contrast_ci_low": contrast_ci_low,
            "contrast_ci_high": contrast_ci_high,
            "contrast_positive": bool(target_delta - mean_other_delta > 0),
            "contrast_ci_excludes_zero": bool(
                contrast_ci_low > 0 or contrast_ci_high < 0
            ),
            "positive_contrast_ci_excludes_zero": bool(contrast_ci_low > 0),
            "baseline_specialization_margin": float(
                record["specialization_margin"]
            ),
            "baseline_routing_specialization_margin": float(
                record["routing_specialization_margin"]
            ),
            "target_routing_frequency": float(record["target_routing_frequency"]),
            "target_routing_coverage": float(record["target_routing_coverage"]),
            "inference_source": provenance[identity]["source"],
            "bootstrap_replicates": bootstrap_replicates,
            "domain_effects": {domain: row_by_domain[domain] for domain in BALANCED_DOMAINS},
        }
        summaries.append(summary)
        for comparison in others:
            comparison_index = BALANCED_DOMAINS.index(comparison)
            pair_bootstrap = target_bootstrap - domain_bootstrap[:, comparison_index]
            _, pair_ci_low, pair_ci_high = confidence_interval(pair_bootstrap)
            comparison_delta = float(deltas[identifier][comparison].mean())
            pairwise_rows.append(
                {
                    "intervention_id": identifier,
                    "pair_id": intervention["pair_id"],
                    "role": intervention["role"],
                    "target_domain": target,
                    "comparison_domain": comparison,
                    "layer": identity[0],
                    "expert_id": identity[1],
                    "target_delta_nll": target_delta,
                    "comparison_delta_nll": comparison_delta,
                    "target_minus_comparison_delta_nll": target_delta
                    - comparison_delta,
                    "contrast_ci_low": pair_ci_low,
                    "contrast_ci_high": pair_ci_high,
                    "positive_contrast_ci_excludes_zero": bool(pair_ci_low > 0),
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )

    summary_by_id = {row["intervention_id"]: row for row in summaries}
    specialized_control_rows = _paired_control_analysis(
        preregistration,
        summary_by_id,
        deltas,
        bootstrap_replicates,
        seed,
    )
    aggregate_rows, aggregate_bootstrap = _aggregate_analysis(
        preregistration,
        summary_by_id,
        deltas,
        bootstrap_replicates,
        seed,
    )
    correlations = _correlation_analysis(
        panel, summary_by_id, deltas, bootstrap_replicates, seed
    )
    decision = _decision(aggregate_rows, specialized_control_rows)
    arrays = _per_example_arrays(panel, baselines, masked, deltas)
    results = {
        "analysis": {
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "primary_contrast": (
                "target delta NLL minus mean of all three non-target delta NLLs"
            ),
            "domains_resampled_independently": True,
            "paired_specialist_control_indices_within_domain": True,
            "aggregate_intervals_condition_on_the_frozen_expert_panel": True,
        },
        "intervention_panel": [
            {key: value for key, value in row.items() if key != "baseline_record"}
            for row in panel
        ],
        "masking_results": domain_rows,
        "intervention_contrasts": summaries,
        "pairwise_domain_contrasts": pairwise_rows,
        "specialized_vs_control": specialized_control_rows,
        "aggregate_results": aggregate_rows,
        "aggregate_bootstrap": aggregate_bootstrap,
        "correlation_results": correlations,
        "decision": decision,
    }
    return results, arrays


def _validate_loss_inputs(
    panel: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, LossStatistics],
    masked: Mapping[tuple[int, int, str], LossStatistics],
) -> None:
    if set(baselines) != set(BALANCED_DOMAINS):
        raise RuntimeError("Baseline loss artifacts do not cover all four domains")
    for domain in BALANCED_DOMAINS:
        baselines[domain].validate()
        if len(baselines[domain].loss_sums) != 100:
            raise RuntimeError(f"Baseline loss artifact has wrong size for {domain}")
        if not np.all(baselines[domain].token_counts == 64):
            raise RuntimeError(f"Baseline token counts are not exactly 64 for {domain}")
    for intervention in panel:
        for domain in BALANCED_DOMAINS:
            key = (intervention["layer"], intervention["expert_id"], domain)
            if key not in masked:
                raise RuntimeError(f"Missing intervention loss artifact: {key}")
            masked[key].validate()
            if not np.array_equal(
                masked[key].token_counts, baselines[domain].token_counts
            ):
                raise RuntimeError(f"Loss token counts differ for {key}")


def _bootstrap_mean(values: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return values[indices].mean(axis=1)


def _bootstrap_domain_means(
    values: Mapping[str, np.ndarray], replicates: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.zeros((replicates, len(BALANCED_DOMAINS)), dtype=np.float64)
    for domain_index, domain in enumerate(BALANCED_DOMAINS):
        domain_values = values[domain]
        indices = rng.integers(
            0, len(domain_values), size=(replicates, len(domain_values))
        )
        output[:, domain_index] = domain_values[indices].mean(axis=1)
    return output


def _paired_control_analysis(
    preregistration: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, np.ndarray]],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in preregistration["matched_controls"]:
        specialized_id = f"specialized_{pair['pair_id']}"
        control = pair["control"]
        control_id = (
            f"control_{pair['pair_id']}_L{control['layer']}_E{control['expert_id']}"
        )
        specialized_summary = summaries[specialized_id]
        control_summary = summaries[control_id]
        target = pair["target_domain"]
        rng = np.random.default_rng(_seed(seed, f"paired:{pair['pair_id']}"))
        specialized_means = np.zeros((replicates, len(BALANCED_DOMAINS)))
        control_means = np.zeros_like(specialized_means)
        for domain_index, domain in enumerate(BALANCED_DOMAINS):
            size = len(deltas[specialized_id][domain])
            indices = rng.integers(0, size, size=(replicates, size))
            specialized_means[:, domain_index] = deltas[specialized_id][domain][
                indices
            ].mean(axis=1)
            control_means[:, domain_index] = deltas[control_id][domain][indices].mean(
                axis=1
            )
        target_index = BALANCED_DOMAINS.index(target)
        other_indices = [index for index in range(4) if index != target_index]
        specialized_bootstrap = specialized_means[:, target_index] - specialized_means[
            :, other_indices
        ].mean(axis=1)
        control_bootstrap = control_means[:, target_index] - control_means[
            :, other_indices
        ].mean(axis=1)
        difference_bootstrap = specialized_bootstrap - control_bootstrap
        _, difference_low, difference_high = confidence_interval(difference_bootstrap)
        difference = (
            specialized_summary["target_minus_mean_other_contrast"]
            - control_summary["target_minus_mean_other_contrast"]
        )
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "target_domain": target,
                "specialized_layer": pair["specialized"]["layer"],
                "specialized_expert_id": pair["specialized"]["expert_id"],
                "control_layer": control["layer"],
                "control_expert_id": control["expert_id"],
                "specialized_contrast": specialized_summary[
                    "target_minus_mean_other_contrast"
                ],
                "specialized_contrast_ci_low": specialized_summary["contrast_ci_low"],
                "specialized_contrast_ci_high": specialized_summary["contrast_ci_high"],
                "control_contrast": control_summary["target_minus_mean_other_contrast"],
                "control_contrast_ci_low": control_summary["contrast_ci_low"],
                "control_contrast_ci_high": control_summary["contrast_ci_high"],
                "specialized_minus_control_difference": difference,
                "difference_ci_low": difference_low,
                "difference_ci_high": difference_high,
                "difference_positive": bool(difference > 0),
                "positive_difference_ci_excludes_zero": bool(difference_low > 0),
                "specialized_specialization_margin": pair["specialized"][
                    "specialization_margin"
                ],
                "control_specialization_margin": control["specialization_margin"],
                "specialized_target_routing_frequency": pair["specialized"][
                    "target_routing_frequency"
                ],
                "control_target_routing_frequency": control[
                    "target_routing_frequency"
                ],
                "target_routing_frequency_absolute_difference": pair[
                    "target_routing_frequency_absolute_difference"
                ],
                "target_routing_coverage_absolute_difference": pair[
                    "target_routing_coverage_absolute_difference"
                ],
                "bootstrap_replicates": replicates,
            }
        )
    return rows


def _aggregate_analysis(
    preregistration: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, np.ndarray]],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_ids_by_domain = {
        domain: [
            pair["pair_id"]
            for pair in preregistration["matched_controls"]
            if pair["target_domain"] == domain
        ]
        for domain in BALANCED_DOMAINS
    }
    rows: list[dict[str, Any]] = []
    bootstrap_output: dict[str, Any] = {}
    for scope in (*BALANCED_DOMAINS, "overall"):
        pair_ids = (
            pair_ids_by_domain[scope]
            if scope != "overall"
            else [pair["pair_id"] for pair in preregistration["matched_controls"]]
        )
        rng = np.random.default_rng(_seed(seed, f"aggregate:{scope}"))
        indices_by_domain = {
            domain: rng.integers(0, 100, size=(replicates, 100))
            for domain in BALANCED_DOMAINS
        }
        specialized_bootstraps = []
        control_bootstraps = []
        specialized_points = []
        control_points = []
        pair_difference_points = []
        for pair_id in pair_ids:
            pair = next(
                item
                for item in preregistration["matched_controls"]
                if item["pair_id"] == pair_id
            )
            target = pair["target_domain"]
            specialized_id = f"specialized_{pair_id}"
            control = pair["control"]
            control_id = (
                f"control_{pair_id}_L{control['layer']}_E{control['expert_id']}"
            )
            specialist_means = []
            control_means = []
            for domain in BALANCED_DOMAINS:
                indices = indices_by_domain[domain]
                specialist_means.append(
                    deltas[specialized_id][domain][indices].mean(axis=1)
                )
                control_means.append(deltas[control_id][domain][indices].mean(axis=1))
            specialist_means_array = np.stack(specialist_means, axis=1)
            control_means_array = np.stack(control_means, axis=1)
            target_index = BALANCED_DOMAINS.index(target)
            other_indices = [index for index in range(4) if index != target_index]
            specialist_contrast = (
                specialist_means_array[:, target_index]
                - specialist_means_array[:, other_indices].mean(axis=1)
            )
            control_contrast = (
                control_means_array[:, target_index]
                - control_means_array[:, other_indices].mean(axis=1)
            )
            specialized_bootstraps.append(specialist_contrast)
            control_bootstraps.append(control_contrast)
            specialized_point = summaries[specialized_id][
                "target_minus_mean_other_contrast"
            ]
            control_point = summaries[control_id]["target_minus_mean_other_contrast"]
            specialized_points.append(specialized_point)
            control_points.append(control_point)
            pair_difference_points.append(specialized_point - control_point)

        specialized_matrix = np.stack(specialized_bootstraps, axis=1)
        control_matrix = np.stack(control_bootstraps, axis=1)
        difference_matrix = specialized_matrix - control_matrix
        mean_specialized_bootstrap = specialized_matrix.mean(axis=1)
        median_specialized_bootstrap = np.median(specialized_matrix, axis=1)
        mean_control_bootstrap = control_matrix.mean(axis=1)
        mean_difference_bootstrap = difference_matrix.mean(axis=1)
        _, mean_specialized_low, mean_specialized_high = confidence_interval(
            mean_specialized_bootstrap
        )
        _, median_specialized_low, median_specialized_high = confidence_interval(
            median_specialized_bootstrap
        )
        _, mean_control_low, mean_control_high = confidence_interval(
            mean_control_bootstrap
        )
        _, mean_difference_low, mean_difference_high = confidence_interval(
            mean_difference_bootstrap
        )
        specialized_summary_rows = [
            summaries[f"specialized_{pair_id}"] for pair_id in pair_ids
        ]
        row = {
            "scope": "domain" if scope != "overall" else "overall",
            "target_domain": scope if scope != "overall" else "all",
            "num_specialized_experts": len(pair_ids),
            "mean_specialized_contrast": float(np.mean(specialized_points)),
            "mean_specialized_contrast_ci_low": mean_specialized_low,
            "mean_specialized_contrast_ci_high": mean_specialized_high,
            "median_specialized_contrast": float(np.median(specialized_points)),
            "median_specialized_contrast_ci_low": median_specialized_low,
            "median_specialized_contrast_ci_high": median_specialized_high,
            "specialized_positive_proportion": float(
                np.mean(np.asarray(specialized_points) > 0)
            ),
            "specialized_ci_excludes_zero_proportion": float(
                np.mean(
                    [
                        row["contrast_ci_excludes_zero"]
                        for row in specialized_summary_rows
                    ]
                )
            ),
            "specialized_positive_ci_excludes_zero_proportion": float(
                np.mean(
                    [
                        row["positive_contrast_ci_excludes_zero"]
                        for row in specialized_summary_rows
                    ]
                )
            ),
            "mean_control_contrast": float(np.mean(control_points)),
            "mean_control_contrast_ci_low": mean_control_low,
            "mean_control_contrast_ci_high": mean_control_high,
            "mean_specialized_minus_control_difference": float(
                np.mean(pair_difference_points)
            ),
            "mean_difference_ci_low": mean_difference_low,
            "mean_difference_ci_high": mean_difference_high,
            "pair_difference_positive_proportion": float(
                np.mean(np.asarray(pair_difference_points) > 0)
            ),
            "bootstrap_replicates": replicates,
        }
        rows.append(row)
        bootstrap_output[scope] = {
            "mean_specialized_contrast": mean_specialized_bootstrap,
            "median_specialized_contrast": median_specialized_bootstrap,
            "mean_control_contrast": mean_control_bootstrap,
            "mean_specialized_minus_control_difference": mean_difference_bootstrap,
        }
    return rows, bootstrap_output


def _correlation_analysis(
    panel: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, np.ndarray]],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = {
        "functional_specialization_margin": np.asarray(
            [row["baseline_record"]["specialization_margin"] for row in panel],
            dtype=np.float64,
        ),
        "routing_specialization_margin": np.asarray(
            [row["baseline_record"]["routing_specialization_margin"] for row in panel],
            dtype=np.float64,
        ),
        "target_routing_frequency": np.asarray(
            [row["baseline_record"]["target_routing_frequency"] for row in panel],
            dtype=np.float64,
        ),
    }
    point_contrasts = np.asarray(
        [summaries[row["intervention_id"]]["target_minus_mean_other_contrast"] for row in panel]
    )
    rng = np.random.default_rng(_seed(seed, "correlations"))
    indices_by_domain = {
        domain: rng.integers(0, 100, size=(replicates, 100))
        for domain in BALANCED_DOMAINS
    }
    bootstrap_contrasts = np.zeros((replicates, len(panel)), dtype=np.float64)
    for intervention_index, intervention in enumerate(panel):
        means = np.stack(
            [
                deltas[intervention["intervention_id"]][domain][indices_by_domain[domain]].mean(
                    axis=1
                )
                for domain in BALANCED_DOMAINS
            ],
            axis=1,
        )
        target_index = BALANCED_DOMAINS.index(intervention["target_domain"])
        other_indices = [index for index in range(4) if index != target_index]
        bootstrap_contrasts[:, intervention_index] = means[:, target_index] - means[
            :, other_indices
        ].mean(axis=1)
    rows: list[dict[str, Any]] = []
    for metric, values in metrics.items():
        boot = np.asarray(
            [safe_spearman(values, bootstrap_contrasts[index]) for index in range(replicates)]
        )
        _, low, high = confidence_interval(boot)
        rows.append(
            {
                "predictor": metric,
                "outcome": "target_minus_mean_other_delta_nll",
                "num_interventions": len(panel),
                "spearman": safe_spearman(values, point_contrasts),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "bootstrap_replicates": replicates,
                "bootstrap_scope": (
                    "example resampling conditional on the 24 frozen interventions"
                ),
            }
        )
    return rows


def _decision(
    aggregate_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    domains = [row for row in aggregate_rows if row["scope"] == "domain"]
    overall = next(row for row in aggregate_rows if row["scope"] == "overall")
    positive_domain_count = sum(
        row["mean_specialized_contrast"] > 0
        and row["mean_specialized_minus_control_difference"] > 0
        for row in domains
    )
    positive_non_coding = any(
        row["target_domain"] != "coding" and row["mean_specialized_contrast"] > 0
        for row in domains
    )
    positive_pair_proportion = float(
        np.mean([row["specialized_minus_control_difference"] > 0 for row in paired_rows])
    )
    if (
        positive_domain_count >= 3
        and overall["mean_specialized_minus_control_difference"] > 0
        and positive_pair_proportion >= 0.75
    ):
        label = "STRONG GO"
    elif (
        overall["mean_specialized_contrast"] > 0
        and overall["mean_specialized_minus_control_difference"] > 0
        and positive_non_coding
    ):
        label = "GO WITH QUALIFICATIONS"
    else:
        label = "WEAK / NO GO"
    return {
        "label": label,
        "positive_domain_count": int(positive_domain_count),
        "positive_non_coding_domain_present": bool(positive_non_coding),
        "positive_specialized_minus_control_pair_proportion": positive_pair_proportion,
        "distributionally_robust_quantization_next_experiment_justified": label
        in ("STRONG GO", "GO WITH QUALIFICATIONS"),
        "rule_source": "selected_experts_preregistered.json",
    }


def _per_example_arrays(
    panel: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, LossStatistics],
    masked: Mapping[tuple[int, int, str], LossStatistics],
    deltas: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    baseline_nll = np.stack(
        [baselines[domain].per_token_nll for domain in BALANCED_DOMAINS]
    )
    token_counts = np.stack([baselines[domain].token_counts for domain in BALANCED_DOMAINS])
    masked_nll = np.stack(
        [
            np.stack(
                [
                    masked[(row["layer"], row["expert_id"], domain)].per_token_nll
                    for domain in BALANCED_DOMAINS
                ]
            )
            for row in panel
        ]
    )
    loss_changes = np.stack(
        [
            np.stack([deltas[row["intervention_id"]][domain] for domain in BALANCED_DOMAINS])
            for row in panel
        ]
    )
    return {
        "intervention_ids": np.asarray(
            [row["intervention_id"] for row in panel], dtype=np.str_
        ),
        "pair_ids": np.asarray([row["pair_id"] for row in panel], dtype=np.str_),
        "roles": np.asarray([row["role"] for row in panel], dtype=np.str_),
        "target_domains": np.asarray(
            [row["target_domain"] for row in panel], dtype=np.str_
        ),
        "layers": np.asarray([row["layer"] for row in panel], dtype=np.int16),
        "expert_ids": np.asarray([row["expert_id"] for row in panel], dtype=np.int16),
        "domain_names": np.asarray(BALANCED_DOMAINS, dtype=np.str_),
        "baseline_per_example_nll": baseline_nll,
        "masked_per_example_nll": masked_nll,
        "per_example_loss_changes": loss_changes,
        "token_counts": token_counts.astype(np.uint32),
    }


def _seed(seed: int, label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return int((seed + suffix) % (2**32 - 1))


def write_balanced_outputs(
    results: dict[str, Any], arrays: Mapping[str, np.ndarray], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "masking_results.csv",
        results["masking_results"],
        masking_result_fields(),
    )
    write_csv(
        output_dir / "pairwise_domain_contrasts.csv",
        results["pairwise_domain_contrasts"],
        pairwise_fields(),
    )
    write_csv(
        output_dir / "specialized_vs_control.csv",
        results["specialized_vs_control"],
        specialized_control_fields(),
    )
    write_csv(
        output_dir / "aggregate_results.csv",
        results["aggregate_results"],
        aggregate_fields(),
    )
    atomic_save_npz(output_dir / "per_example_loss_changes.npz", **arrays)


def masking_result_fields() -> list[str]:
    return [
        "intervention_id",
        "pair_id",
        "role",
        "target_domain",
        "layer",
        "expert_id",
        "domain",
        "is_target_domain",
        "examples",
        "evaluated_tokens",
        "baseline_nll",
        "masked_nll",
        "delta_nll",
        "delta_nll_ci_low",
        "delta_nll_ci_high",
        "positive_delta_example_fraction",
        "masked_routes",
        "observed_routing_coverage",
        "baseline_routing_coverage",
        "baseline_routing_frequency",
        "baseline_normalized_contribution",
        "baseline_functional_rank",
        "baseline_specialization_margin",
        "baseline_routing_specialization_margin",
        "target_routing_frequency",
        "target_routing_coverage",
        "inference_source",
        "bootstrap_replicates",
    ]


def pairwise_fields() -> list[str]:
    return [
        "intervention_id",
        "pair_id",
        "role",
        "target_domain",
        "comparison_domain",
        "layer",
        "expert_id",
        "target_delta_nll",
        "comparison_delta_nll",
        "target_minus_comparison_delta_nll",
        "contrast_ci_low",
        "contrast_ci_high",
        "positive_contrast_ci_excludes_zero",
        "bootstrap_replicates",
    ]


def specialized_control_fields() -> list[str]:
    return [
        "pair_id",
        "target_domain",
        "specialized_layer",
        "specialized_expert_id",
        "control_layer",
        "control_expert_id",
        "specialized_contrast",
        "specialized_contrast_ci_low",
        "specialized_contrast_ci_high",
        "control_contrast",
        "control_contrast_ci_low",
        "control_contrast_ci_high",
        "specialized_minus_control_difference",
        "difference_ci_low",
        "difference_ci_high",
        "difference_positive",
        "positive_difference_ci_excludes_zero",
        "specialized_specialization_margin",
        "control_specialization_margin",
        "specialized_target_routing_frequency",
        "control_target_routing_frequency",
        "target_routing_frequency_absolute_difference",
        "target_routing_coverage_absolute_difference",
        "bootstrap_replicates",
    ]


def aggregate_fields() -> list[str]:
    return [
        "scope",
        "target_domain",
        "num_specialized_experts",
        "mean_specialized_contrast",
        "mean_specialized_contrast_ci_low",
        "mean_specialized_contrast_ci_high",
        "median_specialized_contrast",
        "median_specialized_contrast_ci_low",
        "median_specialized_contrast_ci_high",
        "specialized_positive_proportion",
        "specialized_ci_excludes_zero_proportion",
        "specialized_positive_ci_excludes_zero_proportion",
        "mean_control_contrast",
        "mean_control_contrast_ci_low",
        "mean_control_contrast_ci_high",
        "mean_specialized_minus_control_difference",
        "mean_difference_ci_low",
        "mean_difference_ci_high",
        "pair_difference_positive_proportion",
        "bootstrap_replicates",
    ]


def write_balanced_summary(results: Mapping[str, Any], output_path: Path) -> str:
    prereg = results["preregistration"]
    analysis = results["balanced_analysis"]
    summaries = analysis["intervention_contrasts"]
    pairs = analysis["specialized_vs_control"]
    aggregates = analysis["aggregate_results"]
    correlations = analysis["correlation_results"]
    integrity = results["integrity_validation"]
    summary_by_identity = {
        (row["role"], row["layer"], row["expert_id"]): row for row in summaries
    }
    pair_by_specialist = {
        (row["specialized_layer"], row["specialized_expert_id"]): row for row in pairs
    }
    lines = [
        "# Balanced Causal Validation of Domain-Specialized OLMoE Experts",
        "",
        "## Experimental Setup",
        "",
        f"- Model: `{results['run_config']['model']}`",
        f"- Revision: `{results['run_config']['model_revision']}`",
        "- Hardware and arithmetic: NVIDIA A40, CUDA, BF16, batch size 1",
        "- Controlled corpus: 100 examples/domain, 64 measured positions/example, "
        "6,400 measured positions/domain, shared `Input:\\n` prefix",
        "- Intervention: zero one selected expert gate coefficient at measured source "
        "positions without rerouting or weight changes",
        "- Primary contrast: target-domain delta NLL minus the mean delta NLL of all "
        "three non-target domains",
        "- Uncertainty: 1,000 fixed-seed bootstrap replicates from saved per-example losses",
        "",
        "## Pre-Registered Expert Selection",
        "",
        f"The panel was frozen before masking with fingerprint "
        f"`{prereg['preregistration_fingerprint']}`. Selection read baseline routing, "
        "gate-mass, and functional-contribution artifacts only; masked outcomes were excluded.",
        "",
        "| Target | Specialist | Target rank | Margin | Target route coverage | Control | "
        "Control margin | Control route coverage |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for pair in prereg["matched_controls"]:
        specialist = pair["specialized"]
        control = pair["control"]
        lines.append(
            f"| {pair['target_domain'].title()} | L{specialist['layer']}/E"
            f"{specialist['expert_id']} | {specialist['target_rank']:.0f} | "
            f"{specialist['specialization_margin']:.6f} | "
            f"{specialist['target_routing_coverage']:.4f} | "
            f"L{control['layer']}/E{control['expert_id']} | "
            f"{control['specialization_margin']:.6f} | "
            f"{control['target_routing_coverage']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All four domains satisfied the strict pre-registered eligibility tier; no "
            "specialist threshold was relaxed. All controls are unique and from the same "
            "layer, and all satisfied the primary 25%-of-specialist margin cap. Routing "
            "matches are imperfect for several extremely high-coverage specialists and are "
            "reported rather than hidden.",
            "",
            "## Integrity Validation",
            "",
            f"Integrity status: **{'PASSED' if integrity['passed'] else 'FAILED'}**.",
            "",
            f"- Controlled collection fingerprint: "
            f"`{integrity['collection_fingerprint']}`",
            f"- Selection-input fingerprint: `{integrity['selection_input_fingerprint']}`",
            f"- Runtime revision exact: {str(integrity['model_revision_exact']).lower()}",
            f"- Dataset revisions and input hashes exact: "
            f"{str(integrity['source_static_audit_passed']).lower()}",
            f"- Fresh baseline reproduced all four source baselines: "
            f"{str(integrity['baseline_reproduction_passed']).lower()}",
            f"- Selected-route counts matched stored routing tensors: "
            f"{str(integrity['routing_match_passed']).lower()}",
            f"- Smoke and intervention hook-leak checks passed: "
            f"{str(integrity['hook_checks_passed']).lower()}",
        ]
    )
    for domain in BALANCED_DOMAINS:
        lines.extend(
            [
                "",
                f"## {domain.title()}-Specialized Experts",
                "",
                "| Specialist | Target delta NLL | Target-minus-mean-other | 95% CI | "
                "Specialist-minus-control | 95% CI |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        domain_pairs = [
            pair for pair in prereg["matched_controls"] if pair["target_domain"] == domain
        ]
        for pair in domain_pairs:
            specialist = pair["specialized"]
            summary = summary_by_identity[
                ("specialized", specialist["layer"], specialist["expert_id"])
            ]
            paired = pair_by_specialist[(specialist["layer"], specialist["expert_id"])]
            lines.append(
                f"| L{specialist['layer']}/E{specialist['expert_id']} | "
                f"{summary['target_delta_nll']:+.6f} | "
                f"{summary['target_minus_mean_other_contrast']:+.6f} | "
                f"[{summary['contrast_ci_low']:+.6f}, "
                f"{summary['contrast_ci_high']:+.6f}] | "
                f"{paired['specialized_minus_control_difference']:+.6f} | "
                f"[{paired['difference_ci_low']:+.6f}, "
                f"{paired['difference_ci_high']:+.6f}] |"
            )

    lines.extend(
        [
            "",
            "## Matched-Routing Controls",
            "",
            "| Target | Specialist | Control | Specialist contrast | Control contrast | "
            "Difference | 95% CI | Route-frequency gap |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        lines.append(
            f"| {row['target_domain'].title()} | L{row['specialized_layer']}/E"
            f"{row['specialized_expert_id']} | L{row['control_layer']}/E"
            f"{row['control_expert_id']} | {row['specialized_contrast']:+.6f} | "
            f"{row['control_contrast']:+.6f} | "
            f"{row['specialized_minus_control_difference']:+.6f} | "
            f"[{row['difference_ci_low']:+.6f}, {row['difference_ci_high']:+.6f}] | "
            f"{row['target_routing_frequency_absolute_difference']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Causal Effects",
            "",
            "| Scope | Mean specialist contrast | 95% CI | Mean control contrast | 95% CI | "
            "Mean specialist-control | 95% CI | Positive specialists |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        label = (
            row["target_domain"].title()
            if row["scope"] == "domain"
            else "All domains"
        )
        lines.append(
            f"| {label} | {row['mean_specialized_contrast']:+.6f} | "
            f"[{row['mean_specialized_contrast_ci_low']:+.6f}, "
            f"{row['mean_specialized_contrast_ci_high']:+.6f}] | "
            f"{row['mean_control_contrast']:+.6f} | "
            f"[{row['mean_control_contrast_ci_low']:+.6f}, "
            f"{row['mean_control_contrast_ci_high']:+.6f}] | "
            f"{row['mean_specialized_minus_control_difference']:+.6f} | "
            f"[{row['mean_difference_ci_low']:+.6f}, "
            f"{row['mean_difference_ci_high']:+.6f}] | "
            f"{row['specialized_positive_proportion']:.1%} |"
        )
    lines.extend(
        [
            "",
            "Aggregate bootstrap intervals condition on the 12 fixed pre-registered "
            "specialists and resample examples within domain; they do not estimate "
            "checkpoint, dataset-choice, or expert-selection uncertainty.",
            "",
            "## Specialization Score vs Causal Sensitivity",
            "",
            "| Baseline predictor | Spearman | 95% bootstrap CI |",
            "|---|---:|---:|",
        ]
    )
    for row in correlations:
        lines.append(
            f"| {row['predictor'].replace('_', ' ').title()} | "
            f"{row['spearman']:+.3f} | [{row['bootstrap_ci_low']:+.3f}, "
            f"{row['bootstrap_ci_high']:+.3f}] |"
        )

    failures = []
    for pair in prereg["matched_controls"]:
        specialist = pair["specialized"]
        summary = summary_by_identity[
            ("specialized", specialist["layer"], specialist["expert_id"])
        ]
        paired = pair_by_specialist[(specialist["layer"], specialist["expert_id"])]
        reasons = []
        if summary["target_delta_nll"] <= 0:
            reasons.append("target masking did not increase NLL")
        elif (
            summary["target_delta_nll_ci_low"]
            <= 0
            <= summary["target_delta_nll_ci_high"]
        ):
            reasons.append("target-domain delta NLL CI includes zero")
        if summary["target_minus_mean_other_contrast"] <= 0:
            reasons.append("another-domain mean effect was at least as large")
        if summary["contrast_ci_low"] <= 0 <= summary["contrast_ci_high"]:
            reasons.append("primary contrast CI includes zero")
        if paired["specialized_minus_control_difference"] <= 0:
            reasons.append("matched control contrast was at least as large")
        elif paired["difference_ci_low"] <= 0 <= paired["difference_ci_high"]:
            reasons.append("specialist-minus-control difference CI includes zero")
        if reasons:
            failures.append(
                f"- {pair['target_domain'].title()} L{specialist['layer']}/E"
                f"{specialist['expert_id']}: " + "; ".join(reasons) + "."
            )
    lines.extend(["", "## Failures / Counterexamples", ""])
    lines.extend(failures or ["No pre-registered specialist met a counterexample criterion."])
    decision = analysis["decision"]
    overall = next(row for row in aggregates if row["scope"] == "overall")
    lines.extend(
        [
            "",
            "## Scientific Interpretation",
            "",
            "The balanced panel asks whether baseline functional specialization predicts "
            "domain-conditioned causal reliance beyond routing frequency alone. The answer "
            f"under the frozen decision rule is **{decision['label']}**: "
            f"{decision['positive_domain_count']} of 4 domains have both a positive mean "
            "specialist contrast and a positive mean specialist-minus-control difference; "
            f"the overall specialist-minus-control effect is "
            f"{overall['mean_specialized_minus_control_difference']:+.6f} nats/token.",
            "",
            "This conclusion applies to the fixed 100-example, 64-position controlled "
            "subsets of one OLMoE checkpoint. Selected-route masking is a local causal "
            "sensitivity test, not expert deletion and not a simulation of quantization.",
            "",
            "## Decision",
            "",
            f"### {decision['label']}",
            "",
            "The label follows the decision rule frozen in `selected_experts_preregistered.json`; "
            "failed experts were not replaced.",
            "",
            "## Recommendation",
            "",
        ]
    )
    if decision["distributionally_robust_quantization_next_experiment_justified"]:
        lines.append(
            "The evidence justifies proceeding to a separately designed, reversible "
            "distributionally robust mixed-precision quantization experiment. No "
            "quantization, pruning, fine-tuning, or weight modification was performed here."
        )
    else:
        lines.append(
            "The evidence does not yet justify distributionally robust mixed-precision "
            "quantization. Resolve the causal-specificity weakness before changing weights "
            "or precision."
        )
    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return text


def create_balanced_figures(results: Mapping[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis = results["balanced_analysis"]
    summaries = analysis["intervention_contrasts"]
    specialized = [row for row in summaries if row["role"] == "specialized"]
    pairs = analysis["specialized_vs_control"]
    aggregates = [
        row for row in analysis["aggregate_results"] if row["scope"] == "domain"
    ]
    paths: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=False)
    for axis, target in zip(axes.flat, BALANCED_DOMAINS):
        target_rows = [row for row in specialized if row["target_domain"] == target]
        x = np.arange(len(target_rows))
        width = 0.19
        for domain_index, domain in enumerate(BALANCED_DOMAINS):
            means = [row["domain_effects"][domain]["delta_nll"] for row in target_rows]
            lows = [row["domain_effects"][domain]["delta_nll_ci_low"] for row in target_rows]
            highs = [row["domain_effects"][domain]["delta_nll_ci_high"] for row in target_rows]
            errors = np.asarray(
                [[mean - low for mean, low in zip(means, lows)],
                 [high - mean for mean, high in zip(means, highs)]]
            )
            alpha = 1.0 if domain == target else 0.45
            axis.bar(
                x + (domain_index - 1.5) * width,
                means,
                width,
                yerr=errors,
                capsize=2,
                color=DOMAIN_COLORS[domain],
                alpha=alpha,
                label=domain.title(),
                linewidth=0.5,
                edgecolor="black",
            )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, [f"L{row['layer']}/E{row['expert_id']}" for row in target_rows])
        axis.set_title(f"{target.title()} specialists")
        axis.set_ylabel("Delta NLL (nats/token)")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.suptitle("Domain-specialized selected-route masking effects", y=0.985)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    paths.extend(
        _save_both(
            figure, figure_dir / "figure_1_domain_specialized_masking_effects", plt
        )
    )

    figure, axis = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(pairs))
    width = 0.36
    spec = np.asarray([row["specialized_contrast"] for row in pairs])
    control = np.asarray([row["control_contrast"] for row in pairs])
    spec_err = np.asarray(
        [
            spec - np.asarray([row["specialized_contrast_ci_low"] for row in pairs]),
            np.asarray([row["specialized_contrast_ci_high"] for row in pairs]) - spec,
        ]
    )
    control_err = np.asarray(
        [
            control - np.asarray([row["control_contrast_ci_low"] for row in pairs]),
            np.asarray([row["control_contrast_ci_high"] for row in pairs]) - control,
        ]
    )
    colors = [DOMAIN_COLORS[row["target_domain"]] for row in pairs]
    axis.bar(x - width / 2, spec, width, yerr=spec_err, capsize=2, color=colors, label="Specialist")
    axis.bar(
        x + width / 2,
        control,
        width,
        yerr=control_err,
        capsize=2,
        facecolor="white",
        edgecolor=colors,
        hatch="//",
        label="Matched control",
    )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(
        x,
        [
            f"{row['target_domain'][0].upper()}\nL{row['specialized_layer']}/E"
            f"{row['specialized_expert_id']}"
            for row in pairs
        ],
    )
    axis.set_ylabel("Target-minus-mean-other delta NLL")
    axis.set_title("Specialized experts versus same-layer routing controls")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths.extend(_save_both(figure, figure_dir / "figure_2_specialized_vs_controls", plt))

    figure, axis = plt.subplots(figsize=(9, 6))
    x = np.arange(4)
    means = np.asarray([row["mean_specialized_contrast"] for row in aggregates])
    errors = np.asarray(
        [
            means - np.asarray([row["mean_specialized_contrast_ci_low"] for row in aggregates]),
            np.asarray([row["mean_specialized_contrast_ci_high"] for row in aggregates]) - means,
        ]
    )
    controls = np.asarray([row["mean_control_contrast"] for row in aggregates])
    axis.bar(
        x,
        means,
        yerr=errors,
        capsize=4,
        color=[DOMAIN_COLORS[domain] for domain in BALANCED_DOMAINS],
        edgecolor="black",
        linewidth=0.6,
        label="Specialists",
    )
    axis.scatter(
        x,
        controls,
        marker="D",
        s=45,
        facecolor="white",
        edgecolor="black",
        label="Controls",
    )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, [domain.title() for domain in BALANCED_DOMAINS])
    axis.set_ylabel("Mean target-minus-mean-other delta NLL")
    axis.set_title("Aggregate causal specificity by target domain")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    paths.extend(_save_both(figure, figure_dir / "figure_3_aggregate_domain_effects", plt))

    panel = intervention_panel(results["preregistration"])
    summary_by_id = {row["intervention_id"]: row for row in summaries}
    figure, axis = plt.subplots(figsize=(9, 6.5))
    for role, marker in (("specialized", "o"), ("control", "s")):
        for domain in BALANCED_DOMAINS:
            rows = [
                row
                for row in panel
                if row["role"] == role and row["target_domain"] == domain
            ]
            axis.scatter(
                [row["baseline_record"]["specialization_margin"] for row in rows],
                [
                    summary_by_id[row["intervention_id"]][
                        "target_minus_mean_other_contrast"
                    ]
                    for row in rows
                ],
                color=DOMAIN_COLORS[domain],
                marker=marker,
                s=65,
                edgecolor="black",
                linewidth=0.5,
            )
    functional_correlation = next(
        row
        for row in analysis["correlation_results"]
        if row["predictor"] == "functional_specialization_margin"
    )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.axvline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Baseline functional specialization margin")
    axis.set_ylabel("Target-minus-mean-other delta NLL")
    axis.set_title(
        "Functional specialization versus causal sensitivity\n"
        f"Spearman rho = {functional_correlation['spearman']:.3f}"
    )
    domain_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOMAIN_COLORS[domain],
               markeredgecolor="black", label=domain.title())
        for domain in BALANCED_DOMAINS
    ]
    role_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="none", label="Specialist"),
        Line2D([0], [0], marker="s", color="black", linestyle="none", label="Control"),
    ]
    axis.legend(handles=domain_handles + role_handles, frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    paths.extend(_save_both(figure, figure_dir / "figure_4_specialization_vs_causal", plt))

    figure, axis = plt.subplots(figsize=(9, 6.5))
    for role, marker in (("specialized", "o"), ("control", "s")):
        for domain in BALANCED_DOMAINS:
            rows = [
                row
                for row in panel
                if row["role"] == role and row["target_domain"] == domain
            ]
            axis.scatter(
                [row["baseline_record"]["target_routing_frequency"] for row in rows],
                [
                    summary_by_id[row["intervention_id"]][
                        "target_minus_mean_other_contrast"
                    ]
                    for row in rows
                ],
                color=DOMAIN_COLORS[domain],
                marker=marker,
                s=65,
                edgecolor="black",
                linewidth=0.5,
            )
    routing_correlation = next(
        row
        for row in analysis["correlation_results"]
        if row["predictor"] == "target_routing_frequency"
    )
    axis.axhline(0, color="#555555", linewidth=0.8)
    axis.set_xlabel("Target-domain routing frequency (share of top-8 assignments)")
    axis.set_ylabel("Target-minus-mean-other delta NLL")
    axis.set_title(
        "Routing frequency versus causal specificity\n"
        f"Spearman rho = {routing_correlation['spearman']:.3f}"
    )
    axis.legend(handles=domain_handles + role_handles, frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    paths.extend(_save_both(figure, figure_dir / "figure_5_routing_vs_causal", plt))
    return paths


def _save_both(figure: Any, base_path: Path, plt: Any) -> list[Path]:
    paths = [base_path.with_suffix(".png"), base_path.with_suffix(".pdf")]
    figure.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return paths
