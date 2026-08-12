from __future__ import annotations

import itertools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, package_versions, read_json, write_csv
from .metrics import DomainStatistics
from .statistics import (
    NORMALIZED_METRIC_KEYS,
    bootstrap_spearman_pair,
    confidence_interval,
    descending_ranks,
    importance_percentiles,
    nanmean,
    safe_kendall,
    safe_spearman,
    topk_similarity,
)


def analyze_results(
    input_dir: Path,
    bootstrap_replicates: int = 100,
    bootstrap_seed: int | None = None,
    specialized_per_layer: int = 10,
) -> dict[str, Any]:
    collection_config = read_json(input_dir / "collection_config.json")
    architecture = read_json(input_dir / "architecture.json")
    smoke_path = input_dir / "smoke_validation.json"
    smoke = read_json(smoke_path) if smoke_path.exists() else None
    domains = list(collection_config["domains"])
    if len(domains) < 2:
        raise RuntimeError("At least two completed domains are required for comparison")
    layer_metadata = architecture["layers"]
    layer_names = [item["block_name"] for item in layer_metadata]
    layer_ids = [int(item["model_layer_index"]) for item in layer_metadata]
    statistics: dict[str, DomainStatistics] = {}
    dataset_metadata: dict[str, dict[str, Any]] = {}
    for domain in domains:
        statistics[domain] = DomainStatistics.load(
            input_dir / "domains" / f"{domain}.npz"
        )
        dataset_metadata[domain] = read_json(
            input_dir / "domains" / f"{domain}.metadata.json"
        )
        if statistics[domain].layer_names != layer_names:
            raise RuntimeError(f"Layer mapping mismatch in domain {domain!r}")
    _validate_domain_shapes(statistics)

    aggregate = {domain: stats.aggregate() for domain, stats in statistics.items()}
    metrics = [
        "routing_frequency",
        "gate_mass",
        "functional_contribution",
    ]
    if all(stats.gradient_sums is not None for stats in statistics.values()):
        metrics.append("gradient_attribution")
    elif any(stats.gradient_sums is not None for stats in statistics.values()):
        raise RuntimeError("Gradient attribution is present for only a subset of domains")

    expert_rows, ranks, percentiles = _expert_importance_rows(
        domains, aggregate, layer_metadata, metrics
    )
    seed = int(collection_config["seed"] if bootstrap_seed is None else bootstrap_seed)
    correlation_rows = _cross_domain_correlations(
        domains,
        statistics,
        aggregate,
        layer_ids,
        metrics,
        bootstrap_replicates,
        seed,
    )
    topk_rows = _topk_rows(domains, aggregate, layer_ids, metrics)
    routing_functional_rows = _routing_vs_functional(
        domains, aggregate, layer_ids
    )
    specialized_rows = _specialized_experts(
        domains,
        aggregate,
        ranks,
        percentiles,
        layer_metadata,
        specialized_per_layer,
    )

    _write_analysis_csvs(
        input_dir,
        expert_rows,
        correlation_rows,
        topk_rows,
        routing_functional_rows,
        specialized_rows,
        domains,
        include_gradient="gradient_attribution" in metrics,
    )
    results = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "checkpoint": collection_config["model"],
            "requested_revision": collection_config.get("model_revision"),
            "resolved_revision": collection_config.get("resolved_model_revision"),
        },
        "experiment": {
            "domains": domains,
            "seed": collection_config["seed"],
            "max_sequence_length": collection_config["max_length"],
            "requested_examples_per_domain": collection_config["num_examples"],
            "quick_mode": collection_config["quick"],
            "include_reference_answers": collection_config["include_reference_answers"],
            "compute_gradient_attribution": collection_config[
                "compute_gradient_attribution"
            ],
            "device": collection_config["device"],
            "device_description": collection_config["device_description"],
            "dtype": collection_config["dtype"],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
        },
        "package_versions": collection_config.get(
            "package_versions", package_versions()
        ),
        "datasets": dataset_metadata,
        "token_counts": {
            domain: {
                "examples": statistics[domain].num_examples,
                "tokens": int(statistics[domain].token_counts.sum()),
                "mean_tokens_per_example": float(
                    statistics[domain].token_counts.mean()
                ),
            }
            for domain in domains
        },
        "architecture": architecture,
        "metric_definitions": {
            "routing_frequency": (
                "Valid-token top-k assignments to an expert divided by all valid-token "
                "top-k assignments, computed independently per MoE layer."
            ),
            "gate_mass": (
                "Sum of the selected routing coefficients actually applied by the model, "
                "divided by valid token count. For OLMoE norm_topk_prob=false, selected "
                "coefficients are not renormalized to sum to one."
            ),
            "functional_contribution": (
                "Sum over valid routed tokens of the L2 norm of gate_weight times the "
                "individual expert output, divided by valid token count. This is an "
                "activation-magnitude proxy, not causal importance."
            ),
            "gradient_attribution": (
                "Optional sum of abs(gate_weight * d(next-token CE sum)/d(gate_weight)), "
                "divided by valid token count."
                if "gradient_attribution" in metrics
                else "Not collected."
            ),
            "normalized_vectors": "Each nonnegative layer/domain expert vector divided by its sum.",
        },
        "smoke_validation": smoke,
        "expert_importance": expert_rows,
        "cross_domain_correlations": correlation_rows,
        "topk_overlap": topk_rows,
        "routing_vs_functional_correlation": routing_functional_rows,
        "domain_specialized_experts": specialized_rows,
    }
    atomic_write_json(input_dir / "results.json", results)
    return results


def _expert_importance_rows(
    domains: list[str],
    aggregate: dict[str, dict[str, np.ndarray]],
    layer_metadata: list[dict[str, Any]],
    metrics: list[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    rows: list[dict[str, Any]] = []
    ranks: dict[str, dict[str, np.ndarray]] = {}
    percentiles: dict[str, dict[str, np.ndarray]] = {}
    for domain in domains:
        ranks[domain] = {}
        percentiles[domain] = {}
        for metric in metrics:
            values = aggregate[domain][NORMALIZED_METRIC_KEYS[metric]]
            ranks[domain][metric] = np.stack(
                [descending_ranks(layer) for layer in values]
            )
            percentiles[domain][metric] = np.stack(
                [importance_percentiles(layer) for layer in values]
            )
        num_layers, num_experts = aggregate[domain]["routing_frequency"].shape
        for layer in range(num_layers):
            metadata = layer_metadata[layer]
            for expert_id in range(num_experts):
                row = {
                    "domain": domain,
                    "layer": int(metadata["model_layer_index"]),
                    "layer_ordinal": layer,
                    "layer_name": metadata["block_name"],
                    "expert_id": expert_id,
                    "routing_frequency": float(
                        aggregate[domain]["routing_frequency"][layer, expert_id]
                    ),
                    "routing_rank": float(
                        ranks[domain]["routing_frequency"][layer, expert_id]
                    ),
                    "gate_mass": float(aggregate[domain]["gate_mass"][layer, expert_id]),
                    "gate_rank": float(ranks[domain]["gate_mass"][layer, expert_id]),
                    "functional_contribution": float(
                        aggregate[domain]["functional_contribution"][layer, expert_id]
                    ),
                    "functional_rank": float(
                        ranks[domain]["functional_contribution"][layer, expert_id]
                    ),
                    "normalized_routing": float(
                        aggregate[domain]["normalized_routing"][layer, expert_id]
                    ),
                    "normalized_gate": float(
                        aggregate[domain]["normalized_gate"][layer, expert_id]
                    ),
                    "normalized_contribution": float(
                        aggregate[domain]["normalized_contribution"][layer, expert_id]
                    ),
                }
                if "gradient_attribution" in metrics:
                    row.update(
                        {
                            "gradient_attribution": float(
                                aggregate[domain]["gradient_attribution"][layer, expert_id]
                            ),
                            "gradient_rank": float(
                                ranks[domain]["gradient_attribution"][layer, expert_id]
                            ),
                            "normalized_gradient": float(
                                aggregate[domain]["normalized_gradient"][layer, expert_id]
                            ),
                        }
                    )
                rows.append(row)
    return rows, ranks, percentiles


def _cross_domain_correlations(
    domains: list[str],
    statistics: dict[str, DomainStatistics],
    aggregate: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    metrics: list[str],
    bootstrap_replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for metric in metrics:
        normalized_key = NORMALIZED_METRIC_KEYS[metric]
        for domain_a, domain_b in itertools.combinations(domains, 2):
            boot = bootstrap_spearman_pair(
                statistics[domain_a],
                statistics[domain_b],
                metric,
                bootstrap_replicates,
                rng,
            )
            layer_spearman: list[float] = []
            layer_kendall: list[float] = []
            for ordinal, layer_id in enumerate(layer_ids):
                first = aggregate[domain_a][normalized_key][ordinal]
                second = aggregate[domain_b][normalized_key][ordinal]
                spearman = safe_spearman(first, second)
                kendall = safe_kendall(first, second)
                bootstrap_mean, low, high = confidence_interval(
                    boot[:, ordinal] if len(boot) else np.asarray([])
                )
                rows.append(
                    {
                        "metric": metric,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "layer": layer_id,
                        "layer_ordinal": ordinal,
                        "spearman": spearman,
                        "kendall_tau": kendall,
                        "bootstrap_mean_spearman": bootstrap_mean,
                        "bootstrap_ci_low": low,
                        "bootstrap_ci_high": high,
                        "bootstrap_replicates": bootstrap_replicates,
                    }
                )
                layer_spearman.append(spearman)
                layer_kendall.append(kendall)
            if len(boot):
                with np.errstate(invalid="ignore"):
                    bootstrap_aggregate = np.nanmean(boot, axis=1)
            else:
                bootstrap_aggregate = np.asarray([])
            bootstrap_mean, low, high = confidence_interval(bootstrap_aggregate)
            rows.append(
                {
                    "metric": metric,
                    "domain_a": domain_a,
                    "domain_b": domain_b,
                    "layer": "average",
                    "layer_ordinal": "average",
                    "spearman": nanmean(layer_spearman),
                    "kendall_tau": nanmean(layer_kendall),
                    "bootstrap_mean_spearman": bootstrap_mean,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )
    return rows


def _topk_rows(
    domains: list[str],
    aggregate: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    metrics: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        normalized_key = NORMALIZED_METRIC_KEYS[metric]
        for domain_a, domain_b in itertools.combinations(domains, 2):
            for fraction in (0.10, 0.25, 0.50):
                per_layer: list[dict[str, Any]] = []
                for ordinal, layer_id in enumerate(layer_ids):
                    k, intersection, overlap, jaccard = topk_similarity(
                        aggregate[domain_a][normalized_key][ordinal],
                        aggregate[domain_b][normalized_key][ordinal],
                        fraction,
                    )
                    row = {
                        "metric": metric,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "threshold": fraction,
                        "top_k": k,
                        "layer": layer_id,
                        "layer_ordinal": ordinal,
                        "intersection": intersection,
                        "overlap_fraction": overlap,
                        "jaccard_similarity": jaccard,
                    }
                    rows.append(row)
                    per_layer.append(row)
                rows.append(
                    {
                        "metric": metric,
                        "domain_a": domain_a,
                        "domain_b": domain_b,
                        "threshold": fraction,
                        "top_k": per_layer[0]["top_k"],
                        "layer": "average",
                        "layer_ordinal": "average",
                        "intersection": nanmean(item["intersection"] for item in per_layer),
                        "overlap_fraction": nanmean(
                            item["overlap_fraction"] for item in per_layer
                        ),
                        "jaccard_similarity": nanmean(
                            item["jaccard_similarity"] for item in per_layer
                        ),
                    }
                )
    return rows


def _routing_vs_functional(
    domains: list[str],
    aggregate: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in domains:
        values = []
        for ordinal, layer_id in enumerate(layer_ids):
            spearman = safe_spearman(
                aggregate[domain]["routing_frequency"][ordinal],
                aggregate[domain]["functional_contribution"][ordinal],
            )
            values.append(spearman)
            rows.append(
                {
                    "domain": domain,
                    "layer": layer_id,
                    "layer_ordinal": ordinal,
                    "spearman": spearman,
                }
            )
        rows.append(
            {
                "domain": domain,
                "layer": "average",
                "layer_ordinal": "average",
                "spearman": nanmean(values),
            }
        )
    return rows


def _specialized_experts(
    domains: list[str],
    aggregate: dict[str, dict[str, np.ndarray]],
    ranks: dict[str, dict[str, np.ndarray]],
    percentiles: dict[str, dict[str, np.ndarray]],
    layer_metadata: list[dict[str, Any]],
    specialized_per_layer: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = {
        domain: aggregate[domain]["normalized_contribution"] for domain in domains
    }
    num_layers, num_experts = next(iter(values.values())).shape
    epsilon = 1e-12
    for layer in range(num_layers):
        layer_rows = []
        for expert_id in range(num_experts):
            per_domain = np.asarray(
                [values[domain][layer, expert_id] for domain in domains]
            )
            maximum_index = int(np.argmax(per_domain))
            minimum_index = int(np.argmin(per_domain))
            maximum = float(per_domain[maximum_index])
            minimum = float(per_domain[minimum_index])
            row: dict[str, Any] = {
                "layer": int(layer_metadata[layer]["model_layer_index"]),
                "layer_ordinal": layer,
                "layer_name": layer_metadata[layer]["block_name"],
                "expert_id": expert_id,
                "specialization_metric": "functional_contribution",
                "specialization_score": (maximum + epsilon) / (minimum + epsilon),
                "absolute_normalized_range": maximum - minimum,
                "rank_range": float(
                    max(
                        ranks[domain]["functional_contribution"][layer, expert_id]
                        for domain in domains
                    )
                    - min(
                        ranks[domain]["functional_contribution"][layer, expert_id]
                        for domain in domains
                    )
                ),
                "max_domain": domains[maximum_index],
                "min_domain": domains[minimum_index],
                "max_normalized_importance": maximum,
                "min_normalized_importance": minimum,
                "minimum_was_zero": minimum == 0.0,
            }
            for domain in domains:
                row[f"{domain}_normalized_contribution"] = float(
                    values[domain][layer, expert_id]
                )
                row[f"{domain}_rank"] = float(
                    ranks[domain]["functional_contribution"][layer, expert_id]
                )
                row[f"{domain}_percentile"] = float(
                    percentiles[domain]["functional_contribution"][layer, expert_id]
                )
            layer_rows.append(row)
        layer_rows.sort(
            key=lambda row: (
                row["specialization_score"],
                row["absolute_normalized_range"],
                row["rank_range"],
            ),
            reverse=True,
        )
        rows.extend(layer_rows[: min(specialized_per_layer, num_experts)])
    return rows


def _write_analysis_csvs(
    output_dir: Path,
    expert_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    topk_rows: list[dict[str, Any]],
    routing_functional_rows: list[dict[str, Any]],
    specialized_rows: list[dict[str, Any]],
    domains: list[str],
    include_gradient: bool,
) -> None:
    expert_fields = [
        "domain",
        "layer",
        "layer_ordinal",
        "layer_name",
        "expert_id",
        "routing_frequency",
        "routing_rank",
        "gate_mass",
        "gate_rank",
        "functional_contribution",
        "functional_rank",
        "normalized_routing",
        "normalized_gate",
        "normalized_contribution",
    ]
    if include_gradient:
        expert_fields += [
            "gradient_attribution",
            "gradient_rank",
            "normalized_gradient",
        ]
    write_csv(
        output_dir / "expert_importance_by_domain.csv",
        expert_rows,
        expert_fields,
    )
    write_csv(
        output_dir / "cross_domain_correlations.csv",
        correlation_rows,
        [
            "metric",
            "domain_a",
            "domain_b",
            "layer",
            "layer_ordinal",
            "spearman",
            "kendall_tau",
            "bootstrap_mean_spearman",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "bootstrap_replicates",
        ],
    )
    write_csv(
        output_dir / "topk_overlap.csv",
        topk_rows,
        [
            "metric",
            "domain_a",
            "domain_b",
            "threshold",
            "top_k",
            "layer",
            "layer_ordinal",
            "intersection",
            "overlap_fraction",
            "jaccard_similarity",
        ],
    )
    write_csv(
        output_dir / "routing_vs_functional_correlation.csv",
        routing_functional_rows,
        ["domain", "layer", "layer_ordinal", "spearman"],
    )
    specialized_fields = [
        "layer",
        "layer_ordinal",
        "layer_name",
        "expert_id",
        "specialization_metric",
        "specialization_score",
        "absolute_normalized_range",
        "rank_range",
        "max_domain",
        "min_domain",
        "max_normalized_importance",
        "min_normalized_importance",
        "minimum_was_zero",
    ]
    for domain in domains:
        specialized_fields.extend(
            [
                f"{domain}_normalized_contribution",
                f"{domain}_rank",
                f"{domain}_percentile",
            ]
        )
    write_csv(
        output_dir / "domain_specialized_experts.csv",
        specialized_rows,
        specialized_fields,
    )


def _validate_domain_shapes(statistics: dict[str, DomainStatistics]) -> None:
    shapes = {domain: stats.routing_counts.shape[1:] for domain, stats in statistics.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Domain layer/expert shapes differ: {shapes}")
    for domain, stats in statistics.items():
        if np.any(stats.token_counts == 0):
            raise RuntimeError(f"Domain {domain!r} contains unprocessed examples")
        if np.any(stats.routing_counts.sum(axis=(0, 2)) == 0):
            raise RuntimeError(f"Domain {domain!r} has an MoE layer with no routing counts")
        if np.any(stats.contribution_sums.sum(axis=(0, 2)) <= 0):
            raise RuntimeError(
                f"Domain {domain!r} has an MoE layer with no functional contribution"
            )
