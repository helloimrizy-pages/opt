from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


SIMPLE_POLICY_ORDER = ("lru", "lfu", "lfu_decay", "static_hotset")
REGIME_ORDER = ("stationary", "abrupt", "repeated", "mixed")


def analyze_headroom(
    result_rows: Sequence[Mapping[str, Any]],
    per_sequence_rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    config = frozen["preregistered_config"]
    primary = [
        dict(row)
        for row in result_rows
        if row["cost_model"] == config["primary_cost_model"]
        and abs(float(row["lambda"]) - float(config["primary_lambda"])) < 1e-12
    ]
    selected_alpha = float(frozen["selected_lfu_decay_alpha"])
    primary_selected = [
        row
        for row in primary
        if row["policy"] != "lfu_decay"
        or abs(float(row["alpha"]) - selected_alpha) < 1e-12
    ]
    by_condition = {row["condition_id"]: row for row in primary_selected}
    sequence_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sequence_rows:
        if row["condition_id"] in by_condition:
            sequence_by_condition[row["condition_id"]].append(dict(row))
    for condition_rows in sequence_by_condition.values():
        condition_rows.sort(key=lambda row: int(row["sequence_position"]))

    workload_rows: list[dict[str, Any]] = []
    workloads = sorted({row["workload"] for row in primary_selected})
    capacities = list(map(int, config["cache_capacities"]))
    bootstrap_replicates = int(config["bootstrap_replicates"])
    bootstrap_seed = int(config["bootstrap_seed"])
    for workload in workloads:
        for capacity in capacities:
            records = [
                row
                for row in primary_selected
                if row["workload"] == workload and int(row["capacity"]) == capacity
            ]
            workload_rows.append(
                _condition_headroom(
                    records,
                    sequence_by_condition,
                    replicates=bootstrap_replicates,
                    seed=_derived_seed(bootstrap_seed, f"workload:{workload}:{capacity}"),
                )
            )

    regime_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        regime_workloads = sorted(
            {
                row["workload"]
                for row in primary_selected
                if row["regime"] == regime
            }
        )
        if not regime_workloads:
            continue
        for capacity in capacities:
            headroom = _aggregate_regime(
                regime,
                regime_workloads,
                capacity,
                primary_selected,
                sequence_by_condition,
                replicates=bootstrap_replicates,
                seed=_derived_seed(bootstrap_seed, f"regime:{regime}:{capacity}"),
            )
            regime_rows.append(headroom)
            table_rows.append(
                _table_row(
                    regime,
                    regime_workloads,
                    capacity,
                    primary_selected,
                    headroom,
                    selected_alpha,
                )
            )

    random_rows = _random_statistics(primary, capacities)
    decision = stage0_decision(regime_rows, config)
    association_rows, correlations = _diagnostic_associations(workload_rows, diagnostics)
    return {
        "schema_version": "race_stage0_analysis_v1",
        "primary_definition": {
            "cost_model": config["primary_cost_model"],
            "lambda": config["primary_lambda"],
            "strongest_simple_candidates": list(SIMPLE_POLICY_ORDER),
            "lfu_decay_alpha": selected_alpha,
            "bootstrap_unit": config["bootstrap_unit"],
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "workload_bootstrap_strata": ["segment_index", "domain"],
            "regime_bootstrap_cluster": (
                "source_sequence_id, stratified by domain across workload components"
            ),
            "best_simple_reselected_inside_each_paired_bootstrap_replicate": True,
        },
        "workload_headroom": workload_rows,
        "regime_headroom": regime_rows,
        "table_rows": table_rows,
        "random_statistics": random_rows,
        "routing_diagnostics": list(diagnostics),
        "diagnostic_associations": association_rows,
        "diagnostic_correlations": correlations,
        "decision": decision,
    }


def _condition_headroom(
    records: Sequence[Mapping[str, Any]],
    sequence_by_condition: Mapping[str, list[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    oracle = _one(records, "oracle")
    simple = [_one(records, name) for name in SIMPLE_POLICY_ORDER]
    best = min(simple, key=lambda row: (float(row["total_cost"]), SIMPLE_POLICY_ORDER.index(row["policy"])))
    simple_rows = [_sequence_rows(row, sequence_by_condition) for row in simple]
    oracle_rows = _sequence_rows(oracle, sequence_by_condition)
    _validate_aligned_rows(simple_rows + [oracle_rows])
    vectors = [_cost_vector(rows) for rows in simple_rows]
    oracle_vector = _cost_vector(oracle_rows)
    multiplicities = _stratified_workload_multiplicities(
        oracle_rows, replicates=replicates, seed=seed
    )
    simple_boot = np.stack(
        [(multiplicities * vector[None, :]).sum(axis=1) for vector in vectors], axis=1
    )
    best_boot = simple_boot.min(axis=1)
    oracle_boot = (multiplicities * oracle_vector[None, :]).sum(axis=1)
    headroom_boot = _safe_ratio(best_boot - oracle_boot, best_boot)
    absolute_boot = best_boot - oracle_boot
    difference = _sequence_vector(best, sequence_by_condition) - oracle_vector
    return {
        "workload": oracle["workload"],
        "regime": oracle["regime"],
        "capacity": int(oracle["capacity"]),
        "best_simple_policy": _method_label(best),
        "best_simple_cost": float(best["total_cost"]),
        "oracle_cost": float(oracle["total_cost"]),
        "absolute_gap": float(best["total_cost"] - oracle["total_cost"]),
        "headroom": _scalar_ratio(
            float(best["total_cost"] - oracle["total_cost"]), float(best["total_cost"])
        ),
        "headroom_ci_low": _quantile(headroom_boot, 0.025),
        "headroom_ci_high": _quantile(headroom_boot, 0.975),
        "absolute_gap_ci_low": _quantile(absolute_boot, 0.025),
        "absolute_gap_ci_high": _quantile(absolute_boot, 0.975),
        "mean_sequence_absolute_gap": float(np.mean(difference)),
        "median_sequence_absolute_gap": float(np.median(difference)),
        "paired_standardized_effect": _standardized_effect(difference),
        "sequences": len(difference),
    }


def _aggregate_regime(
    regime: str,
    workloads: Sequence[str],
    capacity: int,
    primary: Sequence[Mapping[str, Any]],
    sequence_by_condition: Mapping[str, list[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for workload in workloads:
        records = [
            row
            for row in primary
            if row["workload"] == workload and int(row["capacity"]) == capacity
        ]
        oracle = _one(records, "oracle")
        simple = [_one(records, name) for name in SIMPLE_POLICY_ORDER]
        best = min(
            simple,
            key=lambda row: (float(row["total_cost"]), SIMPLE_POLICY_ORDER.index(row["policy"])),
        )
        simple_rows = [_sequence_rows(row, sequence_by_condition) for row in simple]
        oracle_rows = _sequence_rows(oracle, sequence_by_condition)
        _validate_aligned_rows(simple_rows + [oracle_rows])
        components.append(
            {
                "workload": workload,
                "oracle": oracle,
                "best": best,
                "simple_vectors": [_cost_vector(rows) for rows in simple_rows],
                "oracle_vector": _cost_vector(oracle_rows),
                "oracle_rows": oracle_rows,
            }
        )

    multiplicity_by_sequence = _regime_cluster_multiplicities(
        [component["oracle_rows"] for component in components],
        replicates=replicates,
        seed=seed,
    )
    best_point_total = 0.0
    oracle_point_total = 0.0
    best_boot_total = np.zeros(replicates, dtype=np.float64)
    oracle_boot_total = np.zeros(replicates, dtype=np.float64)
    differences: list[np.ndarray] = []
    best_names: list[str] = []
    total_sequences = 0
    for component in components:
        oracle = component["oracle"]
        best = component["best"]
        oracle_rows = component["oracle_rows"]
        oracle_vector = component["oracle_vector"]
        multiplicities = np.stack(
            [
                multiplicity_by_sequence[int(row["source_sequence_id"])]
                for row in oracle_rows
            ],
            axis=1,
        )
        simple_boot = np.stack(
            [
                (multiplicities * vector[None, :]).sum(axis=1)
                for vector in component["simple_vectors"]
            ],
            axis=1,
        )
        best_boot_total += simple_boot.min(axis=1)
        oracle_boot_total += (multiplicities * oracle_vector[None, :]).sum(axis=1)
        best_point_total += float(best["total_cost"])
        oracle_point_total += float(oracle["total_cost"])
        differences.append(_sequence_vector(best, sequence_by_condition) - oracle_vector)
        best_names.append(f"{component['workload']}:{_method_label(best)}")
        total_sequences += len(oracle_vector)
    headroom_boot = _safe_ratio(best_boot_total - oracle_boot_total, best_boot_total)
    absolute_boot = best_boot_total - oracle_boot_total
    difference = np.concatenate(differences)
    return {
        "regime": regime,
        "capacity": capacity,
        "workloads": len(workloads),
        "sequences": total_sequences,
        "unique_source_sequences": len(multiplicity_by_sequence),
        "best_simple_policy_by_workload": ";".join(best_names),
        "best_simple_cost": best_point_total,
        "oracle_cost": oracle_point_total,
        "absolute_gap": best_point_total - oracle_point_total,
        "headroom": _scalar_ratio(best_point_total - oracle_point_total, best_point_total),
        "headroom_ci_low": _quantile(headroom_boot, 0.025),
        "headroom_ci_high": _quantile(headroom_boot, 0.975),
        "absolute_gap_ci_low": _quantile(absolute_boot, 0.025),
        "absolute_gap_ci_high": _quantile(absolute_boot, 0.975),
        "mean_sequence_absolute_gap": float(np.mean(difference)),
        "median_sequence_absolute_gap": float(np.median(difference)),
        "paired_standardized_effect": _standardized_effect(difference),
    }


def _table_row(
    regime: str,
    workloads: Sequence[str],
    capacity: int,
    primary: Sequence[Mapping[str, Any]],
    headroom: Mapping[str, Any],
    selected_alpha: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "regime": regime,
        "capacity": capacity,
        "lfu_decay_alpha": selected_alpha,
        "oracle_headroom": headroom["headroom"],
        "oracle_headroom_ci_low": headroom["headroom_ci_low"],
        "oracle_headroom_ci_high": headroom["headroom_ci_high"],
        "best_simple_policy_by_workload": headroom["best_simple_policy_by_workload"],
    }
    for policy in (*SIMPLE_POLICY_ORDER, "oracle"):
        rows = [
            row
            for row in primary
            if row["workload"] in workloads
            and int(row["capacity"]) == capacity
            and row["policy"] == policy
        ]
        output[policy] = float(sum(float(row["total_cost"]) for row in rows))
        output[f"{policy}_requests"] = int(sum(int(row["requests"]) for row in rows))
        output[f"{policy}_normalized"] = _scalar_ratio(
            output[policy], float(output[f"{policy}_requests"])
        )
    return output


def _random_statistics(
    primary: Sequence[Mapping[str, Any]], capacities: Sequence[int]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        workloads = sorted({row["workload"] for row in primary if row["regime"] == regime})
        if not workloads:
            continue
        for capacity in capacities:
            by_seed: dict[int, float] = defaultdict(float)
            for row in primary:
                if (
                    row["policy"] == "random"
                    and row["regime"] == regime
                    and int(row["capacity"]) == capacity
                ):
                    by_seed[int(row["seed"])] += float(row["total_cost"])
            values = np.asarray(list(by_seed.values()), dtype=np.float64)
            output.append(
                {
                    "regime": regime,
                    "capacity": capacity,
                    "seeds": len(values),
                    "mean_total_cost": float(values.mean()),
                    "std_total_cost": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "minimum_total_cost": float(values.min()),
                    "maximum_total_cost": float(values.max()),
                }
            )
    return output


def stage0_decision(
    regime_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    criteria = config["go_no_go"]
    outcomes: dict[str, Any] = {}
    for regime in REGIME_ORDER:
        rows = [row for row in regime_rows if row["regime"] == regime]
        if not rows:
            continue
        strong = [
            int(row["capacity"])
            for row in rows
            if float(row["headroom"]) >= float(criteria["strong_point_headroom"])
            and float(row["headroom_ci_low"]) >= float(criteria["strong_ci_lower"])
        ]
        weak = [
            int(row["capacity"])
            for row in rows
            if float(row["headroom"]) >= float(criteria["weak_point_headroom"])
            and float(row["headroom_ci_low"]) > float(criteria["weak_ci_lower"])
        ]
        outcomes[regime] = {
            "strong_budgets": strong,
            "weak_or_better_budgets": weak,
            "strong_budget_count": len(strong),
            "weak_or_better_budget_count": len(weak),
        }
    if not bool(config.get("decision_enabled", False)):
        decision = "PILOT_ONLY_NO_STAGE0_DECISION"
        reason = "Pilot/smoke runs validate mechanics only; the full frozen trace is required."
    else:
        strong_regimes = [
            name
            for name, item in outcomes.items()
            if item["strong_budget_count"] >= int(criteria["strong_required_budgets"])
        ]
        weak_regimes = [
            name
            for name, item in outcomes.items()
            if item["weak_or_better_budget_count"] >= int(criteria["weak_required_budgets"])
        ]
        if strong_regimes:
            decision = "RACE_STAGE0_STRONG_GO"
            reason = (
                "At least one workload regime has >=15% oracle headroom with a paired "
                "CI lower bound >=5% at 3/5 or more cache budgets."
            )
        elif weak_regimes:
            decision = "RACE_STAGE0_WEAK_GO"
            reason = (
                "No regime met STRONG GO, but at least one has >=5% positive-CI "
                "headroom at 3/5 or more cache budgets."
            )
        else:
            decision = "RACE_STAGE0_NO_GO"
            reason = (
                "Neither the preregistered STRONG nor WEAK oracle-headroom rule was met."
            )
    return {
        "decision": decision,
        "reason": reason,
        "criteria": dict(criteria),
        "regime_outcomes": outcomes,
    }


def _diagnostic_associations(
    workload_rows: Sequence[Mapping[str, Any]], diagnostics: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_workload = {row["workload"]: row for row in diagnostics}
    metrics = (
        "normalized_frequency_entropy_mean",
        "top_10_traffic_share_mean",
        "gini_mean",
        "consecutive_jaccard_mean",
        "reuse_within_10_events",
        "activity_lag1_autocorrelation_mean",
        "adjacent_segment_js_mean",
        "pairwise_domain_js_mean",
    )
    rows: list[dict[str, Any]] = []
    for headroom in workload_rows:
        diagnostic = by_workload[headroom["workload"]]
        row = dict(headroom)
        row.update({name: diagnostic[name] for name in metrics})
        rows.append(row)
    correlations: list[dict[str, Any]] = []
    for metric in metrics:
        pairs = [
            (float(row[metric]), float(row["headroom"]))
            for row in rows
            if row[metric] is not None
            and np.isfinite(float(row[metric]))
            and np.isfinite(float(row["headroom"]))
        ]
        if len(pairs) < 3:
            correlation = float("nan")
            p_value = float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = spearmanr(
                    [item[0] for item in pairs], [item[1] for item in pairs]
                )
            correlation = float(result.statistic)
            p_value = float(result.pvalue)
        correlations.append(
            {
                "diagnostic": metric,
                "spearman_with_headroom": correlation,
                "p_value_descriptive": p_value,
                "conditions": len(pairs),
            }
        )
    return rows, correlations


def _one(records: Sequence[Mapping[str, Any]], policy: str) -> Mapping[str, Any]:
    matches = [row for row in records if row["policy"] == policy]
    if len(matches) != 1:
        raise ValueError(f"Expected one {policy} record, found {len(matches)}")
    return matches[0]


def _sequence_vector(
    record: Mapping[str, Any], sequence_by_condition: Mapping[str, list[dict[str, Any]]]
) -> np.ndarray:
    return _cost_vector(_sequence_rows(record, sequence_by_condition))


def _sequence_rows(
    record: Mapping[str, Any], sequence_by_condition: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows = sequence_by_condition.get(record["condition_id"], [])
    if not rows:
        raise ValueError(f"No per-sequence values for {record['condition_id']}")
    return rows


def _cost_vector(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([row["misses"] for row in rows], dtype=np.float64)


def _row_identity(row: Mapping[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["sequence_position"]),
        int(row["source_sequence_id"]),
        str(row["domain"]),
        int(row["segment_index"]),
    )


def _validate_aligned_rows(rows_by_policy: Sequence[Sequence[Mapping[str, Any]]]) -> None:
    identities = [tuple(_row_identity(row) for row in rows) for rows in rows_by_policy]
    if not identities or not identities[0] or any(value != identities[0] for value in identities[1:]):
        raise ValueError("Paired per-sequence rows are not identically aligned")


def _stratified_workload_multiplicities(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> np.ndarray:
    """Bootstrap prompts within fixed workload segment/domain strata."""

    rng = np.random.default_rng(seed)
    output = np.zeros((replicates, len(rows)), dtype=np.int32)
    strata: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[(int(row["segment_index"]), str(row["domain"]))].append(index)
    replicate_index = np.arange(replicates, dtype=np.int64)[:, None]
    for key in sorted(strata):
        positions = np.asarray(strata[key], dtype=np.int64)
        draws = rng.integers(0, len(positions), size=(replicates, len(positions)))
        selected = positions[draws]
        np.add.at(output, (replicate_index, selected), 1)
    return output


def _regime_cluster_multiplicities(
    workload_rows: Sequence[Sequence[Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Reuse one prompt's bootstrap multiplicity everywhere it occurs in a regime."""

    domain_by_sequence: dict[int, str] = {}
    for rows in workload_rows:
        for row in rows:
            sequence = int(row["source_sequence_id"])
            domain = str(row["domain"])
            previous = domain_by_sequence.setdefault(sequence, domain)
            if previous != domain:
                raise ValueError(f"Source sequence {sequence} changes domain across workloads")
    by_domain: dict[str, list[int]] = defaultdict(list)
    for sequence, domain in domain_by_sequence.items():
        by_domain[domain].append(sequence)
    rng = np.random.default_rng(seed)
    output: dict[int, np.ndarray] = {}
    replicate_index = np.arange(replicates, dtype=np.int64)[:, None]
    for domain in sorted(by_domain):
        identifiers = np.asarray(sorted(by_domain[domain]), dtype=np.int64)
        counts = np.zeros((replicates, len(identifiers)), dtype=np.int32)
        draws = rng.integers(0, len(identifiers), size=(replicates, len(identifiers)))
        np.add.at(counts, (replicate_index, draws), 1)
        for index, sequence in enumerate(identifiers):
            output[int(sequence)] = counts[:, index]
    return output


def _method_label(record: Mapping[str, Any]) -> str:
    if record["policy"] == "lfu_decay":
        return f"lfu_decay(alpha={float(record['alpha']):.2f})"
    return str(record["policy"])


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full_like(numerator, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


def _scalar_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else float("nan")


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else float("nan")


def _standardized_effect(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    deviation = float(np.std(values, ddof=1))
    if deviation == 0:
        return float("inf") if float(np.mean(values)) > 0 else 0.0
    return float(np.mean(values) / deviation)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"race-stage0-bootstrap-v1\0{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
