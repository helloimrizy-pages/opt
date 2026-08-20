from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from residency_headroom.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
    write_csv,
)

from .calibration import FAMILY_ORDER, load_and_verify_stage1_frozen
from .frozen import load_and_verify_frozen_inputs, stage0_references
from .metrics import comparison_metrics
from .simulation import method_id


REGIME_ORDER = ("stationary", "abrupt", "repeated", "mixed")
FAMILY_LABELS = {
    "persistence": "Persistence",
    "last_gate": "LastGate",
    "gate_ewma": "GateEWMA",
    "markov_1": "Markov1",
    "markov_h": "MarkovH",
    "markov_plus_ewma": "Hybrid",
}


def analyze_and_report(
    repository_root: Path,
    preregistration_path: Path,
    frozen_config_path: Path,
    evaluation_dir: Path,
    stage1_root: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    frozen_inputs = load_and_verify_frozen_inputs(repository_root, preregistration_path)
    stage1_frozen = load_and_verify_stage1_frozen(frozen_config_path)
    manifest = read_json(evaluation_dir / "evaluation_manifest.json")
    if manifest["frozen_config_file_sha256"] != stage1_frozen["file_sha256"]:
        raise ValueError("Evaluation and Stage 1 frozen config hashes differ")
    if not read_json(evaluation_dir / "sanity_checks.json")["passed"]:
        raise ValueError("Cannot analyze an evaluation that failed sanity checks")
    stage1_rows = read_jsonl(evaluation_dir / "results.jsonl")
    stage1_sequences = read_jsonl(evaluation_dir / "per_sequence_results.jsonl")
    quality_rows = read_jsonl(evaluation_dir / "prediction_quality.jsonl")
    stage0_result_path = repository_root / frozen_inputs.preregistration["stage0_reference"][
        "result_path"
    ]
    stage0_per_sequence_path = repository_root / frozen_inputs.preregistration[
        "stage0_reference"
    ]["per_sequence_result_path"]
    references = stage0_references(stage0_result_path)
    needed_stage0_conditions = {
        str(item[role]["condition_id"])
        for item in references.values()
        for role in ("simple", "oracle")
    }
    stage0_sequences = [
        row
        for row in read_jsonl(stage0_per_sequence_path)
        if str(row["condition_id"]) in needed_stage0_conditions
    ]
    stage1_lookup = {
        (str(row["workload"]), int(row["capacity"]), str(row["method_id"])): row
        for row in stage1_rows
    }
    stage0_sequence_lookup = _sequence_lookup(stage0_sequences)
    stage1_sequence_lookup = _sequence_lookup(stage1_sequences)
    selected_specs = {
        family: dict(spec)
        for family, spec in stage1_frozen["selected_by_family"].items()
    }
    selected_ids = {family: method_id(spec) for family, spec in selected_specs.items()}
    selected_predictor_id = str(stage1_frozen["selected_predictor_id"])
    capacities = tuple(map(int, stage1_frozen["cache_capacities"]))
    decision_capacities = tuple(map(int, stage1_frozen["decision_capacities"]))
    workloads_by_regime = {
        regime: tuple(workload.name for workload in frozen_inputs.workloads if workload.regime == regime)
        for regime in REGIME_ORDER
    }

    detailed_rows = _detailed_causal_rows(
        frozen_inputs.workloads,
        capacities,
        references,
        stage1_lookup,
        selected_ids,
    )
    regime_rows = [
        _aggregate_costs(
            regime,
            workloads_by_regime[regime],
            capacity,
            references,
            stage1_lookup,
            selected_ids,
            selected_predictor_id,
        )
        for regime in REGIME_ORDER
        for capacity in capacities
    ]
    suite_workloads = tuple(workload.name for workload in frozen_inputs.workloads)
    suite_rows = [
        _aggregate_costs(
            "all_frozen_workloads",
            suite_workloads,
            capacity,
            references,
            stage1_lookup,
            selected_ids,
            selected_predictor_id,
        )
        for capacity in capacities
    ]

    bootstrap_replicates = int(stage1_frozen["bootstrap"]["bootstrap_replicates"])
    bootstrap_seed = int(stage1_frozen["bootstrap"]["bootstrap_seed"])
    bootstrap_rows = []
    for scope, workloads in [
        *((regime, workloads_by_regime[regime]) for regime in REGIME_ORDER),
        ("all_frozen_workloads", suite_workloads),
    ]:
        for capacity in capacities:
            bootstrap_rows.append(
                _bootstrap_comparison(
                    scope,
                    workloads,
                    capacity,
                    selected_predictor_id,
                    references,
                    stage1_lookup,
                    stage0_sequence_lookup,
                    stage1_sequence_lookup,
                    replicates=bootstrap_replicates,
                    seed=_derived_seed(bootstrap_seed, f"{scope}:{capacity}"),
                )
            )
    bootstrap_by_key = {
        (row["bootstrap_scope"], int(row["bootstrap_capacity"])): row
        for row in bootstrap_rows
    }
    for row in regime_rows + suite_rows:
        row.update(bootstrap_by_key[(row["scope"], int(row["capacity"]))])

    all_spec_ids = sorted({str(row["method_id"]) for row in stage1_rows if row["causal"]})
    sensitivity_rows = []
    for method_identifier in all_spec_ids:
        for capacity in capacities:
            values = _aggregate_one_method(
                suite_workloads,
                capacity,
                method_identifier,
                references,
                stage1_lookup,
            )
            sensitivity_rows.append(
                {
                    "method_id": method_identifier,
                    "capacity": capacity,
                    **values,
                }
            )
    lookahead_rows = []
    for horizon in stage1_frozen["limited_lookahead_horizons"]:
        identifier = f"lookahead_oracle_h{int(horizon)}"
        for capacity in capacities:
            values = _aggregate_one_method(
                suite_workloads, capacity, identifier, references, stage1_lookup
            )
            lookahead_rows.append(
                {"horizon": int(horizon), "capacity": capacity, **values}
            )
    perfect_rows = [
        {
            "capacity": capacity,
            **_aggregate_one_method(
                suite_workloads,
                capacity,
                "perfect_score_simple_policy",
                references,
                stage1_lookup,
            ),
        }
        for capacity in capacities
    ]
    quality_analysis = _quality_analysis(
        quality_rows, sensitivity_rows, decision_capacities
    )
    decision = _stage1_decision(suite_rows, stage1_frozen["decision_rule"])
    analysis = {
        "schema_version": "race_stage1_analysis_v1",
        "created_at_utc": utc_now(),
        "verdict": decision["verdict"],
        "decision": decision,
        "selected_predictor": stage1_frozen["selected_predictor"],
        "selected_predictor_id": selected_predictor_id,
        "selected_by_family": selected_specs,
        "primary_aggregation": stage1_frozen["bootstrap"]["primary_suite_aggregation"],
        "causal_by_workload": detailed_rows,
        "regime_results": regime_rows,
        "suite_results": suite_rows,
        "causal_sensitivity": sensitivity_rows,
        "lookahead_results": lookahead_rows,
        "perfect_score_results": perfect_rows,
        "prediction_quality": quality_analysis,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_interpretation": stage1_frozen["bootstrap"]["conditionality"],
        "trace_hash": frozen_inputs.trace.trace_hash,
        "preregistration_hash": frozen_inputs.preregistration_hash,
        "frozen_config_file_sha256": stage1_frozen["file_sha256"],
        "evaluation_manifest_sha256": sha256_file(
            evaluation_dir / "evaluation_manifest.json"
        ),
    }
    report_dir = stage1_root / "reports"
    table_dir = stage1_root / "tables"
    figure_dir = stage1_root / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / "analysis.json", analysis)
    _write_tables(table_dir, detailed_rows, regime_rows, suite_rows, sensitivity_rows, lookahead_rows)
    _write_figures(
        figure_dir,
        regime_rows,
        suite_rows,
        sensitivity_rows,
        lookahead_rows,
        quality_analysis,
        decision_capacities,
        selected_predictor_id,
    )
    report = _render_report(
        analysis,
        frozen_inputs,
        stage1_frozen,
        table_dir,
        figure_dir,
    )
    report_path = report_dir / "race_stage1_prediction_headroom_report.md"
    atomic_write_text(report_path, report)
    atomic_write_text(
        report_path.with_suffix(".sha256"),
        f"{sha256_file(report_path)}  {report_path.name}\n",
    )
    audit = _analysis_audit(analysis, report_path, table_dir, figure_dir)
    atomic_write_json(report_dir / "analysis_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Stage 1 analysis audit failed")
    return analysis


def _detailed_causal_rows(
    workloads: Iterable[Any],
    capacities: Sequence[int],
    references: Mapping[tuple[str, int], Mapping[str, Any]],
    lookup: Mapping[tuple[str, int, str], Mapping[str, Any]],
    selected_ids: Mapping[str, str],
) -> list[dict[str, Any]]:
    output = []
    for workload in workloads:
        for capacity in capacities:
            reference = references[(workload.name, capacity)]
            simple = float(reference["simple"]["misses"])
            oracle = float(reference["oracle"]["misses"])
            row: dict[str, Any] = {
                "workload": workload.name,
                "regime": workload.regime,
                "capacity": capacity,
                "stage0_best_policy": reference["simple"]["policy"],
                "stage0_best": simple,
                "oracle": oracle,
            }
            for family in FAMILY_ORDER:
                cost = float(lookup[(workload.name, capacity, selected_ids[family])]["misses"])
                row[FAMILY_LABELS[family]] = cost
                row[f"{FAMILY_LABELS[family]}_gap_closed"] = comparison_metrics(
                    simple, cost, oracle
                )["oracle_gap_closed"]
            output.append(row)
    return output


def _aggregate_costs(
    scope: str,
    workloads: Sequence[str],
    capacity: int,
    references: Mapping[tuple[str, int], Mapping[str, Any]],
    lookup: Mapping[tuple[str, int, str], Mapping[str, Any]],
    selected_ids: Mapping[str, str],
    selected_predictor_id: str,
) -> dict[str, Any]:
    simple = float(sum(float(references[(name, capacity)]["simple"]["misses"]) for name in workloads))
    oracle = float(sum(float(references[(name, capacity)]["oracle"]["misses"]) for name in workloads))
    row: dict[str, Any] = {
        "scope": scope,
        "capacity": capacity,
        "spare_residency": capacity - 8,
        "workloads": len(workloads),
        "baseline_cost": simple,
        "oracle_cost": oracle,
        "stage0_oracle_headroom": (simple - oracle) / simple,
    }
    for family in FAMILY_ORDER:
        identifier = selected_ids[family]
        cost = float(sum(float(lookup[(name, capacity, identifier)]["misses"]) for name in workloads))
        row[f"{family}_cost"] = cost
        row[f"{family}_gap_closed"] = comparison_metrics(simple, cost, oracle)[
            "oracle_gap_closed"
        ]
    predictor = float(
        sum(float(lookup[(name, capacity, selected_predictor_id)]["misses"]) for name in workloads)
    )
    row.update({"predictor_cost": predictor, **comparison_metrics(simple, predictor, oracle)})
    return row


def _aggregate_one_method(
    workloads: Sequence[str],
    capacity: int,
    identifier: str,
    references: Mapping[tuple[str, int], Mapping[str, Any]],
    lookup: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    simple = float(sum(float(references[(name, capacity)]["simple"]["misses"]) for name in workloads))
    oracle = float(sum(float(references[(name, capacity)]["oracle"]["misses"]) for name in workloads))
    cost = float(sum(float(lookup[(name, capacity, identifier)]["misses"]) for name in workloads))
    return {
        "baseline_cost": simple,
        "method_cost": cost,
        "oracle_cost": oracle,
        **comparison_metrics(simple, cost, oracle),
    }


def _sequence_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["condition_id"])].append(dict(row))
    for values in output.values():
        values.sort(key=lambda row: int(row["sequence_position"]))
    return output


def _bootstrap_comparison(
    scope: str,
    workloads: Sequence[str],
    capacity: int,
    method_identifier: str,
    references: Mapping[tuple[str, int], Mapping[str, Any]],
    stage1_lookup: Mapping[tuple[str, int, str], Mapping[str, Any]],
    stage0_sequences: Mapping[str, list[dict[str, Any]]],
    stage1_sequences: Mapping[str, list[dict[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    components = []
    domain_by_sequence: dict[int, str] = {}
    differences = []
    for workload in workloads:
        reference = references[(workload, capacity)]
        method = stage1_lookup[(workload, capacity, method_identifier)]
        simple_rows = stage0_sequences[str(reference["simple"]["condition_id"])]
        oracle_rows = stage0_sequences[str(reference["oracle"]["condition_id"])]
        method_rows = stage1_sequences[str(method["condition_id"])]
        identities = [
            [(int(row["source_sequence_id"]), str(row["domain"])) for row in values]
            for values in (simple_rows, method_rows, oracle_rows)
        ]
        if identities[0] != identities[1] or identities[0] != identities[2]:
            raise ValueError("Paired Stage 0/Stage 1 per-sequence rows are not aligned")
        for sequence, domain in identities[0]:
            previous = domain_by_sequence.setdefault(sequence, domain)
            if previous != domain:
                raise ValueError("A source sequence changes domain across frozen workloads")
        simple_vector = np.asarray([row["misses"] for row in simple_rows], dtype=np.float64)
        method_vector = np.asarray([row["misses"] for row in method_rows], dtype=np.float64)
        oracle_vector = np.asarray([row["misses"] for row in oracle_rows], dtype=np.float64)
        differences.extend(method_vector - oracle_vector)
        components.append((identities[0], simple_vector, method_vector, oracle_vector))
    multiplicities = _cluster_multiplicities(domain_by_sequence, replicates, seed)
    simple_boot = np.zeros(replicates)
    method_boot = np.zeros(replicates)
    oracle_boot = np.zeros(replicates)
    for identities, simple_vector, method_vector, oracle_vector in components:
        weights = np.stack([multiplicities[sequence] for sequence, _domain in identities], axis=1)
        simple_boot += (weights * simple_vector[None, :]).sum(axis=1)
        method_boot += (weights * method_vector[None, :]).sum(axis=1)
        oracle_boot += (weights * oracle_vector[None, :]).sum(axis=1)
    denominator = simple_boot - oracle_boot
    gap_closed = np.full(replicates, np.nan)
    np.divide(simple_boot - method_boot, denominator, out=gap_closed, where=denominator != 0)
    residual = np.divide(
        method_boot - oracle_boot,
        simple_boot,
        out=np.full(replicates, np.nan),
        where=simple_boot != 0,
    )
    improvement = np.divide(
        simple_boot - method_boot,
        simple_boot,
        out=np.full(replicates, np.nan),
        where=simple_boot != 0,
    )
    difference_values = np.asarray(differences, dtype=np.float64)
    return {
        "bootstrap_scope": scope,
        "bootstrap_capacity": capacity,
        "gap_closed_ci_low": _quantile(gap_closed, 0.025),
        "gap_closed_ci_high": _quantile(gap_closed, 0.975),
        "residual_headroom_ci_low": _quantile(residual, 0.025),
        "residual_headroom_ci_high": _quantile(residual, 0.975),
        "baseline_improvement_ci_low": _quantile(improvement, 0.025),
        "baseline_improvement_ci_high": _quantile(improvement, 0.975),
        "mean_sequence_predictor_oracle_gap": float(difference_values.mean()),
        "median_sequence_predictor_oracle_gap": float(np.median(difference_values)),
        "paired_standardized_effect": _standardized_effect(difference_values),
        "bootstrap_replicates": replicates,
        "unique_source_sequences": len(domain_by_sequence),
    }


def _cluster_multiplicities(
    domain_by_sequence: Mapping[int, str], replicates: int, seed: int
) -> dict[int, np.ndarray]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for sequence, domain in domain_by_sequence.items():
        by_domain[domain].append(sequence)
    rng = np.random.default_rng(seed)
    output: dict[int, np.ndarray] = {}
    replicate_index = np.arange(replicates, dtype=np.int64)[:, None]
    for domain in sorted(by_domain):
        identifiers = np.asarray(sorted(by_domain[domain]), dtype=np.int64)
        counts = np.zeros((replicates, len(identifiers)), dtype=np.int32)
        draws = rng.integers(0, len(identifiers), size=counts.shape)
        np.add.at(counts, (replicate_index, draws), 1)
        for index, sequence in enumerate(identifiers):
            output[int(sequence)] = counts[:, index]
    return output


def _stage1_decision(
    suite_rows: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]
) -> dict[str, Any]:
    nondegenerate = [row for row in suite_rows if int(row["capacity"]) != 8]
    strong_a = [
        int(row["capacity"])
        for row in nondegenerate
        if float(row["oracle_gap_closed"]) < float(rule["strong_go"]["gap_closed_strictly_below"])
    ]
    strong_b = [
        int(row["capacity"])
        for row in nondegenerate
        if float(row["residual_headroom"]) >= float(rule["strong_go"]["residual_headroom_at_least"])
    ]
    no_gap = [
        int(row["capacity"])
        for row in nondegenerate
        if float(row["oracle_gap_closed"]) >= float(rule["no_go"]["gap_closed_at_least"])
    ]
    no_residual = [
        int(row["capacity"])
        for row in nondegenerate
        if float(row["residual_headroom"]) < float(rule["no_go"]["or_residual_headroom_strictly_below"])
    ]
    no_go = (
        len(no_gap) >= int(rule["no_go"]["gap_closed_required_capacities"])
        or len(no_residual) >= int(rule["no_go"]["residual_required_capacities"])
    )
    strong_go = (
        len(strong_a) >= int(rule["strong_go"]["gap_closed_required_capacities"])
        and len(strong_b) >= int(rule["strong_go"]["residual_required_capacities"])
    )
    if no_go:
        verdict = "RACE_STAGE1_NO_GO"
        reason = (
            "The preregistered NO-GO trigger was met: simple causal prediction captured "
            "at least 75% of the oracle gap or left under 5% residual headroom at 3/4 "
            "non-degenerate capacities."
        )
    elif strong_go:
        verdict = "RACE_STAGE1_STRONG_GO"
        reason = (
            "At 3/4 or more non-degenerate capacities, the globally calibration-selected "
            "causal predictor closed under 50% of the Stage 0 oracle gap and left at least "
            "10% of Stage 0 baseline cost between itself and the oracle."
        )
    else:
        verdict = "RACE_STAGE1_WEAK_GO"
        reason = "Neither the frozen STRONG-GO rule nor either frozen NO-GO trigger was met."
    return {
        "verdict": verdict,
        "reason": reason,
        "strong_condition_a_capacities": strong_a,
        "strong_condition_b_capacities": strong_b,
        "no_go_gap_closed_capacities": no_gap,
        "no_go_low_residual_capacities": no_residual,
        "rule": dict(rule),
    }


def _quality_analysis(
    quality_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    decision_capacities: Sequence[int],
) -> dict[str, Any]:
    gap_by_method: dict[str, list[float]] = defaultdict(list)
    for row in sensitivity_rows:
        if int(row["capacity"]) in decision_capacities and row["oracle_gap_closed"] is not None:
            gap_by_method[str(row["method_id"])].append(float(row["oracle_gap_closed"]))
    joined = []
    for row in quality_rows:
        values = gap_by_method.get(str(row["method_id"]), [])
        if not values:
            continue
        joined.append(
            {
                **dict(row),
                "mean_suite_gap_closed_capacities_12_32": float(np.mean(values)),
            }
        )
    if len(joined) >= 3:
        correlation = spearmanr(
            [float(row["average_precision"]) for row in joined],
            [float(row["mean_suite_gap_closed_capacities_12_32"]) for row in joined],
        )
        statistic = float(correlation.statistic)
        p_value = float(correlation.pvalue)
    else:
        statistic = float("nan")
        p_value = float("nan")
    return {
        "workload": "mixed_interleaved",
        "joined_rows": joined,
        "spearman_average_precision_vs_gap_closed": statistic,
        "p_value_descriptive": p_value,
        "interpretation": "Descriptive configuration-level association; configurations are not independent experiments.",
    }


def _write_tables(
    table_dir: Path,
    detailed: Sequence[Mapping[str, Any]],
    regime: Sequence[Mapping[str, Any]],
    suite: Sequence[Mapping[str, Any]],
    sensitivity: Sequence[Mapping[str, Any]],
    lookahead: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(table_dir / "causal_by_workload.csv", detailed, list(detailed[0]))
    write_csv(table_dir / "table1_causal_costs_by_regime.csv", regime, list(regime[0]))
    gap_rows = [
        {
            "capacity": row["capacity"],
            **{
                FAMILY_LABELS[family]: (
                    None
                    if row[f"{family}_gap_closed"] is None
                    else 100.0 * float(row[f"{family}_gap_closed"])
                )
                for family in FAMILY_ORDER
            },
        }
        for row in suite
    ]
    write_csv(table_dir / "table2_oracle_gap_closed.csv", gap_rows, list(gap_rows[0]))
    table3 = [
        {
            "capacity": row["capacity"],
            "baseline_cost": row["baseline_cost"],
            "predictor_cost": row["predictor_cost"],
            "oracle_cost": row["oracle_cost"],
            "baseline_improvement": row["baseline_improvement"],
            "gap_closed": row["oracle_gap_closed"],
            "gap_closed_ci_low": row["gap_closed_ci_low"],
            "gap_closed_ci_high": row["gap_closed_ci_high"],
            "residual_headroom": row["residual_headroom"],
            "residual_headroom_ci_low": row["residual_headroom_ci_low"],
            "residual_headroom_ci_high": row["residual_headroom_ci_high"],
        }
        for row in suite
    ]
    write_csv(table_dir / "table3_residual_headroom.csv", table3, list(table3[0]))
    write_csv(table_dir / "causal_sensitivity.csv", sensitivity, list(sensitivity[0]))
    write_csv(table_dir / "lookahead_curve.csv", lookahead, list(lookahead[0]))
    atomic_write_text(
        table_dir / "required_tables.md",
        _markdown_required_tables(regime, gap_rows, table3),
    )


def _markdown_required_tables(
    regime: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    table3: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# RACE Stage 1 required tables", "", "## Table 1 — Causal policy costs by workload family", ""]
    lines.append("| Regime | Capacity | Stage0 Best | Persistence | LastGate | GateEWMA | Markov1 | MarkovH | Hybrid | Oracle |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in regime:
        lines.append(
            "| {scope} | {capacity} | {baseline_cost:.0f} | {persistence_cost:.0f} | "
            "{last_gate_cost:.0f} | {gate_ewma_cost:.0f} | {markov_1_cost:.0f} | "
            "{markov_h_cost:.0f} | {markov_plus_ewma_cost:.0f} | {oracle_cost:.0f} |".format(**row)
        )
    lines.extend(["", "## Table 2 — Oracle gap closed (%)", ""])
    lines.append("| Capacity | Persistence | LastGate | GateEWMA | Markov1 | MarkovH | Hybrid |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in gap_rows:
        values = ["N/A" if row[label] is None else f"{float(row[label]):.2f}%" for label in FAMILY_LABELS.values()]
        lines.append(f"| {row['capacity']} | " + " | ".join(values) + " |")
    lines.extend(["", "## Table 3 — Residual headroom for calibration-selected predictor", ""])
    lines.append("| Capacity | Baseline | Predictor | Oracle | Baseline improvement | Gap closed (95% CI) | Residual headroom (95% CI) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in table3:
        gap = "N/A" if row["gap_closed"] is None else (
            f"{100*row['gap_closed']:.2f}% [{100*row['gap_closed_ci_low']:.2f}, {100*row['gap_closed_ci_high']:.2f}]"
        )
        residual = (
            f"{100*row['residual_headroom']:.2f}% "
            f"[{100*row['residual_headroom_ci_low']:.2f}, {100*row['residual_headroom_ci_high']:.2f}]"
        )
        lines.append(
            f"| {row['capacity']} | {row['baseline_cost']:.0f} | {row['predictor_cost']:.0f} | "
            f"{row['oracle_cost']:.0f} | {100*row['baseline_improvement']:.2f}% | {gap} | {residual} |"
        )
    return "\n".join(lines) + "\n"


def _write_figures(
    figure_dir: Path,
    regime_rows: Sequence[Mapping[str, Any]],
    suite_rows: Sequence[Mapping[str, Any]],
    sensitivity_rows: Sequence[Mapping[str, Any]],
    lookahead_rows: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    decision_capacities: Sequence[int],
    selected_predictor_id: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, regime in zip(axes.flat, REGIME_ORDER):
        rows = [row for row in regime_rows if row["scope"] == regime]
        x = [row["capacity"] for row in rows]
        axis.plot(x, [1.0] * len(x), marker="o", label="Stage0 best")
        axis.plot(x, [row["predictor_cost"] / row["baseline_cost"] for row in rows], marker="o", label="Selected causal")
        axis.plot(x, [row["oracle_cost"] / row["baseline_cost"] for row in rows], marker="o", label="Full oracle")
        axis.set_title(regime.capitalize())
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Cache capacity")
    axes[1, 1].set_xlabel("Cache capacity")
    axes[0, 0].set_ylabel("Cost / Stage0 best")
    axes[1, 0].set_ylabel("Cost / Stage0 best")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure1_normalized_transfer_cost")

    fig, axis = plt.subplots(figsize=(8, 5))
    family_ids = sorted({row["method_id"] for row in sensitivity_rows})
    major = [
        identifier
        for identifier in family_ids
        if identifier in {selected_predictor_id, "persistence", "last_gate", "markov_1"}
        or identifier.startswith("markov_h_h")
    ]
    for identifier in major:
        rows = [row for row in sensitivity_rows if row["method_id"] == identifier and row["capacity"] in decision_capacities]
        if not rows:
            continue
        axis.plot(
            [int(row["capacity"]) - 8 for row in rows],
            [100 * float(row["oracle_gap_closed"]) for row in rows],
            marker="o",
            label=identifier,
            linewidth=2.5 if identifier == selected_predictor_id else 1.0,
        )
    axis.axhline(50, color="black", linestyle="--", linewidth=1, label="50% threshold")
    axis.axhline(75, color="gray", linestyle=":", linewidth=1, label="75% threshold")
    axis.set_xlabel("Spare residency slots S = B - 8")
    axis.set_ylabel("Oracle gap closed (%)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure2_gap_closed_vs_spare")

    fig, axis = plt.subplots(figsize=(8, 5))
    for capacity in decision_capacities:
        rows = [row for row in lookahead_rows if int(row["capacity"]) == capacity]
        axis.plot(
            [row["horizon"] for row in rows],
            [100 * float(row["oracle_gap_closed"]) for row in rows],
            marker="o",
            label=f"capacity {capacity}",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32], labels=["1", "2", "4", "8", "16", "32"])
    axis.set_xlabel("Exact future lookahead H (same-layer events)")
    axis.set_ylabel("Full-oracle gap recovered (%)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure3_lookahead_curve")

    fig, axis = plt.subplots(figsize=(8, 5))
    for row in quality["joined_rows"]:
        x = 100 * float(row["average_precision"])
        y = 100 * float(row["mean_suite_gap_closed_capacities_12_32"])
        axis.scatter(x, y, s=28)
        axis.annotate(str(row["method_id"]), (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points")
    axis.set_xlabel("Next-event average precision (%)")
    axis.set_ylabel("Mean oracle gap closed, capacities 12–32 (%)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "figure4_prediction_quality_vs_residency")


def _save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _render_report(
    analysis: Mapping[str, Any],
    frozen_inputs: Any,
    frozen_config: Mapping[str, Any],
    table_dir: Path,
    figure_dir: Path,
) -> str:
    verdict = str(analysis["verdict"])
    suite = analysis["suite_results"]
    nondegenerate = [row for row in suite if int(row["capacity"]) != 8]
    gap_values = [100 * float(row["oracle_gap_closed"]) for row in nondegenerate]
    residual_values = [100 * float(row["residual_headroom"]) for row in nondegenerate]
    selected = str(analysis["selected_predictor_id"])
    lines = [
        verdict,
        "",
        "# RACE Stage 1: Simple Prediction Headroom Test",
        "",
        "## A. Executive verdict",
        "",
        (
            f"The globally calibration-selected causal policy was `{selected}`. Across the "
            f"frozen ten-workload suite it closed {min(gap_values):.2f}%–{max(gap_values):.2f}% "
            f"of the validated Stage 0 oracle gap at capacities 12–32, leaving "
            f"{min(residual_values):.2f}%–{max(residual_values):.2f}% of Stage 0 baseline "
            f"cost between the predictor and oracle. {analysis['decision']['reason']}"
        ),
        "",
        "## B. Frozen Stage 0 reference",
        "",
        f"- Verdict: `RACE_STAGE0_STRONG_GO`",
        f"- Source/base commit: `{frozen_inputs.preregistration['stage0_reference']['source_base_commit']}`",
        f"- Actual Stage 0 runtime commit: `{frozen_inputs.preregistration['stage0_reference']['actual_runtime_commit']}`",
        f"- Trace logical hash: `{frozen_inputs.trace.trace_hash}`",
        "- Cache model: 16 independent layer caches, 64 experts per layer, atomic top-8 requests, mandatory admission, and no prefetch.",
        "",
        "Validated Stage 0 oracle headroom (%):",
        "",
        "| Capacity | Stationary | Abrupt | Repeated | Mixed |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    regime_lookup = {
        (row["scope"], int(row["capacity"])): row for row in analysis["regime_results"]
    }
    for capacity in (8, 12, 16, 24, 32):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                capacity,
                *[
                    f"{100*regime_lookup[(regime, capacity)]['stage0_oracle_headroom']:.2f}%"
                    for regime in REGIME_ORDER
                ],
            )
        )
    lines.extend(
        [
            "",
            "## C. Predictor implementations",
            "",
            "All causal methods use one eviction mechanism: retain old candidates by prediction score, then LRU recency, then expert ID. Persistence uses the previous same-layer request; LastGate uses the last observed requested-expert gate weight; GateEWMA decays absent experts toward zero; Markov1 and MarkovH use fixed calibration-only binary transition probabilities; and Hybrid combines selected MarkovH with request-indicator EWMA. No method prefetches.",
            "",
            "## D. Calibration procedure",
            "",
            f"The transition models and all selections used only the 80 frozen calibration sequences (`{frozen_inputs.calibration.hash}`). Evaluation uses the disjoint 320-sequence split. Gate alpha, Markov horizon, hybrid beta, and then one verdict predictor were each selected once by calibration misses summed over capacities 12–32; nothing was reselected by evaluation workload or capacity.",
            "",
            f"Selected family configurations: `{frozen_config['selected_by_family']}`.",
            "",
            "## E. Main results",
            "",
            f"The complete required tables are in `{table_dir.name}/required_tables.md`; exact per-workload values are in `{table_dir.name}/causal_by_workload.csv`. Table 3 below is the decision-driving frozen-suite aggregate.",
            "",
            "| Capacity | Baseline cost | Predictor cost | Oracle cost | Improvement | Gap closed (95% CI) | Residual (95% CI) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in suite:
        if row["oracle_gap_closed"] is None:
            gap = "N/A — zero Stage 0 oracle gap"
        else:
            gap = f"{100*row['oracle_gap_closed']:.2f}% [{100*row['gap_closed_ci_low']:.2f}, {100*row['gap_closed_ci_high']:.2f}]"
        residual = f"{100*row['residual_headroom']:.2f}% [{100*row['residual_headroom_ci_low']:.2f}, {100*row['residual_headroom_ci_high']:.2f}]"
        lines.append(
            f"| {row['capacity']} | {row['baseline_cost']:.0f} | {row['predictor_cost']:.0f} | "
            f"{row['oracle_cost']:.0f} | {100*row['baseline_improvement']:.2f}% | {gap} | {residual} |"
        )
    lines.extend(
        [
            "",
            "Capacity 8 is reported only as a degenerate sanity condition and is excluded from every decision count.",
            "",
            "## F. Oracle-gap closure",
            "",
            f"The frozen decision counted gap closure below 50% at capacities {analysis['decision']['strong_condition_a_capacities']} and residual headroom of at least 10% at capacities {analysis['decision']['strong_condition_b_capacities']}. The NO-GO gap-closure trigger held at {analysis['decision']['no_go_gap_closed_capacities']}; the low-residual trigger held at {analysis['decision']['no_go_low_residual_capacities']}.",
            "",
            "Figure 2 plots gap closure against spare residency `S=B-8`, including the 50% and 75% thresholds.",
            "",
            "## G. Lookahead analysis",
            "",
        ]
    )
    for capacity in (12, 16, 24, 32):
        rows = [row for row in analysis["lookahead_results"] if int(row["capacity"]) == capacity]
        lines.append(
            f"- Capacity {capacity}: "
            + ", ".join(
                f"H={row['horizon']} → {100*row['oracle_gap_closed']:.2f}%" for row in rows
            )
            + "."
        )
    lines.extend(
        [
            "",
            "These policies are non-causal diagnostics. Each first action is exact for its visible finite horizon under the frozen unit-cost semantics; the full perfect-score rule matches the Stage 0 exact oracle cost on every full condition and on enumerated/random tiny traces.",
            "",
            "## H. Predictor-quality analysis",
            "",
            f"Prediction quality was measured on the frozen mixed workload. The descriptive Spearman association between next-event average precision and mean residency gap closure was {analysis['prediction_quality']['spearman_average_precision_vs_gap_closed']:.3f} (p={analysis['prediction_quality']['p_value_descriptive']:.3g}). This configuration-level association is diagnostic, not an independent-sample hypothesis test.",
            "",
            "## I. Residual opportunity",
            "",
            "The perfect-score policy uses the same simple retention mechanism and reproduces the full oracle cost, so under this equal-cost Stage 0 model the decision mechanism itself contributes no measurable residual when supplied exact next-use scores. The selected causal-to-lookahead differences therefore diagnose prediction/horizon error rather than speculative transfer timing. Residual size grows or shrinks with spare capacity as shown in Figure 2 and the exact CSVs.",
            "",
            "## J. Limitations",
            "",
            "- Results are trace simulations of expert residency, misses, admissions, and transfers; no end-to-end latency improvement or hardware speedup is claimed.",
            "- Causal predictors are deliberately simple and fixed; this stage does not test learned neural forecasting.",
            "- The full oracle and limited-lookahead policies are non-causal diagnostic comparators, not deployable methods.",
            "- Bootstrap intervals reweight saved per-sequence contributions conditional on the frozen workload ordering; stateful cache trajectories are not regenerated under reordered bootstrap workloads.",
            "- Workload-suite aggregation sums the ten preregistered paths, so source prompts recurring across regimes receive the same clustered bootstrap multiplicity but contribute once per frozen occurrence.",
            "- The absent raw pilot trace remains a Stage 0 archival limitation; the validated full trace and replay artifacts used here remain intact.",
            "",
            "## K. Next action",
            "",
            _next_action(verdict),
            "",
            "## Reproducibility hashes",
            "",
            f"- Stage 1 preregistration: `{analysis['preregistration_hash']}`",
            f"- Stage 1 frozen config file: `{analysis['frozen_config_file_sha256']}`",
            f"- Transition model: `{frozen_config['transition_model_hash']}`",
            f"- Evaluation manifest file: `{analysis['evaluation_manifest_sha256']}`",
            "",
            "## Core answer",
            "",
            f"After giving simple causal routing prediction a strong and fair chance, the selected `{selected}` policy leaves {min(residual_values):.2f}%–{max(residual_values):.2f}% of Stage 0 simple-baseline cost as unexploited oracle headroom across capacities 12–32.",
        ]
    )
    return "\n".join(lines) + "\n"


def _next_action(verdict: str) -> str:
    if verdict == "RACE_STAGE1_STRONG_GO":
        return "Proceed to RACE algorithm design."
    if verdict == "RACE_STAGE1_WEAK_GO":
        return "Inspect residual gap before implementing full RACE."
    return "Do not build RACE as currently formulated."


def _analysis_audit(
    analysis: Mapping[str, Any], report: Path, tables: Path, figures: Path
) -> dict[str, Any]:
    required = [
        report,
        report.with_suffix(".sha256"),
        tables / "required_tables.md",
        tables / "causal_by_workload.csv",
        tables / "table1_causal_costs_by_regime.csv",
        tables / "table2_oracle_gap_closed.csv",
        tables / "table3_residual_headroom.csv",
        figures / "figure1_normalized_transfer_cost.png",
        figures / "figure2_gap_closed_vs_spare.png",
        figures / "figure3_lookahead_curve.png",
        figures / "figure4_prediction_quality_vs_residency.png",
    ]
    checks = {
        "report_begins_with_exact_verdict": report.read_text(encoding="utf-8").startswith(
            str(analysis["verdict"]) + "\n"
        ),
        "all_required_outputs_exist": all(path.exists() and path.stat().st_size > 0 for path in required),
        "five_capacities_present": len(analysis["suite_results"]) == 5,
        "capacity8_gap_is_null": analysis["suite_results"][0]["oracle_gap_closed"] is None,
        "four_decision_capacities": len(
            [row for row in analysis["suite_results"] if row["capacity"] != 8]
        )
        == 4,
    }
    return {
        "schema_version": "race_stage1_analysis_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "verdict": analysis["verdict"],
    }


def _quantile(values: np.ndarray, probability: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else float("nan")


def _standardized_effect(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    deviation = float(np.std(values, ddof=1))
    if deviation == 0:
        return float("inf") if float(np.mean(values)) > 0 else 0.0
    return float(np.mean(values) / deviation)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"race-stage1-bootstrap-v1\0{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
