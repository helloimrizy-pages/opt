"""Stage 2 analysis: frozen decision rule, paired bootstrap, tables and figures."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

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

from .advisers import MARKOV_HORIZONS, pool_names
from .calibration import load_and_verify_stage2_frozen
from .frozen import load_and_verify_stage2_inputs
from .metrics import comparison_metrics
from .report import render_report

REGIME_ORDER = ("stationary", "abrupt", "repeated", "mixed")
PRIMARY_LABELS = ("RACE_UNIFORM", "RACE_STATIC", "RACE_ONLINE", "RACE_COST")
TRAJECTORY_WORKLOAD = "mixed_interleaved"


def analyze_and_report(
    repository_root: Path,
    preregistration_path: Path,
    frozen_config_path: Path,
    evaluation_dir: Path,
    stage2_root: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    inputs = load_and_verify_stage2_inputs(repository_root, preregistration_path)
    frozen = load_and_verify_stage2_frozen(frozen_config_path)
    manifest = read_json(evaluation_dir / "evaluation_manifest.json")
    if manifest["frozen_config_file_sha256"] != frozen["file_sha256"]:
        raise ValueError("Evaluation and Stage 2 frozen config hashes differ")
    if not read_json(evaluation_dir / "sanity_checks.json")["passed"]:
        raise ValueError("Cannot analyze an evaluation that failed sanity checks")

    rows = read_jsonl(evaluation_dir / "results.jsonl")
    diagnostics = read_jsonl(evaluation_dir / "diagnostics.jsonl")
    capacities = tuple(int(value) for value in frozen["cache_capacities"])
    decision_capacities = tuple(int(value) for value in frozen["decision_capacities"])
    variant_labels = [item["label"] for item in frozen["variants"]]
    label_by_variant_id = {item["variant_id"]: item["label"] for item in frozen["variants"]}
    primary_label = next(item["label"] for item in frozen["variants"] if item["is_primary"])
    primary_variant_id = frozen["primary_variant_id"]

    cost = {
        (str(row["workload"]), int(row["capacity"]), label_by_variant_id[str(row["variant_id"])]): int(row["misses"])
        for row in rows
    }
    condition_ids = {
        (str(row["workload"]), int(row["capacity"]), label_by_variant_id[str(row["variant_id"])]): str(row["condition_id"])
        for row in rows
    }
    workloads = list(inputs.workloads)
    workloads_by_regime = {
        regime: tuple(item.name for item in workloads if item.regime == regime)
        for regime in REGIME_ORDER
    }
    suite = tuple(item.name for item in workloads)

    def aggregate(names: Sequence[str], capacity: int, label: str) -> int:
        return int(sum(cost[(name, capacity, label)] for name in names))

    def stage0_simple(names: Sequence[str], capacity: int) -> float:
        return float(
            sum(
                float(inputs.stage0_references[(name, capacity)]["simple"]["misses"])
                for name in names
            )
        )

    def stage0_oracle(names: Sequence[str], capacity: int) -> float:
        return float(
            sum(
                float(inputs.stage0_references[(name, capacity)]["oracle"]["misses"])
                for name in names
            )
        )

    def stage1(names: Sequence[str], capacity: int) -> float:
        return float(sum(float(inputs.stage1_costs[(name, capacity)]) for name in names))

    scopes: list[tuple[str, tuple[str, ...]]] = [
        ("all_frozen_workloads", suite),
        *((regime, workloads_by_regime[regime]) for regime in REGIME_ORDER),
    ]
    scope_rows: list[dict[str, Any]] = []
    for scope, names in scopes:
        for capacity in capacities:
            simple = stage0_simple(names, capacity)
            oracle = stage0_oracle(names, capacity)
            reference = stage1(names, capacity)
            entry: dict[str, Any] = {
                "scope": scope,
                "capacity": capacity,
                "spare_residency": capacity - int(inputs.trace.top_k),
                "workloads": len(names),
                "stage0_simple_cost": simple,
                "stage1_cost": reference,
                "oracle_cost": oracle,
                "requests": int(
                    sum(
                        int(row["requests"])
                        for row in rows
                        if str(row["workload"]) in names
                        and int(row["capacity"]) == capacity
                        and label_by_variant_id[str(row["variant_id"])] == primary_label
                    )
                ),
            }
            for label in variant_labels:
                value = float(aggregate(names, capacity, label))
                entry[f"{label}_cost"] = value
                entry[f"{label}_metrics"] = comparison_metrics(
                    simple=simple, stage1=reference, race=value, oracle=oracle
                )
            entry.update(entry[f"{primary_label}_metrics"])
            scope_rows.append(entry)

    workload_rows: list[dict[str, Any]] = []
    for workload in workloads:
        for capacity in capacities:
            simple = stage0_simple([workload.name], capacity)
            oracle = stage0_oracle([workload.name], capacity)
            reference = stage1([workload.name], capacity)
            entry = {
                "workload": workload.name,
                "regime": workload.regime,
                "capacity": capacity,
                "stage0_simple_policy": inputs.stage0_references[(workload.name, capacity)][
                    "simple"
                ]["policy"],
                "stage0_simple_cost": simple,
                "stage1_cost": reference,
                "oracle_cost": oracle,
            }
            for label in variant_labels:
                entry[f"{label}_cost"] = float(cost[(workload.name, capacity, label)])
            entry.update(
                comparison_metrics(
                    simple=simple,
                    stage1=reference,
                    race=float(cost[(workload.name, capacity, primary_label)]),
                    oracle=oracle,
                )
            )
            workload_rows.append(entry)

    bootstrap_rows = _bootstrap(
        repository_root,
        inputs,
        frozen,
        evaluation_dir,
        scopes,
        capacities,
        condition_ids,
        primary_label,
    )
    bootstrap_by_key = {
        (row["scope"], int(row["capacity"])): row for row in bootstrap_rows
    }
    for row in scope_rows:
        row.update(bootstrap_by_key[(row["scope"], int(row["capacity"]))])

    suite_rows = [row for row in scope_rows if row["scope"] == "all_frozen_workloads"]
    regression_rows = [
        {
            "workload": row["workload"],
            "regime": row["regime"],
            "capacity": row["capacity"],
            "stage1_cost": row["stage1_cost"],
            "race_cost": row[f"{primary_label}_cost"],
            "regression_ratio": row["regression_ratio"],
            "flagged": bool(row["regression_ratio"] > 1.03),
        }
        for row in workload_rows
    ]
    decision = _stage2_decision(
        suite_rows, frozen["success_criteria"], decision_capacities, primary_label, regression_rows
    )
    ablation = _ablation_table(scope_rows, decision_capacities, variant_labels)
    ranking = _ranking_table(diagnostics, label_by_variant_id, capacities)
    horizons = _horizon_table(diagnostics, label_by_variant_id, capacities, frozen)
    weights = _weight_table(diagnostics, label_by_variant_id, capacities)
    regret = _regret_table(diagnostics, label_by_variant_id, capacities)
    delayed = _delayed_feedback_table(diagnostics, label_by_variant_id, capacities)
    leverage = _ranking_vs_cost(ranking, scope_rows, decision_capacities, variant_labels)
    trajectory_layers = _trajectory_layers(diagnostics, label_by_variant_id, primary_label)

    analysis = {
        "schema_version": "race_stage2_analysis_v1",
        "created_at_utc": utc_now(),
        "verdict": decision["verdict"],
        "decision": decision,
        "primary_variant_label": primary_label,
        "primary_variant_id": primary_variant_id,
        "primary_variant_parameters": {
            "eta": frozen["selected_eta"],
            "initialization": frozen["selected_initialization"],
            "loss": frozen["primary_loss"],
            "weight_scope": frozen["weight_scope_primary"],
            "H_max": frozen["H_max"],
            "adviser_pool": frozen["adviser_pools"]["primary"]["order"],
        },
        "variant_labels": variant_labels,
        "capacities": list(capacities),
        "decision_capacities": list(decision_capacities),
        "suite_results": suite_rows,
        "scope_results": scope_rows,
        "workload_results": workload_rows,
        "regression_rows": regression_rows,
        "ablation": ablation,
        "ranking_diagnostics": ranking,
        "horizon_weights": horizons,
        "weight_adaptation": weights,
        "regret_accounting": regret,
        "delayed_feedback": delayed,
        "ranking_vs_cost": leverage,
        "trajectory_layers": trajectory_layers,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_interpretation": frozen["statistics"]["conditionality"],
        "trace_hash": inputs.trace.trace_hash,
        "preregistration_hash": inputs.preregistration_hash,
        "frozen_config_file_sha256": frozen["file_sha256"],
        "evaluation_manifest_sha256": sha256_file(evaluation_dir / "evaluation_manifest.json"),
        "stage1_winner_method_id": frozen["stage1_winner_method_id"],
        "stage0_archive_manifest_sha256": frozen["stage0_archive_manifest_sha256"],
        "stage1_archive_manifest_sha256": frozen["stage1_archive_manifest_sha256"],
    }

    report_dir = stage2_root / "reports"
    table_dir = stage2_root / "tables"
    figure_dir = stage2_root / "figures"
    for directory in (report_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / "analysis.json", analysis)
    _write_tables(table_dir, analysis, frozen)
    _write_figures(figure_dir, analysis, evaluation_dir, frozen)
    report = render_report(analysis, inputs, frozen)
    report_path = report_dir / "race_stage2_report.md"
    atomic_write_text(report_path, report)
    atomic_write_text(
        report_path.with_suffix(".sha256"),
        f"{sha256_file(report_path)}  {report_path.name}\n",
    )
    audit = _analysis_audit(analysis, report_path, table_dir, figure_dir)
    atomic_write_json(report_dir / "analysis_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Stage 2 analysis audit failed: {audit['checks']}")
    return analysis


def _bootstrap(
    repository_root: Path,
    inputs: Any,
    frozen: Mapping[str, Any],
    evaluation_dir: Path,
    scopes: Sequence[tuple[str, Sequence[str]]],
    capacities: Sequence[int],
    condition_ids: Mapping[tuple[str, int, str], str],
    primary_label: str,
) -> list[dict[str, Any]]:
    replicates = int(frozen["statistics"]["bootstrap_replicates"])
    seed = int(frozen["statistics"]["bootstrap_seed"])
    prereg = inputs.preregistration
    stage0_needed = {
        str(item[role]["condition_id"])
        for item in inputs.stage0_references.values()
        for role in ("simple", "oracle")
    }
    stage0_sequences = _sequence_lookup(
        _filtered_rows(
            repository_root / prereg["stage0_reference"]["per_sequence_result_path"],
            stage0_needed,
        )
    )
    stage1_needed = set(inputs.stage1_sequence_condition_ids.values())
    stage1_sequences = _sequence_lookup(
        _filtered_rows(
            repository_root / prereg["stage1_reference"]["per_sequence_results_path"],
            stage1_needed,
        )
    )
    stage2_needed = {
        condition_ids[key] for key in condition_ids if key[2] == primary_label
    }
    stage2_sequences = _sequence_lookup(
        _filtered_rows(evaluation_dir / "per_sequence_results.jsonl", stage2_needed)
    )

    output: list[dict[str, Any]] = []
    for scope, names in scopes:
        for capacity in capacities:
            components = []
            domain_by_sequence: dict[int, str] = {}
            for name in names:
                reference = inputs.stage0_references[(name, capacity)]
                simple_rows = stage0_sequences[str(reference["simple"]["condition_id"])]
                oracle_rows = stage0_sequences[str(reference["oracle"]["condition_id"])]
                stage1_rows = stage1_sequences[
                    str(inputs.stage1_sequence_condition_ids[(name, capacity)])
                ]
                race_rows = stage2_sequences[str(condition_ids[(name, capacity, primary_label)])]
                identities = [
                    [(int(row["source_sequence_id"]), str(row["domain"])) for row in values]
                    for values in (simple_rows, oracle_rows, stage1_rows, race_rows)
                ]
                if any(item != identities[0] for item in identities[1:]):
                    raise ValueError(
                        f"Paired per-sequence rows are misaligned for {name} at capacity {capacity}"
                    )
                for sequence, domain in identities[0]:
                    previous = domain_by_sequence.setdefault(sequence, domain)
                    if previous != domain:
                        raise ValueError("A source sequence changes domain across workloads")
                components.append(
                    (
                        identities[0],
                        np.asarray([row["misses"] for row in simple_rows], dtype=np.float64),
                        np.asarray([row["misses"] for row in oracle_rows], dtype=np.float64),
                        np.asarray([row["misses"] for row in stage1_rows], dtype=np.float64),
                        np.asarray([row["misses"] for row in race_rows], dtype=np.float64),
                    )
                )
            multiplicities = _cluster_multiplicities(
                domain_by_sequence, replicates, _derived_seed(seed, f"{scope}:{capacity}")
            )
            totals = np.zeros((4, replicates))
            differences: list[float] = []
            for identities, simple, oracle, reference, race in components:
                weights = np.stack(
                    [multiplicities[sequence] for sequence, _domain in identities], axis=1
                )
                totals[0] += (weights * simple[None, :]).sum(axis=1)
                totals[1] += (weights * oracle[None, :]).sum(axis=1)
                totals[2] += (weights * reference[None, :]).sum(axis=1)
                totals[3] += (weights * race[None, :]).sum(axis=1)
                differences.extend(reference - race)
            simple_boot, oracle_boot, stage1_boot, race_boot = totals
            improvement = _safe_divide(stage1_boot - race_boot, stage1_boot)
            gap_closed = _safe_divide(simple_boot - race_boot, simple_boot - oracle_boot)
            residual_recovered = _safe_divide(stage1_boot - race_boot, stage1_boot - oracle_boot)
            stage1_gap = _safe_divide(simple_boot - stage1_boot, simple_boot - oracle_boot)
            paired = np.asarray(differences, dtype=np.float64)
            output.append(
                {
                    "scope": scope,
                    "capacity": capacity,
                    "race_improvement_ci_low": _quantile(improvement, 0.025),
                    "race_improvement_ci_high": _quantile(improvement, 0.975),
                    "original_oracle_gap_closed_ci_low": _quantile(gap_closed, 0.025),
                    "original_oracle_gap_closed_ci_high": _quantile(gap_closed, 0.975),
                    "stage1_residual_recovered_ci_low": _quantile(residual_recovered, 0.025),
                    "stage1_residual_recovered_ci_high": _quantile(residual_recovered, 0.975),
                    "stage1_original_oracle_gap_closed_ci_low": _quantile(stage1_gap, 0.025),
                    "stage1_original_oracle_gap_closed_ci_high": _quantile(stage1_gap, 0.975),
                    "mean_paired_sequence_gain_vs_stage1": float(paired.mean()),
                    "median_paired_sequence_gain_vs_stage1": float(np.median(paired)),
                    "paired_standardized_effect": _standardized_effect(paired),
                    "paired_sequences_improved": int((paired > 0).sum()),
                    "paired_sequences_worsened": int((paired < 0).sum()),
                    "paired_units": int(paired.size),
                    "bootstrap_replicates": replicates,
                    "unique_source_sequences": len(domain_by_sequence),
                }
            )
    return output


def _stage2_decision(
    suite_rows: Sequence[Mapping[str, Any]],
    criteria: Mapping[str, Any],
    decision_capacities: Sequence[int],
    primary_label: str,
    regression_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [row for row in suite_rows if int(row["capacity"]) in set(decision_capacities)]
    strong = criteria["strong_success"]
    very = criteria["very_strong_success"]

    def capacities_meeting(threshold: float, metric: str, require_ci: bool) -> list[int]:
        found = []
        for row in rows:
            value = row[metric]
            if value is None or float(value) < float(threshold):
                continue
            if require_ci and not (float(row["race_improvement_ci_low"]) > 0.0):
                continue
            found.append(int(row["capacity"]))
        return found

    strong_a = capacities_meeting(
        strong["condition_a"]["threshold"], "race_improvement_over_stage1", True
    )
    strong_b = capacities_meeting(
        strong["condition_b"]["threshold"], "original_oracle_gap_closed", False
    )
    very_a = capacities_meeting(
        very["condition_a"]["threshold"], "race_improvement_over_stage1", False
    )
    very_b = capacities_meeting(
        very["condition_b"]["threshold"], "original_oracle_gap_closed", False
    )
    weak_capacities = capacities_meeting(0.05, "race_improvement_over_stage1", False)
    below_five = [
        int(row["capacity"])
        for row in rows
        if float(row["race_improvement_over_stage1"]) < 0.05
    ]
    suite_regressions = [
        int(row["capacity"]) for row in rows if float(row["regression_ratio"]) > 1.03
    ]
    flagged = [row for row in regression_rows if row["flagged"]]
    online_beats_simpler = [
        int(row["capacity"])
        for row in rows
        if float(row[f"{primary_label}_cost"]) < float(row["RACE_STATIC_cost"])
        and float(row[f"{primary_label}_cost"]) < float(row["RACE_UNIFORM_cost"])
    ]

    required = int(strong["condition_a"]["required_capacities"])
    no_go = (
        len(below_five) >= 2
        or bool(suite_regressions)
        or len(online_beats_simpler) < required
    )
    strong_pass = len(strong_a) >= required and len(strong_b) >= int(
        strong["condition_b"]["required_capacities"]
    )
    very_pass = len(very_a) >= int(very["condition_a"]["required_capacities"]) and len(
        very_b
    ) >= int(very["condition_b"]["required_capacities"])
    weak_pass = len(weak_capacities) >= required

    triggers = []
    if len(below_five) >= 2:
        triggers.append(
            "improvement over the frozen Stage 1 winner stayed below 5% at capacities "
            f"{below_five}"
        )
    if suite_regressions:
        triggers.append(
            f"the frozen-suite cost exceeded 1.03x the Stage 1 winner at capacities {suite_regressions}"
        )
    if len(online_beats_simpler) < required:
        triggers.append(
            "the primary online variant beat both RACE_STATIC and RACE_UNIFORM at only "
            f"{len(online_beats_simpler)} of {len(rows)} non-degenerate capacities"
        )
    if no_go:
        verdict = "RACE_STAGE2_NO_GO"
        reason = (
            "The preregistered NO-GO rule fired because "
            + "; ".join(triggers)
            + "."
        )
    elif very_pass:
        verdict = "RACE_STAGE2_VERY_STRONG_SUCCESS"
        reason = (
            f"RACE improved on the frozen Stage 1 winner by at least 15% at capacities {very_a} "
            f"and closed at least 50% of the original Stage 0 oracle gap at {very_b}."
        )
    elif strong_pass:
        verdict = "RACE_STAGE2_STRONG_SUCCESS"
        reason = (
            f"RACE improved on the frozen Stage 1 winner by at least 10% with a positive paired "
            f"bootstrap interval at capacities {strong_a} and closed at least 30% of the original "
            f"Stage 0 oracle gap at {strong_b}."
        )
    elif weak_pass:
        verdict = "RACE_STAGE2_WEAK"
        reason = (
            f"RACE produced a consistent improvement of at least 5% over the frozen Stage 1 winner "
            f"at capacities {weak_capacities} but missed the preregistered strong thresholds."
        )
    else:
        verdict = "RACE_STAGE2_NO_GO"
        reason = (
            "RACE did not reach a consistent 5% improvement over the frozen Stage 1 winner at "
            "three of the four non-degenerate capacities."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "primary_variant_label": primary_label,
        "strong_condition_a_capacities": strong_a,
        "strong_condition_b_capacities": strong_b,
        "very_strong_condition_a_capacities": very_a,
        "very_strong_condition_b_capacities": very_b,
        "weak_capacities": weak_capacities,
        "capacities_below_five_percent": below_five,
        "suite_regression_capacities": suite_regressions,
        "flagged_workload_regressions": len(flagged),
        "online_beats_static_and_uniform_capacities": online_beats_simpler,
        "no_go_triggers": triggers,
        "criteria": dict(criteria),
    }


def _ablation_table(
    scope_rows: Sequence[Mapping[str, Any]],
    decision_capacities: Sequence[int],
    variant_labels: Sequence[str],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in scope_rows
        if row["scope"] == "all_frozen_workloads" and int(row["capacity"]) in set(decision_capacities)
    ]
    comparisons = [
        ("A_multi_horizon", "stage1_cost", "RACE_UNIFORM_cost", "Stage 1 winner -> RACE_UNIFORM"),
        ("B_static_weights", "RACE_UNIFORM_cost", "RACE_STATIC_cost", "RACE_UNIFORM -> RACE_STATIC"),
        ("C_online_adaptation", "RACE_STATIC_cost", "RACE_ONLINE_cost", "RACE_STATIC -> RACE_ONLINE"),
        ("D_cost_sensitivity", "RACE_ONLINE_cost", "RACE_COST_cost", "RACE_ONLINE -> RACE_COST"),
    ]
    if "RACE_ONLINE_EXTENDED" in variant_labels:
        comparisons.extend(
            [
                (
                    "E_adviser_diversity_uniform",
                    "RACE_UNIFORM_cost",
                    "RACE_UNIFORM_EXTENDED_cost",
                    "RACE_UNIFORM -> RACE_UNIFORM_EXTENDED",
                ),
                (
                    "F_adviser_diversity_online",
                    "RACE_ONLINE_cost",
                    "RACE_ONLINE_EXTENDED_cost",
                    "RACE_ONLINE -> RACE_ONLINE_EXTENDED",
                ),
                (
                    "G_extended_vs_stage1",
                    "stage1_cost",
                    "RACE_ONLINE_EXTENDED_cost",
                    "Stage 1 winner -> RACE_ONLINE_EXTENDED",
                ),
            ]
        )
    if "RACE_STATIC_PERLAYER" in variant_labels:
        comparisons.append(
            (
                "H_per_layer_static",
                "RACE_STATIC_cost",
                "RACE_STATIC_PERLAYER_cost",
                "RACE_STATIC -> RACE_STATIC_PERLAYER",
            )
        )
    if "RACE_ONLINE_GLOBAL" in variant_labels:
        comparisons.append(
            (
                "I_weight_scope",
                "RACE_ONLINE_cost",
                "RACE_ONLINE_GLOBAL_cost",
                "RACE_ONLINE (per layer) -> RACE_ONLINE_GLOBAL",
            )
        )
    output = []
    for key, before, after, description in comparisons:
        for row in rows:
            left = float(row[before])
            right = float(row[after])
            output.append(
                {
                    "question": key,
                    "comparison": description,
                    "capacity": int(row["capacity"]),
                    "before_cost": left,
                    "after_cost": right,
                    "relative_change": (left - right) / left,
                }
            )
    return output


def _ranking_table(
    diagnostics: Sequence[Mapping[str, Any]],
    label_by_variant_id: Mapping[str, str],
    capacities: Sequence[int],
) -> list[dict[str, Any]]:
    fields = (
        "eviction_events",
        "events_without_eviction",
        "comparable_pairs_capped",
        "comparable_pairs_true",
        "concordant_capped",
        "discordant_capped",
        "tied_capped",
        "concordant_true",
        "discordant_true",
        "tied_true",
        "oracle_consistent_events",
        "oracle_optimal_events",
    )
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(fields, 0.0)
    )
    for row in diagnostics:
        ranking = row.get("ranking") or {}
        if not ranking:
            continue
        key = (label_by_variant_id[str(row["variant_id"])], int(row["capacity"]))
        bucket = grouped[key]
        for name in fields:
            bucket[name] += float(ranking[name])

    def ratio(numerator: float, denominator: float) -> float | None:
        return float(numerator / denominator) if denominator else None

    output = []
    for (label, capacity), bucket in sorted(grouped.items()):
        events = bucket["eviction_events"]
        output.append(
            {
                "variant": label,
                "capacity": capacity,
                "eviction_events": int(events),
                "events_without_eviction": int(bucket["events_without_eviction"]),
                "comparable_pairs_capped": int(bucket["comparable_pairs_capped"]),
                "comparable_pairs_true": int(bucket["comparable_pairs_true"]),
                "pairwise_ordering_accuracy_capped": ratio(
                    bucket["concordant_capped"] + 0.5 * bucket["tied_capped"],
                    bucket["comparable_pairs_capped"],
                ),
                "pairwise_ordering_accuracy_true": ratio(
                    bucket["concordant_true"] + 0.5 * bucket["tied_true"],
                    bucket["comparable_pairs_true"],
                ),
                "pairwise_concordance_capped": ratio(
                    bucket["concordant_capped"] - bucket["discordant_capped"],
                    bucket["concordant_capped"] + bucket["discordant_capped"],
                ),
                "pairwise_concordance_true": ratio(
                    bucket["concordant_true"] - bucket["discordant_true"],
                    bucket["concordant_true"] + bucket["discordant_true"],
                ),
                "oracle_consistent_eviction_rate": ratio(
                    bucket["oracle_consistent_events"], events
                ),
                "oracle_optimal_eviction_rate": ratio(
                    bucket["oracle_optimal_events"], events
                ),
            }
        )
    return output


def _horizon_table(
    diagnostics: Sequence[Mapping[str, Any]],
    label_by_variant_id: Mapping[str, str],
    capacities: Sequence[int],
    frozen: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, float]]] = defaultdict(list)
    for row in diagnostics:
        weights = row.get("adviser_mean_weights") or {}
        if not weights:
            continue
        grouped[(label_by_variant_id[str(row["variant_id"])], int(row["capacity"]))].append(weights)
    output = []
    for (label, capacity), entries in sorted(grouped.items()):
        names = sorted({name for entry in entries for name in entry})
        mean = {
            name: float(np.mean([entry.get(name, 0.0) for entry in entries]))
            for name in names
        }
        markov = {
            f"H{horizon}": mean.get(f"MARKOV_H{horizon}", 0.0)
            for horizon in MARKOV_HORIZONS
        }
        total = sum(markov.values())
        effective_horizon = (
            sum(horizon * markov[f"H{horizon}"] for horizon in MARKOV_HORIZONS) / total
            if total > 0
            else None
        )
        output.append(
            {
                "variant": label,
                "capacity": capacity,
                "spare_residency": capacity - 8,
                **{f"mean_weight_{name}": value for name, value in mean.items()},
                **{f"markov_{key}": value for key, value in markov.items()},
                "markov_mass": total,
                "weighted_mean_markov_horizon": effective_horizon,
            }
        )
    return output


def _weight_table(
    diagnostics: Sequence[Mapping[str, Any]],
    label_by_variant_id: Mapping[str, str],
    capacities: Sequence[int],
) -> list[dict[str, Any]]:
    output = []
    for row in diagnostics:
        label = label_by_variant_id[str(row["variant_id"])]
        for stream in row.get("weights_by_stream", []):
            output.append(
                {
                    "variant": label,
                    "workload": row["workload"],
                    "regime": row["regime"],
                    "capacity": int(row["capacity"]),
                    "layer": int(stream["layer"]),
                    "decisions": int(stream["decisions"]),
                    "dominant_adviser": stream["dominant_adviser"],
                    "dominant_adviser_mean_weight": stream["dominant_adviser_mean_weight"],
                    "start_weights": stream["start_weights"],
                    "end_weights": stream["end_weights"],
                    "mean_weights": stream["mean_weights"],
                    "weight_variance": stream["weight_variance"],
                    "end_entropy_nats": stream["end_entropy_nats"],
                    "end_effective_advisers": stream["end_effective_advisers"],
                    "mean_entropy_nats": stream["mean_entropy_nats"],
                    "mean_effective_advisers": stream["mean_effective_advisers"],
                }
            )
    return output


def _regret_table(
    diagnostics: Sequence[Mapping[str, Any]],
    label_by_variant_id: Mapping[str, str],
    capacities: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in diagnostics:
        learning = row.get("learning") or {}
        if not learning.get("adviser_order"):
            continue
        key = (label_by_variant_id[str(row["variant_id"])], int(row["capacity"]))
        bucket = grouped.setdefault(
            key,
            {
                "adviser_order": learning["adviser_order"],
                "rank": np.zeros(len(learning["adviser_order"])),
                "cost": np.zeros(len(learning["adviser_order"])),
                "mixture_rank": 0.0,
                "mixture_cost": 0.0,
                "examples": 0,
            },
        )
        bucket["rank"] += np.asarray(learning["cumulative_adviser_rank_loss"], dtype=np.float64)
        bucket["cost"] += np.asarray(learning["cumulative_adviser_cost_loss"], dtype=np.float64)
        bucket["mixture_rank"] += float(learning["cumulative_mixture_rank_loss"])
        bucket["mixture_cost"] += float(learning["cumulative_mixture_cost_loss"])
        bucket["examples"] += int(learning["examples_resolved"])
    output = []
    for (label, capacity), bucket in sorted(grouped.items()):
        names = bucket["adviser_order"]
        rank = bucket["rank"]
        costs = bucket["cost"]
        examples = max(bucket["examples"], 1)
        output.append(
            {
                "variant": label,
                "capacity": capacity,
                "resolved_examples": bucket["examples"],
                "cumulative_mixture_rank_loss": bucket["mixture_rank"],
                "best_fixed_adviser_rank_loss": float(rank.min()),
                "best_fixed_adviser_by_rank_loss": names[int(np.argmin(rank))],
                "empirical_rank_regret": float(bucket["mixture_rank"] - rank.min()),
                "empirical_rank_regret_per_example": float(
                    (bucket["mixture_rank"] - rank.min()) / examples
                ),
                "cumulative_mixture_cost_loss": bucket["mixture_cost"],
                "best_fixed_adviser_cost_loss": float(costs.min()),
                "best_fixed_adviser_by_cost_loss": names[int(np.argmin(costs))],
                "empirical_cost_regret": float(bucket["mixture_cost"] - costs.min()),
                "mean_mixture_rank_loss": float(bucket["mixture_rank"] / examples),
                "mean_adviser_rank_loss": (rank / examples).tolist(),
                "adviser_order": list(names),
            }
        )
    return output


def _delayed_feedback_table(
    diagnostics: Sequence[Mapping[str, Any]],
    label_by_variant_id: Mapping[str, str],
    capacities: Sequence[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "generated": 0,
            "resolved": 0,
            "unresolved": 0,
            "skipped": 0,
            "updates": 0,
            "delay_sum": 0.0,
            "delay_max": 0,
            "minimum_offset": None,
            "updates_by_layer": defaultdict(int),
        }
    )
    for row in diagnostics:
        learning = row.get("learning") or {}
        if not learning:
            continue
        key = (label_by_variant_id[str(row["variant_id"])], int(row["capacity"]))
        bucket = grouped[key]
        bucket["generated"] += int(learning["examples_generated"])
        bucket["resolved"] += int(learning["examples_resolved"])
        bucket["unresolved"] += int(learning["examples_unresolved_at_stream_end"])
        bucket["skipped"] += int(learning["examples_skipped_no_comparable_pair"])
        bucket["updates"] += int(learning["applied_updates"])
        average = learning["average_feedback_delay_same_layer_events"]
        if average is not None:
            bucket["delay_sum"] += float(average) * int(learning["examples_resolved"])
        bucket["delay_max"] = max(
            bucket["delay_max"], int(learning["maximum_feedback_delay_same_layer_events"])
        )
        offset = learning["minimum_update_minus_decision_offset"]
        if offset is not None:
            bucket["minimum_offset"] = (
                offset
                if bucket["minimum_offset"] is None
                else min(bucket["minimum_offset"], offset)
            )
        for layer, value in (learning.get("updates_by_layer") or {}).items():
            bucket["updates_by_layer"][int(layer)] += int(value)
    output = []
    for (label, capacity), bucket in sorted(grouped.items()):
        resolved = bucket["resolved"]
        output.append(
            {
                "variant": label,
                "capacity": capacity,
                "examples_generated": bucket["generated"],
                "examples_resolved": resolved,
                "examples_unresolved_at_stream_end": bucket["unresolved"],
                "unresolved_fraction": (
                    bucket["unresolved"] / bucket["generated"] if bucket["generated"] else 0.0
                ),
                "examples_skipped_no_comparable_pair": bucket["skipped"],
                "applied_updates": bucket["updates"],
                "average_feedback_delay_same_layer_events": (
                    bucket["delay_sum"] / resolved if resolved else None
                ),
                "maximum_feedback_delay_same_layer_events": bucket["delay_max"],
                "minimum_update_minus_decision_offset": bucket["minimum_offset"],
                "layers_updated": len(bucket["updates_by_layer"]),
                "updates_per_layer_min": (
                    min(bucket["updates_by_layer"].values())
                    if bucket["updates_by_layer"]
                    else 0
                ),
                "updates_per_layer_max": (
                    max(bucket["updates_by_layer"].values())
                    if bucket["updates_by_layer"]
                    else 0
                ),
            }
        )
    return output


def _ranking_vs_cost(
    ranking: Sequence[Mapping[str, Any]],
    scope_rows: Sequence[Mapping[str, Any]],
    decision_capacities: Sequence[int],
    variant_labels: Sequence[str],
) -> dict[str, Any]:
    suite = {
        int(row["capacity"]): row
        for row in scope_rows
        if row["scope"] == "all_frozen_workloads"
    }
    points = []
    for row in ranking:
        capacity = int(row["capacity"])
        if capacity not in set(decision_capacities):
            continue
        entry = suite[capacity]
        label = str(row["variant"])
        if f"{label}_cost" not in entry:
            continue
        points.append(
            {
                "variant": label,
                "capacity": capacity,
                "pairwise_ordering_accuracy_capped": row["pairwise_ordering_accuracy_capped"],
                "oracle_consistent_eviction_rate": row["oracle_consistent_eviction_rate"],
                "improvement_over_stage1": (
                    float(entry["stage1_cost"]) - float(entry[f"{label}_cost"])
                )
                / float(entry["stage1_cost"]),
            }
        )
    correlations = {}
    if len(points) >= 3:
        from scipy.stats import spearmanr

        for name in (
            "pairwise_ordering_accuracy_capped",
            "oracle_consistent_eviction_rate",
        ):
            xs = [float(item[name]) for item in points if item[name] is not None]
            ys = [
                float(item["improvement_over_stage1"])
                for item in points
                if item[name] is not None
            ]
            if len(xs) >= 3:
                result = spearmanr(xs, ys)
                correlations[name] = {
                    "spearman": float(result.statistic),
                    "p_value_descriptive": float(result.pvalue),
                    "points": len(xs),
                }
    return {
        "points": points,
        "correlations": correlations,
        "interpretation": (
            "Descriptive configuration-level association across variants and capacities; the "
            "configurations are not independent experiments."
        ),
    }


def _trajectory_layers(
    diagnostics: Sequence[Mapping[str, Any]],
    label_by_variant_id: Mapping[str, str],
    primary_label: str,
) -> dict[str, Any]:
    baseline: dict[int, list[int]] = {}
    primary: dict[int, list[int]] = {}
    for row in diagnostics:
        if row["workload"] != TRAJECTORY_WORKLOAD or int(row["capacity"]) != 32:
            continue
        label = label_by_variant_id[str(row["variant_id"])]
        if label == "RACE_UNIFORM":
            baseline[int(row["capacity"])] = list(row["layer_misses"])
        elif label == primary_label:
            primary[int(row["capacity"])] = list(row["layer_misses"])
    if 32 not in baseline or 32 not in primary:
        return {"rule": "unavailable", "layers": []}
    reduction = [
        (float(before) - float(after)) / float(before) if before else 0.0
        for before, after in zip(baseline[32], primary[32])
    ]
    order = sorted(range(len(reduction)), key=lambda index: (-reduction[index], index))
    best = order[0]
    median = sorted(range(len(reduction)), key=lambda index: (reduction[index], index))[
        len(reduction) // 2
    ]
    return {
        "rule": (
            "largest and median per-layer miss reduction of the frozen primary variant relative "
            "to RACE_UNIFORM on mixed_interleaved at capacity 32"
        ),
        "workload": TRAJECTORY_WORKLOAD,
        "capacity": 32,
        "per_layer_reduction": reduction,
        "layers": [best, median],
        "largest_layer": best,
        "median_layer": median,
    }


def _write_tables(
    table_dir: Path, analysis: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
    primary = analysis["primary_variant_label"]
    labels = analysis["variant_labels"]
    suite = analysis["suite_results"]
    main = []
    for row in suite:
        entry = {
            "capacity": row["capacity"],
            "stage0_simple_cost": row["stage0_simple_cost"],
            "stage1_winner_cost": row["stage1_cost"],
        }
        for label in labels:
            entry[f"{label}_cost"] = row[f"{label}_cost"]
        entry["oracle_cost"] = row["oracle_cost"]
        entry["stage1_winner_normalized"] = row["stage1_cost"] / row["stage0_simple_cost"]
        for label in labels:
            entry[f"{label}_normalized"] = row[f"{label}_cost"] / row["stage0_simple_cost"]
        entry["oracle_normalized"] = row["oracle_cost"] / row["stage0_simple_cost"]
        main.append(entry)
    write_csv(table_dir / "table1_main_costs.csv", main, list(main[0]))

    success = [
        {
            "capacity": row["capacity"],
            "improvement_vs_stage1": row["race_improvement_over_stage1"],
            "improvement_ci_low": row["race_improvement_ci_low"],
            "improvement_ci_high": row["race_improvement_ci_high"],
            "original_oracle_gap_closed": row["original_oracle_gap_closed"],
            "original_oracle_gap_closed_ci_low": row["original_oracle_gap_closed_ci_low"],
            "original_oracle_gap_closed_ci_high": row["original_oracle_gap_closed_ci_high"],
            "stage1_original_oracle_gap_closed": row["stage1_original_oracle_gap_closed"],
            "stage1_residual_recovered": row["stage1_residual_recovered"],
            "stage1_residual_recovered_ci_low": row["stage1_residual_recovered_ci_low"],
            "stage1_residual_recovered_ci_high": row["stage1_residual_recovered_ci_high"],
            "residual_headroom": row["residual_headroom"],
        }
        for row in suite
        if int(row["capacity"]) != 8
    ]
    write_csv(table_dir / "table2_success_metrics.csv", success, list(success[0]))

    regime = [
        {
            "scope": row["scope"],
            "capacity": row["capacity"],
            "stage0_simple_cost": row["stage0_simple_cost"],
            "stage1_cost": row["stage1_cost"],
            "race_cost": row[f"{primary}_cost"],
            "oracle_cost": row["oracle_cost"],
            "improvement_vs_stage1": row["race_improvement_over_stage1"],
            "original_oracle_gap_closed": row["original_oracle_gap_closed"],
            "stage1_residual_recovered": row["stage1_residual_recovered"],
            "regression_ratio": row["regression_ratio"],
        }
        for row in analysis["scope_results"]
        if row["scope"] != "all_frozen_workloads"
    ]
    write_csv(table_dir / "table3_workload_regimes.csv", regime, list(regime[0]))
    write_csv(
        table_dir / "table4_ablations.csv",
        analysis["ablation"],
        list(analysis["ablation"][0]),
    )
    write_csv(
        table_dir / "table5_ranking_diagnostics.csv",
        analysis["ranking_diagnostics"],
        list(analysis["ranking_diagnostics"][0]),
    )
    write_csv(
        table_dir / "table6_horizon_weights.csv",
        analysis["horizon_weights"],
        list(analysis["horizon_weights"][0]),
    )
    write_csv(
        table_dir / "table7_delayed_feedback.csv",
        analysis["delayed_feedback"],
        list(analysis["delayed_feedback"][0]),
    )
    write_csv(
        table_dir / "table8_regret.csv",
        analysis["regret_accounting"],
        list(analysis["regret_accounting"][0]),
    )
    write_csv(
        table_dir / "table9_per_workload.csv",
        analysis["workload_results"],
        [
            "workload",
            "regime",
            "capacity",
            "stage0_simple_policy",
            "stage0_simple_cost",
            "stage1_cost",
            *[f"{label}_cost" for label in labels],
            "oracle_cost",
            "race_improvement_over_stage1",
            "original_oracle_gap_closed",
            "stage1_residual_recovered",
            "regression_ratio",
        ],
    )
    write_csv(
        table_dir / "table10_regressions.csv",
        analysis["regression_rows"],
        list(analysis["regression_rows"][0]),
    )
    weight_rows = [
        row
        for row in analysis["weight_adaptation"]
        if row["workload"] == TRAJECTORY_WORKLOAD
    ]
    if weight_rows:
        write_csv(
            table_dir / "table11_weight_adaptation_mixed.csv",
            [
                {
                    key: (json.dumps(value) if isinstance(value, list) else value)
                    for key, value in row.items()
                }
                for row in weight_rows
            ],
            list(weight_rows[0]),
        )
    atomic_write_text(
        table_dir / "required_tables.md", _markdown_tables(analysis, frozen)
    )


def _markdown_tables(analysis: Mapping[str, Any], frozen: Mapping[str, Any]) -> str:
    primary = analysis["primary_variant_label"]
    suite = analysis["suite_results"]
    lines = [
        "# RACE Stage 2 required tables",
        "",
        "## Table 1 — Main frozen-suite comparison (unit expert transfers)",
        "",
        "| Capacity | Stage1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite:
        lines.append(
            "| {} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} |".format(
                row["capacity"],
                row["stage1_cost"],
                row["RACE_UNIFORM_cost"],
                row["RACE_STATIC_cost"],
                row["RACE_ONLINE_cost"],
                row["RACE_COST_cost"],
                row["oracle_cost"],
            )
        )
    lines += [
        "",
        "Normalized by the Stage 0 strongest-simple cost at the same capacity:",
        "",
        "| Capacity | Stage1 winner | RACE Uniform | RACE Static | RACE Online | RACE Cost | Oracle |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite:
        base = row["stage0_simple_cost"]
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["capacity"],
                row["stage1_cost"] / base,
                row["RACE_UNIFORM_cost"] / base,
                row["RACE_STATIC_cost"] / base,
                row["RACE_ONLINE_cost"] / base,
                row["RACE_COST_cost"] / base,
                row["oracle_cost"] / base,
            )
        )
    lines += [
        "",
        f"## Table 2 — Success metrics for the frozen primary variant `{primary}`",
        "",
        "| Capacity | Improvement vs Stage1 (95% CI) | Original oracle gap closed (95% CI) | Stage1 residual recovered (95% CI) |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in suite:
        if int(row["capacity"]) == 8:
            continue
        lines.append(
            "| {} | {:.2f}% [{:.2f}, {:.2f}] | {:.2f}% [{:.2f}, {:.2f}] | {:.2f}% [{:.2f}, {:.2f}] |".format(
                row["capacity"],
                100 * row["race_improvement_over_stage1"],
                100 * row["race_improvement_ci_low"],
                100 * row["race_improvement_ci_high"],
                100 * row["original_oracle_gap_closed"],
                100 * row["original_oracle_gap_closed_ci_low"],
                100 * row["original_oracle_gap_closed_ci_high"],
                100 * row["stage1_residual_recovered"],
                100 * row["stage1_residual_recovered_ci_low"],
                100 * row["stage1_residual_recovered_ci_high"],
            )
        )
    lines += [
        "",
        "## Table 3 — Frozen primary variant by workload regime",
        "",
        "| Regime | Capacity | Stage0 simple | Stage1 | RACE | Oracle | Improvement vs Stage1 | Gap closed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["scope_results"]:
        if row["scope"] == "all_frozen_workloads":
            continue
        gap = row["original_oracle_gap_closed"]
        lines.append(
            "| {} | {} | {:,.0f} | {:,.0f} | {:,.0f} | {:,.0f} | {:.2f}% | {} |".format(
                row["scope"],
                row["capacity"],
                row["stage0_simple_cost"],
                row["stage1_cost"],
                row[f"{primary}_cost"],
                row["oracle_cost"],
                100 * row["race_improvement_over_stage1"],
                "N/A" if gap is None else f"{100*gap:.2f}%",
            )
        )
    lines += [
        "",
        "## Table 4 — Ablation decomposition (frozen ten-workload suite)",
        "",
        "| Question | Comparison | Capacity | Before | After | Relative change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis["ablation"]:
        lines.append(
            "| {} | {} | {} | {:,.0f} | {:,.0f} | {:+.2f}% |".format(
                row["question"],
                row["comparison"],
                row["capacity"],
                row["before_cost"],
                row["after_cost"],
                100 * row["relative_change"],
            )
        )
    return "\n".join(lines) + "\n"


def _write_figures(
    figure_dir: Path,
    analysis: Mapping[str, Any],
    evaluation_dir: Path,
    frozen: Mapping[str, Any],
) -> None:
    primary = analysis["primary_variant_label"]
    suite = analysis["suite_results"]
    capacities = [int(row["capacity"]) for row in suite]

    figure, axis = plt.subplots(figsize=(7.5, 5))
    base = [row["stage0_simple_cost"] for row in suite]
    axis.plot(capacities, [1.0] * len(suite), marker="o", label="Stage 0 strongest simple")
    axis.plot(
        capacities,
        [row["stage1_cost"] / value for row, value in zip(suite, base)],
        marker="o",
        label="Stage 1 winner",
    )
    axis.plot(
        capacities,
        [row[f"{primary}_cost"] / value for row, value in zip(suite, base)],
        marker="o",
        linewidth=2.5,
        label=f"RACE primary ({primary})",
    )
    axis.plot(
        capacities,
        [row["oracle_cost"] / value for row, value in zip(suite, base)],
        marker="o",
        label="Offline oracle",
    )
    axis.set_xlabel("Cache capacity B")
    axis.set_ylabel("Normalized transfer cost (Stage 0 simple = 1)")
    axis.set_xticks(capacities)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, figure_dir / "figure1_normalized_transfer_cost")

    figure, axis = plt.subplots(figsize=(7.5, 5))
    decision = [row for row in suite if int(row["capacity"]) != 8]
    spare = [int(row["capacity"]) - 8 for row in decision]
    axis.plot(
        spare,
        [100 * row["stage1_original_oracle_gap_closed"] for row in decision],
        marker="o",
        label="Stage 1 winner",
    )
    axis.plot(
        spare,
        [100 * row["original_oracle_gap_closed"] for row in decision],
        marker="o",
        linewidth=2.5,
        label=f"RACE primary ({primary})",
    )
    axis.axhline(30, color="black", linestyle="--", linewidth=1, label="30% Stage 2 threshold")
    axis.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% very-strong label")
    axis.set_xlabel("Spare residency slots S = B - 8")
    axis.set_ylabel("Original Stage 0 oracle gap closed (%)")
    axis.set_xticks(spare)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    _save(figure, figure_dir / "figure2_oracle_gap_vs_spare")

    trajectory_path = evaluation_dir / "weight_trajectories.jsonl"
    layers = analysis["trajectory_layers"].get("layers", [])
    if trajectory_path.exists() and layers:
        rows = read_jsonl(trajectory_path)
        names = list(frozen["adviser_pools"]["primary"]["order"])
        chosen = [value for value in dict.fromkeys(layers)]
        figure, axes = plt.subplots(
            1, max(len(chosen), 1), figsize=(6.2 * max(len(chosen), 1), 4.4), squeeze=False
        )
        for index, layer in enumerate(chosen):
            axis = axes[0][index]
            series = sorted(
                (
                    row
                    for row in rows
                    if int(row["capacity"]) == 32 and int(row["layer"]) == int(layer)
                ),
                key=lambda row: int(row["same_layer_position"]),
            )
            if not series:
                continue
            positions = [int(row["same_layer_position"]) for row in series]
            values = np.asarray([row["weights"] for row in series], dtype=np.float64)
            for adviser in range(values.shape[1]):
                axis.plot(positions, values[:, adviser], label=names[adviser], linewidth=1.2)
            axis.set_title(f"layer {layer} (capacity 32, mixed_interleaved)")
            axis.set_xlabel("Same-layer event index")
            axis.set_ylabel("Adviser weight")
            axis.grid(alpha=0.25)
        axes[0][0].legend(fontsize=6, ncol=2)
        figure.tight_layout()
        _save(figure, figure_dir / "figure3_adviser_weight_trajectories")

    points = analysis["ranking_vs_cost"]["points"]
    if points:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        for axis, key, label in (
            (axes[0], "pairwise_ordering_accuracy_capped", "Pairwise ordering accuracy (capped target)"),
            (axes[1], "oracle_consistent_eviction_rate", "Oracle-consistent eviction rate"),
        ):
            for item in points:
                if item[key] is None:
                    continue
                axis.scatter(100 * float(item[key]), 100 * float(item["improvement_over_stage1"]), s=26)
                axis.annotate(
                    f"{item['variant']}@{item['capacity']}",
                    (100 * float(item[key]), 100 * float(item["improvement_over_stage1"])),
                    fontsize=5,
                    xytext=(3, 2),
                    textcoords="offset points",
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xlabel(f"{label} (%)")
            axis.set_ylabel("Improvement over Stage 1 winner (%)")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        _save(figure, figure_dir / "figure4_ranking_vs_transfer_cost")


def _save(figure: Any, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _analysis_audit(
    analysis: Mapping[str, Any], report: Path, tables: Path, figures: Path
) -> dict[str, Any]:
    required = [
        report,
        report.with_suffix(".sha256"),
        tables / "required_tables.md",
        tables / "table1_main_costs.csv",
        tables / "table2_success_metrics.csv",
        tables / "table3_workload_regimes.csv",
        tables / "table4_ablations.csv",
        tables / "table5_ranking_diagnostics.csv",
        tables / "table6_horizon_weights.csv",
        figures / "figure1_normalized_transfer_cost.png",
        figures / "figure2_oracle_gap_vs_spare.png",
        figures / "figure4_ranking_vs_transfer_cost.png",
    ]
    text = report.read_text(encoding="utf-8")
    checks = {
        "report_begins_with_exact_verdict": text.startswith(str(analysis["verdict"]) + "\n"),
        "verdict_is_one_of_four": analysis["verdict"]
        in {
            "RACE_STAGE2_VERY_STRONG_SUCCESS",
            "RACE_STAGE2_STRONG_SUCCESS",
            "RACE_STAGE2_WEAK",
            "RACE_STAGE2_NO_GO",
        },
        "all_required_outputs_exist": all(
            path.exists() and path.stat().st_size > 0 for path in required
        ),
        "five_capacities_present": len(analysis["suite_results"]) == 5,
        "capacity8_gap_is_null": analysis["suite_results"][0]["original_oracle_gap_closed"]
        is None,
        "four_decision_capacities": len(
            [row for row in analysis["suite_results"] if int(row["capacity"]) != 8]
        )
        == 4,
        "regressions_reported": "regression_rows" in analysis,
    }
    return {
        "schema_version": "race_stage2_analysis_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "verdict": analysis["verdict"],
    }


def _filtered_rows(path: Path, wanted: set[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("condition_id")) in wanted:
                rows.append(row)
    return rows


def _sequence_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["condition_id"])].append(dict(row))
    for values in output.values():
        values.sort(key=lambda row: int(row["sequence_position"]))
    return output


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


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=output, where=denominator != 0)
    return output


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
    digest = hashlib.sha256(f"race-stage2-bootstrap-v1\0{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little")
