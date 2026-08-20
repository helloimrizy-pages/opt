"""Stage 3 analysis: frozen decision ladder, paired bootstrap, tables and figures."""

from __future__ import annotations

import hashlib
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

from .calibration import load_and_verify_stage3_frozen
from .frozen import load_and_verify_stage3_inputs, stage1_per_sequence_rows
from .report import render_report

REGIME_ORDER = ("stationary", "abrupt", "repeated", "mixed")


def metrics(*, simple: float, stage1: float, stage3: float, oracle: float) -> dict[str, Any]:
    gap = simple - oracle
    residual = stage1 - oracle
    return {
        "stage0_simple_cost": float(simple),
        "stage1_cost": float(stage1),
        "stage3_cost": float(stage3),
        "oracle_cost": float(oracle),
        "improvement_over_stage1": (stage1 - stage3) / stage1,
        "original_oracle_gap_closed": None if abs(gap) < 1e-12 else (simple - stage3) / gap,
        "stage1_original_oracle_gap_closed": None if abs(gap) < 1e-12 else (simple - stage1) / gap,
        "stage1_residual_recovered": None if abs(residual) < 1e-12 else (stage1 - stage3) / residual,
        "residual_headroom": (stage3 - oracle) / simple,
        "regression_ratio": stage3 / stage1,
        "normalized_cost": stage3 / simple,
    }


def analyze_and_report(
    repository_root: Path,
    preregistration_path: Path,
    frozen_config_path: Path,
    evaluation_dir: Path,
    stage3_root: Path,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    inputs = load_and_verify_stage3_inputs(repository_root, preregistration_path)
    frozen = load_and_verify_stage3_frozen(frozen_config_path)
    manifest = read_json(evaluation_dir / "evaluation_manifest.json")
    if manifest["frozen_config_file_sha256"] != frozen["file_sha256"]:
        raise ValueError("Evaluation and Stage 3 frozen config hashes differ")
    if not read_json(evaluation_dir / "sanity_checks.json")["passed"]:
        raise ValueError("Cannot analyze an evaluation that failed sanity checks")

    rows = read_jsonl(evaluation_dir / "results.jsonl")
    diagnostics = read_jsonl(evaluation_dir / "diagnostics.jsonl")
    capacities = tuple(int(v) for v in frozen["cache_capacities"])
    decision = tuple(int(v) for v in frozen["decision_capacities"])
    variants = list(frozen["variants"])
    primary = str(frozen["primary_variant"])
    cost = {(str(r["workload"]), int(r["capacity"]), str(r["variant"])): int(r["misses"]) for r in rows}
    condition = {
        (str(r["workload"]), int(r["capacity"]), str(r["variant"])): str(r["condition_id"])
        for r in rows
    }
    workloads = list(inputs.workloads)
    by_regime = {rg: tuple(w.name for w in workloads if w.regime == rg) for rg in REGIME_ORDER}
    suite = tuple(w.name for w in workloads)

    def sum_ref(names, capacity, role):
        return float(sum(float(inputs.stage0_references[(n, capacity)][role]["misses"]) for n in names))

    scopes = [("all_frozen_workloads", suite)] + [(rg, by_regime[rg]) for rg in REGIME_ORDER]
    scope_rows: list[dict[str, Any]] = []
    for scope, names in scopes:
        for capacity in capacities:
            simple = sum_ref(names, capacity, "simple")
            oracle = sum_ref(names, capacity, "oracle")
            stage1 = float(sum(inputs.stage1_costs[(n, capacity)] for n in names))
            entry: dict[str, Any] = {
                "scope": scope, "capacity": capacity, "spare_residency": capacity - inputs.trace.top_k,
                "workloads": len(names),
            }
            for variant in variants:
                value = float(sum(cost[(n, capacity, variant)] for n in names))
                entry[f"{variant}_cost"] = value
                entry[f"{variant}_metrics"] = metrics(
                    simple=simple, stage1=stage1, stage3=value, oracle=oracle
                )
            entry.update(entry[f"{primary}_metrics"])
            scope_rows.append(entry)

    workload_rows = []
    for workload in workloads:
        for capacity in capacities:
            simple = sum_ref([workload.name], capacity, "simple")
            oracle = sum_ref([workload.name], capacity, "oracle")
            stage1 = float(inputs.stage1_costs[(workload.name, capacity)])
            entry = {
                "workload": workload.name, "regime": workload.regime, "capacity": capacity,
                "stage0_simple_policy": inputs.stage0_references[(workload.name, capacity)]["simple"]["policy"],
            }
            for variant in variants:
                entry[f"{variant}_cost"] = float(cost[(workload.name, capacity, variant)])
            entry.update(metrics(simple=simple, stage1=stage1,
                                 stage3=float(cost[(workload.name, capacity, primary)]), oracle=oracle))
            workload_rows.append(entry)

    bootstrap = _bootstrap(repository_root, inputs, frozen, evaluation_dir, scopes,
                           capacities, condition, primary)
    lookup = {(b["scope"], int(b["capacity"])): b for b in bootstrap}
    for row in scope_rows:
        row.update(lookup[(row["scope"], int(row["capacity"]))])
    suite_rows = [r for r in scope_rows if r["scope"] == "all_frozen_workloads"]
    regressions = [
        {"workload": r["workload"], "regime": r["regime"], "capacity": r["capacity"],
         "stage1_cost": r["stage1_cost"], "stage3_cost": r[f"{primary}_cost"],
         "regression_ratio": r["regression_ratio"], "flagged": bool(r["regression_ratio"] > 1.03)}
        for r in workload_rows
    ]
    decision_record = _decide(suite_rows, frozen["success_criteria"], decision)
    ranking = _ranking_table(diagnostics, capacities)
    ablation = _ablation(scope_rows, decision, variants, primary)

    analysis = {
        "schema_version": "race_stage3_analysis_v1",
        "created_at_utc": utc_now(),
        "verdict": decision_record["verdict"],
        "decision": decision_record,
        "primary_variant": primary,
        "variants": variants,
        "capacities": list(capacities),
        "decision_capacities": list(decision),
        "suite_results": suite_rows,
        "scope_results": scope_rows,
        "workload_results": workload_rows,
        "regression_rows": regressions,
        "ablation": ablation,
        "ranking_diagnostics": ranking,
        "bootstrap_rows": bootstrap,
        "bootstrap_interpretation": frozen["statistics"]["conditionality"],
        "calibration_selection": read_json(repository_root / frozen["selection_path"]),
        "trace_hash": inputs.trace.trace_hash,
        "preregistration_hash": inputs.preregistration_hash,
        "frozen_config_file_sha256": frozen["file_sha256"],
        "evaluation_manifest_sha256": sha256_file(evaluation_dir / "evaluation_manifest.json"),
        "stage1_winner_method_id": frozen["stage1_winner_method_id"],
        "stage0_archive_manifest_sha256": frozen["stage0_archive_manifest_sha256"],
        "stage1_archive_manifest_sha256": frozen["stage1_archive_manifest_sha256"],
        "stage2_archive_manifest_sha256": frozen["stage2_archive_manifest_sha256"],
    }
    report_dir, table_dir, figure_dir = (stage3_root / "reports", stage3_root / "tables",
                                        stage3_root / "figures")
    for directory in (report_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_dir / "analysis.json", analysis)
    _write_tables(table_dir, analysis)
    _write_figures(figure_dir, analysis)
    report = render_report(analysis, inputs, frozen)
    report_path = report_dir / "race_stage3_report.md"
    atomic_write_text(report_path, report)
    atomic_write_text(report_path.with_suffix(".sha256"),
                      f"{sha256_file(report_path)}  {report_path.name}\n")
    audit = {
        "schema_version": "race_stage3_analysis_audit_v1",
        "checks": {
            "report_begins_with_exact_verdict": report.startswith(analysis["verdict"] + "\n"),
            "verdict_in_ladder": analysis["verdict"] in {
                "RACE_STAGE3_VERY_STRONG_SUCCESS", "RACE_STAGE3_STRONG_SUCCESS",
                "RACE_STAGE3_PARTIAL_SUCCESS", "RACE_STAGE3_WEAK", "RACE_STAGE3_NO_GO"},
            "five_capacities": len(suite_rows) == 5,
            "capacity8_gap_null": suite_rows[0]["original_oracle_gap_closed"] is None,
            "stage2_criteria_reported": "stage2_criteria" in decision_record,
        },
    }
    audit["passed"] = all(audit["checks"].values())
    atomic_write_json(report_dir / "analysis_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"Stage 3 analysis audit failed: {audit['checks']}")
    return analysis


def _decide(suite_rows, criteria, decision_capacities) -> dict[str, Any]:
    rows = [r for r in suite_rows if int(r["capacity"]) in set(decision_capacities)]
    positive = [int(r["capacity"]) for r in rows
                if float(r["improvement_over_stage1"]) > 0 and float(r["improvement_ci_low"]) > 0]
    regressed = [int(r["capacity"]) for r in rows if float(r["regression_ratio"]) > 1.03]

    def meet(threshold, field, require_ci):
        found = []
        for r in rows:
            value = r[field]
            if value is None or float(value) < threshold:
                continue
            if require_ci and not float(r["improvement_ci_low"]) > 0:
                continue
            found.append(int(r["capacity"]))
        return found

    very_a = meet(0.15, "improvement_over_stage1", False)
    very_b = meet(0.50, "original_oracle_gap_closed", False)
    strong_a = meet(0.10, "improvement_over_stage1", True)
    strong_b = meet(0.30, "original_oracle_gap_closed", True)
    partial_b = meet(0.20, "original_oracle_gap_closed", False)
    stage2 = {
        "condition_a_capacities": strong_a,
        "condition_b_capacities": meet(0.30, "original_oracle_gap_closed", False),
        "stage2_strong_success_would_pass": len(strong_a) >= 3
        and len(meet(0.30, "original_oracle_gap_closed", False)) >= 3,
        "note": "Stage 2's criteria applied verbatim to the Stage 3 primary variant.",
    }
    if regressed or len(positive) < 3:
        verdict, reason = "RACE_STAGE3_NO_GO", (
            f"The preregistered NO-GO rule fired: suite regressions above 3% at {regressed}, "
            f"and a strictly positive improvement with a paired interval excluding zero at only "
            f"{len(positive)} of {len(rows)} non-degenerate capacities."
        )
    elif len(very_a) >= 3 and len(very_b) >= 3:
        verdict, reason = "RACE_STAGE3_VERY_STRONG_SUCCESS", (
            f"At least 15% improvement at {very_a} and at least 50% oracle-gap closure at {very_b}.")
    elif len(strong_a) >= 3 and len(strong_b) >= 3:
        verdict, reason = "RACE_STAGE3_STRONG_SUCCESS", (
            f"At least 10% improvement with a positive paired interval at {strong_a} and at least "
            f"30% oracle-gap closure at {strong_b}.")
    elif len(positive) == len(rows) and len(partial_b) >= 3:
        verdict, reason = "RACE_STAGE3_PARTIAL_SUCCESS", (
            f"A strictly positive improvement over the frozen Stage 1 winner whose paired 95% "
            f"interval excludes zero at all {len(rows)} non-degenerate capacities, at least 20% "
            f"of the original Stage 0 oracle gap closed at {partial_b}, and no capacity regressing "
            f"above the 3% threshold.")
    else:
        verdict, reason = "RACE_STAGE3_WEAK", (
            f"A strictly positive improvement with a paired interval excluding zero at {positive}, "
            f"but the preregistered PARTIAL_SUCCESS rule was not met.")
    return {
        "verdict": verdict, "reason": reason,
        "capacities_with_positive_improvement": positive,
        "suite_regression_capacities": regressed,
        "very_strong_a": very_a, "very_strong_b": very_b,
        "strong_a": strong_a, "strong_b": strong_b, "partial_b": partial_b,
        "stage2_criteria": stage2, "criteria": dict(criteria),
    }


def _ablation(scope_rows, decision_capacities, variants, primary) -> list[dict[str, Any]]:
    rows = [r for r in scope_rows if r["scope"] == "all_frozen_workloads"
            and int(r["capacity"]) in set(decision_capacities)]
    pairs = [("A_stage1_to_primary", "stage1_cost", f"{primary}_cost",
              "Stage 1 winner -> STAGE3_RANKER")]
    if "STAGE3_RANKER_NO_REQUEST_SCOPE" in variants:
        pairs.append(("B_request_scope_value", "STAGE3_RANKER_NO_REQUEST_SCOPE_cost",
                      f"{primary}_cost", "without request scope -> with request scope"))
        pairs.append(("B2_stage1_to_no_request_scope", "stage1_cost",
                      "STAGE3_RANKER_NO_REQUEST_SCOPE_cost",
                      "Stage 1 winner -> STAGE3_RANKER_NO_REQUEST_SCOPE (identical information)"))
    if "STAGE3_RANKER_POOLED" in variants:
        pairs.append(("C_per_capacity_value", "STAGE3_RANKER_POOLED_cost", f"{primary}_cost",
                      "pooled model -> per-capacity models"))
    if "STAGE3_RANKER_ROUND1_DATA" in variants:
        pairs.append(("D_retraining_value", "STAGE3_RANKER_ROUND1_DATA_cost", f"{primary}_cost",
                      "round-1 data -> round-2 data"))
    out = []
    for key, before, after, description in pairs:
        for row in rows:
            left, right = float(row[before]), float(row[after])
            out.append({"question": key, "comparison": description, "capacity": int(row["capacity"]),
                        "before_cost": left, "after_cost": right,
                        "relative_change": (left - right) / left})
    return out


def _ranking_table(diagnostics, capacities) -> list[dict[str, Any]]:
    fields = ("eviction_events", "comparable_pairs_capped", "comparable_pairs_true",
              "concordant_capped", "discordant_capped", "tied_capped",
              "concordant_true", "discordant_true", "tied_true",
              "oracle_consistent_events", "oracle_optimal_events")
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: dict.fromkeys(fields, 0.0))
    for row in diagnostics:
        ranking = row.get("ranking") or {}
        if not ranking:
            continue
        bucket = grouped[(str(row["variant"]), int(row["capacity"]))]
        for name in fields:
            bucket[name] += float(ranking[name])

    def ratio(num, den):
        return float(num / den) if den else None

    return [
        {
            "variant": variant, "capacity": capacity,
            "eviction_events": int(b["eviction_events"]),
            "pairwise_ordering_accuracy_capped": ratio(
                b["concordant_capped"] + 0.5 * b["tied_capped"], b["comparable_pairs_capped"]),
            "pairwise_ordering_accuracy_true": ratio(
                b["concordant_true"] + 0.5 * b["tied_true"], b["comparable_pairs_true"]),
            "oracle_consistent_eviction_rate": ratio(b["oracle_consistent_events"], b["eviction_events"]),
            "oracle_optimal_eviction_rate": ratio(b["oracle_optimal_events"], b["eviction_events"]),
        }
        for (variant, capacity), b in sorted(grouped.items())
    ]


def _bootstrap(repository_root, inputs, frozen, evaluation_dir, scopes, capacities,
               condition, primary) -> list[dict[str, Any]]:
    replicates = int(frozen["statistics"]["bootstrap_replicates"])
    seed = int(frozen["statistics"]["bootstrap_seed"])
    prereg = inputs.preregistration
    stage0_needed = {str(item[role]["condition_id"])
                     for item in inputs.stage0_references.values() for role in ("simple", "oracle")}
    stage0_rows = _lookup(_filtered_file(
        repository_root / prereg["stage0_reference"]["per_sequence_result_path"], stage0_needed))
    stage1_rows = _lookup(stage1_per_sequence_rows(
        repository_root, prereg, set(inputs.stage1_condition_ids.values())))
    stage3_needed = {condition[k] for k in condition if k[2] == primary}
    stage3_rows = _lookup(_filtered_file(
        evaluation_dir / "per_sequence_results.jsonl", stage3_needed))

    output = []
    for scope, names in scopes:
        for capacity in capacities:
            parts, domains = [], {}
            for name in names:
                reference = inputs.stage0_references[(name, capacity)]
                blocks = [
                    stage0_rows[str(reference["simple"]["condition_id"])],
                    stage0_rows[str(reference["oracle"]["condition_id"])],
                    stage1_rows[str(inputs.stage1_condition_ids[(name, capacity)])],
                    stage3_rows[str(condition[(name, capacity, primary)])],
                ]
                identity = [[(int(r["source_sequence_id"]), str(r["domain"])) for r in b] for b in blocks]
                if any(i != identity[0] for i in identity[1:]):
                    raise ValueError(f"Paired rows misaligned for {name} at capacity {capacity}")
                for sequence, domain in identity[0]:
                    domains.setdefault(sequence, domain)
                parts.append((identity[0], *[np.asarray([r["misses"] for r in b], dtype=np.float64)
                                             for b in blocks]))
            multiplicity = _multiplicities(domains, replicates, _seed(seed, f"{scope}:{capacity}"))
            totals = np.zeros((4, replicates))
            paired = []
            for identity, simple, oracle, stage1, stage3 in parts:
                weights = np.stack([multiplicity[s] for s, _ in identity], axis=1)
                for index, vector in enumerate((simple, oracle, stage1, stage3)):
                    totals[index] += (weights * vector[None, :]).sum(axis=1)
                paired.extend(stage1 - stage3)
            simple_b, oracle_b, stage1_b, stage3_b = totals
            improvement = _divide(stage1_b - stage3_b, stage1_b)
            gap = _divide(simple_b - stage3_b, simple_b - oracle_b)
            residual = _divide(stage1_b - stage3_b, stage1_b - oracle_b)
            values = np.asarray(paired, dtype=np.float64)
            output.append({
                "scope": scope, "capacity": capacity,
                "improvement_ci_low": _quantile(improvement, 0.025),
                "improvement_ci_high": _quantile(improvement, 0.975),
                "original_oracle_gap_closed_ci_low": _quantile(gap, 0.025),
                "original_oracle_gap_closed_ci_high": _quantile(gap, 0.975),
                "stage1_residual_recovered_ci_low": _quantile(residual, 0.025),
                "stage1_residual_recovered_ci_high": _quantile(residual, 0.975),
                "mean_paired_sequence_gain": float(values.mean()),
                "paired_sequences_improved": int((values > 0).sum()),
                "paired_sequences_worsened": int((values < 0).sum()),
                "paired_units": int(values.size),
                "bootstrap_replicates": replicates,
                "unique_source_sequences": len(domains),
            })
    return output


def _write_tables(table_dir: Path, analysis: Mapping[str, Any]) -> None:
    primary = analysis["primary_variant"]
    suite = analysis["suite_results"]
    main = []
    for row in suite:
        entry = {"capacity": row["capacity"], "stage0_simple_cost": row["stage0_simple_cost"],
                 "stage1_winner_cost": row["stage1_cost"]}
        for variant in analysis["variants"]:
            entry[f"{variant}_cost"] = row[f"{variant}_cost"]
        entry["oracle_cost"] = row["oracle_cost"]
        main.append(entry)
    write_csv(table_dir / "table1_main_costs.csv", main, list(main[0]))
    success = [{"capacity": r["capacity"], "improvement_vs_stage1": r["improvement_over_stage1"],
                "improvement_ci_low": r["improvement_ci_low"], "improvement_ci_high": r["improvement_ci_high"],
                "original_oracle_gap_closed": r["original_oracle_gap_closed"],
                "stage1_gap_closed": r["stage1_original_oracle_gap_closed"],
                "stage1_residual_recovered": r["stage1_residual_recovered"],
                "residual_headroom": r["residual_headroom"]}
               for r in suite if int(r["capacity"]) != 8]
    write_csv(table_dir / "table2_success_metrics.csv", success, list(success[0]))
    regime = [{"scope": r["scope"], "capacity": r["capacity"], "stage1_cost": r["stage1_cost"],
               "stage3_cost": r[f"{primary}_cost"], "oracle_cost": r["oracle_cost"],
               "improvement_vs_stage1": r["improvement_over_stage1"],
               "original_oracle_gap_closed": r["original_oracle_gap_closed"]}
              for r in analysis["scope_results"] if r["scope"] != "all_frozen_workloads"]
    write_csv(table_dir / "table3_workload_regimes.csv", regime, list(regime[0]))
    for name, key in (("table4_ablations.csv", "ablation"),
                      ("table5_ranking_diagnostics.csv", "ranking_diagnostics"),
                      ("table6_regressions.csv", "regression_rows")):
        data = analysis[key]
        fields = sorted({k for row in data for k in row})
        write_csv(table_dir / name, data, fields)


def _write_figures(figure_dir: Path, analysis: Mapping[str, Any]) -> None:
    primary = analysis["primary_variant"]
    suite = analysis["suite_results"]
    caps = [int(r["capacity"]) for r in suite]
    base = [r["stage0_simple_cost"] for r in suite]
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.plot(caps, [1.0] * len(suite), marker="o", label="Stage 0 strongest simple")
    axis.plot(caps, [r["stage1_cost"] / b for r, b in zip(suite, base)], marker="o", label="Stage 1 winner")
    axis.plot(caps, [r[f"{primary}_cost"] / b for r, b in zip(suite, base)], marker="o",
              linewidth=2.5, label=f"Stage 3 primary ({primary})")
    axis.plot(caps, [r["oracle_cost"] / b for r, b in zip(suite, base)], marker="o", label="Offline oracle")
    axis.set_xlabel("Cache capacity B"); axis.set_ylabel("Normalized transfer cost")
    axis.set_xticks(caps); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    figure.tight_layout(); _save(figure, figure_dir / "figure1_normalized_transfer_cost")

    rows = [r for r in suite if int(r["capacity"]) != 8]
    spare = [int(r["capacity"]) - 8 for r in rows]
    figure, axis = plt.subplots(figsize=(7.5, 5))
    axis.plot(spare, [100 * r["stage1_original_oracle_gap_closed"] for r in rows], marker="o",
              label="Stage 1 winner")
    axis.plot(spare, [100 * r["original_oracle_gap_closed"] for r in rows], marker="o",
              linewidth=2.5, label="Stage 3 primary")
    axis.axhline(30, color="black", linestyle="--", linewidth=1, label="30% (Stage 2/3 strong)")
    axis.axhline(20, color="gray", linestyle=":", linewidth=1, label="20% (Stage 3 partial)")
    axis.set_xlabel("Spare residency slots S = B - 8")
    axis.set_ylabel("Original Stage 0 oracle gap closed (%)")
    axis.set_xticks(spare); axis.grid(alpha=0.25); axis.legend(fontsize=8)
    figure.tight_layout(); _save(figure, figure_dir / "figure2_oracle_gap_vs_spare")

    figure, axis = plt.subplots(figsize=(7.5, 5))
    for variant in analysis["variants"]:
        points = [(r["pairwise_ordering_accuracy_capped"], r["capacity"])
                  for r in analysis["ranking_diagnostics"]
                  if r["variant"] == variant and int(r["capacity"]) != 8]
        if points:
            axis.plot([p[1] for p in points], [100 * p[0] for p in points], marker="o", label=variant)
    axis.set_xlabel("Cache capacity B")
    axis.set_ylabel("Pairwise ordering accuracy, capped target (%)")
    axis.grid(alpha=0.25); axis.legend(fontsize=7)
    figure.tight_layout(); _save(figure, figure_dir / "figure3_ranking_accuracy")


def _save(figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _filtered_file(path: Path, wanted: set[str]) -> list[dict[str, Any]]:
    import json
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("condition_id")) in wanted:
                    rows.append(row)
    return rows


def _lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[str(row["condition_id"])].append(dict(row))
    for values in output.values():
        values.sort(key=lambda r: int(r["sequence_position"]))
    return output


def _multiplicities(domains, replicates, seed) -> dict[int, np.ndarray]:
    by_domain: dict[str, list[int]] = defaultdict(list)
    for sequence, domain in domains.items():
        by_domain[domain].append(sequence)
    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    index = np.arange(replicates, dtype=np.int64)[:, None]
    for domain in sorted(by_domain):
        ids = np.asarray(sorted(by_domain[domain]), dtype=np.int64)
        counts = np.zeros((replicates, len(ids)), dtype=np.int32)
        np.add.at(counts, (index, rng.integers(0, len(ids), size=counts.shape)), 1)
        for position, sequence in enumerate(ids):
            out[int(sequence)] = counts[:, position]
    return out


def _divide(numerator, denominator):
    out = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _quantile(values, probability) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, probability)) if finite.size else float("nan")


def _seed(seed: int, label: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"race-stage3-bootstrap-v1\0{seed}\0{label}".encode()).digest()[:8], "little")
