"""Stage 2C analysis driver, tables, decisions, figures, and summaries.

Reads only frozen allocations, the frozen fragility record, and saved
per-example loss checkpoints; computes the preregistered statistics; and
writes machine-readable results, the Stage 2C decision file, and
matplotlib-only figures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fragility import STAGE2C_REGIMES, STAGE2C_STAGE, load_frozen_fragility
from .fragility_evaluation import stage2c_phase_records
from .fragility_optimization import (
    FRAGILITY_ROBUST_METHOD,
    load_frozen_stage2c_registry,
    load_stage2c_allocation,
    predicted_residual_risk,
)
from .fragility_statistics import (
    STAGE2C_BOOTSTRAP_REPLICATES,
    STAGE2C_BOOTSTRAP_SEED,
    STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
    fragility_transfer_check,
    protection_shift_analysis,
    stage2c_development_decision,
    stage2c_development_gates,
    stage2c_final_decision,
    stage2c_final_regime_assessment,
)
from .io_utils import atomic_save_npz, atomic_write_json, read_json, write_csv
from .protection_evaluation import allocation_slug, load_allocation_losses
from .protection_optimization import PROTECTION_FRACTIONS
from .protection_reporting import result_csv_fields
from .protection_statistics import (
    MethodStatistics,
    build_replicate_indices,
    compute_method_statistics,
    mean_random_replicate_worst,
    paired_comparison,
    random_baseline_statistics,
)
from .specialist_preservation import STAGE2B_DOMAINS

BASE_REFERENCE_BY_REGIME = {
    "4to8": "uniform_4bit_reference",
    "3to8": "uniform_3bit_reference",
}
PAIRED_COMPARATORS = (
    "robust_functional",
    "global_importance",
    "average_specialization",
    "robust_routing",
)
PHASE_DIRS = {"development": "development_seed45", "final": "final_seed44"}
DIAGNOSTIC_METHODS = (
    "fragility_robust",
    "robust_functional",
    "average_specialization",
    "global_importance",
)
FIGURE_METHODS = (
    ("global_importance", "Global-Importance"),
    ("average_specialization", "Average-Specialization"),
    ("robust_functional", "Robust-Functional"),
    ("fragility_robust", "Fragility-Robust"),
)


def phase_dir_name(phase: str) -> str:
    if phase not in PHASE_DIRS:
        raise ValueError(f"Unknown Stage 2C phase {phase!r}")
    return PHASE_DIRS[phase]


def analyze_stage2c_phase(
    phase: str,
    results_dir: Path,
    stage2b_allocations_dir: Path,
    run_fingerprint: str,
    authorized_regimes: list[str] | None = None,
    replicates: int = STAGE2C_BOOTSTRAP_REPLICATES,
    seed: int = STAGE2C_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute every preregistered Stage 2C statistic for one phase."""

    allocations_dir = results_dir / "allocations"
    registry = load_frozen_stage2c_registry(allocations_dir, stage2b_allocations_dir)
    fragility_record = load_frozen_fragility(results_dir / "calibration")
    if registry["fragility_sha256"] != fragility_record["fragility_sha256"]:
        raise RuntimeError("Registry and frozen fragility record disagree")
    references, competitors = stage2c_phase_records(
        registry, allocations_dir, stage2b_allocations_dir, phase, authorized_regimes
    )
    losses_dir = results_dir / phase_dir_name(phase) / "losses"

    reference_by_method = {record["method"]: record for record in references}
    bf16_losses = load_allocation_losses(
        losses_dir, reference_by_method["bf16_reference"], run_fingerprint
    )
    bf16_nll = {
        domain: statistics.per_token_nll for domain, statistics in bf16_losses.items()
    }
    example_counts = {domain: len(values) for domain, values in bf16_nll.items()}
    indices = build_replicate_indices(example_counts, replicates, seed)

    base_nll_by_regime: dict[str, dict[str, np.ndarray]] = {}
    reference_stats: dict[str, MethodStatistics] = {}
    per_example: dict[str, np.ndarray] = {}
    for domain, values in bf16_nll.items():
        per_example[f"bf16_reference__{domain}"] = values
    for record in references:
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        nll = {domain: item.per_token_nll for domain, item in losses.items()}
        for regime, reference_name in BASE_REFERENCE_BY_REGIME.items():
            if record["method"] == reference_name:
                base_nll_by_regime[regime] = nll
        if record["method"] != "bf16_reference":
            for domain, values in nll.items():
                per_example[f"{record['method']}__{domain}"] = values
        reference_stats[record["method"]] = compute_method_statistics(
            record, nll, bf16_nll, nll, indices
        )

    statistics: dict[tuple[str, float], dict[str, MethodStatistics]] = {}
    for record in competitors:
        key = (record["regime"], record["budget_fraction"])
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        nll = {domain: item.per_token_nll for domain, item in losses.items()}
        for domain, values in nll.items():
            per_example[f"{allocation_slug(record)}__{domain}"] = values
        statistics.setdefault(key, {})[record["method"]] = compute_method_statistics(
            record, nll, bf16_nll, base_nll_by_regime[record["regime"]], indices
        )

    comparisons: list[dict[str, Any]] = []
    random_summaries: list[dict[str, Any]] = []
    for key, methods in sorted(statistics.items()):
        regime, budget = key
        fragility_robust = methods[FRAGILITY_ROBUST_METHOD]
        randoms = [
            item for name, item in methods.items() if name.startswith("random_seed")
        ]
        random_summary = random_baseline_statistics(randoms)
        random_summaries.append(
            {
                "regime": regime,
                "budget_fraction": budget,
                **{k: v for k, v in random_summary.items() if not isinstance(v, dict)},
                **{
                    f"worst_{name}": value
                    for name, value in random_summary[
                        "individual_worst_relative_delta"
                    ].items()
                },
            }
        )
        random_worst_reps = mean_random_replicate_worst(randoms)
        random_point = random_summary["mean_worst_relative_delta"]
        difference = fragility_robust.replicate_worst - random_worst_reps
        low = float(np.quantile(difference, 0.025))
        high = float(np.quantile(difference, 0.975))
        comparisons.append(
            {
                "first": FRAGILITY_ROBUST_METHOD,
                "second": "random_mean",
                "regime": regime,
                "budget_fraction": budget,
                "metric": "worst_relative_delta",
                "difference": fragility_robust.worst_relative_delta - random_point,
                "difference_ci_low": low,
                "difference_ci_high": high,
                "favors_first": bool(
                    fragility_robust.worst_relative_delta < random_point
                ),
                "ci_excludes_zero": bool(high < 0 or low > 0),
                "ci_favors_first": bool(high < 0),
            }
        )
        for other in PAIRED_COMPARATORS:
            for metric in ("worst_relative_delta", "mean_relative_delta"):
                comparisons.append(
                    paired_comparison(fragility_robust, methods[other], metric)
                )

    analysis: dict[str, Any] = {
        "stage": STAGE2C_STAGE,
        "phase": phase,
        "run_fingerprint": run_fingerprint,
        "registry_sha256": registry["registry_sha256"],
        "fragility_sha256": fragility_record["fragility_sha256"],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "example_counts": example_counts,
        "reference_rows": [item.summary_row() for item in reference_stats.values()],
        "method_rows": [
            item.summary_row()
            for _, methods in sorted(statistics.items())
            for item in methods.values()
        ],
        "comparisons": comparisons,
        "random_baseline_summaries": random_summaries,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    q_norm_by_regime = {
        regime: {
            domain: fragility_record["regimes"][regime]["domains"][domain][
                "normalized_fragility"
            ]
            for domain in STAGE2B_DOMAINS
        }
        for regime in STAGE2C_REGIMES
        if fragility_record["regimes"][regime]["regime_valid"]
    }

    if phase == "development":
        gates_by_regime = {}
        for regime in STAGE2C_REGIMES:
            if regime not in q_norm_by_regime:
                continue
            methods = statistics[(regime, STAGE2C_DEVELOPMENT_BUDGET_FRACTION)]
            gates_by_regime[regime] = stage2c_development_gates(
                methods[FRAGILITY_ROBUST_METHOD],
                methods["robust_functional"],
                [i for n, i in methods.items() if n.startswith("random_seed")],
                methods["global_importance"],
                methods["average_specialization"],
            )
        analysis["development_gates"] = gates_by_regime
        analysis["development_decision"] = stage2c_development_decision(gates_by_regime)
    else:
        assessments = {}
        for regime in authorized_regimes or []:
            comparisons_vs_average = {}
            comparisons_vs_global = {}
            for budget in PROTECTION_FRACTIONS:
                for comparison in comparisons:
                    if (
                        comparison["regime"] == regime
                        and comparison["budget_fraction"] == budget
                        and comparison["metric"] == "worst_relative_delta"
                    ):
                        if comparison["second"] == "average_specialization":
                            comparisons_vs_average[budget] = comparison
                        elif comparison["second"] == "global_importance":
                            comparisons_vs_global[budget] = comparison
            assessments[regime] = stage2c_final_regime_assessment(
                statistics,
                comparisons_vs_average,
                comparisons_vs_global,
                regime,
                PROTECTION_FRACTIONS,
            )
        analysis["final_regime_assessments"] = assessments
        analysis["final_decision"] = stage2c_final_decision(assessments)

        coverage_by_method: dict[str, dict[str, dict[str, float]]] = {}
        for method in (FRAGILITY_ROBUST_METHOD, "robust_functional"):
            coverage_by_method[method] = {}
            for regime in authorized_regimes or []:
                record = _find_allocation(
                    registry,
                    allocations_dir,
                    stage2b_allocations_dir,
                    method,
                    regime,
                    STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
                )
                coverage_by_method[method][regime] = dict(
                    record["functional_specialist_coverage"]
                )
        analysis["protection_shift_analysis"] = protection_shift_analysis(
            coverage_by_method[FRAGILITY_ROBUST_METHOD],
            coverage_by_method["robust_functional"],
            {
                regime: q_norm_by_regime[regime]
                for regime in (authorized_regimes or [])
            },
        )
        analysis["fragility_transfer_check"] = fragility_transfer_check(
            {
                regime: q_norm_by_regime[regime]
                for regime in (authorized_regimes or [])
            },
            {
                regime: dict(
                    reference_stats[BASE_REFERENCE_BY_REGIME[regime]].relative_delta
                )
                for regime in (authorized_regimes or [])
            },
        )

    analysis["_per_example"] = per_example
    analysis["_statistics"] = statistics
    analysis["_reference_statistics"] = reference_stats
    return analysis


def _find_allocation(
    registry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    method: str,
    regime: str,
    budget_fraction: float,
) -> dict[str, Any]:
    for entry in list(registry["new_entries"]) + list(registry["reused_entries"]):
        if (
            entry["method"] == method
            and entry["regime"] == regime
            and entry["budget_fraction"] == budget_fraction
        ):
            return load_stage2c_allocation(
                entry, allocations_dir, stage2b_allocations_dir
            )
    raise RuntimeError(
        f"Allocation {method}/{regime}/{budget_fraction} is not in the registry"
    )


def write_stage2c_phase_outputs(
    analysis: dict[str, Any], output_dir: Path
) -> dict[str, Path]:
    """Persist one phase's machine-readable outputs."""

    phase = analysis["phase"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    per_example = analysis.pop("_per_example")
    analysis.pop("_statistics", None)
    analysis.pop("_reference_statistics", None)

    rows = analysis["reference_rows"] + analysis["method_rows"]
    csv_path = output_dir / f"{phase}_results.csv"
    write_csv(csv_path, rows, result_csv_fields())
    paths["results_csv"] = csv_path
    json_path = output_dir / f"{phase}_results.json"
    atomic_write_json(json_path, analysis)
    paths["results_json"] = json_path
    losses_path = output_dir / f"{phase}_per_example_losses.npz"
    atomic_save_npz(losses_path, **per_example)
    paths["per_example_losses"] = losses_path

    comparisons_path = output_dir / "bootstrap_comparisons.csv"
    write_csv(
        comparisons_path,
        analysis["comparisons"],
        [
            "first", "second", "regime", "budget_fraction", "metric", "difference",
            "difference_ci_low", "difference_ci_high", "favors_first",
            "ci_excludes_zero", "ci_favors_first",
        ],
    )
    paths["comparisons_csv"] = comparisons_path
    return paths


def write_stage2c_development_decision(
    analysis: Mapping[str, Any], results_dir: Path, preregistration_sha256: str
) -> Path:
    payload = {
        "stage": STAGE2C_STAGE,
        "phase": "development",
        "decision": analysis["development_decision"]["decision"],
        "development_decision": analysis["development_decision"],
        "development_gates": analysis["development_gates"],
        "run_fingerprint": analysis["run_fingerprint"],
        "registry_sha256": analysis["registry_sha256"],
        "fragility_sha256": analysis["fragility_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "bootstrap_replicates": analysis["bootstrap_replicates"],
        "bootstrap_seed": analysis["bootstrap_seed"],
        "method_never_modified_by_gate": True,
        "seed44_untouched_at_decision_time": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = results_dir / "stage2c_decision.json"
    atomic_write_json(path, payload)
    return path


def write_stage2c_final_decision(
    analysis: Mapping[str, Any], results_dir: Path
) -> Path:
    path = results_dir / "stage2c_decision.json"
    payload = read_json(path)
    if payload.get("decision") != "FINAL_CONFIRMATION_GO":
        raise RuntimeError(
            "Final results cannot be recorded without FINAL_CONFIRMATION_GO"
        )
    payload["phase"] = "final"
    payload["final_decision"] = analysis["final_decision"]
    payload["final_regime_assessments"] = analysis["final_regime_assessments"]
    payload["final_run_fingerprint"] = analysis["run_fingerprint"]
    payload["final_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, payload)
    return path


def write_preevaluation_diagnostics(
    results_dir: Path, stage2b_allocations_dir: Path
) -> tuple[Path, Path]:
    """Optimization-space diagnostic table written before any seed-45 NLL.

    Compares Fragility-Robust, Robust-Functional, Average-Specialization, and
    Global-Importance coverage plus predicted maximum residual risk. This is
    score-space information only and must not alter Stage 2C.
    """

    allocations_dir = results_dir / "allocations"
    registry = load_frozen_stage2c_registry(allocations_dir, stage2b_allocations_dir)
    fragility_record = load_frozen_fragility(results_dir / "calibration")
    rows: list[dict[str, Any]] = []
    lines = [
        "# Stage 2C pre-evaluation diagnostics (optimization space only)",
        "",
        "Generated before any seed-45 development NLL was inspected.",
        "",
        "| Regime | Budget | Method | General | Math | Coding | Reasoning | "
        "Min Coverage | Predicted Max Residual Risk |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for regime in STAGE2C_REGIMES:
        if not fragility_record["regimes"][regime]["regime_valid"]:
            continue
        q_norm = np.asarray(
            [
                fragility_record["regimes"][regime]["domains"][domain][
                    "normalized_fragility"
                ]
                for domain in STAGE2B_DOMAINS
            ]
        )
        for fraction in PROTECTION_FRACTIONS:
            for method in DIAGNOSTIC_METHODS:
                record = _find_allocation(
                    registry, allocations_dir, stage2b_allocations_dir,
                    method, regime, fraction,
                )
                coverage = np.asarray(
                    [
                        record["functional_specialist_coverage"][domain]
                        for domain in STAGE2B_DOMAINS
                    ]
                )
                max_risk = float(predicted_residual_risk(q_norm, coverage).max())
                row = {
                    "regime": regime,
                    "budget_fraction": fraction,
                    "method": method,
                    **{
                        f"coverage_{domain}": float(coverage[index])
                        for index, domain in enumerate(STAGE2B_DOMAINS)
                    },
                    "coverage_min": float(coverage.min()),
                    "predicted_max_residual_risk": max_risk,
                }
                rows.append(row)
                lines.append(
                    f"| {regime} | {int(round(fraction * 100))}% | {method} | "
                    + " | ".join(f"{value:.4f}" for value in coverage)
                    + f" | {coverage.min():.4f} | {max_risk:.4f} |"
                )
    csv_path = results_dir / "preevaluation_diagnostics.csv"
    write_csv(
        csv_path,
        rows,
        [
            "regime", "budget_fraction", "method",
            *[f"coverage_{domain}" for domain in STAGE2B_DOMAINS],
            "coverage_min", "predicted_max_residual_risk",
        ],
    )
    md_path = results_dir / "preevaluation_diagnostics.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def render_stage2c_table(analysis: Mapping[str, Any], value_prefix: str) -> list[str]:
    """Markdown table per regime/budget with domain values and worst column."""

    lines: list[str] = []
    rows = analysis["method_rows"]
    budgets = sorted({row["budget_fraction"] for row in rows})
    for regime in STAGE2C_REGIMES:
        for budget in budgets:
            block = [
                row
                for row in rows
                if row["regime"] == regime and row["budget_fraction"] == budget
            ]
            if not block:
                continue
            lines.append(
                f"### {regime}, {int(round(budget * 100))}% protection budget"
            )
            lines.append("")
            lines.append(
                "| Method | General | Math | Coding | Reasoning | Mean | Worst |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in block:
                key = "relative_delta" if value_prefix == "relative" else "delta_nll"
                mean_key = (
                    "mean_relative_delta"
                    if value_prefix == "relative"
                    else "mean_raw_delta_nll"
                )
                worst_key = (
                    "worst_relative_delta"
                    if value_prefix == "relative"
                    else "worst_raw_delta_nll"
                )
                values = [row[f"{key}_{domain}"] for domain in STAGE2B_DOMAINS]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row["method_label"],
                            *[f"{value:+.6f}" for value in values],
                            f"{row[mean_key]:+.6f}",
                            f"{row[worst_key]:+.6f}",
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return lines


def write_stage2c_summary(results_dir: Path) -> Path:
    """Write SUMMARY.md describing whatever Stage 2C artifacts currently exist."""

    def _optional(path: Path) -> Any | None:
        return read_json(path) if path.is_file() else None

    fragility = _optional(results_dir / "calibration" / "calibration_fragility.json")
    registry = _optional(results_dir / "allocations" / "allocation_registry.json")
    splits = _optional(results_dir / "splits" / "split_manifest.json")
    decision = _optional(results_dir / "stage2c_decision.json")
    development = _optional(
        results_dir / PHASE_DIRS["development"] / "development_results.json"
    )
    final = _optional(results_dir / PHASE_DIRS["final"] / "final_results.json")
    audit = _optional(results_dir / "audits" / "independent_audit.json")

    lines: list[str] = [
        "# Stage 2C: Fragility-weighted robust specialist preservation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Stage 2C balances predicted residual domain vulnerability",
        "(ResidualRisk_d = q_norm[d] * (1 - Coverage_d)) instead of raw",
        "specialist coverage. Fragility comes only from domain-level uniform",
        "base-precision calibration NLL; no expert-level delta-NLL surrogate is",
        "used, and the frozen Stage 2A SURROGATE_NO_GO and Stage 2B",
        "ROBUST_PRESERVATION_NO_GO decisions are preserved unchanged.",
        "",
        "Storage numbers are exact projected format bytes for QDQ-simulated",
        "formats; no runtime speedup, latency, or measured-memory claim is made.",
        "",
    ]
    if fragility is not None:
        lines += ["## Calibration fragility", ""]
        for regime in STAGE2C_REGIMES:
            entry = fragility["regimes"][regime]
            lines.append(
                f"- {regime} (base {entry['base_bits']}-bit), regime valid: "
                f"{entry['regime_valid']}"
            )
            for domain in STAGE2B_DOMAINS:
                values = entry["domains"][domain]
                normalized = values["normalized_fragility"]
                normalized_text = (
                    f"{normalized:.4f}" if normalized is not None else "n/a"
                )
                lines.append(
                    f"  - {domain}: BF16 NLL {values['bf16_nll']:.6f}, base NLL "
                    f"{values['base_nll']:.6f}, relative fragility "
                    f"{values['relative_delta']:+.6f}, normalized {normalized_text}"
                )
        lines += [
            "",
            f"- Fragility SHA-256: `{fragility['fragility_sha256']}`",
            "",
        ]
    if registry is not None:
        lines += [
            "## Frozen allocations",
            "",
            f"- New Fragility-Robust allocations: {len(registry['new_entries'])}",
            f"- Reused frozen Stage 2B comparators: {len(registry['reused_entries'])}",
            f"- Registry SHA-256: `{registry['registry_sha256']}`",
            f"- Stage 2B registry SHA-256: `{registry['stage2b_registry_sha256']}`",
            "",
        ]
    if splits is not None:
        lines += [
            "## Seed-45 development split",
            "",
            f"- Seed: {splits['development_seed']}, "
            f"{splits['examples_per_domain']} examples/domain, "
            f"{splits['measured_tokens_per_example']} measured positions/example",
        ]
        for domain, entry in splits["domains"].items():
            lines.append(
                f"- {domain}: prior pool fully excluded = "
                f"{entry['prior_pool_fully_excluded']}; disjointness verified = "
                f"{entry['disjointness_verified']}"
            )
        lines.append("")
    if development is not None:
        lines += ["## Development results (seed 45, 20% budget)", ""]
        lines += render_stage2c_table(development, "relative")
    if decision is not None:
        lines += [f"**Stage 2C decision: {decision['decision']}**", ""]
        for regime, gates in decision.get("development_gates", {}).items():
            summary = ", ".join(
                f"{key}={'PASS' if gates[key]['passed'] else 'FAIL'}"
                for key in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e")
            )
            lines.append(f"- {regime}: {summary}")
        lines.append("")
    if final is not None:
        lines += [
            "## Final results (seed 44)",
            "",
            f"**Final decision: {final['final_decision']['decision']}**",
            "",
            "See `final_seed44/FINAL_SUMMARY.md` for the complete tables.",
            "",
        ]
    if audit is not None:
        lines += [
            "## Independent audit",
            "",
            f"- Passed: {audit.get('passed')}",
            f"- Checks: {audit.get('checks_passed')} passed, "
            f"{audit.get('checks_failed')} failed",
            "",
        ]
    path = results_dir / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_stage2c_final_summary(
    results_dir: Path, analysis: Mapping[str, Any]
) -> Path:
    lines: list[str] = [
        "# Stage 2C final evaluation summary (seed 44)",
        "",
        f"Final decision: **{analysis['final_decision']['decision']}**",
        "",
        analysis["final_decision"]["rule"],
        "",
        "## Relative NLL degradation (RelativeDelta) by regime and budget",
        "",
    ]
    lines += render_stage2c_table(analysis, "relative")
    lines += ["## Raw delta NLL by regime and budget", ""]
    lines += render_stage2c_table(analysis, "raw")
    lines += ["## Regime assessments", ""]
    for regime, assessment in analysis["final_regime_assessments"].items():
        lines += [
            f"### {regime}",
            "",
            f"- Budgets with all-four point wins: "
            f"{assessment['budgets_with_all_four_point_wins']} of 4",
            f"- Average worst-domain improvement over Average-Specialization: "
            f"{assessment['average_improvement_over_average_specialization']:+.6f}",
            f"- CI wins vs Average-Specialization: "
            f"{assessment['ci_wins_vs_average_specialization']}",
            f"- Point wins vs Global-Importance: "
            f"{assessment['point_wins_vs_global_importance']}",
            f"- Systematic negative-recovery domains: "
            f"{assessment['systematic_negative_recovery_domains'] or 'none'}",
            f"- Strong success: {assessment['strong_success']}",
            f"- Qualified success: {assessment['qualified_success']}",
            "",
        ]
    lines += ["## Mechanism: protection shift vs fragility", ""]
    for regime, entry in analysis["protection_shift_analysis"]["regimes"].items():
        lines.append(
            f"- {regime}: Spearman(fragility, coverage shift) = "
            f"{entry['spearman_fragility_vs_shift']:+.4f}"
        )
        for domain in STAGE2B_DOMAINS:
            lines.append(
                f"  - {domain}: shift {entry['protection_shift_by_domain'][domain]:+.4f}, "
                f"q_norm {entry['q_norm_by_domain'][domain]:.4f}"
            )
    lines.append("")
    lines += ["## Fragility transfer (calibration vs seed-44 uniform base)", ""]
    for regime, entry in analysis["fragility_transfer_check"]["regimes"].items():
        lines.append(
            f"- {regime}: Spearman = {entry['spearman']:+.4f}; most fragile "
            f"(calibration) = {entry['most_fragile_domain_calibration']}, most "
            f"degraded (final) = {entry['most_degraded_domain_final']}"
        )
    lines.append("")
    path = results_dir / PHASE_DIRS["final"] / "FINAL_SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _figure_targets(base: Path, name: str) -> list[Path]:
    return [base / f"{name}.png", base / f"{name}.pdf"]


def create_stage2c_figures(
    results_dir: Path,
    stage2b_allocations_dir: Path,
    figures_dir: Path,
    analysis: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Matplotlib-only Stage 2C figures; generates whatever data allows."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    fragility_path = results_dir / "calibration" / "calibration_fragility.json"
    fragility_record = (
        load_frozen_fragility(results_dir / "calibration")
        if fragility_path.is_file()
        else None
    )
    allocations_dir = results_dir / "allocations"
    registry = (
        load_frozen_stage2c_registry(allocations_dir, stage2b_allocations_dir)
        if (allocations_dir / "allocation_registry.json").is_file()
        else None
    )
    positions = np.arange(len(STAGE2B_DOMAINS))
    domain_labels = [d.capitalize() for d in STAGE2B_DOMAINS]

    # Figure 1: calibration domain fragility for both base precisions.
    if fragility_record is not None:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for axis, regime in zip(axes, STAGE2C_REGIMES, strict=True):
            entry = fragility_record["regimes"][regime]
            values = [
                entry["domains"][domain]["relative_delta"]
                for domain in STAGE2B_DOMAINS
            ]
            axis.bar(positions, values, color="tab:blue")
            axis.set_xticks(positions, domain_labels, rotation=15)
            axis.set_title(
                f"Base {entry['base_bits']}-bit ({regime}); valid={entry['regime_valid']}"
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            for index, domain in enumerate(STAGE2B_DOMAINS):
                normalized = entry["domains"][domain]["normalized_fragility"]
                if normalized is not None:
                    axis.annotate(
                        f"q̃={normalized:.2f}",
                        (index, values[index]),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
        axes[0].set_ylabel("Calibration relative fragility (raw)")
        figure.suptitle("Figure 1: Calibration domain fragility")
        figure.tight_layout()
        for path in _figure_targets(figures_dir, "fig1_calibration_fragility"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 2: coverage shift Robust-Functional vs Fragility-Robust at 20%.
    if registry is not None and fragility_record is not None:
        for regime in STAGE2C_REGIMES:
            if not fragility_record["regimes"][regime]["regime_valid"]:
                continue
            robust = _find_allocation(
                registry, allocations_dir, stage2b_allocations_dir,
                "robust_functional", regime, STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
            )
            fragile = _find_allocation(
                registry, allocations_dir, stage2b_allocations_dir,
                FRAGILITY_ROBUST_METHOD, regime, STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
            )
            figure, axis = plt.subplots(figsize=(8, 4.5))
            width = 0.35
            robust_values = [
                robust["functional_specialist_coverage"][d] for d in STAGE2B_DOMAINS
            ]
            fragile_values = [
                fragile["functional_specialist_coverage"][d] for d in STAGE2B_DOMAINS
            ]
            axis.bar(
                positions - width / 2, robust_values, width,
                label="Robust-Functional (Stage 2B)",
            )
            axis.bar(
                positions + width / 2, fragile_values, width,
                label="Fragility-Robust (Stage 2C)",
            )
            for index, domain in enumerate(STAGE2B_DOMAINS):
                q_norm = fragility_record["regimes"][regime]["domains"][domain][
                    "normalized_fragility"
                ]
                axis.annotate(
                    f"q̃={q_norm:.2f}",
                    (index, max(robust_values[index], fragile_values[index])),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            axis.set_xticks(positions, domain_labels)
            axis.set_ylabel("Functional specialist coverage")
            axis.set_title(f"Figure 2: Coverage shift at 20% budget ({regime})")
            axis.legend(fontsize=8)
            figure.tight_layout()
            for path in _figure_targets(figures_dir, f"fig2_coverage_shift_{regime}"):
                figure.savefig(path, dpi=200)
                created.append(path)
            plt.close(figure)

    # Figure 5: fragility vs protection shift (score space).
    if registry is not None and fragility_record is not None:
        figure, axis = plt.subplots(figsize=(6.5, 5))
        plotted = False
        for regime in STAGE2C_REGIMES:
            if not fragility_record["regimes"][regime]["regime_valid"]:
                continue
            robust = _find_allocation(
                registry, allocations_dir, stage2b_allocations_dir,
                "robust_functional", regime, STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
            )
            fragile = _find_allocation(
                registry, allocations_dir, stage2b_allocations_dir,
                FRAGILITY_ROBUST_METHOD, regime, STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
            )
            xs, ys = [], []
            for domain in STAGE2B_DOMAINS:
                xs.append(
                    fragility_record["regimes"][regime]["domains"][domain][
                        "normalized_fragility"
                    ]
                )
                ys.append(
                    fragile["functional_specialist_coverage"][domain]
                    - robust["functional_specialist_coverage"][domain]
                )
            axis.scatter(xs, ys, label=regime)
            for x, y, domain in zip(xs, ys, STAGE2B_DOMAINS, strict=True):
                axis.annotate(f"{domain} ({regime})", (x, y), fontsize=7, alpha=0.85)
            plotted = True
        if plotted:
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xlabel("Calibration normalized fragility q̃")
            axis.set_ylabel("Coverage change vs Robust-Functional (20% budget)")
            axis.set_title("Figure 5: Fragility vs protection shift")
            axis.legend(fontsize=8)
            figure.tight_layout()
            for path in _figure_targets(figures_dir, "fig5_fragility_vs_shift"):
                figure.savefig(path, dpi=200)
                created.append(path)
        plt.close(figure)

    if analysis is None:
        return created
    method_rows = analysis["method_rows"]
    phase = analysis["phase"]

    # Figure 4: domain degradation profiles at the 20% budget.
    regimes_present = sorted({row["regime"] for row in method_rows})
    for regime in regimes_present:
        block = [
            row
            for row in method_rows
            if row["regime"] == regime
            and row["budget_fraction"] == STAGE2C_DEVELOPMENT_BUDGET_FRACTION
        ]
        if not block:
            continue
        figure, axis = plt.subplots(figsize=(8.5, 4.5))
        width = 0.2
        for offset, (method, label) in enumerate(FIGURE_METHODS):
            row = next(r for r in block if r["method"] == method)
            values = [row[f"relative_delta_{d}"] for d in STAGE2B_DOMAINS]
            axis.bar(positions + (offset - 1.5) * width, values, width, label=label)
        axis.set_xticks(positions, domain_labels)
        axis.set_ylabel("Relative NLL degradation")
        axis.set_title(f"Figure 4: Domain degradation at 20% budget ({regime}, {phase})")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(figures_dir, f"fig4_domain_profiles_{regime}_{phase}"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    if phase != "final" or registry is None:
        return created

    # Figure 3: headline worst-domain quality/memory curve (final only).
    curve_methods = list(FIGURE_METHODS) + [("random_mean", "Random mean")]
    records_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}
    for entry in list(registry["new_entries"]) + list(registry["reused_entries"]):
        if entry["method_kind"] == "uniform_reference":
            continue
        records_by_key[
            (entry["method"], entry["regime"], entry["budget_fraction"])
        ] = load_stage2c_allocation(entry, allocations_dir, stage2b_allocations_dir)
    for regime in regimes_present:
        figure, axis = plt.subplots(figsize=(7, 5))
        for method, label in curve_methods:
            points = []
            for budget in PROTECTION_FRACTIONS:
                if method == "random_mean":
                    randoms = [
                        row
                        for row in method_rows
                        if row["regime"] == regime
                        and row["budget_fraction"] == budget
                        and row["method"].startswith("random_seed")
                    ]
                    if not randoms:
                        continue
                    worst = float(
                        np.mean([row["worst_relative_delta"] for row in randoms])
                    )
                    bits = float(
                        np.mean(
                            [
                                records_by_key[(row["method"], regime, budget)][
                                    "effective_bits_per_weight"
                                ]
                                for row in randoms
                            ]
                        )
                    )
                else:
                    matches = [
                        row
                        for row in method_rows
                        if row["method"] == method
                        and row["regime"] == regime
                        and row["budget_fraction"] == budget
                    ]
                    if not matches:
                        continue
                    worst = matches[0]["worst_relative_delta"]
                    bits = records_by_key[(method, regime, budget)][
                        "effective_bits_per_weight"
                    ]
                points.append((bits, worst))
            points.sort()
            axis.plot(
                [p[0] for p in points], [p[1] for p in points], marker="o", label=label
            )
        axis.set_xlabel("Effective expert-weight bits per weight (projected)")
        axis.set_ylabel("Worst-domain relative NLL degradation")
        axis.set_title(f"Figure 3: Worst-domain quality vs memory ({regime})")
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(figures_dir, f"fig3_quality_memory_{regime}"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 6: calibration fragility vs seed-44 uniform-base vulnerability.
    if fragility_record is not None and "fragility_transfer_check" in analysis:
        figure, axis = plt.subplots(figsize=(6.5, 5))
        for regime, entry in analysis["fragility_transfer_check"]["regimes"].items():
            xs = [
                fragility_record["regimes"][regime]["domains"][domain][
                    "relative_delta"
                ]
                for domain in STAGE2B_DOMAINS
            ]
            ys = [
                entry["final_uniform_base_relative_delta"][domain]
                for domain in STAGE2B_DOMAINS
            ]
            axis.scatter(xs, ys, label=f"{regime} (Spearman {entry['spearman']:+.2f})")
            for x, y, domain in zip(xs, ys, STAGE2B_DOMAINS, strict=True):
                axis.annotate(f"{domain}", (x, y), fontsize=7, alpha=0.85)
        axis.set_xlabel("Calibration relative fragility (raw)")
        axis.set_ylabel("Seed-44 uniform-base relative NLL degradation")
        axis.set_title("Figure 6: Calibration vs final domain vulnerability")
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(figures_dir, "fig6_calibration_vs_final"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)
    return created
