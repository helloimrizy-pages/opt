"""Stage 2B analysis driver, tables, decisions, and figures.

Reads only frozen allocations plus saved per-example loss checkpoints, computes
the preregistered statistics, and writes machine-readable results, decision
files, and matplotlib-only figures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .io_utils import atomic_save_npz, atomic_write_json, write_csv
from .protection_allocations import load_allocation, load_frozen_registry
from .protection_evaluation import allocation_slug, load_allocation_losses
from .protection_optimization import BASE_BITS_BY_REGIME, PROTECTION_FRACTIONS
from .protection_statistics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEVELOPMENT_BUDGET_FRACTION,
    MethodStatistics,
    build_replicate_indices,
    compute_method_statistics,
    coverage_recovery_diagnostic,
    development_decision,
    development_gates,
    final_decision,
    final_regime_assessment,
    mean_random_replicate_worst,
    paired_comparison,
    random_baseline_statistics,
)
from .specialist_preservation import STAGE2B_DOMAINS

BASE_REFERENCE_BY_REGIME = {
    "4to8": "uniform_4bit_reference",
    "3to8": "uniform_3bit_reference",
}
COMPARATOR_METHODS = (
    "global_importance",
    "average_specialization",
    "general_only",
    "math_only",
    "coding_only",
    "reasoning_only",
    "robust_routing",
)
SINGLE_DOMAIN_ROWS = ("general_only", "math_only", "coding_only", "reasoning_only")


def phase_records(
    registry: Mapping[str, Any], allocations_dir: Path, phase: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (uniform reference records, matched-budget records) for a phase."""

    references: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for entry in registry["entries"]:
        record = load_allocation(allocations_dir, entry["file"])
        if record["method_kind"] == "uniform_reference":
            references.append(record)
        elif phase == "development":
            if record["budget_fraction"] == DEVELOPMENT_BUDGET_FRACTION:
                competitors.append(record)
        else:
            competitors.append(record)
    order = {"bf16_reference": 0, "uniform_8bit_reference": 1,
             "uniform_4bit_reference": 2, "uniform_3bit_reference": 3}
    references.sort(key=lambda record: order.get(record["method"], 9))
    return references, competitors


def analyze_phase(
    phase: str,
    allocations_dir: Path,
    losses_dir: Path,
    run_fingerprint: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute every preregistered statistic for one evaluation phase."""

    registry = load_frozen_registry(allocations_dir)
    references, competitors = phase_records(registry, allocations_dir, phase)
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
    for record in references:
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        nll = {domain: item.per_token_nll for domain, item in losses.items()}
        for regime, reference_name in BASE_REFERENCE_BY_REGIME.items():
            if record["method"] == reference_name:
                base_nll_by_regime[regime] = nll
        reference_stats[record["method"]] = compute_method_statistics(
            record, nll, bf16_nll, nll, indices
        )

    statistics: dict[tuple[str, float], dict[str, MethodStatistics]] = {}
    coverage_min: dict[tuple[str, float], dict[str, float]] = {}
    per_example: dict[str, np.ndarray] = {}
    for domain, values in bf16_nll.items():
        per_example[f"bf16_reference__{domain}"] = values
    for record in references:
        if record["method"] == "bf16_reference":
            continue
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        for domain, item in losses.items():
            per_example[f"{record['method']}__{domain}"] = item.per_token_nll
    for record in competitors:
        key = (record["regime"], record["budget_fraction"])
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        nll = {domain: item.per_token_nll for domain, item in losses.items()}
        for domain, values in nll.items():
            per_example[f"{allocation_slug(record)}__{domain}"] = values
        statistics.setdefault(key, {})[record["method"]] = compute_method_statistics(
            record, nll, bf16_nll, base_nll_by_regime[record["regime"]], indices
        )
        coverage_min.setdefault(key, {})[record["method"]] = record[
            "functional_specialist_coverage_min"
        ]

    comparisons: list[dict[str, Any]] = []
    random_summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for key, methods in sorted(statistics.items()):
        regime, budget = key
        robust = methods["robust_functional"]
        randoms = [
            item for name, item in methods.items() if name.startswith("random_seed")
        ]
        random_summary = random_baseline_statistics(randoms)
        random_summaries.append(
            {"regime": regime, "budget_fraction": budget, **{
                k: v for k, v in random_summary.items()
                if not isinstance(v, dict)
            },
             **{f"worst_{name}": value
                for name, value in random_summary[
                    "individual_worst_relative_delta"].items()}}
        )
        random_worst_reps = mean_random_replicate_worst(randoms)
        random_point = random_summary["mean_worst_relative_delta"]
        difference = robust.replicate_worst - random_worst_reps
        low, high = float(np.quantile(difference, 0.025)), float(
            np.quantile(difference, 0.975)
        )
        comparisons.append(
            {
                "first": "robust_functional",
                "second": "random_mean",
                "regime": regime,
                "budget_fraction": budget,
                "metric": "worst_relative_delta",
                "difference": robust.worst_relative_delta - random_point,
                "difference_ci_low": low,
                "difference_ci_high": high,
                "favors_first": bool(robust.worst_relative_delta < random_point),
                "ci_excludes_zero": bool(high < 0 or low > 0),
                "ci_favors_first": bool(high < 0),
            }
        )
        for other in COMPARATOR_METHODS:
            for metric in ("worst_relative_delta", "mean_relative_delta"):
                comparisons.append(paired_comparison(robust, methods[other], metric))
        diagnostics.append(
            {
                "regime": regime,
                "budget_fraction": budget,
                **coverage_recovery_diagnostic(methods, coverage_min[key]),
            }
        )

    analysis: dict[str, Any] = {
        "phase": phase,
        "run_fingerprint": run_fingerprint,
        "registry_sha256": registry["registry_sha256"],
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
        "coverage_recovery_diagnostics": diagnostics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if phase == "development":
        gates_by_regime = {}
        for regime in BASE_BITS_BY_REGIME:
            methods = statistics[(regime, DEVELOPMENT_BUDGET_FRACTION)]
            gates_by_regime[regime] = development_gates(
                methods["robust_functional"],
                [i for n, i in methods.items() if n.startswith("random_seed")],
                methods["global_importance"],
                methods["average_specialization"],
            )
        analysis["development_gates"] = gates_by_regime
        analysis["development_decision"] = development_decision(gates_by_regime)
    else:
        assessments = {}
        for regime in BASE_BITS_BY_REGIME:
            comparisons_vs_average = {}
            for budget in PROTECTION_FRACTIONS:
                match = [
                    c
                    for c in comparisons
                    if c["regime"] == regime
                    and c["budget_fraction"] == budget
                    and c["second"] == "average_specialization"
                    and c["metric"] == "worst_relative_delta"
                ]
                comparisons_vs_average[budget] = match[0]
            assessments[regime] = final_regime_assessment(
                statistics, comparisons_vs_average, regime, PROTECTION_FRACTIONS
            )
        analysis["final_regime_assessments"] = assessments
        analysis["final_decision"] = final_decision(assessments)
        transfer_rows = []
        for regime in BASE_BITS_BY_REGIME:
            methods = statistics[(regime, DEVELOPMENT_BUDGET_FRACTION)]
            for method in SINGLE_DOMAIN_ROWS:
                for domain in STAGE2B_DOMAINS:
                    transfer_rows.append(
                        {
                            "regime": regime,
                            "budget_fraction": DEVELOPMENT_BUDGET_FRACTION,
                            "calibration_method": method,
                            "evaluation_domain": domain,
                            "relative_delta": methods[method].relative_delta[domain],
                        }
                    )
        analysis["single_domain_transfer"] = transfer_rows

    analysis["_per_example"] = per_example
    analysis["_statistics"] = statistics
    analysis["_reference_statistics"] = reference_stats
    return analysis


def result_csv_fields() -> list[str]:
    fields = [
        "method", "method_label", "method_kind", "regime", "budget_fraction",
        "mean_relative_delta", "mean_relative_delta_ci_low",
        "mean_relative_delta_ci_high", "median_relative_delta",
        "worst_relative_delta", "worst_relative_delta_ci_low",
        "worst_relative_delta_ci_high", "worst_domain", "worst_raw_delta_nll",
        "mean_raw_delta_nll", "mean_recovery", "min_recovery",
    ]
    for domain in STAGE2B_DOMAINS:
        fields.extend(
            [
                f"relative_delta_{domain}",
                f"relative_delta_{domain}_ci_low",
                f"relative_delta_{domain}_ci_high",
                f"delta_nll_{domain}",
                f"recovery_{domain}",
                f"recovery_{domain}_ci_low",
                f"recovery_{domain}_ci_high",
            ]
        )
    return fields


def write_phase_outputs(analysis: dict[str, Any], output_dir: Path) -> dict[str, Path]:
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

    comparison_fields = [
        "first", "second", "regime", "budget_fraction", "metric", "difference",
        "difference_ci_low", "difference_ci_high", "favors_first",
        "ci_excludes_zero", "ci_favors_first",
    ]
    comparisons_path = output_dir / "bootstrap_comparisons.csv"
    write_csv(comparisons_path, analysis["comparisons"], comparison_fields)
    paths["comparisons_csv"] = comparisons_path

    if phase == "final":
        transfer_path = output_dir / "single_domain_transfer.csv"
        write_csv(
            transfer_path,
            analysis["single_domain_transfer"],
            [
                "regime", "budget_fraction", "calibration_method",
                "evaluation_domain", "relative_delta",
            ],
        )
        paths["transfer_csv"] = transfer_path
    return paths


def write_development_decision(
    analysis: Mapping[str, Any], results_root: Path
) -> Path:
    decision_payload = {
        "stage": "stage2b_robust_specialist_preservation",
        "phase": "development",
        "decision": analysis["development_decision"]["decision"],
        "development_decision": analysis["development_decision"],
        "development_gates": analysis["development_gates"],
        "run_fingerprint": analysis["run_fingerprint"],
        "registry_sha256": analysis["registry_sha256"],
        "bootstrap_replicates": analysis["bootstrap_replicates"],
        "bootstrap_seed": analysis["bootstrap_seed"],
        "method_never_modified_by_gate": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = results_root / "stage2b_decision.json"
    atomic_write_json(path, decision_payload)
    return path


def write_final_decision(analysis: Mapping[str, Any], results_root: Path) -> Path:
    path = results_root / "stage2b_decision.json"
    from .io_utils import read_json

    payload = read_json(path)
    if payload.get("decision") != "FULL_EVALUATION_GO":
        raise RuntimeError(
            "Final results cannot be recorded without a FULL_EVALUATION_GO decision"
        )
    payload["phase"] = "final"
    payload["final_decision"] = analysis["final_decision"]
    payload["final_regime_assessments"] = analysis["final_regime_assessments"]
    payload["final_run_fingerprint"] = analysis["run_fingerprint"]
    payload["final_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, payload)
    return path


def _figure_targets(base: Path, name: str) -> list[Path]:
    return [base / f"{name}.png", base / f"{name}.pdf"]


def create_phase_figures(
    analysis: Mapping[str, Any],
    allocations_dir: Path,
    figures_dir: Path,
) -> list[Path]:
    """Matplotlib-only figures; seaborn is deliberately not used."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    registry = load_frozen_registry(allocations_dir)
    records = {
        entry["file"]: load_allocation(allocations_dir, entry["file"])
        for entry in registry["entries"]
    }
    created: list[Path] = []
    method_rows = analysis["method_rows"]
    phase = analysis["phase"]
    show_methods = [
        ("global_importance", "Global-Importance"),
        ("average_specialization", "Average-Specialization"),
        ("robust_routing", "Robust-Routing"),
        ("robust_functional", "Robust-Functional"),
    ]

    # Figure 1: specialist coverage at the representative 20% budget.
    for regime in BASE_BITS_BY_REGIME:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        width = 0.2
        positions = np.arange(len(STAGE2B_DOMAINS))
        for offset, (method, label) in enumerate(show_methods):
            record = next(
                r
                for r in records.values()
                if r["method"] == method
                and r["regime"] == regime
                and r["budget_fraction"] == DEVELOPMENT_BUDGET_FRACTION
            )
            coverage = [
                record["functional_specialist_coverage"][d] for d in STAGE2B_DOMAINS
            ]
            axis.bar(positions + (offset - 1.5) * width, coverage, width, label=label)
        axis.set_xticks(positions, [d.capitalize() for d in STAGE2B_DOMAINS])
        axis.set_ylabel("Functional specialist coverage")
        axis.set_title(f"Specialist coverage at 20% budget ({regime})")
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(figures_dir, f"fig1_specialist_coverage_{regime}"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 2: worst-domain degradation versus effective bits per weight.
    if phase == "final":
        curve_methods = show_methods + [("random_mean", "Random mean")]
        for regime in BASE_BITS_BY_REGIME:
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
                        worst = float(
                            np.mean([row["worst_relative_delta"] for row in randoms])
                        )
                        bits = float(
                            np.mean(
                                [
                                    records[f"{row['method']}_{regime}_budget{int(round(budget*100))}.json"][
                                        "effective_bits_per_weight"
                                    ]
                                    for row in randoms
                                ]
                            )
                        )
                    else:
                        row = next(
                            r
                            for r in method_rows
                            if r["method"] == method
                            and r["regime"] == regime
                            and r["budget_fraction"] == budget
                        )
                        worst = row["worst_relative_delta"]
                        bits = records[
                            f"{method}_{regime}_budget{int(round(budget*100))}.json"
                        ]["effective_bits_per_weight"]
                    points.append((bits, worst))
                points.sort()
                axis.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    label=label,
                )
            axis.set_xlabel("Effective expert-weight bits per weight (projected)")
            axis.set_ylabel("Worst-domain relative NLL degradation")
            axis.set_title(f"Worst-domain degradation vs memory ({regime})")
            axis.legend(fontsize=8)
            figure.tight_layout()
            for path in _figure_targets(figures_dir, f"fig2_pareto_{regime}"):
                figure.savefig(path, dpi=200)
                created.append(path)
            plt.close(figure)

    # Figure 3: per-domain degradation profiles at 20%.
    for regime in BASE_BITS_BY_REGIME:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        width = 0.2
        positions = np.arange(len(STAGE2B_DOMAINS))
        for offset, (method, label) in enumerate(show_methods):
            row = next(
                r
                for r in method_rows
                if r["method"] == method
                and r["regime"] == regime
                and r["budget_fraction"] == DEVELOPMENT_BUDGET_FRACTION
            )
            values = [row[f"relative_delta_{d}"] for d in STAGE2B_DOMAINS]
            axis.bar(positions + (offset - 1.5) * width, values, width, label=label)
        axis.set_xticks(positions, [d.capitalize() for d in STAGE2B_DOMAINS])
        axis.set_ylabel("Relative NLL degradation")
        axis.set_title(f"Domain degradation at 20% budget ({regime})")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(figures_dir, f"fig3_domain_profiles_{regime}"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 4: single-domain transfer heatmap.
    if phase == "final":
        for regime in BASE_BITS_BY_REGIME:
            matrix = np.zeros((len(SINGLE_DOMAIN_ROWS), len(STAGE2B_DOMAINS)))
            for row in analysis["single_domain_transfer"]:
                if row["regime"] != regime:
                    continue
                i = SINGLE_DOMAIN_ROWS.index(row["calibration_method"])
                j = STAGE2B_DOMAINS.index(row["evaluation_domain"])
                matrix[i, j] = row["relative_delta"]
            figure, axis = plt.subplots(figsize=(6, 5))
            image = axis.imshow(matrix, cmap="viridis")
            axis.set_xticks(range(len(STAGE2B_DOMAINS)),
                            [d.capitalize() for d in STAGE2B_DOMAINS])
            axis.set_yticks(range(len(SINGLE_DOMAIN_ROWS)),
                            [m.replace("_only", "-only").capitalize()
                             for m in SINGLE_DOMAIN_ROWS])
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center",
                              color="white", fontsize=7)
            figure.colorbar(image, label="Relative NLL degradation")
            axis.set_title(f"Single-domain protection transfer at 20% ({regime})")
            figure.tight_layout()
            for path in _figure_targets(figures_dir, f"fig4_transfer_{regime}"):
                figure.savefig(path, dpi=200)
                created.append(path)
            plt.close(figure)

    # Figure 5: expert protection maps at 20%.
    map_methods = [
        ("global_importance", "Global-Importance"),
        ("average_specialization", "Average-Specialization"),
        ("robust_functional", "Robust-Functional"),
    ]
    for regime in BASE_BITS_BY_REGIME:
        figure, axes = plt.subplots(1, len(map_methods), figsize=(13, 4.2))
        for axis, (method, label) in zip(axes, map_methods, strict=True):
            record = next(
                r
                for r in records.values()
                if r["method"] == method
                and r["regime"] == regime
                and r["budget_fraction"] == DEVELOPMENT_BUDGET_FRACTION
            )
            bits = np.asarray(record["expert_bits"])
            protected = (bits == record["protected_bits"]).astype(float)
            axis.imshow(protected, aspect="auto", cmap="Greys", vmin=0, vmax=1)
            axis.set_title(label, fontsize=9)
            axis.set_xlabel("Expert ID")
            axis.set_ylabel("MoE layer")
        figure.suptitle(f"Protected experts at 20% budget ({regime})")
        figure.tight_layout()
        for path in _figure_targets(figures_dir, f"fig5_protection_map_{regime}"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 6: coverage versus empirical outcome diagnostic.
    figure, axis = plt.subplots(figsize=(6.5, 5))
    for regime in BASE_BITS_BY_REGIME:
        xs, ys, labels = [], [], []
        for row in method_rows:
            if row["regime"] != regime or row["method"].startswith("random_seed"):
                continue
            if phase == "final" and row["budget_fraction"] != DEVELOPMENT_BUDGET_FRACTION:
                continue
            file_name = (
                f"{row['method']}_{regime}_budget"
                f"{int(round(row['budget_fraction'] * 100))}.json"
            )
            xs.append(records[file_name]["functional_specialist_coverage_min"])
            ys.append(row["min_recovery"])
            labels.append(row["method"])
        axis.scatter(xs, ys, label=regime)
        for x, y, label in zip(xs, ys, labels, strict=True):
            axis.annotate(label, (x, y), fontsize=6, alpha=0.8)
    axis.set_xlabel("Minimum functional specialist coverage")
    axis.set_ylabel("Minimum domain recovery vs all-base model")
    axis.set_title("Coverage vs empirical recovery (diagnostic only)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    for path in _figure_targets(figures_dir, "fig6_coverage_vs_recovery"):
        figure.savefig(path, dpi=200)
        created.append(path)
    plt.close(figure)
    return created


def render_main_table(analysis: Mapping[str, Any], value_prefix: str) -> list[str]:
    """Markdown table per regime/budget with domain values and worst column."""

    lines: list[str] = []
    rows = analysis["method_rows"]
    budgets = sorted({row["budget_fraction"] for row in rows})
    for regime in BASE_BITS_BY_REGIME:
        for budget in budgets:
            block = [
                row
                for row in rows
                if row["regime"] == regime and row["budget_fraction"] == budget
            ]
            if not block:
                continue
            lines.append(f"### {regime}, {int(round(budget * 100))}% protection budget")
            lines.append("")
            lines.append(
                "| Method | Protected experts | General | Math | Coding | "
                "Reasoning | Mean | Worst |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for row in block:
                key = (
                    "relative_delta" if value_prefix == "relative" else "delta_nll"
                )
                values = [row[f"{key}_{domain}"] for domain in STAGE2B_DOMAINS]
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
                protected = _protected_count_for_row(row)
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            row["method_label"],
                            str(protected),
                            *[f"{value:+.6f}" for value in values],
                            f"{row[mean_key]:+.6f}",
                            f"{row[worst_key]:+.6f}",
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return lines


_PROTECTED_COUNTS: dict[str, int] = {}


def attach_protected_counts(allocations_dir: Path) -> None:
    registry = load_frozen_registry(allocations_dir)
    for entry in registry["entries"]:
        record = load_allocation(allocations_dir, entry["file"])
        if record["method_kind"] == "uniform_reference":
            continue
        slug = (
            f"{record['method']}|{record['regime']}|{record['budget_fraction']}"
        )
        _PROTECTED_COUNTS[slug] = record["protected_expert_count"]


def _protected_count_for_row(row: Mapping[str, Any]) -> int:
    return _PROTECTED_COUNTS.get(
        f"{row['method']}|{row['regime']}|{row['budget_fraction']}", -1
    )
