"""Stage 3 analysis driver, tables, decisions, figures, and summaries.

Reads only frozen allocations, the frozen damage matrix, and saved per-example
loss checkpoints; computes the preregistered statistics; and writes
machine-readable results, the Stage 3 decision file, and matplotlib-only
figures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fragility_statistics import (
    STAGE2C_BOOTSTRAP_REPLICATES,
    STAGE2C_BOOTSTRAP_SEED,
    STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
)
from .io_utils import atomic_save_npz, atomic_write_json, read_json, write_csv
from .measured_damage import (
    STAGE3_REGIMES,
    STAGE3_STAGE,
    additivity_decision,
    additivity_gates_for_regime,
    load_frozen_damage,
    predicted_domain_delta_nll,
)
from .measured_damage_evaluation import (
    stage3_phase_dir_name,
    stage3_phase_records,
)
from .measured_damage_optimization import (
    MEASURED_DAMAGE_METHOD,
    load_frozen_stage3_registry,
)
from .measured_damage_statistics import (
    prediction_transfer_check,
    stage3_development_decision,
    stage3_development_gates,
    stage3_final_decision,
    stage3_final_regime_assessment,
)
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
    "fragility_robust",
    "global_importance",
    "average_specialization",
    "robust_routing",
)
FIGURE_METHODS = (
    ("global_importance", "Global-Importance"),
    ("average_specialization", "Average-Specialization"),
    ("robust_functional", "Robust-Functional"),
    ("fragility_robust", "Fragility-Robust"),
    (MEASURED_DAMAGE_METHOD, "Measured-Damage-Robust"),
)
ADDITIVITY_REPORT_FILE = "additivity_report.json"


def analyze_stage3_additivity(
    results_dir: Path,
    stage2b_allocations_dir: Path,
    stage2c_allocations_dir: Path,
    run_fingerprint: str,
) -> dict[str, Any]:
    """Recompute the complete additivity report from frozen artifacts."""

    allocations_dir = results_dir / "allocations"
    registry = load_frozen_stage3_registry(
        allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir
    )
    damage_record, damage_arrays = load_frozen_damage(results_dir / "damage")
    if registry["damage_sha256"] != damage_record["damage_sha256"]:
        raise RuntimeError("Registry and frozen damage matrix disagree")
    _, probes = stage3_phase_records(
        registry, allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir,
        "additivity",
    )
    losses_dir = results_dir / stage3_phase_dir_name("additivity") / "losses"
    bf16 = np.asarray(
        [damage_record["bf16_nll"][domain] for domain in STAGE2B_DOMAINS]
    )
    rows: list[dict[str, Any]] = []
    for record in probes:
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        measured = np.asarray(
            [
                float(losses[domain].per_token_nll.mean()) - bf16[index]
                for index, domain in enumerate(STAGE2B_DOMAINS)
            ]
        )
        predicted = predicted_domain_delta_nll(
            np.asarray(record["expert_bits"], dtype=np.int64),
            damage_arrays["delta_nll"],
        )
        rows.append(
            {
                "slug": allocation_slug(record),
                "method": record["method"],
                "method_label": record["method_label"],
                "regime": record["regime"],
                "budget_fraction": record["budget_fraction"],
                "predicted": predicted.tolist(),
                "measured": measured.tolist(),
                "predicted_worst": float(predicted.max()),
                "measured_worst": float(measured.max()),
            }
        )

    gates_by_regime = {
        regime: additivity_gates_for_regime(
            [row for row in rows if row["regime"] == regime]
        )
        for regime in STAGE3_REGIMES
    }
    decision = additivity_decision(gates_by_regime)

    uniform_diagnostics: dict[str, Any] = {}
    for bits in (3, 4, 8):
        uniform_bits = np.full((16, 64), bits, dtype=np.int64)
        predicted = predicted_domain_delta_nll(
            uniform_bits, damage_arrays["delta_nll"]
        )
        measured = np.asarray(
            [
                damage_record["uniform_nll"][f"uniform{bits}"][domain] - bf16[index]
                for index, domain in enumerate(STAGE2B_DOMAINS)
            ]
        )
        uniform_diagnostics[f"uniform{bits}"] = {
            "predicted_delta_nll": {
                domain: float(predicted[index])
                for index, domain in enumerate(STAGE2B_DOMAINS)
            },
            "measured_delta_nll": {
                domain: float(measured[index])
                for index, domain in enumerate(STAGE2B_DOMAINS)
            },
            "predicted_over_measured_ratio_by_domain": {
                domain: float(
                    predicted[index] / measured[index]
                    if measured[index] != 0
                    else float("nan")
                )
                for index, domain in enumerate(STAGE2B_DOMAINS)
            },
        }

    return {
        "stage": STAGE3_STAGE,
        "phase": "additivity",
        "run_fingerprint": run_fingerprint,
        "registry_sha256": registry["registry_sha256"],
        "damage_sha256": damage_record["damage_sha256"],
        "probe_rows": rows,
        "gates_by_regime": gates_by_regime,
        "additivity_decision": decision,
        "uniform_state_diagnostics_not_gated": uniform_diagnostics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_additivity_outputs(report: Mapping[str, Any], results_dir: Path) -> Path:
    phase_dir = results_dir / stage3_phase_dir_name("additivity")
    phase_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(phase_dir / ADDITIVITY_REPORT_FILE, report)
    rows = []
    for row in report["probe_rows"]:
        entry = {
            "slug": row["slug"],
            "method": row["method"],
            "regime": row["regime"],
            "predicted_worst": row["predicted_worst"],
            "measured_worst": row["measured_worst"],
        }
        for index, domain in enumerate(STAGE2B_DOMAINS):
            entry[f"predicted_{domain}"] = row["predicted"][index]
            entry[f"measured_{domain}"] = row["measured"][index]
        rows.append(entry)
    write_csv(
        phase_dir / "additivity_probes.csv",
        rows,
        [
            "slug", "method", "regime",
            *[f"predicted_{domain}" for domain in STAGE2B_DOMAINS],
            *[f"measured_{domain}" for domain in STAGE2B_DOMAINS],
            "predicted_worst", "measured_worst",
        ],
    )
    return phase_dir / ADDITIVITY_REPORT_FILE


def load_additivity_report(results_dir: Path) -> dict[str, Any]:
    path = results_dir / stage3_phase_dir_name("additivity") / ADDITIVITY_REPORT_FILE
    if not path.is_file():
        raise RuntimeError(
            "The additivity report is missing; run "
            "scripts/check_damage_additivity.py first"
        )
    return read_json(path)


def analyze_stage3_phase(
    phase: str,
    results_dir: Path,
    stage2b_allocations_dir: Path,
    stage2c_allocations_dir: Path,
    run_fingerprint: str,
    authorized_regimes: list[str],
    replicates: int = STAGE2C_BOOTSTRAP_REPLICATES,
    seed: int = STAGE2C_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute every preregistered Stage 3 statistic for one held-out phase."""

    allocations_dir = results_dir / "allocations"
    registry = load_frozen_stage3_registry(
        allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir
    )
    damage_record, damage_arrays = load_frozen_damage(results_dir / "damage")
    if registry["damage_sha256"] != damage_record["damage_sha256"]:
        raise RuntimeError("Registry and frozen damage matrix disagree")
    references, competitors = stage3_phase_records(
        registry, allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir,
        phase, authorized_regimes,
    )
    losses_dir = results_dir / stage3_phase_dir_name(phase) / "losses"

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
    predicted_by_slug: dict[str, dict[str, float]] = {}
    realized_by_slug: dict[str, dict[str, float]] = {}
    for record in competitors:
        key = (record["regime"], record["budget_fraction"])
        losses = load_allocation_losses(losses_dir, record, run_fingerprint)
        nll = {domain: item.per_token_nll for domain, item in losses.items()}
        slug = allocation_slug(record)
        for domain, values in nll.items():
            per_example[f"{slug}__{domain}"] = values
        statistics.setdefault(key, {})[record["method"]] = compute_method_statistics(
            record, nll, bf16_nll, base_nll_by_regime[record["regime"]], indices
        )
        predicted = predicted_domain_delta_nll(
            np.asarray(record["expert_bits"], dtype=np.int64),
            damage_arrays["delta_nll"],
        )
        predicted_by_slug[slug] = {
            domain: float(predicted[index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        }
        realized_by_slug[slug] = {
            domain: float(nll[domain].mean() - bf16_nll[domain].mean())
            for domain in STAGE2B_DOMAINS
        }

    comparisons: list[dict[str, Any]] = []
    random_summaries: list[dict[str, Any]] = []
    for key, methods in sorted(statistics.items()):
        regime, budget = key
        measured = methods[MEASURED_DAMAGE_METHOD]
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
        difference = measured.replicate_worst - random_worst_reps
        low = float(np.quantile(difference, 0.025))
        high = float(np.quantile(difference, 0.975))
        comparisons.append(
            {
                "first": MEASURED_DAMAGE_METHOD,
                "second": "random_mean",
                "regime": regime,
                "budget_fraction": budget,
                "metric": "worst_relative_delta",
                "difference": measured.worst_relative_delta - random_point,
                "difference_ci_low": low,
                "difference_ci_high": high,
                "favors_first": bool(measured.worst_relative_delta < random_point),
                "ci_excludes_zero": bool(high < 0 or low > 0),
                "ci_favors_first": bool(high < 0),
            }
        )
        for other in PAIRED_COMPARATORS:
            for metric in ("worst_relative_delta", "mean_relative_delta"):
                comparisons.append(
                    paired_comparison(measured, methods[other], metric)
                )

    analysis: dict[str, Any] = {
        "stage": STAGE3_STAGE,
        "phase": phase,
        "run_fingerprint": run_fingerprint,
        "registry_sha256": registry["registry_sha256"],
        "damage_sha256": damage_record["damage_sha256"],
        "authorized_regimes": list(authorized_regimes),
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
        "prediction_transfer_check": prediction_transfer_check(
            predicted_by_slug, realized_by_slug
        ),
        "prediction_transfer_points": {
            slug: {
                "predicted": predicted_by_slug[slug],
                "realized": realized_by_slug[slug],
            }
            for slug in sorted(predicted_by_slug)
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if phase == "development":
        additivity = load_additivity_report(results_dir)
        gates_by_regime = {}
        for regime in authorized_regimes:
            methods = statistics[(regime, STAGE2C_DEVELOPMENT_BUDGET_FRACTION)]
            gates_by_regime[regime] = stage3_development_gates(
                methods[MEASURED_DAMAGE_METHOD],
                methods["robust_functional"],
                methods["fragility_robust"],
                [i for n, i in methods.items() if n.startswith("random_seed")],
                methods["global_importance"],
                methods["average_specialization"],
            )
        analysis["development_gates"] = gates_by_regime
        analysis["development_decision"] = stage3_development_decision(
            gates_by_regime,
            additivity["additivity_decision"]["authorized_regimes"],
        )
    else:
        assessments = {}
        for regime in authorized_regimes:
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
            assessments[regime] = stage3_final_regime_assessment(
                statistics,
                comparisons_vs_average,
                comparisons_vs_global,
                regime,
                PROTECTION_FRACTIONS,
            )
        analysis["final_regime_assessments"] = assessments
        analysis["final_decision"] = stage3_final_decision(assessments)

    analysis["_per_example"] = per_example
    analysis["_statistics"] = statistics
    analysis["_reference_statistics"] = reference_stats
    return analysis


def write_stage3_phase_outputs(
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


def write_stage3_additivity_decision(
    report: Mapping[str, Any], results_dir: Path, preregistration_sha256: str
) -> Path:
    """Write the stage decision after a failed additivity gate.

    Called only when no regime is authorized: the stage stops here and the
    development split is never evaluated.
    """

    decision = report["additivity_decision"]
    if decision["authorized_regimes"]:
        raise RuntimeError(
            "The additivity decision authorizes regimes; the stage decision is "
            "written by the development runner instead"
        )
    payload = {
        "stage": STAGE3_STAGE,
        "phase": "additivity",
        "decision": decision["decision"],
        "additivity_decision": decision,
        "additivity_gates": report["gates_by_regime"],
        "run_fingerprint": report["run_fingerprint"],
        "registry_sha256": report["registry_sha256"],
        "damage_sha256": report["damage_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "seed46_never_evaluated": True,
        "seed44_untouched_at_decision_time": True,
        "method_never_modified_by_gate": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = results_dir / "stage3_decision.json"
    atomic_write_json(path, payload)
    return path


def write_stage3_development_decision(
    analysis: Mapping[str, Any],
    additivity_report: Mapping[str, Any],
    results_dir: Path,
    preregistration_sha256: str,
) -> Path:
    payload = {
        "stage": STAGE3_STAGE,
        "phase": "development",
        "decision": analysis["development_decision"]["decision"],
        "development_decision": analysis["development_decision"],
        "development_gates": analysis["development_gates"],
        "additivity_decision": additivity_report["additivity_decision"],
        "additivity_gates": additivity_report["gates_by_regime"],
        "run_fingerprint": analysis["run_fingerprint"],
        "registry_sha256": analysis["registry_sha256"],
        "damage_sha256": analysis["damage_sha256"],
        "preregistration_sha256": preregistration_sha256,
        "bootstrap_replicates": analysis["bootstrap_replicates"],
        "bootstrap_seed": analysis["bootstrap_seed"],
        "method_never_modified_by_gate": True,
        "seed44_untouched_at_decision_time": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = results_dir / "stage3_decision.json"
    atomic_write_json(path, payload)
    return path


def write_stage3_final_decision(
    analysis: Mapping[str, Any], results_dir: Path
) -> Path:
    path = results_dir / "stage3_decision.json"
    payload = read_json(path)
    if payload.get("decision") != "FINAL_CONFIRMATION_GO":
        raise RuntimeError(
            "Final results cannot be recorded without FINAL_CONFIRMATION_GO"
        )
    payload["phase"] = "final"
    payload["final_decision"] = analysis["final_decision"]
    payload["final_regime_assessments"] = analysis["final_regime_assessments"]
    payload["final_prediction_transfer_check"] = analysis[
        "prediction_transfer_check"
    ]
    payload["final_run_fingerprint"] = analysis["run_fingerprint"]
    payload["final_created_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, payload)
    return path


def render_stage3_table(analysis: Mapping[str, Any], value_prefix: str) -> list[str]:
    """Markdown table per regime/budget with domain values and worst column."""

    lines: list[str] = []
    rows = analysis["method_rows"]
    budgets = sorted({row["budget_fraction"] for row in rows})
    for regime in STAGE3_REGIMES:
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


def write_stage3_summary(results_dir: Path) -> Path:
    """Write SUMMARY.md describing whatever Stage 3 artifacts currently exist."""

    def _optional(path: Path) -> Any | None:
        return read_json(path) if path.is_file() else None

    damage = _optional(results_dir / "damage" / "damage_matrix.json")
    registry = _optional(results_dir / "allocations" / "allocation_registry.json")
    splits = _optional(results_dir / "splits" / "split_manifest.json")
    additivity = _optional(
        results_dir / stage3_phase_dir_name("additivity") / ADDITIVITY_REPORT_FILE
    )
    decision = _optional(results_dir / "stage3_decision.json")
    development = _optional(
        results_dir
        / stage3_phase_dir_name("development")
        / "development_results.json"
    )
    final = _optional(
        results_dir / stage3_phase_dir_name("final") / "final_results.json"
    )
    audit = _optional(results_dir / "audits" / "independent_audit.json")

    lines: list[str] = [
        "# Stage 3: Measured expert-damage preservation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Stage 3 measures the ground-truth per-expert quantization damage",
        "m[l,e,d,b] directly (single-expert QDQ delta NLL on the frozen",
        "calibration subset) and protects experts by minimizing the largest",
        "additively predicted domain delta NLL. Nothing is estimated or",
        "fitted; the frozen Stage 2A SURROGATE_NO_GO, Stage 2B",
        "ROBUST_PRESERVATION_NO_GO, and Stage 2C FRAGILITY_ROBUST_NO_GO",
        "decisions are preserved unchanged.",
        "",
        "Storage numbers are exact projected format bytes for QDQ-simulated",
        "formats; no runtime speedup, latency, or measured-memory claim is made.",
        "",
    ]
    if damage is not None:
        lines += ["## Measured damage matrix", ""]
        lines.append(
            f"- {damage['num_moe_layers']}x{damage['num_experts']} experts, "
            f"domains {damage['domains']}, bit widths {damage['profile_bits']}"
        )
        for bits_key, summary in damage["summary"].items():
            totals = summary["total_delta_nll_by_domain"]
            lines.append(
                f"- {bits_key}: total delta NLL by domain "
                + ", ".join(
                    f"{domain} {totals[domain]:+.6f}" for domain in STAGE2B_DOMAINS
                )
                + f"; negative cells {summary['negative_damage_cells']}"
            )
        lines += ["", f"- Damage SHA-256: `{damage['damage_sha256']}`", ""]
    if registry is not None:
        lines += [
            "## Frozen allocations",
            "",
            f"- New Measured-Damage-Robust allocations: {len(registry['new_entries'])}",
            f"- Reused frozen Stage 2B comparators: "
            f"{len(registry['reused_stage2b_entries'])}",
            f"- Reused frozen Stage 2C comparators: "
            f"{len(registry['reused_stage2c_entries'])}",
            f"- Registry SHA-256: `{registry['registry_sha256']}`",
            "",
        ]
    if splits is not None:
        lines += [
            "## Seed-46 development split",
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
    if additivity is not None:
        lines += ["## Additivity gates (calibration probes)", ""]
        for regime, gates in additivity["gates_by_regime"].items():
            per_domain = gates["gate_add_1"]["spearman_by_domain"]
            lines.append(
                f"- {regime}: gate_add_1="
                f"{'PASS' if gates['gate_add_1']['passed'] else 'FAIL'} "
                "(per-domain Spearman "
                + ", ".join(
                    f"{domain} {per_domain[domain]:+.3f}"
                    for domain in STAGE2B_DOMAINS
                )
                + f"), gate_add_2="
                f"{'PASS' if gates['gate_add_2']['passed'] else 'FAIL'} "
                f"(worst-delta Spearman {gates['gate_add_2']['spearman']:+.3f})"
            )
        lines.append(
            f"- Additivity decision: "
            f"**{additivity['additivity_decision']['decision']}**, authorized "
            f"regimes: {additivity['additivity_decision']['authorized_regimes']}"
        )
        lines.append("")
    if development is not None:
        lines += ["## Development results (seed 46, 20% budget)", ""]
        lines += render_stage3_table(development, "relative")
        transfer = development.get("prediction_transfer_check")
        if transfer is not None:
            lines += [
                "### Prediction transfer (descriptive)",
                "",
                f"- Worst-delta Spearman (predicted vs realized): "
                f"{transfer['worst_delta_spearman']:+.4f}",
                f"- Overall Spearman: {transfer['overall_spearman']:+.4f}",
                "",
            ]
    if decision is not None:
        lines += [f"**Stage 3 decision: {decision['decision']}**", ""]
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


def write_stage3_final_summary(
    results_dir: Path, analysis: Mapping[str, Any]
) -> Path:
    lines: list[str] = [
        "# Stage 3 final evaluation summary (seed 44)",
        "",
        f"Final decision: **{analysis['final_decision']['decision']}**",
        "",
        analysis["final_decision"]["rule"],
        "",
        "## Relative NLL degradation (RelativeDelta) by regime and budget",
        "",
    ]
    lines += render_stage3_table(analysis, "relative")
    lines += ["## Raw delta NLL by regime and budget", ""]
    lines += render_stage3_table(analysis, "raw")
    lines += ["## Regime assessments", ""]
    for regime, assessment in analysis["final_regime_assessments"].items():
        lines += [
            f"### {regime}",
            "",
            f"- Budgets with all-five point wins: "
            f"{assessment['budgets_with_all_five_point_wins']} of 4",
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
    transfer = analysis["prediction_transfer_check"]
    lines += [
        "## Prediction transfer (calibration-predicted vs seed-44 realized)",
        "",
        f"- Worst-delta Spearman: {transfer['worst_delta_spearman']:+.4f}",
        f"- Overall Spearman: {transfer['overall_spearman']:+.4f}",
        f"- Total predicted/realized ratio: "
        f"{transfer['total_predicted_over_realized_ratio']:.4f}",
        "",
    ]
    path = results_dir / stage3_phase_dir_name("final") / "FINAL_SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _figure_targets(base: Path, name: str) -> list[Path]:
    return [base / f"{name}.png", base / f"{name}.pdf"]


def create_stage3_figures(
    results_dir: Path,
    figures_dir: Path,
    analysis: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Matplotlib-only Stage 3 figures; generates whatever data allows."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    positions = np.arange(len(STAGE2B_DOMAINS))
    domain_labels = [d.capitalize() for d in STAGE2B_DOMAINS]

    # Figure 1: measured per-expert damage maps at 4-bit.
    damage_path = results_dir / "damage" / "damage_matrix.json"
    if damage_path.is_file():
        _, arrays = load_frozen_damage(results_dir / "damage")
        delta = arrays["delta_nll"]
        bit_index = 1  # 4-bit within (3, 4, 8)
        figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
        for axis, (domain_index, domain) in zip(
            axes.reshape(-1), enumerate(STAGE2B_DOMAINS), strict=True
        ):
            image = axis.imshow(
                delta[:, :, domain_index, bit_index],
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
            )
            axis.set_title(f"{domain.capitalize()} 4-bit measured delta NLL")
            axis.set_xlabel("Expert ID")
            axis.set_ylabel("MoE layer")
            figure.colorbar(image, ax=axis)
        figure.suptitle("Figure 1: Measured per-expert 4-bit damage")
        figure.tight_layout()
        for path in _figure_targets(figures_dir, "fig1_measured_damage_4bit"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 2: additivity predicted vs measured probe deltas.
    additivity_path = (
        results_dir / stage3_phase_dir_name("additivity") / ADDITIVITY_REPORT_FILE
    )
    if additivity_path.is_file():
        report = read_json(additivity_path)
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        for regime, marker in zip(STAGE3_REGIMES, ("o", "s"), strict=True):
            rows = [r for r in report["probe_rows"] if r["regime"] == regime]
            predicted = np.asarray([r["predicted"] for r in rows]).reshape(-1)
            measured = np.asarray([r["measured"] for r in rows]).reshape(-1)
            axis.scatter(
                measured, predicted, marker=marker, alpha=0.75, label=regime
            )
        limits = axis.get_xlim()
        axis.plot(limits, limits, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel("Measured probe delta NLL (calibration)")
        axis.set_ylabel("Additively predicted delta NLL")
        axis.set_title("Figure 2: Additivity of measured expert damage")
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(figures_dir, "fig2_additivity_probes"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    if analysis is None:
        return created
    method_rows = analysis["method_rows"]
    phase = analysis["phase"]

    # Figure 3: domain degradation profiles at the 20% budget.
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
        figure, axis = plt.subplots(figsize=(9.5, 4.5))
        width = 0.16
        for offset, (method, label) in enumerate(FIGURE_METHODS):
            matches = [r for r in block if r["method"] == method]
            if not matches:
                continue
            values = [matches[0][f"relative_delta_{d}"] for d in STAGE2B_DOMAINS]
            axis.bar(positions + (offset - 2.0) * width, values, width, label=label)
        axis.set_xticks(positions, domain_labels)
        axis.set_ylabel("Relative NLL degradation")
        axis.set_title(
            f"Figure 3: Domain degradation at 20% budget ({regime}, {phase})"
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend(fontsize=8)
        figure.tight_layout()
        for path in _figure_targets(
            figures_dir, f"fig3_domain_profiles_{regime}_{phase}"
        ):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)

    # Figure 4: calibration-predicted vs held-out realized worst-domain delta.
    transfer = analysis.get("prediction_transfer_check")
    points = analysis.get("prediction_transfer_points")
    if transfer is not None and points:
        figure, axis = plt.subplots(figsize=(6.5, 5.5))
        predicted_worst = [
            max(entry["predicted"].values()) for entry in points.values()
        ]
        realized_worst = [
            max(entry["realized"].values()) for entry in points.values()
        ]
        axis.scatter(realized_worst, predicted_worst, alpha=0.8)
        limits = axis.get_xlim()
        axis.plot(limits, limits, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel(f"Realized worst-domain delta NLL ({phase})")
        axis.set_ylabel("Calibration-predicted worst-domain delta NLL")
        axis.set_title(
            f"Figure 4: Prediction transfer ({phase}); worst-delta Spearman "
            f"{transfer['worst_delta_spearman']:+.3f}"
        )
        figure.tight_layout()
        for path in _figure_targets(figures_dir, f"fig4_prediction_transfer_{phase}"):
            figure.savefig(path, dpi=200)
            created.append(path)
        plt.close(figure)
    return created
