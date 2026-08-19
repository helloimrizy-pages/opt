from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import atomic_write_json, atomic_write_text, write_csv


def write_analysis_outputs(
    analysis: Mapping[str, Any],
    frozen: Mapping[str, Any],
    output_dir: Path,
    *,
    hardware_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "analysis.json", analysis)
    _write_rows(output_dir / "tables" / "regime_headroom.csv", analysis["regime_headroom"])
    _write_rows(output_dir / "tables" / "workload_headroom.csv", analysis["workload_headroom"])
    _write_rows(output_dir / "tables" / "baseline_costs.csv", analysis["table_rows"])
    _write_rows(output_dir / "tables" / "random_statistics.csv", analysis["random_statistics"])
    _write_rows(output_dir / "tables" / "routing_diagnostics.csv", analysis["routing_diagnostics"])
    _write_rows(
        output_dir / "tables" / "diagnostic_associations.csv",
        analysis["diagnostic_associations"],
    )
    _write_rows(
        output_dir / "tables" / "diagnostic_correlations.csv",
        analysis["diagnostic_correlations"],
    )
    figure_paths = create_figures(analysis, output_dir / "figures")
    report = render_report(
        analysis,
        frozen,
        figure_paths=figure_paths,
        hardware_calibration=hardware_calibration,
    )
    report_path = output_dir / "stage3_residency_headroom_report.md"
    atomic_write_text(report_path, report)
    audit = audit_report_package(analysis, frozen, output_dir, figure_paths)
    atomic_write_json(output_dir / "analysis_audit.json", audit)
    audit_name = (
        "pilot_audit_report.md"
        if frozen["preregistered_config"]["run_kind"] != "full"
        else "full_audit_report.md"
    )
    atomic_write_text(output_dir / audit_name, render_audit(audit, frozen))
    if not audit["passed"]:
        raise RuntimeError("Generated Stage 0 analysis package failed its audit")
    return {
        "report": str(report_path),
        "audit": str(output_dir / "analysis_audit.json"),
        "figures": [str(path) for path in figure_paths],
    }


def create_figures(analysis: Mapping[str, Any], figures_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "lru": "#4C78A8",
        "lfu": "#F58518",
        "lfu_decay": "#54A24B",
        "static_hotset": "#B279A2",
        "oracle": "#E45756",
    }
    labels = {
        "lru": "LRU",
        "lfu": "LFU",
        "lfu_decay": "LFU-decay",
        "static_hotset": "Static Hotset",
        "oracle": "Offline oracle",
    }
    regimes = [
        regime
        for regime in ("stationary", "abrupt", "repeated", "mixed")
        if any(row["regime"] == regime for row in analysis["table_rows"])
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, regime in zip(axes.flat, regimes):
        rows = sorted(
            (row for row in analysis["table_rows"] if row["regime"] == regime),
            key=lambda row: int(row["capacity"]),
        )
        capacities = [int(row["capacity"]) for row in rows]
        for policy in colors:
            axis.plot(
                capacities,
                [float(row[f"{policy}_normalized"]) for row in rows],
                marker="o",
                color=colors[policy],
                label=labels[policy],
            )
        axis.set_title(regime.capitalize())
        axis.grid(alpha=0.25)
        axis.set_xticks(capacities)
    for axis in axes[:, 0]:
        axis.set_ylabel("Miss cost / requested experts")
    for axis in axes[-1, :]:
        axis.set_xlabel("Resident experts per layer")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False)
    figure.suptitle("Stage 0 normalized residency cost (all preregistered regimes)")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure1 = figures_dir / "figure1_normalized_cost.png"
    figure.savefig(figure1, dpi=180)
    figure.savefig(figure1.with_suffix(".pdf"))
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for regime in regimes:
        rows = sorted(
            (row for row in analysis["regime_headroom"] if row["regime"] == regime),
            key=lambda row: int(row["capacity"]),
        )
        capacity = np.asarray([row["capacity"] for row in rows], dtype=float)
        headroom = 100 * np.asarray([row["headroom"] for row in rows], dtype=float)
        low = 100 * np.asarray([row["headroom_ci_low"] for row in rows], dtype=float)
        high = 100 * np.asarray([row["headroom_ci_high"] for row in rows], dtype=float)
        axis.plot(capacity, headroom, marker="o", label=regime.capitalize())
        axis.fill_between(capacity, low, high, alpha=0.12)
    axis.axhline(15, color="black", linestyle="--", linewidth=1, label="STRONG GO point threshold")
    axis.axhline(5, color="gray", linestyle=":", linewidth=1, label="WEAK boundary")
    axis.set_xlabel("Resident experts per layer")
    axis.set_ylabel("Offline-oracle headroom (%)")
    axis.set_xticks(sorted({int(row["capacity"]) for row in analysis["regime_headroom"]}))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure2 = figures_dir / "figure2_oracle_headroom.png"
    figure.savefig(figure2, dpi=180)
    figure.savefig(figure2.with_suffix(".pdf"))
    plt.close(figure)

    association = analysis["diagnostic_associations"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, label in (
        (axes[0], "pairwise_domain_js_mean", "Pairwise domain JS divergence"),
        (axes[1], "consecutive_jaccard_mean", "Consecutive-request Jaccard"),
    ):
        for regime in regimes:
            rows = [
                row
                for row in association
                if row["regime"] == regime
                and row[metric] is not None
                and np.isfinite(float(row[metric]))
            ]
            if rows:
                axis.scatter(
                    [float(row[metric]) for row in rows],
                    [100 * float(row["headroom"]) for row in rows],
                    label=regime.capitalize(),
                    alpha=0.75,
                )
        axis.set_xlabel(label)
        axis.set_ylabel("Oracle headroom (%)")
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure3 = figures_dir / "figure3_locality_shift_vs_headroom.png"
    figure.savefig(figure3, dpi=180)
    figure.savefig(figure3.with_suffix(".pdf"))
    plt.close(figure)
    return [figure1, figure2, figure3]


def render_report(
    analysis: Mapping[str, Any],
    frozen: Mapping[str, Any],
    *,
    figure_paths: Sequence[Path],
    hardware_calibration: Mapping[str, Any] | None,
) -> str:
    config = frozen["preregistered_config"]
    decision = analysis["decision"]
    lines = [
        "# RACE Stage 0: Expert Residency Oracle-Headroom Study",
        "",
        "## Executive result",
        "",
        f"**{decision['decision']}**",
        "",
        decision["reason"],
        "",
    ]
    if decision["decision"] == "PILOT_ONLY_NO_STAGE0_DECISION":
        lines.extend(
            [
                "This is a mechanics/runtime pilot and is not the final scientific dataset. "
                "No GO/NO-GO conclusion is authorized from it.",
                "",
            ]
        )
    lines.extend(
        [
            "The primary quantity is equal-cost expert misses at `lambda=0`. The strongest "
            "simple policy is selected separately for each workload/capacity from LRU, LFU, "
            f"LFU-decay with the globally calibrated alpha `{frozen['selected_lfu_decay_alpha']}`, "
            "and Static Hotset. Random is excluded from that selection.",
            "",
            "## Exact reason",
            "",
        ]
    )
    for regime, outcome in decision["regime_outcomes"].items():
        lines.append(
            f"- {regime}: STRONG-support budgets {outcome['strong_budgets']} "
            f"({outcome['strong_budget_count']}/5); >=5% positive-CI budgets "
            f"{outcome['weak_or_better_budgets']} ({outcome['weak_or_better_budget_count']}/5)."
        )
    lines.extend(["", "## Baseline tables", ""])
    for title, regimes in (
        ("Table A — Stationary", ("stationary",)),
        ("Table B — Abrupt shifts", ("abrupt",)),
        ("Table C — Repeated and mixed workloads", ("repeated", "mixed")),
    ):
        lines.extend([f"### {title}", ""])
        rows = [row for row in analysis["table_rows"] if row["regime"] in regimes]
        lines.extend(_markdown_baseline_table(rows))
        lines.append("")
    lines.extend(
        [
            "Costs above are raw expert misses summed over every workload in the named regime. "
            "The plotted normalized cost is misses divided by requested experts.",
            "",
            "## Paired bootstrap confidence intervals",
            "",
            "The independent unit is a source prompt/decode sequence. Policies use identical "
            "fixed traces; workload resampling is stratified by segment/domain, and regime "
            "resampling clusters every repeated appearance of the same source prompt. Each "
            "replicate reselects the strongest simple policy, making the comparison conservative "
            "with respect to simple-baseline selection without treating reused prompts as "
            "independent evidence.",
            "",
            "| Regime | Cache | Best-simple cost | Oracle cost | Absolute gap | Headroom (95% CI) | Mean/median sequence gap | Paired effect |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["regime_headroom"]:
        lines.append(
            f"| {row['regime']} | {row['capacity']} | {row['best_simple_cost']:.0f} | "
            f"{row['oracle_cost']:.0f} | {row['absolute_gap']:.0f} | "
            f"{100*row['headroom']:.2f}% "
            f"[{100*row['headroom_ci_low']:.2f}%, {100*row['headroom_ci_high']:.2f}%] | "
            f"{row['mean_sequence_absolute_gap']:.2f} / "
            f"{row['median_sequence_absolute_gap']:.2f} | "
            f"{_fmt(row['paired_standardized_effect'])} |"
        )
    oracle = frozen["oracle_validation"]
    lines.extend(
        [
            "",
            "## Oracle validation",
            "",
            f"The scalable generalized farthest-in-future policy matched exact dynamic "
            f"programming on **{oracle['exhaustive_cases']} exhaustive** and "
            f"**{oracle['random_cases']} random** tiny set-valued cases. Maximum objective "
            f"difference was `{oracle['maximum_cost_difference']}` across lambdas "
            f"`{oracle['lambda_values']}`.",
            "",
            "The proof obligation is intentionally limited to equal-size/equal-cost experts. "
            "The trace reports whether all OLMoE experts have equal parameter bytes; the "
            "scalable oracle must not be relabeled optimal for heterogeneous weights.",
            "",
            "## Routing and locality diagnostics",
            "",
            "| Workload | Entropy | Top-10 share | Gini | Consecutive Jaccard | Reuse <=10 | Adjacent-segment JS | Domain JS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["routing_diagnostics"]:
        lines.append(
            f"| {row['workload']} | {_fmt(row['normalized_frequency_entropy_mean'])} | "
            f"{_fmt(row['top_10_traffic_share_mean'])} | {_fmt(row['gini_mean'])} | "
            f"{_fmt(row['consecutive_jaccard_mean'])} | {_fmt(row['reuse_within_10_events'])} | "
            f"{_fmt(row['adjacent_segment_js_mean'])} | {_fmt(row['pairwise_domain_js_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Associations with headroom are descriptive: workloads are fixed and cache-budget "
            "conditions are not independent replications.",
            "",
            "## Cost interpretation and limitations",
            "",
            "- These results are trace simulations of expert residency. They are not measured "
            "end-to-end inference latency or a runtime speedup.",
            "- The primary unit cost counts one transfer per missing expert. Byte-weighted cost "
            "uses actual expert parameter bytes from the loaded checkpoint. If those sizes are "
            "equal, byte cost is exactly proportional to miss count.",
            "- Atomic mandatory admission makes admissions equal misses. The reported lambda "
            "grid therefore rescales costs and cannot change rankings; it must not be interpreted "
            "as an independent latency model.",
            "- Sequence-level bootstrap intervals condition on these prompts, this model revision, "
            "this deterministic decode, and the frozen workload construction.",
            "- Static Hotset and LFU-decay use only the disjoint calibration sequences. All "
            "evaluation caches start empty, so initial compulsory transfers are counted.",
        ]
    )
    if hardware_calibration and hardware_calibration.get("available"):
        lines.append(
            "- A separate measured host-device expert transfer calibration is available. It is "
            "not end-to-end inference latency and was not double-counted with admissions."
        )
    else:
        lines.append(
            "- No defensible CUDA host-device measurement was available in this environment; "
            "hardware-weighted cost remains unavailable rather than fabricated."
        )
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path.name}`")
    lines.extend(["", "## Reproducibility", ""])
    lines.extend(
        [
            f"- Source commit: `{config['source_commit']}`",
            f"- Model: `{config['model']}` at `{config['model_revision']}`",
            f"- Trace hash: `{frozen['trace_hash']}`",
            f"- Frozen evaluation hash: `{frozen['config_hash']}`",
            f"- Decode seed: `{config['seed']}`; bootstrap seed: `{config['bootstrap_seed']}`",
            f"- Cache capacities per layer: `{config['cache_capacities']}`",
            "",
            "Exact commands and file layout are in `stage3_residency/README.md` and the "
            "generated trace/evaluation manifests.",
            "",
            "## Next action",
            "",
        ]
    )
    next_action = {
        "RACE_STAGE0_STRONG_GO": "Proceed to design RACE. This result does not claim RACE will work.",
        "RACE_STAGE0_WEAK_GO": "Do not build RACE yet; inspect where headroom concentrates.",
        "RACE_STAGE0_NO_GO": "Do not build RACE.",
        "PILOT_ONLY_NO_STAGE0_DECISION": "Run and audit the frozen full GPU trace/evaluation before deciding.",
    }[decision["decision"]]
    lines.append(next_action)
    lines.append("")
    return "\n".join(lines)


def audit_report_package(
    analysis: Mapping[str, Any],
    frozen: Mapping[str, Any],
    output_dir: Path,
    figure_paths: Sequence[Path],
) -> dict[str, Any]:
    checks = [
        {
            "name": "oracle_validation_passed",
            "passed": bool(frozen["oracle_validation"]["passed"]),
        },
        {
            "name": "all_five_budgets_each_regime",
            "passed": all(
                len([row for row in analysis["regime_headroom"] if row["regime"] == regime])
                == 5
                for regime in {row["regime"] for row in analysis["regime_headroom"]}
            ),
        },
        {
            "name": "oracle_never_exceeds_best_simple",
            "passed": all(float(row["oracle_cost"]) <= float(row["best_simple_cost"])
                          for row in analysis["regime_headroom"]),
        },
        {
            "name": "headroom_formula",
            "passed": all(
                abs(
                    float(row["headroom"])
                    - (float(row["best_simple_cost"]) - float(row["oracle_cost"]))
                    / float(row["best_simple_cost"])
                )
                <= 1e-12
                for row in analysis["regime_headroom"]
                if float(row["best_simple_cost"]) != 0
            ),
        },
        {
            "name": "required_figures_exist",
            "passed": len(figure_paths) == 3
            and all(path.exists() and path.with_suffix(".pdf").exists() for path in figure_paths),
        },
        {
            "name": "report_exists",
            "passed": (output_dir / "stage3_residency_headroom_report.md").exists(),
        },
    ]
    return {
        "schema_version": "race_stage0_analysis_audit_v1",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "decision": analysis["decision"]["decision"],
        "trace_hash": frozen["trace_hash"],
        "config_hash": frozen["config_hash"],
    }


def render_audit(audit: Mapping[str, Any], frozen: Mapping[str, Any]) -> str:
    kind = frozen["preregistered_config"]["run_kind"]
    lines = [
        f"# RACE Stage 0 {kind} audit report",
        "",
        f"Overall: **{'PASS' if audit['passed'] else 'FAIL'}**",
        "",
    ]
    for check in audit["checks"]:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} — {check['name']}")
    lines.extend(
        [
            "",
            f"Trace hash: `{audit['trace_hash']}`",
            f"Frozen config hash: `{audit['config_hash']}`",
            f"Reported decision state: `{audit['decision']}`",
            "",
            "This audit covers artifact mechanics and recomputed tables. It does not convert a "
            "pilot into final evidence and does not measure end-to-end inference latency.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_baseline_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    output = [
        "| Regime | Cache | LRU | LFU | LFU-decay | Static Hotset | Oracle | Oracle headroom (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        output.append(
            f"| {row['regime']} | {row['capacity']} | {row['lru']:.0f} | "
            f"{row['lfu']:.0f} | {row['lfu_decay']:.0f} | "
            f"{row['static_hotset']:.0f} | {row['oracle']:.0f} | "
            f"{100*row['oracle_headroom']:.2f}% "
            f"[{100*row['oracle_headroom_ci_low']:.2f}%, "
            f"{100*row['oracle_headroom_ci_high']:.2f}%] |"
        )
    return output


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, rows, fields)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not np.isfinite(number):
        return "NA" if np.isnan(number) else ("inf" if number > 0 else "-inf")
    return f"{number:.3f}"
