"""Stage 2B human-readable summaries built from on-disk artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .io_utils import read_json
from .protection_reporting import render_main_table
from .specialist_preservation import STAGE2B_DOMAINS


def _read_optional(path: Path) -> Any | None:
    return read_json(path) if path.is_file() else None


def write_stage2b_summary(results_root: Path) -> Path:
    """Write SUMMARY.md describing whatever Stage 2B artifacts currently exist."""

    calibration = _read_optional(results_root / "calibration" / "calibration_metadata.json")
    registry = _read_optional(results_root / "allocations" / "allocation_registry.json")
    splits = _read_optional(results_root / "splits" / "split_manifest.json")
    decision = _read_optional(results_root / "stage2b_decision.json")
    development = _read_optional(results_root / "development" / "development_results.json")
    final = _read_optional(results_root / "final" / "final_results.json")
    audit = _read_optional(results_root / "audits" / "independent_audit.json")

    lines: list[str] = [
        "# Stage 2B: Robust specialist preservation under a fixed protection budget",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "The method protects domain-specialized expert capacity directly and",
        "maximizes the specialist coverage of the worst-protected domain. It does",
        "not predict per-expert quantization delta NLL and uses no AOD, GQS, APD,",
        "reconstruction-error, or fitted surrogate objective. The Stage-2A",
        "SURROGATE_NO_GO result remains frozen.",
        "",
        "Storage numbers are exact projected format bytes and effective",
        "bits/weight for QDQ-simulated formats. No runtime speedup, latency,",
        "GPU-memory, or energy claim is made.",
        "",
    ]
    if calibration is not None:
        lines += [
            "## Calibration",
            "",
            f"- Calibration seed: {calibration['calibration_seed']}",
            f"- Multi-domain budget: {calibration['calibration_examples_per_domain']} "
            f"examples/domain, {calibration['multi_domain_total_calibration_examples']}"
            " total",
            f"- Single-domain budget: "
            f"{calibration['single_domain_total_calibration_examples']} examples",
            f"- Calibration fingerprint: `{calibration['calibration_fingerprint']}`",
            "",
        ]
    if registry is not None:
        lines += [
            "## Frozen allocations",
            "",
            f"- Registry entries: {len(registry['entries'])}",
            f"- Registry SHA-256: `{registry['registry_sha256']}`",
            f"- Protection fractions: {registry['protection_fractions']}",
            f"- Random seeds: {registry['random_seeds']}",
        ]
        for regime, info in registry["regimes"].items():
            lines.append(
                f"- {regime}: base {info['base_bits']}-bit, protected "
                f"{info['protected_bits']}-bit, total increment "
                f"{info['total_increment_bytes']:,} bytes"
            )
        lines.append("")
    if splits is not None:
        lines += [
            "## Held-out splits",
            "",
            f"- Development: seed {splits['development_seed']}, "
            f"{splits['development_examples_per_domain']} examples/domain",
            f"- Final: seed {splits['final_seed']}, "
            f"{splits['final_examples_per_domain']} examples/domain",
            f"- Measured positions/example: {splits['measured_tokens_per_example']}",
        ]
        for domain, entry in splits["domains"].items():
            lines.append(
                f"- {domain}: prior pool fully excluded = "
                f"{entry['prior_pool_fully_excluded']}; disjointness verified = "
                f"{entry['disjointness_verified']}"
            )
        lines.append("")
    if development is not None:
        lines += ["## Development results (20% budget)", ""]
        lines += render_main_table(development, "relative")
        if decision is not None:
            lines += [
                f"**Development decision: {decision['decision']}**",
                "",
            ]
            for regime, gates in decision.get("development_gates", {}).items():
                summary = ", ".join(
                    f"{key}={'PASS' if gates[key]['passed'] else 'FAIL'}"
                    for key in ("gate_a", "gate_b", "gate_c", "gate_d")
                )
                lines.append(f"- {regime}: {summary}")
            lines.append("")
    if final is not None:
        lines += [
            "## Final results",
            "",
            f"**Final decision: {final['final_decision']['decision']}**",
            "",
            "See `final/FINAL_SUMMARY.md` for the complete tables.",
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
    path = results_root / "SUMMARY.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_final_summary(results_root: Path, analysis: Mapping[str, Any]) -> Path:
    """Write FINAL_SUMMARY.md with the complete preregistered final tables."""

    lines: list[str] = [
        "# Stage 2B final evaluation summary",
        "",
        f"Final decision: **{analysis['final_decision']['decision']}**",
        "",
        analysis["final_decision"]["rule"],
        "",
        "## Relative NLL degradation (RelativeDelta) by regime and budget",
        "",
    ]
    lines += render_main_table(analysis, "relative")
    lines += ["## Raw delta NLL by regime and budget", ""]
    lines += render_main_table(analysis, "raw")
    lines += ["## Single-domain transfer at the 20% budget", ""]
    for regime in sorted({row["regime"] for row in analysis["single_domain_transfer"]}):
        lines.append(f"### {regime}")
        lines.append("")
        header = "| Calibration | " + " | ".join(
            d.capitalize() for d in STAGE2B_DOMAINS
        ) + " |"
        lines.append(header)
        lines.append("|---|" + "---:|" * len(STAGE2B_DOMAINS))
        by_method: dict[str, dict[str, float]] = {}
        for row in analysis["single_domain_transfer"]:
            if row["regime"] == regime:
                by_method.setdefault(row["calibration_method"], {})[
                    row["evaluation_domain"]
                ] = row["relative_delta"]
        for method, cells in by_method.items():
            lines.append(
                f"| {method} | "
                + " | ".join(f"{cells[d]:+.6f}" for d in STAGE2B_DOMAINS)
                + " |"
            )
        lines.append("")
    lines += ["## Regime assessments", ""]
    for regime, assessment in analysis["final_regime_assessments"].items():
        lines += [
            f"### {regime}",
            "",
            f"- Budgets with all-three point wins: "
            f"{assessment['budgets_with_all_three_point_wins']} of 4",
            f"- Average worst-domain improvement over Average-Specialization: "
            f"{assessment['average_improvement_over_average_specialization']:+.6f}",
            f"- CI wins vs Average-Specialization: "
            f"{assessment['ci_wins_vs_average_specialization']}",
            f"- Catastrophic domains: {assessment['catastrophic_domains'] or 'none'}",
            f"- Strong success: {assessment['strong_success']}",
            f"- Qualified success: {assessment['qualified_success']}",
            "",
        ]
    path = results_root / "final" / "FINAL_SUMMARY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
