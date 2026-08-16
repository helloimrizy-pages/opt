#!/usr/bin/env python3
"""Tabulate the Stage 3D sweeps, apply the Step 5 thresholds, write the summary.

Reads the three JSONL files, prints one plain-text table per sweep, writes a
CSV per sweep, applies the preregistered thresholds mechanically, and records
which of the stated outcomes each sweep landed in. No interpretation beyond the
thresholds. Runs on any machine; it loads no model.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.io_utils import atomic_write_json, write_csv
from expert_analysis.protection_optimization import BASE_BITS_BY_REGIME, PROTECTED_BITS
from expert_analysis.specialist_preservation import STAGE2B_DOMAINS
from expert_analysis.stage3d_diagnostics import (
    DECISION_THRESHOLDS,
    PRIMARY_REGIME,
    SECONDARY_REGIME,
    STAGE3D_RESULTS_DIRNAME,
    STAGE3D_STAGE,
    SWEEP_A_BUDGET_FRACTION,
    SWEEP_B_BITS,
    SWEEP_B_LAYERS,
    SWEEP_C_EXPERT_BITS,
    read_run_records,
    sweep_a_decision,
    sweep_b_decision,
    sweep_c_report,
    sweep_jsonl_path,
)

# What the project calls the baseline for the paper, stated for Step 4.
PAPER_BASELINE = {
    "name": "uniform base-precision expert-only quantization",
    "definition": (
        "every one of the 1024 expert FFN matrices at the regime's base "
        "precision (4-bit for 4to8, 3-bit for 3to8) using symmetric group-wise "
        "round-to-nearest QDQ at group size 128 with FP16 scales"
    ),
    "quantizes_routers": False,
    "quantizes_gates": False,
    "quantizes_lm_head": False,
    "quantizes_embeddings": False,
    "quantizes_attention": False,
    "left_at_bf16": (
        "MoE router weights, attention projections, embeddings, lm_head, and "
        "all normalization parameters"
    ),
    "reviewer_note": (
        "a baseline that leaves routers, lm_head, and embeddings at BF16 is "
        "weaker than a fully quantized one, so any improvement shown over it "
        "will be discounted. Sweep C measures how much the router half of that "
        "gap is worth."
    ),
}


def _relative_row(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": record["run_id"],
        "sweep": record["sweep"],
        "regime": record.get("regime"),
        "base_bits": record.get("base_bits"),
        "seed": record.get("seed"),
        "protected_expert_count": record.get("protected_expert_count"),
        "worst_domain_relative": record["worst_domain_relative"],
        "worst_domain_relative_domain": record["worst_domain_relative_domain"],
        "worst_domain_raw": record["worst_domain_raw"],
        "worst_domain_raw_domain": record["worst_domain_raw_domain"],
        "mean_relative_delta": record["mean_relative_delta"],
        "wall_clock_seconds": record.get("wall_clock_seconds"),
    }
    for domain in STAGE2B_DOMAINS:
        row[f"loss_{domain}"] = record["loss_by_domain"][domain]
        row[f"relative_delta_{domain}"] = record["relative_delta_by_domain"][domain]
    return row


def _print_table(title: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(item) for item in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [title, "-" * len(title)]
    lines.append("  ".join(item.ljust(widths[index]) for index, item in enumerate(header)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        )
    text = "\n".join(lines)
    print("\n" + text)
    return text


def report_sweep_a(records: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    decision = sweep_a_decision(records)
    blocks: list[str] = []
    for regime, arm in decision["arms"].items():
        selected = [item for item in records if item.get("regime") == regime]
        rows = []
        for record in sorted(
            selected, key=lambda item: (item["seed"] is None, item["seed"] or 0, item["run_id"])
        ):
            rows.append(
                [
                    record["run_id"].removeprefix(f"a_{regime}_"),
                    str(record.get("seed") if record.get("seed") is not None else "-"),
                    str(record["protected_expert_count"]),
                    *[
                        f"{record['relative_delta_by_domain'][domain]:+.5f}"
                        for domain in STAGE2B_DOMAINS
                    ],
                    f"{record['worst_domain_relative']:+.5f}",
                    record["worst_domain_relative_domain"],
                    f"{record['worst_domain_raw']:.5f}",
                    record["worst_domain_raw_domain"],
                ]
            )
        blocks.append(
            _print_table(
                f"Sweep A, {regime} (base {BASE_BITS_BY_REGIME[regime]} bits, "
                f"protected {PROTECTED_BITS} bits, "
                f"{SWEEP_A_BUDGET_FRACTION:.0%} budget)",
                [
                    "set",
                    "seed",
                    "prot",
                    *[f"rel {domain}" for domain in STAGE2B_DOMAINS],
                    "worst rel",
                    "domain",
                    "worst raw",
                    "domain",
                ],
                rows,
            )
        )
        summary = arm["random_worst_domain_relative"]
        positions = arm["deliberate_sets_in_standard_deviations"]
        blocks.append(
            _print_table(
                f"Sweep A, {regime}, {arm['random_sets']} random sets versus the "
                "three deliberate sets",
                ["quantity", "value"],
                [
                    ["random mean worst relative", f"{summary['mean']:+.6f}"],
                    ["random sd worst relative (ddof=1)", f"{summary['sd']:.6f}"],
                    ["random min worst relative", f"{summary['min']:+.6f}"],
                    ["random max worst relative", f"{summary['max']:+.6f}"],
                    [
                        "most-routed worst relative",
                        f"{arm['most_routed_worst_domain_relative']:+.6f}",
                    ],
                    [
                        "least-routed worst relative",
                        f"{arm['least_routed_worst_domain_relative']:+.6f}",
                    ],
                    [
                        "no-protection worst relative",
                        f"{arm['no_protection_worst_domain_relative']:+.6f}",
                    ],
                    [
                        "most-routed, standard deviations from random mean",
                        "n/a" if positions["most_routed"] is None
                        else f"{positions['most_routed']:+.2f}",
                    ],
                    [
                        "least-routed, standard deviations from random mean",
                        "n/a" if positions["least_routed"] is None
                        else f"{positions['least_routed']:+.2f}",
                    ],
                    [
                        "no-protection, standard deviations from random mean",
                        "n/a" if positions["no_protection"] is None
                        else f"{positions['no_protection']:+.2f}",
                    ],
                    ["gap (least minus most)", f"{arm['gap']:+.6f}"],
                    [
                        "gap divided by sd_random",
                        "n/a" if arm["gap_over_sd_random"] is None
                        else f"{arm['gap_over_sd_random']:+.3f}",
                    ],
                    ["arm outcome", arm["outcome"]],
                ],
            )
        )
    return "\n\n".join(blocks), decision


def report_sweep_b(records: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    decision = sweep_b_decision(records)
    by_id = {record["run_id"]: record for record in records}
    rows = []
    for layer in SWEEP_B_LAYERS:
        record = by_id[f"b_layer{layer:02d}"]
        rows.append(
            [
                str(layer),
                *[
                    f"{record['relative_delta_by_domain'][domain]:+.5f}"
                    for domain in STAGE2B_DOMAINS
                ],
                f"{record['worst_domain_relative']:+.5f}",
                record["worst_domain_relative_domain"],
                *[f"{record['delta_by_domain'][domain]:+.5f}" for domain in STAGE2B_DOMAINS],
            ]
        )
    text = _print_table(
        f"Sweep B, one layer's {64} experts at {SWEEP_B_BITS} bits, all else BF16",
        [
            "layer",
            *[f"rel {domain}" for domain in STAGE2B_DOMAINS],
            "worst rel",
            "domain",
            *[f"nll {domain}" for domain in STAGE2B_DOMAINS],
        ],
        rows,
    )
    text += "\n\n" + _print_table(
        "Sweep B threshold",
        ["quantity", "value"],
        [
            [
                "largest worst-domain increase",
                f"{decision['largest_increase']:+.6f} (layer "
                f"{decision['largest_increase_layer']})",
            ],
            [
                "smallest worst-domain increase",
                f"{decision['smallest_increase']:+.6f} (layer "
                f"{decision['smallest_increase_layer']})",
            ],
            [
                "ratio",
                "undefined" if decision["ratio"] is None else f"{decision['ratio']:.3f}",
            ],
            ["outcome", decision["outcome"]],
        ],
    )
    return text, decision


def report_sweep_c(
    sweep_c_records: Sequence[Mapping[str, Any]],
    sweep_a_records: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    report = sweep_c_report(sweep_c_records, sweep_a_records)
    memory = report.get("router_memory") or {}
    rows = [
        [
            f"worst-domain relative, routers at {report['router_bits']} bits",
            f"{report['worst_domain_relative_quantized_routers']:+.6f}",
        ],
        [
            "worst-domain relative, routers at BF16",
            f"{report['worst_domain_relative_bf16_routers']:+.6f}",
        ],
        ["difference", f"{report['worst_domain_relative_difference']:+.6f}"],
    ]
    for domain in STAGE2B_DOMAINS:
        rows.append(
            [f"per-domain difference, {domain}", f"{report['per_domain_difference'][domain]:+.6f}"]
        )
    if memory:
        rows.extend(
            [
                ["router tensors", str(memory["router_tensor_count"])],
                ["router parameters", f"{memory['router_parameters']:,}"],
                ["router bytes at BF16", f"{memory['router_bf16_bytes']:,}"],
                [
                    f"router bytes at {report['router_bits']} bits",
                    f"{memory['router_quantized_bytes']:,}",
                ],
                [
                    "router share of all parameters",
                    f"{memory['router_share_of_all_parameters']:.5%}",
                ],
                [
                    "router share of deployed bytes",
                    f"{memory['router_share_of_deployed_bytes']:.5%}",
                ],
                [
                    "memory saved by quantizing routers",
                    f"{memory['bytes_saved_by_quantizing_routers']:,} bytes "
                    f"({memory['saving_share_of_deployed_bytes']:.5%} of deployed)",
                ],
            ]
        )
    text = _print_table(
        f"Sweep C, every expert at {SWEEP_C_EXPERT_BITS} bits, routers quantized "
        "versus routers at BF16",
        ["quantity", "value"],
        rows,
    )
    return text, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results") / STAGE3D_RESULTS_DIRNAME
    )
    parser.add_argument(
        "--sweeps",
        default="a,b,c",
        help="Comma-separated sweeps to report. Sweep C also needs sweep A.",
    )
    args = parser.parse_args()
    wanted = [item.strip() for item in args.sweeps.split(",") if item.strip()]

    records = {
        sweep: read_run_records(sweep_jsonl_path(args.results_dir, sweep))
        for sweep in ("a", "b", "c")
    }
    sections: list[str] = []
    decisions: dict[str, Any] = {}

    if "a" in wanted:
        if not records["a"]:
            raise RuntimeError("Sweep A has no records yet")
        write_csv(
            args.results_dir / "stage3d_a.csv",
            [_relative_row(record) for record in records["a"]],
            list(_relative_row(records["a"][0])),
        )
        text, decisions["sweep_a"] = report_sweep_a(records["a"])
        sections.append(text)
    if "b" in wanted:
        if not records["b"]:
            raise RuntimeError("Sweep B has no records yet")
        write_csv(
            args.results_dir / "stage3d_b.csv",
            [_relative_row(record) for record in records["b"]],
            list(_relative_row(records["b"][0])),
        )
        text, decisions["sweep_b"] = report_sweep_b(records["b"])
        sections.append(text)
    if "c" in wanted:
        if not records["c"]:
            raise RuntimeError("Sweep C has no records yet")
        if not records["a"]:
            raise RuntimeError("Sweep C needs Sweep A for its BF16-router reference")
        write_csv(
            args.results_dir / "stage3d_c.csv",
            [_relative_row(record) for record in records["c"]],
            list(_relative_row(records["c"][0])),
        )
        text, decisions["sweep_c"] = report_sweep_c(records["c"], records["a"])
        sections.append(text)

    baseline_text = _print_table(
        "Step 4, what the project currently calls the baseline",
        ["property", "value"],
        [
            ["name", PAPER_BASELINE["name"]],
            ["quantizes routers", "no"],
            ["quantizes lm_head", "no"],
            ["quantizes embeddings", "no"],
            ["quantizes attention", "no"],
            ["left at BF16", PAPER_BASELINE["left_at_bf16"]],
        ],
    )
    sections.append(baseline_text)

    outcome_lines = ["Step 5 outcomes:"]
    if "sweep_a" in decisions:
        arm_text = ", ".join(
            f"{regime} {arm['outcome']}"
            for regime, arm in decisions["sweep_a"]["arms"].items()
        )
        outcome_lines.append(
            f"  Sweep A: {decisions['sweep_a']['outcome']} ({arm_text}). "
            f"{decisions['sweep_a']['meaning']}."
        )
        outcome_lines.append(f"    {decisions['sweep_a']['note']}.")
    if "sweep_b" in decisions:
        outcome_lines.append(
            f"  Sweep B: {decisions['sweep_b']['outcome']}. "
            f"{decisions['sweep_b']['meaning']}."
        )
        if decisions["sweep_b"]["note"]:
            outcome_lines.append(f"    {decisions['sweep_b']['note']}.")
    if "sweep_c" in decisions:
        outcome_lines.append(
            "  Sweep C: no threshold. Worst-domain relative difference from "
            f"quantizing routers is "
            f"{decisions['sweep_c']['worst_domain_relative_difference']:+.6f}."
        )
    outcome_text = "\n".join(outcome_lines)
    print("\n" + outcome_text)
    sections.append(outcome_text)

    decision_payload = {
        "stage": STAGE3D_STAGE,
        "paper_baseline": PAPER_BASELINE,
        "thresholds": DECISION_THRESHOLDS,
        "primary_regime": PRIMARY_REGIME,
        "secondary_regime": SECONDARY_REGIME,
        **decisions,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(args.results_dir / "stage3d_decision.json", decision_payload)
    (args.results_dir / "SUMMARY.md").write_text(
        "# Stage 3D diagnostics\n\n```\n" + "\n\n".join(sections) + "\n```\n",
        encoding="utf-8",
    )
    print(f"\nDecision: {args.results_dir / 'stage3d_decision.json'}")
    print(f"Summary: {args.results_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
