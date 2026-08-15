#!/usr/bin/env python3
"""Analyze Stage 2B evaluation losses: metrics, gates, decisions, figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.io_utils import read_json
from expert_analysis.protection_reporting import (
    analyze_phase,
    attach_protected_counts,
    create_phase_figures,
    write_development_decision,
    write_final_decision,
    write_phase_outputs,
)
from expert_analysis.report_stage2b import write_stage2b_summary, write_final_summary
from expert_analysis.stage2b_preflight import verify_frozen_upstream_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final"), required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    verify_frozen_upstream_decisions(args.results_root)
    phase_dir = args.results_dir / args.phase
    run_config = read_json(phase_dir / "run_config.json")
    allocations_dir = args.results_dir / "allocations"
    attach_protected_counts(allocations_dir)
    analysis = analyze_phase(
        args.phase,
        allocations_dir,
        phase_dir / "losses",
        run_config["run_fingerprint"],
    )
    paths = write_phase_outputs(analysis, phase_dir)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    if args.phase == "development":
        decision_path = write_development_decision(analysis, args.results_dir)
        decision = analysis["development_decision"]["decision"]
        print(f"Stage 2B development decision: {decision} ({decision_path})")
        if decision != "FULL_EVALUATION_GO":
            print(
                "NO_GO: all results preserved; the final evaluation set must not "
                "be inspected, the objective must not be altered, and no other "
                "method may be promoted."
            )
    else:
        write_final_decision(analysis, args.results_dir)
        print(f"Final decision: {analysis['final_decision']['decision']}")
        write_final_summary(args.results_dir, analysis)
    if not args.skip_figures:
        figures = create_phase_figures(
            analysis, allocations_dir, args.results_dir / "figures"
        )
        print(f"Created {len(figures)} figure files.")
    write_stage2b_summary(args.results_dir)
    print(f"Updated {args.results_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
