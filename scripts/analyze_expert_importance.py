#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.analysis import analyze_results
from expert_analysis.report import write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze collected domain-conditioned expert statistics."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/expert_domain_importance"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--specialized-per-layer", type=int, default=10)
    args = parser.parse_args()
    if args.bootstrap_replicates < 0 or args.specialized_per_layer < 1:
        parser.error("bootstrap-replicates must be >= 0 and specialized-per-layer >= 1")
    return args


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    results = analyze_results(
        input_dir=input_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        specialized_per_layer=args.specialized_per_layer,
    )
    write_summary(results, input_dir / "SUMMARY.md")
    print(f"Analysis complete: {input_dir / 'results.json'}")
    print(f"Summary written: {input_dir / 'SUMMARY.md'}")
    print(
        f"Next: python scripts/plot_expert_importance.py --input-dir {input_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
