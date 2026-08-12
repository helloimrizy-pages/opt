#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.io_utils import read_json
from expert_analysis.plotting import create_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-quality figures from analyzed expert statistics."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/expert_domain_importance"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    results_path = input_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} does not exist; run analyze_expert_importance.py first"
        )
    paths = create_all_figures(read_json(results_path), input_dir)
    print(f"Created {len(paths)} figure files in {input_dir / 'figures'}")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
