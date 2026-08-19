#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import read_json
from residency_headroom.evaluation import run_evaluation
from residency_headroom.trace import RoutingTrace


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Stage 0 residency simulation")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen_dir = args.frozen_dir.resolve()
    manifest = run_evaluation(
        RoutingTrace.load(args.trace),
        read_json(frozen_dir / "frozen_evaluation_config.json"),
        frozen_dir,
        args.output_dir.resolve(),
    )
    print(
        f"Evaluation complete: {manifest['conditions']} conditions, "
        f"sanity_passed={manifest['sanity_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
