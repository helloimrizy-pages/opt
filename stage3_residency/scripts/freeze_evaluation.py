#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import load_config
from residency_headroom.freeze import freeze_evaluation
from residency_headroom.trace import RoutingTrace


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze trace IDs, workloads, calibrated simple baseline, and oracle validation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-random-cases", type=int, default=500)
    args = parser.parse_args()
    frozen = freeze_evaluation(
        RoutingTrace.load(args.trace),
        load_config(args.config),
        args.output_dir.resolve(),
        oracle_random_cases=args.oracle_random_cases,
    )
    print(f"Frozen evaluation config: {frozen['config_hash']}")
    print(f"Selected global LFU-decay alpha: {frozen['selected_lfu_decay_alpha']}")
    print(
        "Oracle validation: "
        f"{frozen['oracle_validation']['exhaustive_cases']} exhaustive + "
        f"{frozen['oracle_validation']['random_cases']} random cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
