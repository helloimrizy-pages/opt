#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/stage1_prediction/src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/src"))

from race_stage1.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 on frozen Stage 0 workloads")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--workload", action="append", default=None)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    stage1 = REPOSITORY_ROOT / "stage3_residency/stage1_prediction"
    manifest = run_evaluation(
        REPOSITORY_ROOT,
        stage1 / "configs/stage1_preregistered.json",
        stage1 / "results/calibration/stage1_frozen_config.json",
        stage1 / "results/full",
        workers=args.workers,
        workload_names=args.workload,
    )
    print(f"evaluation PASS: {manifest['conditions']} conditions")


if __name__ == "__main__":
    main()
