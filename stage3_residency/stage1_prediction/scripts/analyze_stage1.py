#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/stage1_prediction/src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/src"))

from race_stage1.analysis import analyze_and_report


def main() -> None:
    stage1 = REPOSITORY_ROOT / "stage3_residency/stage1_prediction"
    result = analyze_and_report(
        REPOSITORY_ROOT,
        stage1 / "configs/stage1_preregistered.json",
        stage1 / "results/calibration/stage1_frozen_config.json",
        stage1 / "results/full",
        stage1,
    )
    print(result["verdict"])


if __name__ == "__main__":
    main()
