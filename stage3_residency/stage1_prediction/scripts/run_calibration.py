#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/stage1_prediction/src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency/src"))

from race_stage1.calibration import load_and_verify_stage1_frozen, run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and freeze Stage 1 predictors on calibration only")
    parser.add_argument("--force", action="store_true", help="replace an existing unsealed calibration freeze")
    args = parser.parse_args()
    stage1 = REPOSITORY_ROOT / "stage3_residency/stage1_prediction"
    output = stage1 / "results/calibration"
    frozen = output / "stage1_frozen_config.json"
    if frozen.exists() and not args.force:
        value = load_and_verify_stage1_frozen(frozen)
        print(f"reusing calibration freeze {value['file_sha256']}")
        return
    if (stage1 / "reports/final_archive_manifest.json").exists():
        raise SystemExit("Refusing to replace a sealed Stage 1 archive")
    result = run_calibration(
        REPOSITORY_ROOT,
        stage1 / "configs/stage1_preregistered.json",
        output,
    )
    print(f"selected {result['selection']['selected_predictor_id']}")
    print(f"frozen config {result['frozen_config_file_sha256']}")


if __name__ == "__main__":
    main()
