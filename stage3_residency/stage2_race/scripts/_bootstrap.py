"""Shared path bootstrap for the Stage 2 command-line entry points."""

from __future__ import annotations

import sys
from pathlib import Path


def repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / ".git").exists() and (candidate / "stage3_residency").exists():
            return candidate
    raise SystemExit("Could not locate the repository root")


ROOT = repository_root()
for relative in (
    "stage3_residency/stage2_race/src",
    "stage3_residency/stage1_prediction/src",
    "stage3_residency/src",
    "src",
):
    path = str(ROOT / relative)
    if path not in sys.path:
        sys.path.insert(0, path)

STAGE2 = ROOT / "stage3_residency/stage2_race"
PREREGISTRATION = STAGE2 / "configs/stage2_preregistered.json"
CALIBRATION_DIR = STAGE2 / "results/calibration"
PILOT_DIR = STAGE2 / "results/pilot"
EVALUATION_DIR = STAGE2 / "results/full"
FROZEN_CONFIG = CALIBRATION_DIR / "stage2_frozen_config.json"
