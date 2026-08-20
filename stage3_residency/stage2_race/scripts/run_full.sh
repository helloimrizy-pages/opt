#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${RACE_STAGE2_PYTHON:-python}"
WORKERS="${RACE_STAGE2_WORKERS:-1}"
export PYTHONPATH="stage3_residency/stage2_race/src:stage3_residency/stage1_prediction/src:stage3_residency/src:src${PYTHONPATH:+:${PYTHONPATH}}"

stage3_residency/stage2_race/scripts/run_tests.sh
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/build_dependency_manifest.py
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/fit_models.py
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/run_pilot.py
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/run_calibration.py --workers "${WORKERS}"
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/run_evaluation.py --workers "${WORKERS}"
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/analyze_stage2.py
"${PYTHON_BIN}" stage3_residency/stage2_race/scripts/freeze_archive.py
