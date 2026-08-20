#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${RACE_STAGE3_PYTHON:-python}"
WORKERS="${RACE_STAGE3_WORKERS:-1}"

stage3_residency/stage3_ranking/scripts/run_tests.sh
"${PYTHON_BIN}" stage3_residency/stage3_ranking/scripts/build_dependency_manifest.py
"${PYTHON_BIN}" stage3_residency/stage3_ranking/scripts/run_calibration.py
"${PYTHON_BIN}" stage3_residency/stage3_ranking/scripts/run_pilot.py
"${PYTHON_BIN}" stage3_residency/stage3_ranking/scripts/run_evaluation.py --workers "${WORKERS}"
"${PYTHON_BIN}" stage3_residency/stage3_ranking/scripts/analyze_stage3.py
"${PYTHON_BIN}" stage3_residency/stage3_ranking/scripts/freeze_archive.py
