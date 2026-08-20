#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${RACE_STAGE1_PYTHON:-python}"
WORKERS="${RACE_STAGE1_WORKERS:-1}"
export PYTHONPATH="stage3_residency/stage1_prediction/src:stage3_residency/src:src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" stage3_residency/stage1_prediction/scripts/run_evaluation.py --workers "${WORKERS}" "$@"
