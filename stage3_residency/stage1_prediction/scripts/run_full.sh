#!/usr/bin/env bash
set -euo pipefail

stage3_residency/stage1_prediction/scripts/run_tests.sh
stage3_residency/stage1_prediction/scripts/run_calibration.sh
stage3_residency/stage1_prediction/scripts/run_stage1_full.sh
stage3_residency/stage1_prediction/scripts/analyze_stage1.sh

PYTHON_BIN="${RACE_STAGE1_PYTHON:-python}"
export PYTHONPATH="stage3_residency/stage1_prediction/src:stage3_residency/src:src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" stage3_residency/stage1_prediction/scripts/freeze_archive.py
