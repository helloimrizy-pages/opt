#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${RACE_STAGE3_PYTHON:-python}"
export PYTHONPATH="stage3_residency/stage3_ranking/src:stage3_residency/stage2_race/src:stage3_residency/stage1_prediction/src:stage3_residency/src:stage3_residency/tests:src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m unittest discover -s stage3_residency/tests -v
"${PYTHON_BIN}" -m unittest discover -s stage3_residency/stage1_prediction/tests -v
"${PYTHON_BIN}" -m unittest discover -s stage3_residency/stage2_race/tests -v
"${PYTHON_BIN}" -m unittest discover -s stage3_residency/stage3_ranking/tests -v
