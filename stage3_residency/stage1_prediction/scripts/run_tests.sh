#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${RACE_STAGE1_PYTHON:-python}"
export PYTHONPATH="stage3_residency/stage1_prediction/src:stage3_residency/src:stage3_residency/tests:src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m unittest discover -s stage3_residency/tests -v
"${PYTHON_BIN}" -m unittest discover -s stage3_residency/stage1_prediction/tests -v
