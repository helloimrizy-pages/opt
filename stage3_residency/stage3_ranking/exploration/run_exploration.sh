#!/usr/bin/env bash
# Re-run any calibration-only exploration script from the repository root, e.g.
#   stage3_residency/stage3_ranking/exploration/run_exploration.sh wall.py
# Every script here reads the frozen Stage 0 trace and the frozen Stage 0 CALIBRATION
# workload only. None of them touches an evaluation sequence.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
PYTHON_BIN="${RACE_STAGE3_PYTHON:-python}"
export PYTHONPATH="${ROOT}/stage3_residency/stage3_ranking/src:${ROOT}/stage3_residency/stage2_race/src:${ROOT}/stage3_residency/stage1_prediction/src:${ROOT}/stage3_residency/src:${ROOT}/src:${HERE}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${HERE}"
exec "${PYTHON_BIN}" -u "$@"
