#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
PYTHON_BIN="${RACE_PAPER_PYTHON:-python}"
export PYTHONPATH="${ROOT}/stage3_residency/src:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -u "${HERE}/build_paper_pack.py"
