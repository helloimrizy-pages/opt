#!/usr/bin/env bash
set -euo pipefail

run_kind="${1:-pilot}"

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/analyze.py \
  --trace "stage3_residency/traces/${run_kind}/routing_trace.npz" \
  --frozen-dir "stage3_residency/results/${run_kind}/frozen" \
  --evaluation-dir "stage3_residency/results/${run_kind}/evaluation" \
  --output-dir "stage3_residency/results/${run_kind}/report"
