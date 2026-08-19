#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/generate_traces.py \
  --config stage3_residency/configs/pilot.json \
  --output-dir stage3_residency/traces/pilot \
  --cache-dir .hf_cache

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/freeze_evaluation.py \
  --config stage3_residency/configs/pilot.json \
  --trace stage3_residency/traces/pilot/routing_trace.npz \
  --output-dir stage3_residency/results/pilot/frozen

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/run_evaluation.py \
  --trace stage3_residency/traces/pilot/routing_trace.npz \
  --frozen-dir stage3_residency/results/pilot/frozen \
  --output-dir stage3_residency/results/pilot/evaluation

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/analyze.py \
  --trace stage3_residency/traces/pilot/routing_trace.npz \
  --frozen-dir stage3_residency/results/pilot/frozen \
  --evaluation-dir stage3_residency/results/pilot/evaluation \
  --output-dir stage3_residency/results/pilot/report
