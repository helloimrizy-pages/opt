#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/verify_pilot_gate.py

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/freeze_evaluation.py \
  --config stage3_residency/configs/full_preregistered.json \
  --trace stage3_residency/traces/full/routing_trace.npz \
  --output-dir stage3_residency/results/full/frozen

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/run_evaluation.py \
  --trace stage3_residency/traces/full/routing_trace.npz \
  --frozen-dir stage3_residency/results/full/frozen \
  --output-dir stage3_residency/results/full/evaluation

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/analyze.py \
  --trace stage3_residency/traces/full/routing_trace.npz \
  --frozen-dir stage3_residency/results/full/frozen \
  --evaluation-dir stage3_residency/results/full/evaluation \
  --output-dir stage3_residency/results/full/report
