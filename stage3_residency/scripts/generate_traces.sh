#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/verify_pilot_gate.py

PYTHONPATH=stage3_residency/src:src python stage3_residency/scripts/generate_traces.py \
  --config stage3_residency/configs/full_preregistered.json \
  --output-dir stage3_residency/traces/full \
  --cache-dir .hf_cache
