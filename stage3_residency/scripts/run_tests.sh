#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=stage3_residency/src:src python -m unittest discover \
  -s stage3_residency/tests -v
