#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import atomic_write_json
from residency_headroom.oracle import validate_oracle


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate scalable atomic oracle against exact DP")
    parser.add_argument("--random-cases", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = validate_oracle(random_cases=args.random_cases, seed=args.seed)
    payload = result.as_dict()
    if args.output:
        atomic_write_json(args.output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
