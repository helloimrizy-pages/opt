#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import atomic_write_json
from residency_headroom.trace import RoutingTrace
from residency_headroom.transfer_calibration import measure_host_device_expert_transfer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure isolated host-device copies for one realistic expert tensor"
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    trace = RoutingTrace.load(args.trace)
    sizes = np.asarray(trace.metadata["expert_bytes_by_layer"], dtype=np.int64)
    if not np.all(sizes == sizes.flat[0]):
        raise RuntimeError("This benchmark currently requires identical expert tensor sizes")
    result = measure_host_device_expert_transfer(
        int(sizes.flat[0]), repeats=args.repeats, warmup=args.warmup
    )
    result["trace_hash"] = trace.trace_hash
    atomic_write_json(args.output, result)
    print(
        "Transfer calibration complete"
        if result["available"]
        else f"Transfer calibration unavailable: {result['reason']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
