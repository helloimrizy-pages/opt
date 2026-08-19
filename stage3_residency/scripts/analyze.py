#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import read_json, read_jsonl
from residency_headroom.diagnostics import all_workload_diagnostics
from residency_headroom.reporting import write_analysis_outputs
from residency_headroom.statistics import analyze_headroom
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import build_workloads, split_sequences


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze and report the Stage 0 oracle gap")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-calibration", type=Path, default=None)
    args = parser.parse_args()
    trace = RoutingTrace.load(args.trace)
    frozen = read_json(args.frozen_dir / "frozen_evaluation_config.json")
    sanity = read_json(args.evaluation_dir / "sanity_checks.json")
    if not sanity.get("passed"):
        raise RuntimeError("Refusing to analyze an evaluation that failed sanity checks")
    config = frozen["preregistered_config"]
    split = split_sequences(trace, float(config["calibration_fraction"]))
    workloads = build_workloads(trace, split, config)
    diagnostics = all_workload_diagnostics(trace, workloads)
    analysis = analyze_headroom(
        read_jsonl(args.evaluation_dir / "results.jsonl"),
        read_jsonl(args.evaluation_dir / "per_sequence_results.jsonl"),
        diagnostics,
        frozen,
    )
    hardware = read_json(args.hardware_calibration) if args.hardware_calibration else None
    outputs = write_analysis_outputs(
        analysis, frozen, args.output_dir.resolve(), hardware_calibration=hardware
    )
    print(f"Analysis complete: {analysis['decision']['decision']}")
    print(f"Report: {outputs['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
