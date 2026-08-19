#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import read_json
from residency_headroom.trace import RoutingTrace


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Require an audited real pilot before full Stage 0 compute"
    )
    parser.add_argument(
        "--pilot-root", type=Path, default=Path("stage3_residency/results/pilot")
    )
    parser.add_argument(
        "--pilot-trace",
        type=Path,
        default=Path("stage3_residency/traces/pilot/routing_trace.npz"),
    )
    args = parser.parse_args()
    root = args.pilot_root
    trace = RoutingTrace.load(args.pilot_trace)
    if trace.metadata.get("trace_source") != "real deterministic OLMoE decode-token routing":
        raise RuntimeError("Pilot gate requires a real decode trace")
    frozen = read_json(root / "frozen" / "frozen_evaluation_config.json")
    if frozen["preregistered_config"]["run_kind"] != "pilot":
        raise RuntimeError("Pilot frozen config has the wrong run kind")
    if frozen["trace_hash"] != trace.trace_hash:
        raise RuntimeError("Pilot trace hash differs from the frozen pilot")
    sanity = read_json(root / "evaluation" / "sanity_checks.json")
    audit = read_json(root / "report" / "analysis_audit.json")
    analysis = read_json(root / "report" / "analysis.json")
    if not sanity.get("passed") or not audit.get("passed"):
        raise RuntimeError("Pilot sanity/audit gate did not pass")
    if analysis["decision"]["decision"] != "PILOT_ONLY_NO_STAGE0_DECISION":
        raise RuntimeError("Pilot improperly emitted a scientific decision")
    print(
        f"Pilot gate PASS: trace={trace.trace_hash}, frozen={frozen['config_hash']}, "
        f"sanity={sanity['passed']}, audit={audit['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
