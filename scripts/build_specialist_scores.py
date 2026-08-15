#!/usr/bin/env python3
"""Build the frozen Stage 2B calibration scores from the audited controlled run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.balanced import load_controlled_source
from expert_analysis.io_utils import atomic_write_json
from expert_analysis.specialist_preservation import (
    build_specialist_scores,
    save_specialist_scores,
)
from expert_analysis.stage2b_preflight import verify_frozen_upstream_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root containing the frozen upstream decision artifacts.",
    )
    args = parser.parse_args()

    preflight = verify_frozen_upstream_decisions(args.results_root)
    print("Frozen upstream decisions verified: STRONG GO / GO / SURROGATE_NO_GO")
    source = load_controlled_source(args.source_dir)
    print(
        f"Controlled source audited: fingerprint {source.input_fingerprint[:16]}..."
    )
    scores = build_specialist_scores(source)
    calibration_dir = args.output_dir / "calibration"
    paths = save_specialist_scores(scores, calibration_dir)
    atomic_write_json(args.output_dir / "audits" / "upstream_preflight.json", preflight)
    for domain, coverage in scores.metadata["domains_detail"].items():
        print(
            f"[{domain}] calibration examples: "
            f"{len(coverage['calibration_indices_into_frozen_set'])}"
        )
    print(f"Calibration fingerprint: {scores.metadata['calibration_fingerprint']}")
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
