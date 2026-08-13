#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.balanced import (  # noqa: E402
    BALANCED_DOMAINS,
    build_preregistration,
    load_controlled_source,
    write_preregistration_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the balanced domain-specialist and routing-control panel using "
            "baseline controlled statistics only. This command never reads masking outcomes."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expert_domain_balanced_causal_validation"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = load_controlled_source(args.source_dir)
    payload = build_preregistration(source)
    write_preregistration_artifacts(payload, args.output_dir)
    print("Balanced causal panel is frozen before masking.")
    print(f"Domains: {', '.join(BALANCED_DOMAINS)}")
    for domain in BALANCED_DOMAINS:
        selected = [
            row for row in payload["selected_experts"] if row["target_domain"] == domain
        ]
        labels = ", ".join(f"L{row['layer']}/E{row['expert_id']}" for row in selected)
        print(f"{domain}: {labels}")
    print(f"Preregistration fingerprint: {payload['preregistration_fingerprint']}")
    print(
        "Frozen artifact: "
        f"{args.output_dir.resolve() / 'selected_experts_preregistered.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
