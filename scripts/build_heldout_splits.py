#!/usr/bin/env python3
"""Build the disjoint Stage 2B development (seed 43) and final (seed 44) splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import EXPECTED_MODEL_REVISION, load_controlled_source
from expert_analysis.heldout_splits import build_heldout_splits
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
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    verify_frozen_upstream_decisions(args.results_root)
    source = load_controlled_source(args.source_dir)
    from transformers import AutoTokenizer

    tokenizer_kwargs = {"revision": args.model_revision}
    if args.cache_dir:
        tokenizer_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    manifest = build_heldout_splits(
        source,
        tokenizer,
        args.output_dir / "splits",
        cache_dir=args.cache_dir,
    )
    for domain, entry in manifest["domains"].items():
        print(
            f"[{domain}] development {entry['development']['num_examples']} examples, "
            f"final {entry['final']['num_examples']} examples, prior pool fully "
            f"excluded: {entry['prior_pool_fully_excluded']}"
        )
    print(f"Split manifest: {args.output_dir / 'splits' / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
