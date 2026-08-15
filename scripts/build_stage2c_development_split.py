#!/usr/bin/env python3
"""Build the new Stage 2C seed-45 development split (data only, no model NLL).

The split is disjoint from the frozen calibration data, the original
controlled 100/domain set, the contaminated seed-43 Stage 2B development
split, and the untouched seed-44 final split. Seed-44 artifacts are verified
by hash only; no model evaluation happens here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis import DEFAULT_MODEL
from expert_analysis.balanced import EXPECTED_MODEL_REVISION, load_controlled_source
from expert_analysis.fragility_evaluation import build_stage2c_development_split
from expert_analysis.io_utils import atomic_write_json
from expert_analysis.stage2c_preflight import (
    verify_seed44_untouched,
    verify_stage2c_upstream,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/expert_domain_causal_validation"),
    )
    parser.add_argument(
        "--stage2b-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=EXPECTED_MODEL_REVISION)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    preflight = verify_stage2c_upstream(args.results_root)
    print(
        "Frozen upstream decisions verified: STRONG GO / GO / SURROGATE_NO_GO / "
        "ROBUST_PRESERVATION_NO_GO"
    )
    seed44 = verify_seed44_untouched(args.results_root, args.output_dir)
    print("Seed-44 final split verified untouched (hashes only, no evaluation).")
    atomic_write_json(
        args.output_dir / "audits" / "stage2c_preflight.json",
        {"upstream": preflight, "seed44_untouched": seed44},
    )

    source = load_controlled_source(args.source_dir)
    from transformers import AutoTokenizer

    tokenizer_kwargs = {"revision": args.model_revision}
    if args.cache_dir:
        tokenizer_kwargs["cache_dir"] = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
    manifest = build_stage2c_development_split(
        source,
        tokenizer,
        args.stage2b_dir,
        args.output_dir / "splits",
        cache_dir=args.cache_dir,
    )
    for domain, entry in manifest["domains"].items():
        print(
            f"[{domain}] seed-45 examples: {entry['num_examples']}, prior pool "
            f"fully excluded: {entry['prior_pool_fully_excluded']}, overlaps: "
            f"{sum(entry['overlap_checks'].values())}"
        )
    print(f"Split manifest: {args.output_dir / 'splits' / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
