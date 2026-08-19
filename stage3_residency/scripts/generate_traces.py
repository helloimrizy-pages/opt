#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from residency_headroom.common import load_config
from residency_headroom.trace_generation import generate_trace


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate real OLMoE decode routing traces")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Optional writable datasets cache when the model cache is read-only.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    trace = generate_trace(
        load_config(args.config),
        args.output_dir.resolve(),
        cache_dir=args.cache_dir,
        dataset_cache_dir=args.dataset_cache_dir,
        local_files_only=args.local_files_only,
    )
    validation = trace.validate()
    print(
        f"Trace complete: {validation['sequences']} sequences, "
        f"{validation['generated_tokens']} generated tokens, "
        f"{validation['events']} atomic events, hash={validation['trace_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
