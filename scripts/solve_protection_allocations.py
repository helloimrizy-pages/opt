#!/usr/bin/env python3
"""Solve and freeze every Stage 2B protection allocation before any evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.balanced import load_controlled_source
from expert_analysis.io_utils import atomic_save_npz, read_json
from expert_analysis.protection_allocations import (
    generate_all_allocations,
    load_frozen_registry,
    write_allocation_summaries,
)
from expert_analysis.protection_optimization import (
    BASE_BITS_BY_REGIME,
    PROTECTION_FRACTIONS,
    build_expert_memory_matrix,
    expert_tensor_shapes_from_config,
)
from expert_analysis.specialist_preservation import (
    NUM_MOE_LAYERS,
    build_specialist_scores,
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
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--allow-refreeze",
        action="store_true",
        help="Permit regenerating allocations only if no evaluation losses exist yet.",
    )
    args = parser.parse_args()

    verify_frozen_upstream_decisions(args.results_root)
    allocations_dir = args.output_dir / "allocations"
    registry_path = allocations_dir / "allocation_registry.json"
    losses_exist = any(
        (args.output_dir / phase / "losses").exists()
        for phase in ("development", "final")
    )
    if registry_path.exists():
        if losses_exist:
            raise RuntimeError(
                "Evaluation losses already exist; the frozen allocation registry "
                "must never be regenerated or edited"
            )
        if not args.allow_refreeze:
            print("Allocation registry already frozen; verifying it instead.")
            load_frozen_registry(allocations_dir)
            print("Frozen registry verified.")
            return 0

    source = load_controlled_source(args.source_dir)
    scores = build_specialist_scores(source)
    calibration_metadata_path = (
        args.output_dir / "calibration" / "calibration_metadata.json"
    )
    if calibration_metadata_path.exists():
        saved = read_json(calibration_metadata_path)
        if saved["calibration_fingerprint"] != scores.metadata["calibration_fingerprint"]:
            raise RuntimeError(
                "Recomputed calibration scores do not match the saved calibration "
                "package; refusing to solve against inconsistent scores"
            )

    shapes = expert_tensor_shapes_from_config(source.architecture["config"])
    memory = build_expert_memory_matrix([shapes] * NUM_MOE_LAYERS)
    atomic_save_npz(
        args.output_dir / "calibration" / "memory_matrix.npz",
        bytes_bits3=memory.bytes_by_bits[3],
        bytes_bits4=memory.bytes_by_bits[4],
        bytes_bits8=memory.bytes_by_bits[8],
        bytes_bits16=memory.bytes_by_bits[16],
        weight_count=memory.weight_count,
        group_count=memory.group_count,
        group_size=np.asarray([memory.group_size]),
        tensor_shapes=np.asarray(shapes, dtype=np.int64),
    )
    for regime, base_bits in BASE_BITS_BY_REGIME.items():
        total = memory.total_increment_bytes(base_bits)
        print(f"[{regime}] total protection increment: {total:,} bytes")
        for fraction in PROTECTION_FRACTIONS:
            print(
                f"  budget {int(fraction * 100)}%: "
                f"{memory.protection_budget_bytes(base_bits, fraction):,} bytes"
            )

    registry = generate_all_allocations(scores, memory, allocations_dir)
    coverage_path, summary_path = write_allocation_summaries(
        allocations_dir, args.output_dir
    )
    print(f"Froze {len(registry['entries'])} allocation records.")
    print(f"Registry SHA-256: {registry['registry_sha256']}")
    print(f"Coverage sanity table: {coverage_path}")
    print(f"Allocation summary table: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
