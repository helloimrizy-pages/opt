#!/usr/bin/env python3
"""Solve and freeze every Stage 2C Fragility-Robust allocation.

Requires the frozen calibration fragility record. Generates ONLY the new
Fragility-Robust allocations for every regime/budget (all eight before any
development NLL), reuses the frozen Stage 2B comparator allocations by hash,
and freezes the Stage 2C registry plus the preregistration artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.fragility import load_frozen_fragility, load_frozen_stage2b_scores
from expert_analysis.fragility_evaluation import (
    build_stage2c_preregistration,
    write_stage2c_preregistration,
)
from expert_analysis.fragility_optimization import (
    generate_fragility_robust_allocations,
    load_frozen_stage2c_registry,
    write_stage2c_allocation_summary,
)
from expert_analysis.fragility_reporting import write_preevaluation_diagnostics
from expert_analysis.io_utils import read_json
from expert_analysis.protection_allocations import load_frozen_registry
from expert_analysis.protection_optimization import (
    build_expert_memory_matrix,
)
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS
from expert_analysis.stage2c_preflight import (
    verify_seed44_untouched,
    verify_stage2c_upstream,
)


def load_verified_memory_matrix(stage2b_dir: Path):
    """Rebuild the exact Stage-1 memory accounting and verify the frozen copy."""

    with np.load(
        stage2b_dir / "calibration" / "memory_matrix.npz", allow_pickle=False
    ) as data:
        shapes = [tuple(int(v) for v in shape) for shape in data["tensor_shapes"].tolist()]
        group_size = int(data["group_size"][0])
        saved = {bits: np.asarray(data[f"bytes_bits{bits}"]) for bits in (3, 4, 8, 16)}
    memory = build_expert_memory_matrix(
        [shapes] * NUM_MOE_LAYERS, group_size=group_size
    )
    for bits in (3, 4, 8, 16):
        if not np.array_equal(memory.bytes_by_bits[bits], saved[bits]):
            raise RuntimeError(
                f"Recomputed {bits}-bit memory accounting does not match the frozen "
                "Stage 2B matrix"
            )
    return memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2b-dir",
        type=Path,
        default=Path("results/robust_specialist_preservation"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--allow-refreeze",
        action="store_true",
        help="Permit regenerating allocations only if no evaluation losses exist yet.",
    )
    args = parser.parse_args()

    verify_stage2c_upstream(args.results_root)
    verify_seed44_untouched(args.results_root, args.results_dir)
    allocations_dir = args.results_dir / "allocations"
    stage2b_allocations_dir = args.stage2b_dir / "allocations"
    registry_path = allocations_dir / "allocation_registry.json"
    losses_exist = any(
        (args.results_dir / phase / "losses").exists()
        for phase in ("development_seed45", "final_seed44")
    )
    if registry_path.exists():
        if losses_exist:
            raise RuntimeError(
                "Evaluation losses already exist; the frozen Stage 2C allocation "
                "registry must never be regenerated or edited"
            )
        if not args.allow_refreeze:
            print("Stage 2C allocation registry already frozen; verifying it instead.")
            load_frozen_stage2c_registry(allocations_dir, stage2b_allocations_dir)
            print("Frozen Stage 2C registry verified.")
            return 0

    fragility_record = load_frozen_fragility(args.results_dir / "calibration")
    print(f"Frozen fragility verified: {fragility_record['fragility_sha256'][:16]}...")
    scores = load_frozen_stage2b_scores(args.stage2b_dir)
    print("Frozen Stage 2B specialization artifacts reloaded and hash-verified.")
    memory = load_verified_memory_matrix(args.stage2b_dir)
    print("Exact Stage-1 memory accounting verified against the frozen matrix.")
    stage2b_registry = load_frozen_registry(stage2b_allocations_dir)
    print(f"Frozen Stage 2B registry verified: {stage2b_registry['registry_sha256'][:16]}...")

    split_manifest_path = args.results_dir / "splits" / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise RuntimeError(
            "The seed-45 development split does not exist yet; run "
            "scripts/build_stage2c_development_split.py first so the "
            "preregistration can freeze the split hashes"
        )
    split_manifest = read_json(split_manifest_path)

    registry = generate_fragility_robust_allocations(
        scores,
        memory,
        fragility_record,
        allocations_dir,
        stage2b_registry,
        stage2b_allocations_dir,
    )
    print(
        f"Froze {len(registry['new_entries'])} new Fragility-Robust allocations and "
        f"reused {len(registry['reused_entries'])} frozen Stage 2B records."
    )
    print(f"Stage 2C registry SHA-256: {registry['registry_sha256']}")

    payload = build_stage2c_preregistration(registry, fragility_record, split_manifest)
    prereg_path, prereg_sha = write_stage2c_preregistration(args.results_dir, payload)
    print(f"Preregistration frozen: {prereg_path}")
    print(f"Preregistration SHA-256: {prereg_sha}")

    summary_path = write_stage2c_allocation_summary(
        registry, allocations_dir, stage2b_allocations_dir, args.results_dir
    )
    print(f"Allocation summary table: {summary_path}")
    csv_path, md_path = write_preevaluation_diagnostics(
        args.results_dir, stage2b_allocations_dir
    )
    print(f"Pre-evaluation diagnostics: {csv_path} / {md_path}")
    for entry in registry["new_entries"]:
        print(
            f"  {entry['file']}: {entry['allocation_sha256'][:16]}..."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
