#!/usr/bin/env python3
"""Solve and freeze every Stage 3 Measured-Damage-Robust allocation.

Requires the frozen damage matrix. Generates ONLY the new
Measured-Damage-Robust allocations for every regime/budget (all eight before
any probe or development NLL), reuses the frozen Stage 2B and Stage 2C
comparator allocations by hash, and freezes the Stage 3 registry plus the
preregistration artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from expert_analysis.fragility import load_frozen_stage2b_scores
from expert_analysis.fragility_optimization import load_frozen_stage2c_registry
from expert_analysis.io_utils import read_json
from expert_analysis.measured_damage import load_frozen_damage
from expert_analysis.measured_damage_evaluation import (
    build_stage3_preregistration,
    write_stage3_preregistration,
)
from expert_analysis.measured_damage_optimization import (
    generate_stage3_allocations,
    load_frozen_stage3_registry,
    write_stage3_allocation_summary,
)
from expert_analysis.protection_allocations import load_frozen_registry
from expert_analysis.protection_optimization import build_expert_memory_matrix
from expert_analysis.specialist_preservation import NUM_MOE_LAYERS
from expert_analysis.stage3_preflight import (
    verify_seed44_untouched_stage3,
    verify_stage3_upstream,
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
        "--stage2c-dir",
        type=Path,
        default=Path("results/fragility_robust_preservation"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/measured_damage_preservation"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--allow-refreeze",
        action="store_true",
        help="Permit regenerating allocations only if no evaluation losses exist yet.",
    )
    args = parser.parse_args()

    verify_stage3_upstream(args.results_root)
    verify_seed44_untouched_stage3(args.results_root, args.results_dir)
    allocations_dir = args.results_dir / "allocations"
    stage2b_allocations_dir = args.stage2b_dir / "allocations"
    stage2c_allocations_dir = args.stage2c_dir / "allocations"
    registry_path = allocations_dir / "allocation_registry.json"
    losses_exist = any(
        (args.results_dir / phase / "losses").exists()
        for phase in ("additivity", "development_seed46", "final_seed44")
    )
    if registry_path.exists():
        if losses_exist:
            raise RuntimeError(
                "Evaluation losses already exist; the frozen Stage 3 allocation "
                "registry must never be regenerated or edited"
            )
        if not args.allow_refreeze:
            print("Stage 3 allocation registry already frozen; verifying it instead.")
            load_frozen_stage3_registry(
                allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir
            )
            print("Frozen Stage 3 registry verified.")
            return 0

    damage_record, damage_arrays = load_frozen_damage(args.results_dir / "damage")
    print(f"Frozen damage matrix verified: {damage_record['damage_sha256'][:16]}...")
    scores = load_frozen_stage2b_scores(args.stage2b_dir)
    print("Frozen Stage 2B specialization artifacts reloaded and hash-verified.")
    memory = load_verified_memory_matrix(args.stage2b_dir)
    print("Exact Stage-1 memory accounting verified against the frozen matrix.")
    stage2b_registry = load_frozen_registry(stage2b_allocations_dir)
    print(
        f"Frozen Stage 2B registry verified: "
        f"{stage2b_registry['registry_sha256'][:16]}..."
    )
    stage2c_registry = load_frozen_stage2c_registry(
        stage2c_allocations_dir, stage2b_allocations_dir
    )
    print(
        f"Frozen Stage 2C registry verified: "
        f"{stage2c_registry['registry_sha256'][:16]}..."
    )

    split_manifest_path = args.results_dir / "splits" / "split_manifest.json"
    if not split_manifest_path.is_file():
        raise RuntimeError(
            "The seed-46 development split does not exist yet; run "
            "scripts/build_stage3_development_split.py first so the "
            "preregistration can freeze the split hashes"
        )
    split_manifest = read_json(split_manifest_path)

    registry = generate_stage3_allocations(
        scores,
        memory,
        damage_record,
        damage_arrays["delta_nll"],
        allocations_dir,
        stage2b_registry,
        stage2b_allocations_dir,
        stage2c_registry,
        stage2c_allocations_dir,
    )
    print(
        f"Froze {len(registry['new_entries'])} new Measured-Damage-Robust "
        f"allocations; reused {len(registry['reused_stage2b_entries'])} Stage 2B "
        f"and {len(registry['reused_stage2c_entries'])} Stage 2C frozen records."
    )
    print(f"Stage 3 registry SHA-256: {registry['registry_sha256']}")

    payload = build_stage3_preregistration(registry, damage_record, split_manifest)
    prereg_path, prereg_sha = write_stage3_preregistration(args.results_dir, payload)
    print(f"Preregistration frozen: {prereg_path}")
    print(f"Preregistration SHA-256: {prereg_sha}")

    summary_path = write_stage3_allocation_summary(
        registry,
        allocations_dir,
        stage2b_allocations_dir,
        stage2c_allocations_dir,
        damage_arrays["delta_nll"],
        args.results_dir,
    )
    print(f"Allocation summary table: {summary_path}")
    for entry in registry["new_entries"]:
        print(f"  {entry['file']}: {entry['allocation_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
