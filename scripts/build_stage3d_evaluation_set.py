#!/usr/bin/env python3
"""Freeze the Stage 3D evaluation set and every protection set (no model needed).

Pools the frozen Stage 2B seed-43 and Stage 2C seed-45 development splits into
one 100-example-per-domain evaluation set, proves they are disjoint from each
other and free of seed-44 rows, and writes every Sweep A, B, and C bit
assignment to disk with its hash. Nothing here loads the model or evaluates a
loss, so it runs on any machine.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from expert_analysis.balanced import array_sha256, canonical_sha256
from expert_analysis.io_utils import atomic_save_npz, atomic_write_json, package_versions
from expert_analysis.protection_optimization import BASE_BITS_BY_REGIME, PROTECTED_BITS
from expert_analysis.specialist_preservation import STAGE2B_DOMAINS
from expert_analysis.stage3d_diagnostics import (
    DECISION_THRESHOLDS,
    EVALUATION_SPLIT_SEEDS,
    PRIMARY_REGIME,
    SECONDARY_REGIME,
    STAGE3D_RESULTS_DIRNAME,
    STAGE3D_STAGE,
    SWEEP_A_BUDGET_FRACTION,
    SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME,
    SWEEP_A_RANDOM_SEEDS,
    SWEEP_A_REGIMES,
    budget_protected_count,
    calibration_routing_counts,
    evaluation_set_manifest,
    git_commit,
    load_evaluation_set,
    load_frozen_memory_matrix,
    sweep_a_protection_sets,
    sweep_b_protection_sets,
    sweep_c_protection_sets,
)


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
        "--results-dir", type=Path, default=Path("results") / STAGE3D_RESULTS_DIRNAME
    )
    args = parser.parse_args()

    examples = load_evaluation_set(args.stage2b_dir, args.stage2c_dir)
    manifest = evaluation_set_manifest(examples, args.stage2b_dir, args.stage2c_dir)
    evaluation_dir = args.results_dir / "evaluation_set"
    for domain in STAGE2B_DOMAINS:
        examples[domain].save(evaluation_dir / f"{domain}.npz")
        atomic_write_json(
            evaluation_dir / f"{domain}.metadata.json", examples[domain].metadata
        )
    atomic_write_json(evaluation_dir / "evaluation_set_manifest.json", manifest)
    for domain in STAGE2B_DOMAINS:
        entry = manifest["domains"][domain]
        print(
            f"[{domain}] {entry['num_examples']} examples "
            f"({entry['seed43_rows']} from seed 43, {entry['seed45_rows']} from "
            f"seed 45), {entry['measured_tokens_per_example']} measured tokens each"
        )

    memory = load_frozen_memory_matrix(args.stage2b_dir)
    routing_counts = calibration_routing_counts(args.stage2b_dir)
    protection_sets = (
        sweep_a_protection_sets(memory, routing_counts)
        + sweep_b_protection_sets()
        + sweep_c_protection_sets()
    )

    allocations_dir = args.results_dir / "allocations"
    entries = []
    for protection in protection_sets:
        atomic_save_npz(
            allocations_dir / f"{protection.run_id}.npz",
            expert_bits=np.asarray(protection.bits, dtype=np.int64),
            protected=(
                np.zeros((0, 0), dtype=np.uint8)
                if protection.protected is None
                else protection.protected
            ),
        )
        entries.append(
            {
                "run_id": protection.run_id,
                "sweep": protection.sweep,
                "description": protection.description,
                "selection_rule": protection.selection_rule,
                "regime": protection.regime,
                "base_bits": protection.base_bits,
                "protected_bits": PROTECTED_BITS if protection.regime else None,
                "router_bits": protection.router_bits,
                "seed": protection.seed,
                "protected_expert_count": protection.protected_expert_count,
                "protected_experts": protection.protected_experts,
                "protection_sha256": protection.protection_sha256,
                "bits_matrix_sha256": protection.bits_sha256,
                "expert_bytes": memory.allocation_bytes(protection.bits),
                "effective_bits_per_weight": memory.effective_bits_per_weight(
                    protection.bits
                ),
            }
        )

    budgets = {
        regime: {
            "base_bits": BASE_BITS_BY_REGIME[regime],
            "protected_bits": PROTECTED_BITS,
            "budget_fraction": SWEEP_A_BUDGET_FRACTION,
            "budget_bytes": memory.protection_budget_bytes(
                BASE_BITS_BY_REGIME[regime], SWEEP_A_BUDGET_FRACTION
            ),
            "total_increment_bytes": memory.total_increment_bytes(
                BASE_BITS_BY_REGIME[regime]
            ),
            "protected_expert_count": budget_protected_count(memory, regime),
            "random_sets": SWEEP_A_RANDOM_SEED_COUNT_BY_REGIME[regime],
        }
        for regime in SWEEP_A_REGIMES
    }
    shared = {
        run_id.split("_random_seed")[-1]: entry["protection_sha256"]
        for run_id, entry in (
            (item["run_id"], item) for item in entries if "_random_seed" in item["run_id"]
        )
        if run_id.startswith(f"a_{PRIMARY_REGIME}_")
    }
    for item in entries:
        if "_random_seed" in item["run_id"] and item["run_id"].startswith(
            f"a_{SECONDARY_REGIME}_"
        ):
            seed = item["run_id"].split("_random_seed")[-1]
            if shared.get(seed) != item["protection_sha256"]:
                raise RuntimeError(
                    f"Random set for seed {seed} differs between the two arms; "
                    "the arms are supposed to share identical sets"
                )

    registry = {
        "schema": "stage3d_allocation_registry_v1",
        "stage": STAGE3D_STAGE,
        "evaluation_split_seeds": list(EVALUATION_SPLIT_SEEDS),
        "evaluation_set_input_hashes": {
            domain: manifest["domains"][domain]["input_ids_sha256"]
            for domain in STAGE2B_DOMAINS
        },
        "sweep_a_random_seeds": list(SWEEP_A_RANDOM_SEEDS),
        "sweep_a_random_sets_shared_across_regimes": True,
        "budgets": budgets,
        "routing_counts_sha256": array_sha256(routing_counts),
        "routing_counts_source": (
            "frozen Stage 2B calibration routing counts, summed over the four "
            "domains of the 25-example-per-domain calibration subset"
        ),
        "decision_thresholds": DECISION_THRESHOLDS,
        "entries": entries,
        **git_commit(),
        "package_versions": package_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    registry["registry_sha256"] = canonical_sha256(
        {
            "entries": [
                {
                    "run_id": item["run_id"],
                    "bits_matrix_sha256": item["bits_matrix_sha256"],
                    "protection_sha256": item["protection_sha256"],
                }
                for item in entries
            ],
            "budgets": budgets,
            "evaluation_set_input_hashes": registry["evaluation_set_input_hashes"],
        }
    )
    atomic_write_json(allocations_dir / "allocation_registry.json", registry)

    counts = {sweep: 0 for sweep in ("a", "b", "c")}
    for item in entries:
        counts[item["sweep"]] += 1
    print(
        f"\nFroze {len(entries)} configurations: sweep A {counts['a']}, "
        f"sweep B {counts['b']}, sweep C {counts['c']}."
    )
    for regime, budget in budgets.items():
        print(
            f"  {regime}: {budget['protected_expert_count']} of 1024 experts "
            f"protected at {budget['budget_bytes']} bytes "
            f"({budget['random_sets']} random sets)"
        )
    print(f"Registry: {allocations_dir / 'allocation_registry.json'}")
    print(f"Registry hash: {registry['registry_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
