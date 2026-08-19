#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "stage3_residency" / "src"))

from residency_headroom.common import atomic_write_json
from residency_headroom.simulator import (
    expected_unlimited_misses,
    simulate_oracle,
    simulate_policy,
)
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import make_workload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the real-checkpoint decode smoke mechanics audit"
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace = RoutingTrace.load(args.trace)
    validation = trace.validate(verify_hash=True)
    workload = make_workload(
        "real_decode_smoke",
        "smoke",
        [
            (
                int(item["sequence_id"]),
                0,
                "smoke",
                str(item["domain"]),
            )
            for item in trace.metadata["sequences"]
        ],
    )
    rows = []
    all_results = []
    for capacity in (8, 12, 16, 24, 32):
        results = {
            "oracle": simulate_oracle(trace, workload, capacity),
            "lru": simulate_policy(trace, workload, capacity, "lru"),
            "lfu": simulate_policy(trace, workload, capacity, "lfu"),
            "lfu_decay": simulate_policy(
                trace, workload, capacity, "lfu_decay", alpha=0.95
            ),
            "random": simulate_policy(
                trace, workload, capacity, "random", seed=1001
            ),
        }
        all_results.extend(results.values())
        rows.append(
            {
                "capacity": capacity,
                **{f"{name}_misses": result.misses for name, result in results.items()},
            }
        )

    oracle_values = [row["oracle_misses"] for row in rows]
    unlimited = simulate_oracle(trace, workload, trace.num_experts)
    expected_unlimited = expected_unlimited_misses(trace, workload)
    expert_bytes = np.asarray(trace.metadata["expert_bytes_by_layer"], dtype=np.int64)
    checks = {
        "atomic_trace_schema": validation["passed"],
        "all_layers_per_generated_token": all(
            count == validation["generated_tokens"]
            for count in validation["events_per_layer"].values()
        ),
        "event_accounting": all(
            result.hits + result.misses == result.requests
            and result.admissions == result.misses
            for result in all_results
        ),
        "oracle_dominance": all(
            row["oracle_misses"]
            <= min(value for key, value in row.items() if key.endswith("_misses"))
            for row in rows
        ),
        "oracle_cache_monotonicity": all(
            after <= before for before, after in zip(oracle_values, oracle_values[1:])
        ),
        "unlimited_cache_compulsory_loads_only": unlimited.misses
        == expected_unlimited,
    }
    audit = {
        "schema_version": "race_stage0_real_decode_smoke_audit_v1",
        "passed": all(checks.values()),
        "scientific_decision_enabled": False,
        "trace_validation": validation,
        "trace_source_bundle_hash": trace.metadata.get("stage3_source_bundle_hash"),
        "model": trace.metadata.get("model"),
        "model_revision": trace.metadata.get("resolved_model_revision"),
        "precision": trace.metadata.get("precision"),
        "runtime": trace.metadata.get("runtime"),
        "expert_bytes": int(expert_bytes.flat[0]),
        "all_experts_equal_size": bool(np.all(expert_bytes == expert_bytes.flat[0])),
        "policy_rows": rows,
        "unlimited_cache_misses": unlimited.misses,
        "expected_compulsory_unique_experts": expected_unlimited,
        "checks": checks,
        "policy_parameters": {"lfu_decay_alpha": 0.95, "random_seed": 1001},
        "limitation": (
            "Four prompts and two tokens each; mechanics smoke only, not the "
            "preregistered pilot or a Stage 0 result."
        ),
    }
    atomic_write_json(args.output, audit)
    if not audit["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Smoke audit failed: {failed}")
    print(
        f"Smoke audit PASS: trace={trace.trace_hash}, events={trace.num_events}, "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
