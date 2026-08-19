from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    environment_record,
    hash_arrays,
    read_json,
    sha256_json,
    utc_now,
)
from .freeze import load_static_scores
from .simulator import (
    expected_unlimited_misses,
    per_sequence_rows,
    policy_specs,
    result_rows,
    simulate_oracle,
    simulate_policy,
)
from .trace import RoutingTrace
from .workloads import (
    build_calibration_workload,
    build_workloads,
    calibration_frequency_scores,
    split_sequences,
)


def run_evaluation(
    trace: RoutingTrace,
    frozen: Mapping[str, Any],
    frozen_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run/resume every preregistered simple policy and exact offline oracle."""

    started = time.monotonic()
    _validate_frozen(frozen)
    if trace.trace_hash != frozen["trace_hash"]:
        raise ValueError("Trace hash differs from the frozen evaluation")
    config = frozen["preregistered_config"]
    split = split_sequences(trace, float(config["calibration_fraction"]))
    if split.as_dict() != frozen["sequence_split"]:
        raise ValueError("Reconstructed calibration/evaluation split differs from freeze")
    calibration = build_calibration_workload(
        trace, split, int(config["mixed_workload_seed"])
    )
    workloads = build_workloads(trace, split, config)
    if {item.name: item.hash for item in workloads} != frozen["workload_hashes"]:
        raise ValueError("Reconstructed workload hashes differ from freeze")
    static_scores = load_static_scores(
        frozen_dir / "static_hotset_scores.npz", frozen["static_hotset_score_hash"]
    )
    recomputed_scores = calibration_frequency_scores(trace, calibration)
    if not np.array_equal(static_scores, recomputed_scores):
        raise ValueError("Static Hotset scores do not match calibration requests")
    recomputed_score_hash = hash_arrays(
        {"static_frequency": recomputed_scores},
        {
            "trace_hash": trace.trace_hash,
            "calibration_sequence_ids": list(calibration.sequence_ids),
        },
    )
    if recomputed_score_hash != frozen["static_hotset_score_hash"]:
        raise ValueError("Static Hotset score content hash failed")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "conditions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    all_result_rows: list[dict[str, Any]] = []
    all_sequence_rows: list[dict[str, Any]] = []
    base_records: list[dict[str, Any]] = []
    specs = policy_specs(config) + [{"policy": "oracle"}]
    expected_conditions = len(workloads) * len(config["cache_capacities"]) * len(specs)
    completed = 0
    for workload in workloads:
        domain_label = workload.domains[0] if len(workload.domains) == 1 else "+".join(workload.domains)
        for capacity in map(int, config["cache_capacities"]):
            for spec in specs:
                basis = {
                    "config_hash": frozen["config_hash"],
                    "trace_hash": trace.trace_hash,
                    "workload_hash": workload.hash,
                    "capacity": capacity,
                    **spec,
                }
                condition_hash = sha256_json(basis)
                checkpoint_path = checkpoints_dir / f"{condition_hash}.json"
                if checkpoint_path.exists():
                    checkpoint = read_json(checkpoint_path)
                    if checkpoint.get("condition_basis") != basis:
                        raise RuntimeError(f"Condition checkpoint identity changed: {checkpoint_path}")
                else:
                    if spec["policy"] == "oracle":
                        result = simulate_oracle(trace, workload, capacity)
                    else:
                        result = simulate_policy(
                            trace,
                            workload,
                            capacity,
                            spec["policy"],
                            alpha=spec.get("alpha"),
                            seed=spec.get("seed"),
                            static_scores=(
                                static_scores if spec["policy"] == "static_hotset" else None
                            ),
                        )
                    rows = result_rows(
                        result,
                        lambda_values=config["lambda_values"],
                        cost_models=config["cost_models"],
                        trace_hash=trace.trace_hash,
                        config_hash=frozen["config_hash"],
                        domain_label=domain_label,
                        selected_decay_alpha=float(frozen["selected_lfu_decay_alpha"]),
                    )
                    sequences = per_sequence_rows(
                        result,
                        trace_hash=trace.trace_hash,
                        config_hash=frozen["config_hash"],
                    )
                    checkpoint = {
                        "schema_version": "race_stage0_condition_checkpoint_v1",
                        "condition_hash": condition_hash,
                        "condition_basis": basis,
                        "base_record": result.base_record(),
                        "result_rows": rows,
                        "per_sequence_rows": sequences,
                    }
                    atomic_write_json(checkpoint_path, checkpoint)
                _validate_condition_checkpoint(checkpoint, basis)
                all_result_rows.extend(checkpoint["result_rows"])
                all_sequence_rows.extend(checkpoint["per_sequence_rows"])
                base_records.append(checkpoint["base_record"])
                completed += 1
                atomic_write_json(
                    output_dir / "evaluation_progress.json",
                    {
                        "schema_version": "race_stage0_evaluation_progress_v1",
                        "config_hash": frozen["config_hash"],
                        "trace_hash": trace.trace_hash,
                        "completed_conditions": completed,
                        "expected_conditions": expected_conditions,
                        "last_condition_hash": condition_hash,
                        "elapsed_seconds": time.monotonic() - started,
                        "updated_at_utc": utc_now(),
                    },
                )

    sanity = run_sanity_checks(
        trace,
        workloads,
        config,
        frozen,
        static_scores,
        base_records,
        all_sequence_rows,
    )
    atomic_write_jsonl(output_dir / "results.jsonl", all_result_rows)
    atomic_write_jsonl(output_dir / "per_sequence_results.jsonl", all_sequence_rows)
    atomic_write_json(output_dir / "sanity_checks.json", sanity)
    manifest = {
        "schema_version": "race_stage0_evaluation_manifest_v1",
        "config_hash": frozen["config_hash"],
        "trace_hash": trace.trace_hash,
        "run_kind": config["run_kind"],
        "conditions": expected_conditions,
        "result_rows": len(all_result_rows),
        "per_sequence_rows": len(all_sequence_rows),
        "cost_models": config["cost_models"],
        "lambda_values": config["lambda_values"],
        "policy_specs": specs,
        "selected_lfu_decay_alpha": frozen["selected_lfu_decay_alpha"],
        "sanity_passed": sanity["passed"],
        "environment": environment_record(),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(output_dir / "evaluation_manifest.json", manifest)
    if not sanity["passed"]:
        failures = [item["name"] for item in sanity["checks"] if not item["passed"]]
        raise RuntimeError(f"Stage 0 evaluation sanity checks failed: {failures}")
    return manifest


def run_sanity_checks(
    trace: RoutingTrace,
    workloads: list[Any],
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    static_scores: np.ndarray,
    base_records: list[dict[str, Any]],
    per_sequence: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    add(
        "oracle_exact_validation",
        bool(frozen["oracle_validation"]["passed"])
        and float(frozen["oracle_validation"]["maximum_cost_difference"]) <= 1e-10,
        frozen["oracle_validation"],
    )
    accounting_failures = [
        item
        for item in base_records
        if int(item["hits"]) + int(item["misses"]) != int(item["requests"])
        or int(item["admissions"]) != int(item["misses"])
        or int(item["maximum_occupancy"]) > int(item["capacity"])
    ]
    add("event_accounting_and_capacity", not accounting_failures, accounting_failures[:5])

    grouped = {
        (item["workload"], int(item["capacity"])): [] for item in base_records
    }
    for item in base_records:
        grouped[(item["workload"], int(item["capacity"]))].append(item)
    dominance_failures = []
    for condition, records in grouped.items():
        oracle = next(item for item in records if item["policy"] == "oracle")
        for baseline in records:
            if int(oracle["misses"]) > int(baseline["misses"]):
                dominance_failures.append(
                    {
                        "condition": condition,
                        "baseline": baseline["policy"],
                        "oracle": oracle["misses"],
                        "baseline_misses": baseline["misses"],
                    }
                )
    add("oracle_dominance", not dominance_failures, dominance_failures[:5])

    monotonic_failures = []
    for workload in workloads:
        values = [
            next(
                int(item["misses"])
                for item in base_records
                if item["workload"] == workload.name
                and item["policy"] == "oracle"
                and int(item["capacity"]) == capacity
            )
            for capacity in map(int, config["cache_capacities"])
        ]
        if any(after > before for before, after in zip(values, values[1:])):
            monotonic_failures.append({"workload": workload.name, "oracle_misses": values})
    add("oracle_cache_monotonicity", not monotonic_failures, monotonic_failures)

    minimum = min(map(int, config["cache_capacities"]))
    minimum_failures = []
    for workload in workloads:
        values = {
            int(item["misses"])
            for item in base_records
            if item["workload"] == workload.name and int(item["capacity"]) == minimum
        }
        if len(values) != 1:
            minimum_failures.append({"workload": workload.name, "miss_values": sorted(values)})
    add(
        "minimal_topk_capacity_policy_equivalence",
        not minimum_failures,
        {"capacity": minimum, "failures": minimum_failures},
    )

    unlimited_failures = []
    for workload in workloads:
        result = simulate_oracle(trace, workload, trace.num_experts)
        expected = expected_unlimited_misses(trace, workload)
        if result.misses != expected:
            unlimited_failures.append(
                {"workload": workload.name, "misses": result.misses, "expected": expected}
            )
    add("unlimited_cache_compulsory_loads_only", not unlimited_failures, unlimited_failures)

    zero = simulate_policy(trace, workloads[0], 0, "lru")
    add(
        "zero_cache_streaming_limit",
        zero.misses == zero.requests and zero.hits == 0 and zero.admissions == 0,
        zero.base_record(),
    )

    deterministic_spec = {
        "workload": workloads[0],
        "capacity": int(config["cache_capacities"][0]),
    }
    first = simulate_policy(
        trace, deterministic_spec["workload"], deterministic_spec["capacity"], "lru"
    )
    second = simulate_policy(
        trace, deterministic_spec["workload"], deterministic_spec["capacity"], "lru"
    )
    add(
        "deterministic_replay",
        first.base_record() == second.base_record()
        and first.per_sequence == second.per_sequence,
        {"workload": first.workload, "capacity": first.capacity},
    )

    split_calibration = {
        value for values in frozen["sequence_split"]["calibration"].values() for value in values
    }
    split_evaluation = {
        value for values in frozen["sequence_split"]["evaluation"].values() for value in values
    }
    add(
        "static_hotset_no_evaluation_leakage",
        not (split_calibration & split_evaluation)
        and int(static_scores.sum())
        == sum(
            int(item["generation_length"]) * trace.num_layers * trace.top_k
            for item in trace.metadata["sequences"]
            if int(item["sequence_id"]) in split_calibration
        ),
        {
            "calibration_sequences": len(split_calibration),
            "evaluation_sequences": len(split_evaluation),
            "score_assignments": int(static_scores.sum()),
        },
    )

    byte_failures = []
    if trace.metadata.get("all_experts_equal_size"):
        size = int(np.asarray(trace.metadata["expert_bytes_by_layer"])[0, 0])
        for item in base_records:
            if int(item["bytes_transferred"]) != int(item["misses"]) * size:
                byte_failures.append(item)
    add(
        "byte_cost_proportionality",
        not byte_failures,
        {
            "all_experts_equal_size": trace.metadata.get("all_experts_equal_size"),
            "failures": byte_failures[:5],
        },
    )

    sequence_sums: dict[str, dict[str, int]] = {}
    for row in per_sequence:
        values = sequence_sums.setdefault(
            row["condition_id"],
            {name: 0 for name in ("events", "requests", "hits", "misses", "admissions", "evictions")},
        )
        for name in values:
            values[name] += int(row[name])
    per_sequence_failures = []
    for item in base_records:
        values = sequence_sums[item["condition_id"]]
        if any(values[name] != int(item[name]) for name in values):
            per_sequence_failures.append({"condition_id": item["condition_id"], "sums": values})
    add("per_sequence_aggregation", not per_sequence_failures, per_sequence_failures[:5])

    random_seeds = {
        int(item["seed"])
        for item in base_records
        if item["policy"] == "random" and item["seed"] is not None
    }
    add(
        "random_seed_coverage",
        random_seeds == set(map(int, config["random_policy_seeds"])),
        sorted(random_seeds),
    )
    return {
        "schema_version": "race_stage0_sanity_v1",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def _validate_frozen(frozen: Mapping[str, Any]) -> None:
    basis = {
        key: value
        for key, value in frozen.items()
        if key not in {"config_hash", "frozen_at_utc"}
    }
    computed = sha256_json(basis)
    if computed != frozen.get("config_hash"):
        raise ValueError(
            f"Frozen config hash mismatch: recorded={frozen.get('config_hash')}, "
            f"computed={computed}"
        )
    if not frozen["oracle_validation"]["passed"]:
        raise ValueError("The frozen scalable oracle did not pass exact validation")


def _validate_condition_checkpoint(
    checkpoint: Mapping[str, Any], expected_basis: Mapping[str, Any]
) -> None:
    if checkpoint.get("condition_basis") != expected_basis:
        raise ValueError("Condition checkpoint basis mismatch")
    if checkpoint.get("condition_hash") != sha256_json(expected_basis):
        raise ValueError("Condition checkpoint hash mismatch")
    base = checkpoint["base_record"]
    if int(base["hits"]) + int(base["misses"]) != int(base["requests"]):
        raise ValueError("Condition checkpoint event accounting failed")
    if int(base["capacity"]) > 0 and int(base["admissions"]) != int(base["misses"]):
        raise ValueError("Condition checkpoint admission accounting failed")
