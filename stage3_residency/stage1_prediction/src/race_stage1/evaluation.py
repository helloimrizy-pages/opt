from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from residency_headroom.common import (
    atomic_write_json,
    atomic_write_jsonl,
    environment_record,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)

from .calibration import load_and_verify_stage1_frozen
from .frozen import (
    load_and_verify_frozen_inputs,
    source_bundle_hash,
    stage0_references,
)
from .models import TransitionModels
from .quality import evaluate_prediction_quality
from .simulation import (
    method_id,
    simulate_causal_capacities,
    simulate_lookahead_capacities,
)


def run_evaluation(
    repository_root: Path,
    preregistration_path: Path,
    frozen_config_path: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    workload_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_inputs = load_and_verify_frozen_inputs(repository_root, preregistration_path)
    stage1_frozen = load_and_verify_stage1_frozen(frozen_config_path)
    _verify_stage1_freeze(repository_root, frozen_inputs.preregistration_hash, stage1_frozen)
    names = (
        tuple(workload_names)
        if workload_names is not None
        else tuple(workload.name for workload in frozen_inputs.workloads)
    )
    valid_names = {workload.name for workload in frozen_inputs.workloads}
    if not names or len(set(names)) != len(names) or not set(names).issubset(valid_names):
        raise ValueError("Requested evaluation workloads do not match the frozen Stage 0 suite")

    arguments = [
        (
            str(repository_root),
            str(preregistration_path.resolve()),
            str(frozen_config_path.resolve()),
            str(output_dir.resolve()),
            name,
        )
        for name in names
    ]
    if workers > 1 and len(arguments) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_evaluate_workload_task, args): args[-1] for args in arguments}
            for future in as_completed(futures):
                name = futures[future]
                future.result()
                print(f"evaluation complete: {name}", flush=True)
    else:
        for args in arguments:
            _evaluate_workload_task(args)
            print(f"evaluation complete: {args[-1]}", flush=True)

    workload_order = [workload.name for workload in frozen_inputs.workloads if workload.name in names]
    all_results: list[dict[str, Any]] = []
    all_sequences: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []
    checkpoint_manifests = []
    for name in workload_order:
        directory = output_dir / "checkpoints" / name
        manifest = read_json(directory / "manifest.json")
        _verify_checkpoint(directory, manifest, stage1_frozen["file_sha256"])
        checkpoint_manifests.append(manifest)
        all_results.extend(read_jsonl(directory / "results.jsonl"))
        all_sequences.extend(read_jsonl(directory / "per_sequence_results.jsonl"))
        quality_path = directory / "prediction_quality.jsonl"
        if quality_path.exists():
            all_quality.extend(read_jsonl(quality_path))
    atomic_write_jsonl(output_dir / "results.jsonl", all_results)
    atomic_write_jsonl(output_dir / "per_sequence_results.jsonl", all_sequences)
    atomic_write_jsonl(output_dir / "prediction_quality.jsonl", all_quality)

    sanity = _sanity_checks(
        repository_root,
        frozen_inputs,
        stage1_frozen,
        all_results,
        all_sequences,
        workload_order,
    )
    atomic_write_json(output_dir / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("Stage 1 full-evaluation sanity checks failed")
    manifest = {
        "schema_version": "race_stage1_evaluation_manifest_v1",
        "completed_at_utc": utc_now(),
        "passed": True,
        "workloads": workload_order,
        "conditions": len(all_results),
        "per_sequence_rows": len(all_sequences),
        "prediction_quality_rows": len(all_quality),
        "trace_hash": frozen_inputs.trace.trace_hash,
        "preregistration_hash": frozen_inputs.preregistration_hash,
        "frozen_config_file_sha256": stage1_frozen["file_sha256"],
        "source_bundle_hash": stage1_frozen["stage1_source_bundle_hash"],
        "transition_model_hash": stage1_frozen["transition_model_hash"],
        "results_sha256": sha256_file(output_dir / "results.jsonl"),
        "per_sequence_results_sha256": sha256_file(
            output_dir / "per_sequence_results.jsonl"
        ),
        "prediction_quality_sha256": sha256_file(
            output_dir / "prediction_quality.jsonl"
        ),
        "sanity_checks_sha256": sha256_file(output_dir / "sanity_checks.json"),
        "checkpoint_manifests": checkpoint_manifests,
        "environment": environment_record(),
    }
    atomic_write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest


def _evaluate_workload_task(arguments: tuple[str, str, str, str, str]) -> None:
    repository_root = Path(arguments[0])
    preregistration_path = Path(arguments[1])
    frozen_config_path = Path(arguments[2])
    output_dir = Path(arguments[3])
    workload_name = arguments[4]
    frozen_inputs = load_and_verify_frozen_inputs(repository_root, preregistration_path)
    stage1_frozen = load_and_verify_stage1_frozen(frozen_config_path)
    _verify_stage1_freeze(repository_root, frozen_inputs.preregistration_hash, stage1_frozen)
    workload = next(
        (item for item in frozen_inputs.workloads if item.name == workload_name), None
    )
    if workload is None:
        raise ValueError(f"Unknown frozen workload {workload_name}")
    checkpoint = output_dir / "checkpoints" / workload.name
    manifest_path = checkpoint / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        try:
            _verify_checkpoint(checkpoint, manifest, stage1_frozen["file_sha256"])
            print(f"evaluation checkpoint reused: {workload.name}", flush=True)
            return
        except (ValueError, FileNotFoundError):
            pass

    models = TransitionModels.load(repository_root / stage1_frozen["transition_model_path"])
    if models.trace_hash != frozen_inputs.trace.trace_hash:
        raise ValueError("Transition model references a different trace")
    if models.calibration_workload_hash != frozen_inputs.calibration.hash:
        raise ValueError("Transition model references a different calibration workload")
    capacities = tuple(map(int, stage1_frozen["cache_capacities"]))
    specs = tuple(dict(item) for item in stage1_frozen["all_causal_specs"])
    results = []
    for spec in specs:
        print(f"{workload.name}: {method_id(spec)}", flush=True)
        results.extend(
            simulate_causal_capacities(
                frozen_inputs.trace, workload, capacities, spec, models
            )
        )
    print(f"{workload.name}: limited-lookahead diagnostics", flush=True)
    results.extend(
        simulate_lookahead_capacities(
            frozen_inputs.trace,
            workload,
            capacities,
            stage1_frozen["limited_lookahead_horizons"],
            include_perfect=True,
        )
    )
    result_rows = [
        result.result_record(
            trace_hash=frozen_inputs.trace.trace_hash,
            preregistration_hash=frozen_inputs.preregistration_hash,
            model_hash=stage1_frozen["transition_model_hash"],
        )
        for result in results
    ]
    sequence_rows = [
        row
        for result in results
        for row in result.sequence_records(
            trace_hash=frozen_inputs.trace.trace_hash,
            preregistration_hash=frozen_inputs.preregistration_hash,
            model_hash=stage1_frozen["transition_model_hash"],
        )
    ]
    expected_conditions = (len(specs) + len(stage1_frozen["limited_lookahead_horizons"]) + 1) * len(capacities)
    if len(result_rows) != expected_conditions:
        raise RuntimeError(
            f"Workload {workload.name} produced {len(result_rows)}/{expected_conditions} conditions"
        )
    checkpoint.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(checkpoint / "results.jsonl", result_rows)
    atomic_write_jsonl(checkpoint / "per_sequence_results.jsonl", sequence_rows)
    quality_rows: list[dict[str, Any]] = []
    if workload.name == "mixed_interleaved":
        selected_horizon = int(stage1_frozen["selected_by_family"]["markov_h"]["horizon"])
        for spec in specs:
            print(f"{workload.name}: quality {method_id(spec)}", flush=True)
            quality_rows.append(
                evaluate_prediction_quality(
                    frozen_inputs.trace,
                    workload,
                    spec,
                    models,
                    additional_window_horizon=(
                        selected_horizon
                        if spec["method"] in {"markov_h", "markov_plus_ewma"}
                        else None
                    ),
                )
            )
        atomic_write_jsonl(checkpoint / "prediction_quality.jsonl", quality_rows)
    manifest = {
        "schema_version": "race_stage1_workload_checkpoint_v1",
        "completed_at_utc": utc_now(),
        "workload": workload.name,
        "regime": workload.regime,
        "workload_hash": workload.hash,
        "frozen_config_file_sha256": stage1_frozen["file_sha256"],
        "conditions": len(result_rows),
        "per_sequence_rows": len(sequence_rows),
        "prediction_quality_rows": len(quality_rows),
        "results_sha256": sha256_file(checkpoint / "results.jsonl"),
        "per_sequence_results_sha256": sha256_file(
            checkpoint / "per_sequence_results.jsonl"
        ),
        "prediction_quality_sha256": (
            sha256_file(checkpoint / "prediction_quality.jsonl")
            if quality_rows
            else None
        ),
    }
    atomic_write_json(manifest_path, manifest)


def _verify_stage1_freeze(
    repository_root: Path, preregistration_hash: str, frozen_config: Mapping[str, Any]
) -> None:
    if frozen_config["preregistration_hash"] != preregistration_hash:
        raise ValueError("Stage 1 frozen config references another preregistration")
    selection_path = repository_root / frozen_config["selection_path"]
    if sha256_file(selection_path) != frozen_config["selection_file_sha256"]:
        raise ValueError("Calibration selection changed after freezing")
    model_path = repository_root / frozen_config["transition_model_path"]
    if sha256_file(model_path) != frozen_config["transition_model_hash"]:
        raise ValueError("Transition models changed after freezing")
    current_source = source_bundle_hash(repository_root / "stage3_residency/stage1_prediction")
    if current_source != frozen_config["stage1_source_bundle_hash"]:
        raise ValueError(
            "Stage 1 source/config/tests changed after calibration freeze; "
            "create an explicitly exploratory run instead"
        )


def _verify_checkpoint(
    directory: Path, manifest: Mapping[str, Any], frozen_config_hash: str
) -> None:
    if manifest["frozen_config_file_sha256"] != frozen_config_hash:
        raise ValueError("Checkpoint belongs to another frozen Stage 1 config")
    if sha256_file(directory / "results.jsonl") != manifest["results_sha256"]:
        raise ValueError("Checkpoint result hash mismatch")
    if (
        sha256_file(directory / "per_sequence_results.jsonl")
        != manifest["per_sequence_results_sha256"]
    ):
        raise ValueError("Checkpoint per-sequence hash mismatch")
    quality_hash = manifest.get("prediction_quality_sha256")
    if quality_hash is not None and sha256_file(directory / "prediction_quality.jsonl") != quality_hash:
        raise ValueError("Checkpoint prediction-quality hash mismatch")


def _sanity_checks(
    repository_root: Path,
    frozen_inputs: Any,
    stage1_frozen: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    per_sequence: Sequence[Mapping[str, Any]],
    workload_names: Sequence[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    expected_per_workload = (
        len(stage1_frozen["all_causal_specs"])
        + len(stage1_frozen["limited_lookahead_horizons"])
        + 1
    ) * len(stage1_frozen["cache_capacities"])
    add(
        "condition_count",
        len(results) == len(workload_names) * expected_per_workload,
        {"actual": len(results), "expected": len(workload_names) * expected_per_workload},
    )
    identities = [
        (row["workload"], int(row["capacity"]), row["method_id"]) for row in results
    ]
    add("unique_conditions", len(identities) == len(set(identities)))
    accounting = [
        row["condition_id"]
        for row in results
        if int(row["hits"]) + int(row["misses"]) != int(row["requests"])
        or int(row["misses"]) != int(row["admissions"])
    ]
    add("event_accounting", not accounting, accounting[:10])
    capacity_failures = [
        row["condition_id"]
        for row in results
        if int(row["maximum_occupancy"]) > int(row["capacity"])
    ]
    add("capacity_invariants", not capacity_failures, capacity_failures[:10])
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in per_sequence:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    aggregation_failures = []
    for row in results:
        sequence_rows = by_condition.get(str(row["condition_id"]), [])
        if (
            sum(int(item["misses"]) for item in sequence_rows) != int(row["misses"])
            or sum(int(item["requests"]) for item in sequence_rows) != int(row["requests"])
        ):
            aggregation_failures.append(row["condition_id"])
    add("per_sequence_aggregation", not aggregation_failures, aggregation_failures[:10])

    references = stage0_references(
        repository_root / frozen_inputs.preregistration["stage0_reference"]["result_path"]
    )
    oracle_violations = []
    capacity8_violations = []
    perfect_violations = []
    for row in results:
        key = (str(row["workload"]), int(row["capacity"]))
        oracle_cost = int(references[key]["oracle"]["misses"])
        if oracle_cost > int(row["misses"]):
            oracle_violations.append(
                {"condition": row["condition_id"], "oracle": oracle_cost, "method": row["misses"]}
            )
        if int(row["capacity"]) == frozen_inputs.trace.top_k and int(row["misses"]) != oracle_cost:
            capacity8_violations.append(row["condition_id"])
        if row["method"] == "perfect_score_simple_policy" and int(row["misses"]) != oracle_cost:
            perfect_violations.append(row["condition_id"])
    add("stage0_oracle_dominance", not oracle_violations, oracle_violations[:10])
    add("capacity8_degenerate_sanity", not capacity8_violations, capacity8_violations[:10])
    add("perfect_score_matches_full_oracle", not perfect_violations, perfect_violations[:10])

    lookup = {
        (str(row["workload"]), int(row["capacity"]), str(row["method_id"])): int(row["misses"])
        for row in results
    }
    monotonic_violations = []
    horizon_ids = [f"lookahead_oracle_h{value}" for value in stage1_frozen["limited_lookahead_horizons"]]
    for workload in workload_names:
        for capacity in stage1_frozen["cache_capacities"]:
            values = [lookup[(workload, int(capacity), identifier)] for identifier in horizon_ids]
            for before, after, first, second in zip(values, values[1:], horizon_ids, horizon_ids[1:]):
                if after > before:
                    monotonic_violations.append(
                        {
                            "workload": workload,
                            "capacity": capacity,
                            "shorter": first,
                            "longer": second,
                            "shorter_cost": before,
                            "longer_cost": after,
                        }
                    )
    add(
        "limited_lookahead_nondegradation",
        not monotonic_violations,
        {
            "violations": monotonic_violations,
            "interpretation_if_present": (
                "Receding finite-horizon first actions are each horizon-optimal, but deterministic "
                "terminal ties can produce nonmonotone realized full-path costs."
            ),
        },
    )
    causal_tag_failures = [
        row["condition_id"]
        for row in results
        if (str(row["method"]).startswith("lookahead_") or row["method"] == "perfect_score_simple_policy")
        == bool(row["causal"])
    ]
    add("causal_diagnostic_labels", not causal_tag_failures, causal_tag_failures[:10])
    add(
        "calibration_evaluation_isolation",
        bool(read_json(repository_root / stage1_frozen["selection_path"])["calibration_evaluation_disjoint"]),
    )
    add("stage0_inputs_hash_verified", True)
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "race_stage1_sanity_checks_v1",
        "passed": passed,
        "checks": checks,
        "bootstrap_scope": (
            "Subsequent intervals reweight per-sequence contributions conditional on frozen "
            "stateful workload paths; trajectories are not replayed under reordered bootstrap paths."
        ),
    }
