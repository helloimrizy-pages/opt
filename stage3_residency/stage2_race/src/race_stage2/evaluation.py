"""One frozen Stage 2 evaluation pass over the ten sealed Stage 0 workload paths."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from race_stage1.models import TransitionModels
from residency_headroom.common import (
    atomic_write_json,
    atomic_write_jsonl,
    environment_record,
    read_json,
    read_jsonl,
    sha256_file,
    utc_now,
)

from .calibration import load_and_verify_stage2_frozen
from .frozen import load_and_verify_stage2_inputs, stage2_source_bundle_hash
from .policy import variant_from_spec
from .simulation import simulate_race_variant


TRAJECTORY_WORKLOAD = "mixed_interleaved"
_CONTEXT: dict[tuple[str, str, str], tuple[Any, TransitionModels]] = {}


def _context(root: str, preregistration: str, model_path: str):
    key = (root, preregistration, model_path)
    if key not in _CONTEXT:
        inputs = load_and_verify_stage2_inputs(Path(root), Path(preregistration))
        _CONTEXT[key] = (inputs, TransitionModels.load(Path(model_path)))
    return _CONTEXT[key]


def _evaluate_job(payload: Mapping[str, Any]) -> str:
    inputs, models = _context(
        payload["repository_root"], payload["preregistration_path"], payload["model_path"]
    )
    frozen = load_and_verify_stage2_frozen(Path(payload["frozen_config_path"]))
    if frozen["file_sha256"] != payload["frozen_config_sha256"]:
        raise ValueError("Stage 2 frozen config changed during evaluation")
    workload = next(
        (item for item in inputs.workloads if item.name == payload["workload"]), None
    )
    if workload is None:
        raise ValueError(f"Unknown frozen workload {payload['workload']}")
    if workload.hash != frozen["workload_hashes"][workload.name]:
        raise ValueError(f"Workload {workload.name} hash changed after freezing")
    variant = variant_from_spec(payload["spec"])
    directory = Path(payload["output_dir"]) / "checkpoints" / workload.name
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{payload['label']}"
    manifest_path = directory / f"{stem}.manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("frozen_config_sha256") == payload["frozen_config_sha256"] and all(
            sha256_file(directory / manifest[key + "_file"]) == manifest[key + "_sha256"]
            for key in ("results", "per_sequence", "diagnostics")
        ):
            return f"reused {workload.name}/{stem}"

    trajectory_stride = (
        int(payload["trajectory_stride"])
        if payload["label"] == payload["trajectory_label"]
        and workload.name == TRAJECTORY_WORKLOAD
        else None
    )
    results = simulate_race_variant(
        inputs.trace,
        workload,
        tuple(int(value) for value in frozen["cache_capacities"]),
        variant,
        models,
        enable_diagnostics=True,
        trajectory_stride=trajectory_stride,
    )
    trace_hash = inputs.trace.trace_hash
    config_hash = payload["frozen_config_sha256"]
    result_rows = [
        item.result_record(
            trace_hash=trace_hash,
            preregistration_hash=inputs.preregistration_hash,
            config_hash=config_hash,
        )
        for item in results
    ]
    sequence_rows = [
        row
        for item in results
        for row in item.sequence_records(trace_hash=trace_hash, config_hash=config_hash)
    ]
    diagnostic_rows = [
        item.diagnostic_record(trace_hash=trace_hash, config_hash=config_hash)
        for item in results
    ]
    trajectory_rows = [
        {
            "variant_id": item.variant_id,
            "workload": item.workload,
            "capacity": item.capacity,
            **entry,
        }
        for item in results
        for entry in item.trajectory
    ]
    files = {
        "results": f"{stem}.results.jsonl",
        "per_sequence": f"{stem}.per_sequence.jsonl",
        "diagnostics": f"{stem}.diagnostics.jsonl",
    }
    atomic_write_jsonl(directory / files["results"], result_rows)
    atomic_write_jsonl(directory / files["per_sequence"], sequence_rows)
    atomic_write_jsonl(directory / files["diagnostics"], diagnostic_rows)
    if trajectory_rows:
        atomic_write_jsonl(directory / f"{stem}.trajectory.jsonl", trajectory_rows)
    manifest = {
        "schema_version": "race_stage2_checkpoint_v1",
        "completed_at_utc": utc_now(),
        "workload": workload.name,
        "workload_hash": workload.hash,
        "label": payload["label"],
        "variant_id": variant.variant_id,
        "frozen_config_sha256": payload["frozen_config_sha256"],
        "conditions": len(result_rows),
        "per_sequence_rows": len(sequence_rows),
        "trajectory_rows": len(trajectory_rows),
        **{f"{key}_file": name for key, name in files.items()},
        **{
            f"{key}_sha256": sha256_file(directory / name)
            for key, name in files.items()
        },
    }
    atomic_write_json(manifest_path, manifest)
    return f"completed {workload.name}/{stem}"


def run_evaluation(
    repository_root: Path,
    preregistration_path: Path,
    frozen_config_path: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_and_verify_stage2_inputs(repository_root, preregistration_path)
    frozen = load_and_verify_stage2_frozen(frozen_config_path)
    if frozen["preregistration_hash"] != inputs.preregistration_hash:
        raise ValueError("Stage 2 frozen config references another preregistration")
    current_source = stage2_source_bundle_hash(repository_root)
    if current_source != frozen["stage2_source_bundle_hash"]:
        raise ValueError(
            "Stage 2 source/config/tests changed after the calibration freeze; "
            "create an explicitly exploratory run instead of a frozen evaluation"
        )
    model_path = repository_root / frozen["transition_model_path"]
    if sha256_file(model_path) != frozen["transition_model_hash"]:
        raise ValueError("Stage 2 transition models changed after freezing")

    variants = [
        item
        for item in frozen["variants"]
        if labels is None or item["label"] in set(labels)
    ]
    if not variants:
        raise ValueError("No Stage 2 variant matched the requested labels")
    primary_label = next(item["label"] for item in frozen["variants"] if item["is_primary"])
    base = {
        "repository_root": str(repository_root),
        "preregistration_path": str(preregistration_path.resolve()),
        "model_path": str(model_path),
        "frozen_config_path": str(frozen_config_path.resolve()),
        "frozen_config_sha256": frozen["file_sha256"],
        "output_dir": str(output_dir.resolve()),
        "trajectory_stride": int(
            inputs.preregistration["diagnostics"]["weight_trajectory_stride"]
        ),
        "trajectory_label": primary_label,
    }
    payloads = [
        {
            **base,
            "workload": workload.name,
            "label": item["label"],
            "spec": item["spec"],
        }
        for item in variants
        for workload in inputs.workloads
    ]
    print(f"stage2 evaluation: {len(payloads)} workload/variant jobs", flush=True)
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_evaluate_job, item): item for item in payloads}
            done = 0
            for future in as_completed(futures):
                message = future.result()
                done += 1
                print(f"  [{done}/{len(payloads)}] {message}", flush=True)
    else:
        for done, payload in enumerate(payloads, 1):
            print(f"  [{done}/{len(payloads)}] {_evaluate_job(payload)}", flush=True)

    results: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for workload in inputs.workloads:
        directory = output_dir / "checkpoints" / workload.name
        for item in variants:
            manifest = read_json(directory / f"{item['label']}.manifest.json")
            manifests.append(manifest)
            results.extend(read_jsonl(directory / manifest["results_file"]))
            sequences.extend(read_jsonl(directory / manifest["per_sequence_file"]))
            diagnostics.extend(read_jsonl(directory / manifest["diagnostics_file"]))
            trajectory_path = directory / f"{item['label']}.trajectory.jsonl"
            if trajectory_path.exists():
                trajectories.extend(read_jsonl(trajectory_path))
    atomic_write_jsonl(output_dir / "results.jsonl", results)
    atomic_write_jsonl(output_dir / "per_sequence_results.jsonl", sequences)
    atomic_write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
    atomic_write_jsonl(output_dir / "weight_trajectories.jsonl", trajectories)

    sanity = _sanity_checks(inputs, frozen, results, sequences, variants)
    atomic_write_json(output_dir / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("Stage 2 evaluation sanity checks failed")
    manifest = {
        "schema_version": "race_stage2_evaluation_manifest_v1",
        "completed_at_utc": utc_now(),
        "passed": True,
        "workloads": [workload.name for workload in inputs.workloads],
        "variants": [item["label"] for item in variants],
        "conditions": len(results),
        "per_sequence_rows": len(sequences),
        "diagnostic_rows": len(diagnostics),
        "trajectory_rows": len(trajectories),
        "trace_hash": inputs.trace.trace_hash,
        "preregistration_hash": inputs.preregistration_hash,
        "frozen_config_file_sha256": frozen["file_sha256"],
        "stage2_source_bundle_hash": current_source,
        "transition_model_hash": frozen["transition_model_hash"],
        "results_sha256": sha256_file(output_dir / "results.jsonl"),
        "per_sequence_results_sha256": sha256_file(output_dir / "per_sequence_results.jsonl"),
        "diagnostics_sha256": sha256_file(output_dir / "diagnostics.jsonl"),
        "weight_trajectories_sha256": sha256_file(output_dir / "weight_trajectories.jsonl"),
        "sanity_checks_sha256": sha256_file(output_dir / "sanity_checks.json"),
        "checkpoint_manifests": manifests,
        "environment": environment_record(),
    }
    atomic_write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest


def _sanity_checks(
    inputs: Any,
    frozen: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    sequences: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    capacities = [int(value) for value in frozen["cache_capacities"]]
    expected = len(variants) * len(inputs.workloads) * len(capacities)
    add("condition_count", len(results) == expected, {"actual": len(results), "expected": expected})
    identities = [
        (row["workload"], int(row["capacity"]), row["variant_id"]) for row in results
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
    for row in sequences:
        by_condition.setdefault(str(row["condition_id"]), []).append(row)
    aggregation = [
        row["condition_id"]
        for row in results
        if sum(int(item["misses"]) for item in by_condition.get(str(row["condition_id"]), []))
        != int(row["misses"])
    ]
    add("per_sequence_aggregation", not aggregation, aggregation[:10])

    oracle_violations = []
    degenerate = []
    for row in results:
        key = (str(row["workload"]), int(row["capacity"]))
        oracle = int(inputs.stage0_references[key]["oracle"]["misses"])
        if oracle > int(row["misses"]):
            oracle_violations.append(
                {"condition": row["condition_id"], "oracle": oracle, "race": row["misses"]}
            )
        if int(row["capacity"]) == inputs.trace.top_k and int(row["misses"]) != oracle:
            degenerate.append(row["condition_id"])
    add("stage0_oracle_dominance", not oracle_violations, oracle_violations[:10])
    add("capacity8_degenerate_sanity", not degenerate, degenerate[:10])

    events_by_workload = {}
    for row in results:
        events_by_workload.setdefault(str(row["workload"]), set()).add(int(row["events"]))
    add(
        "event_counts_consistent_per_workload",
        all(len(values) == 1 for values in events_by_workload.values()),
        {key: sorted(value) for key, value in events_by_workload.items()},
    )
    add(
        "frozen_workload_hashes_unchanged",
        {workload.name: workload.hash for workload in inputs.workloads}
        == frozen["workload_hashes"],
    )
    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": "race_stage2_sanity_checks_v1",
        "passed": passed,
        "checks": checks,
        "bootstrap_scope": frozen["statistics"]["conditionality"],
    }
