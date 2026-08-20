"""One frozen Stage 3 evaluation pass over the ten sealed Stage 0 workload paths."""

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

from .calibration import build_variant_scorers, load_and_verify_stage3_frozen
from .features import FeatureState, static_popularity
from .frozen import load_and_verify_stage3_inputs, stage3_source_bundle_hash
from .simulation import simulate_stage3


_CONTEXT: dict[tuple[str, str], tuple[Any, Any, Any]] = {}


def _context(root: str, preregistration: str):
    key = (root, preregistration)
    if key not in _CONTEXT:
        inputs = load_and_verify_stage3_inputs(Path(root), Path(preregistration))
        models = TransitionModels.load(
            Path(root) / inputs.preregistration["stage2_reference"]["transition_model_path"]
        )
        popularity = static_popularity(inputs.trace, inputs.calibration)
        _CONTEXT[key] = (inputs, models, popularity)
    return _CONTEXT[key]


def _evaluate_job(payload: Mapping[str, Any]) -> str:
    inputs, models, popularity = _context(
        payload["repository_root"], payload["preregistration_path"]
    )
    frozen = load_and_verify_stage3_frozen(Path(payload["frozen_config_path"]))
    if frozen["file_sha256"] != payload["frozen_config_sha256"]:
        raise ValueError("Stage 3 frozen config changed during evaluation")
    workload = next((w for w in inputs.workloads if w.name == payload["workload"]), None)
    if workload is None:
        raise ValueError(f"Unknown frozen workload {payload['workload']}")
    if workload.hash != frozen["workload_hashes"][workload.name]:
        raise ValueError(f"Workload {workload.name} hash changed after freezing")

    variant = payload["variant"]
    directory = Path(payload["output_dir"]) / "checkpoints" / workload.name
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / f"{variant}.manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("frozen_config_sha256") == payload["frozen_config_sha256"] and all(
            sha256_file(directory / manifest[key + "_file"]) == manifest[key + "_sha256"]
            for key in ("results", "per_sequence", "diagnostics")
        ):
            return f"reused {workload.name}/{variant}"

    scorers, include_scope, _models = build_variant_scorers(frozen, variant)
    state = FeatureState(
        models, inputs.trace.num_layers, inputs.trace.num_experts, popularity,
        include_request_scope=include_scope,
    )
    results = simulate_stage3(
        inputs.trace,
        workload,
        tuple(int(v) for v in frozen["cache_capacities"]),
        scorers,
        state,
        variant=variant,
        enable_diagnostics=True,
    )
    trace_hash = inputs.trace.trace_hash
    config_hash = payload["frozen_config_sha256"]
    files = {
        "results": f"{variant}.results.jsonl",
        "per_sequence": f"{variant}.per_sequence.jsonl",
        "diagnostics": f"{variant}.diagnostics.jsonl",
    }
    atomic_write_jsonl(
        directory / files["results"],
        [r.result_record(trace_hash=trace_hash, config_hash=config_hash) for r in results],
    )
    atomic_write_jsonl(
        directory / files["per_sequence"],
        [row for r in results
         for row in r.sequence_records(trace_hash=trace_hash, config_hash=config_hash)],
    )
    atomic_write_jsonl(
        directory / files["diagnostics"],
        [r.diagnostic_record(trace_hash=trace_hash, config_hash=config_hash) for r in results],
    )
    atomic_write_json(
        manifest_path,
        {
            "schema_version": "race_stage3_checkpoint_v1",
            "completed_at_utc": utc_now(),
            "workload": workload.name,
            "workload_hash": workload.hash,
            "variant": variant,
            "frozen_config_sha256": config_hash,
            "conditions": len(results),
            **{f"{k}_file": v for k, v in files.items()},
            **{f"{k}_sha256": sha256_file(directory / v) for k, v in files.items()},
        },
    )
    return f"completed {workload.name}/{variant}"


def run_evaluation(
    repository_root: Path,
    preregistration_path: Path,
    frozen_config_path: Path,
    output_dir: Path,
    *,
    workers: int = 1,
    variants: Sequence[str] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_and_verify_stage3_inputs(repository_root, preregistration_path)
    frozen = load_and_verify_stage3_frozen(frozen_config_path)
    if frozen["preregistration_hash"] != inputs.preregistration_hash:
        raise ValueError("Stage 3 frozen config references another preregistration")
    current = stage3_source_bundle_hash(repository_root)
    if current != frozen["stage3_source_bundle_hash"]:
        raise ValueError(
            "Stage 3 source/config/tests changed after the calibration freeze; "
            "create an explicitly exploratory run instead of a frozen evaluation"
        )
    names = list(variants) if variants is not None else list(frozen["variants"])
    payloads = [
        {
            "repository_root": str(repository_root),
            "preregistration_path": str(preregistration_path.resolve()),
            "frozen_config_path": str(frozen_config_path.resolve()),
            "frozen_config_sha256": frozen["file_sha256"],
            "output_dir": str(output_dir.resolve()),
            "workload": workload.name,
            "variant": variant,
        }
        for variant in names
        for workload in inputs.workloads
    ]
    print(f"stage3 evaluation: {len(payloads)} workload/variant jobs", flush=True)
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
    manifests: list[dict[str, Any]] = []
    for workload in inputs.workloads:
        directory = output_dir / "checkpoints" / workload.name
        for variant in names:
            manifest = read_json(directory / f"{variant}.manifest.json")
            manifests.append(manifest)
            results.extend(read_jsonl(directory / manifest["results_file"]))
            sequences.extend(read_jsonl(directory / manifest["per_sequence_file"]))
            diagnostics.extend(read_jsonl(directory / manifest["diagnostics_file"]))
    atomic_write_jsonl(output_dir / "results.jsonl", results)
    atomic_write_jsonl(output_dir / "per_sequence_results.jsonl", sequences)
    atomic_write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)

    sanity = _sanity_checks(inputs, frozen, results, sequences, names)
    atomic_write_json(output_dir / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("Stage 3 evaluation sanity checks failed")
    manifest = {
        "schema_version": "race_stage3_evaluation_manifest_v1",
        "completed_at_utc": utc_now(),
        "passed": True,
        "workloads": [w.name for w in inputs.workloads],
        "variants": names,
        "conditions": len(results),
        "per_sequence_rows": len(sequences),
        "trace_hash": inputs.trace.trace_hash,
        "preregistration_hash": inputs.preregistration_hash,
        "frozen_config_file_sha256": frozen["file_sha256"],
        "stage3_source_bundle_hash": current,
        "results_sha256": sha256_file(output_dir / "results.jsonl"),
        "per_sequence_results_sha256": sha256_file(output_dir / "per_sequence_results.jsonl"),
        "diagnostics_sha256": sha256_file(output_dir / "diagnostics.jsonl"),
        "sanity_checks_sha256": sha256_file(output_dir / "sanity_checks.json"),
        "checkpoint_manifests": manifests,
        "environment": environment_record(),
    }
    atomic_write_json(output_dir / "evaluation_manifest.json", manifest)
    return manifest


def _sanity_checks(inputs, frozen, results, sequences, variants) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    capacities = [int(v) for v in frozen["cache_capacities"]]
    expected = len(variants) * len(inputs.workloads) * len(capacities)
    add("condition_count", len(results) == expected, {"actual": len(results), "expected": expected})
    identities = [(r["workload"], int(r["capacity"]), r["variant"]) for r in results]
    add("unique_conditions", len(identities) == len(set(identities)))
    add(
        "event_accounting",
        not [r for r in results
             if int(r["hits"]) + int(r["misses"]) != int(r["requests"])
             or int(r["misses"]) != int(r["admissions"])],
    )
    add(
        "capacity_invariants",
        not [r for r in results if int(r["maximum_occupancy"]) > int(r["capacity"])],
    )
    by_condition: dict[str, int] = {}
    for row in sequences:
        by_condition[str(row["condition_id"])] = by_condition.get(str(row["condition_id"]), 0) + int(row["misses"])
    add(
        "per_sequence_aggregation",
        not [r for r in results if by_condition.get(str(r["condition_id"]), -1) != int(r["misses"])],
    )
    violations, degenerate = [], []
    for row in results:
        key = (str(row["workload"]), int(row["capacity"]))
        oracle = int(inputs.stage0_references[key]["oracle"]["misses"])
        if oracle > int(row["misses"]):
            violations.append(row["condition_id"])
        if int(row["capacity"]) == inputs.trace.top_k and int(row["misses"]) != oracle:
            degenerate.append(row["condition_id"])
    add("stage0_oracle_dominance", not violations, violations[:10])
    add("capacity8_degenerate_sanity", not degenerate, degenerate[:10])
    add(
        "frozen_workload_hashes_unchanged",
        {w.name: w.hash for w in inputs.workloads} == frozen["workload_hashes"],
    )
    return {
        "schema_version": "race_stage3_sanity_checks_v1",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "bootstrap_scope": frozen["statistics"]["conditionality"],
    }
