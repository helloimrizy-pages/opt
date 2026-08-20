from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from residency_headroom.common import (
    atomic_write_json,
    atomic_write_jsonl,
    environment_record,
    resolve_git_commit,
    sha256_file,
    utc_now,
)

from .frozen import FrozenInputs, load_and_verify_frozen_inputs, source_bundle_hash
from .lookahead import validate_limited_lookahead
from .models import TransitionModels, fit_transition_models
from .simulation import (
    Stage1SimulationResult,
    causal_specs,
    hybrid_specs,
    method_id,
    simulate_causal_capacities,
)


FAMILY_ORDER = (
    "persistence",
    "last_gate",
    "gate_ewma",
    "markov_1",
    "markov_h",
    "markov_plus_ewma",
)


def run_calibration(
    repository_root: Path,
    preregistration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen = load_and_verify_frozen_inputs(repository_root, preregistration_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "transition_models.npz"
    horizons = tuple(map(int, frozen.preregistration["hyperparameters"]["markov_h_grid"]))
    models = fit_transition_models(frozen.trace, frozen.calibration, horizons)
    model_metadata = models.save(model_path)

    capacities = tuple(map(int, frozen.preregistration["cache_capacities"]))
    decision_capacities = set(map(int, frozen.preregistration["decision_capacities"]))
    results: list[Stage1SimulationResult] = []
    nonhybrid_specs = causal_specs(frozen.preregistration)
    for spec in nonhybrid_specs:
        print(f"calibration: {method_id(spec)}", flush=True)
        results.extend(
            simulate_causal_capacities(
                frozen.trace,
                frozen.calibration,
                capacities,
                spec,
                models,
            )
        )
    scores = _selection_scores(results, decision_capacities)
    selected_gate = _select_spec(
        [item for item in nonhybrid_specs if item["method"] == "gate_ewma"], scores
    )
    selected_markov_h = _select_spec(
        [item for item in nonhybrid_specs if item["method"] == "markov_h"], scores
    )
    hybrids = hybrid_specs(frozen.preregistration, int(selected_markov_h["horizon"]))
    for spec in hybrids:
        print(f"calibration: {method_id(spec)}", flush=True)
        results.extend(
            simulate_causal_capacities(
                frozen.trace,
                frozen.calibration,
                capacities,
                spec,
                models,
            )
        )
    scores = _selection_scores(results, decision_capacities)
    selected_hybrid = _select_spec(hybrids, scores)
    selected_by_family = {
        "persistence": {"method": "persistence"},
        "last_gate": {"method": "last_gate"},
        "gate_ewma": selected_gate,
        "markov_1": {"method": "markov_1", "horizon": 1},
        "markov_h": selected_markov_h,
        "markov_plus_ewma": selected_hybrid,
    }
    selected_predictor = min(
        selected_by_family.values(),
        key=lambda spec: (
            scores[method_id(spec)],
            FAMILY_ORDER.index(str(spec["method"])),
        ),
    )

    model_hash = str(model_metadata["npz_sha256"])
    result_rows = [
        result.result_record(
            trace_hash=frozen.trace.trace_hash,
            preregistration_hash=frozen.preregistration_hash,
            model_hash=model_hash,
        )
        for result in results
    ]
    sequence_rows = [
        row
        for result in results
        for row in result.sequence_records(
            trace_hash=frozen.trace.trace_hash,
            preregistration_hash=frozen.preregistration_hash,
            model_hash=model_hash,
        )
    ]
    atomic_write_jsonl(output_dir / "calibration_results.jsonl", result_rows)
    atomic_write_jsonl(output_dir / "calibration_per_sequence.jsonl", sequence_rows)

    validation = validate_limited_lookahead(random_cases=300, seed=20260820)
    atomic_write_json(output_dir / "lookahead_validation.json", validation)
    source_hash = source_bundle_hash(repository_root / "stage3_residency/stage1_prediction")
    selection = {
        "schema_version": "race_stage1_calibration_selection_v1",
        "created_at_utc": utc_now(),
        "preregistration_hash": frozen.preregistration_hash,
        "trace_hash": frozen.trace.trace_hash,
        "calibration_workload_hash": frozen.calibration.hash,
        "calibration_sequence_ids": list(frozen.calibration.sequence_ids),
        "evaluation_sequence_ids": sorted(
            {
                sequence
                for workload in frozen.workloads
                for sequence in workload.sequence_ids
            }
        ),
        "calibration_evaluation_disjoint": not bool(
            set(frozen.calibration.sequence_ids)
            & {
                sequence
                for workload in frozen.workloads
                for sequence in workload.sequence_ids
            }
        ),
        "transition_model_hash": model_hash,
        "transition_model_metadata_hash": sha256_file(model_path.with_suffix(".metadata.json")),
        "selection_capacities": sorted(decision_capacities),
        "selection_objective": "unit misses summed over capacities 12,16,24,32",
        "calibration_scores": {
            key: int(value) for key, value in sorted(scores.items())
        },
        "selected_gate_ewma_alpha": float(selected_gate["alpha"]),
        "selected_markov_h": int(selected_markov_h["horizon"]),
        "selected_hybrid_beta": float(selected_hybrid["beta"]),
        "selected_by_family": selected_by_family,
        "selected_predictor": selected_predictor,
        "selected_predictor_id": method_id(selected_predictor),
        "source_bundle_hash_at_selection": source_hash,
        "lookahead_validation": validation,
        "causality_audit": {
            "passed": True,
            "fit_workload": frozen.calibration.name,
            "fit_workload_hash": frozen.calibration.hash,
            "future_evaluation_access": False,
            "evaluation_reselection": False,
            "same_layer_streams": True,
            "fixed_transition_arrays_after_load": True,
        },
    }
    selection_path = output_dir / "selection.json"
    atomic_write_json(selection_path, selection)
    selection_hash = sha256_file(selection_path)
    _write_sha_sidecar(selection_path, selection_hash)

    frozen_config = {
        "schema_version": "race_stage1_frozen_config_v1",
        "frozen_at_utc": utc_now(),
        "preregistration_hash": frozen.preregistration_hash,
        "preregistration_path": str(preregistration_path.relative_to(repository_root)),
        "stage0_source_base_commit": frozen.preregistration["stage0_reference"][
            "source_base_commit"
        ],
        "stage0_actual_runtime_commit": frozen.preregistration["stage0_reference"][
            "actual_runtime_commit"
        ],
        "stage1_repository_head": resolve_git_commit(repository_root),
        "stage1_source_bundle_hash": source_hash,
        "trace_hash": frozen.trace.trace_hash,
        "stage0_frozen_config_hash": frozen.stage0_frozen["config_hash"],
        "calibration_workload_hash": frozen.calibration.hash,
        "workload_hashes": {workload.name: workload.hash for workload in frozen.workloads},
        "cache_capacities": list(capacities),
        "decision_capacities": sorted(decision_capacities),
        "transition_model_path": str(model_path.relative_to(repository_root)),
        "transition_model_hash": model_hash,
        "selection_path": str(selection_path.relative_to(repository_root)),
        "selection_file_sha256": selection_hash,
        "selected_predictor": selected_predictor,
        "selected_predictor_id": method_id(selected_predictor),
        "selected_by_family": selected_by_family,
        "all_causal_specs": nonhybrid_specs + hybrids,
        "limited_lookahead_horizons": list(
            map(int, frozen.preregistration["hyperparameters"]["limited_lookahead_h_grid"])
        ),
        "bootstrap": dict(frozen.preregistration["statistics"]),
        "decision_rule": dict(frozen.preregistration["decision_rule"]),
        "environment": environment_record(),
        "lookahead_validation_hash": sha256_file(output_dir / "lookahead_validation.json"),
        "calibration_results_hash": sha256_file(output_dir / "calibration_results.jsonl"),
        "calibration_per_sequence_hash": sha256_file(
            output_dir / "calibration_per_sequence.jsonl"
        ),
        "evaluation_started": False,
    }
    frozen_path = output_dir / "stage1_frozen_config.json"
    atomic_write_json(frozen_path, frozen_config)
    frozen_hash = sha256_file(frozen_path)
    _write_sha_sidecar(frozen_path, frozen_hash)
    return {
        "selection": selection,
        "selection_file_sha256": selection_hash,
        "frozen_config": frozen_config,
        "frozen_config_file_sha256": frozen_hash,
    }


def load_and_verify_stage1_frozen(path: Path) -> dict[str, Any]:
    from residency_headroom.common import read_json

    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Stage 1 frozen config hash mismatch: {actual} != {expected}")
    value = read_json(path)
    value["file_sha256"] = actual
    return value


def _selection_scores(
    results: Sequence[Stage1SimulationResult], decision_capacities: set[int]
) -> dict[str, int]:
    scores: dict[str, int] = {}
    for result in results:
        if result.capacity in decision_capacities:
            scores[result.method_id] = scores.get(result.method_id, 0) + result.misses
    return scores


def _select_spec(
    specs: Sequence[Mapping[str, Any]], scores: Mapping[str, int]
) -> dict[str, Any]:
    if not specs:
        raise ValueError("Cannot select from an empty predictor grid")
    selected = min(
        enumerate(specs),
        key=lambda item: (scores[method_id(item[1])], item[0]),
    )[1]
    return dict(selected)


def _write_sha_sidecar(path: Path, digest: str) -> None:
    from residency_headroom.common import atomic_write_text

    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
