"""Stage 2 calibration and configuration freeze.

Every choice made here — static adviser weights, the learning rate, the online
initialization rule and which online loss becomes primary — is made from the frozen
Stage 0 calibration workload path alone. No evaluation sequence is read.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from race_stage1.models import TransitionModels
from residency_headroom.common import (
    atomic_write_json,
    atomic_write_text,
    environment_record,
    read_json,
    resolve_git_commit,
    sha256_file,
    utc_now,
)

from .advisers import (
    MARKOV_HORIZONS,
    adviser_parameters,
    fit_stage2_transition_models,
    pool_names,
    verify_stage1_horizon_reuse,
)
from .frozen import load_and_verify_stage2_inputs, stage2_source_bundle_hash
from .policy import (
    online_variant,
    static_per_layer_variant,
    static_variant,
    uniform_variant,
    variant_from_spec,
    variant_to_spec,
)
from .simulation import CalibrationExampleCollector, simulate_race_variant
from .static_weights import (
    ARMIJO_C,
    INITIAL_STEP,
    ITERATIONS,
    MAX_BACKTRACKS,
    TAU,
    build_pair_dataset,
    learn_static_weights,
)


POOLS = ("primary", "extended")
_CONTEXT: dict[tuple[str, str, str], tuple[Any, TransitionModels]] = {}


def _context(root: str, preregistration: str, model_path: str):
    key = (root, preregistration, model_path)
    if key not in _CONTEXT:
        inputs = load_and_verify_stage2_inputs(Path(root), Path(preregistration))
        models = TransitionModels.load(Path(model_path))
        if models.trace_hash != inputs.trace.trace_hash:
            raise ValueError("Stage 2 transition models reference a different trace")
        if models.calibration_workload_hash != inputs.calibration.hash:
            raise ValueError("Stage 2 transition models reference a different calibration path")
        _CONTEXT[key] = (inputs, models)
    return _CONTEXT[key]


def _calibration_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs, models = _context(
        payload["repository_root"], payload["preregistration_path"], payload["model_path"]
    )
    variant = variant_from_spec(payload["variant"])
    capacities = tuple(int(value) for value in payload["capacities"])
    decision = tuple(int(value) for value in payload["decision_capacities"])
    results = simulate_race_variant(
        inputs.trace,
        inputs.calibration,
        capacities,
        variant,
        models,
        enable_diagnostics=False,
    )
    costs = {int(item.capacity): int(item.misses) for item in results}
    return {
        "label": payload["label"],
        "variant_id": variant.variant_id,
        "variant": variant_to_spec(variant),
        "costs": costs,
        "selection_cost": int(sum(costs[value] for value in decision)),
        "learning": {
            int(item.capacity): {
                "examples_generated": item.learning["examples_generated"],
                "applied_updates": item.learning["applied_updates"],
                "examples_unresolved_at_stream_end": item.learning[
                    "examples_unresolved_at_stream_end"
                ],
                "empirical_rank_regret": item.learning["empirical_rank_regret"],
                "best_fixed_adviser_by_rank_loss": item.learning[
                    "best_fixed_adviser_by_rank_loss"
                ],
            }
            for item in results
        },
        "mean_weights": {
            int(item.capacity): item.adviser_mean_weights for item in results
        },
    }


def _run_jobs(payloads: Sequence[Mapping[str, Any]], workers: int) -> list[dict[str, Any]]:
    if workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(_calibration_job, payloads))
    return [_calibration_job(payload) for payload in payloads]


def ensure_transition_models(
    inputs: Any, repository_root: Path, model_path: Path
) -> tuple[TransitionModels, dict[str, Any], dict[str, Any]]:
    """Fit (or reuse) the Stage 2 Markov horizons from calibration data only."""

    reuse = False
    if model_path.exists() and model_path.with_suffix(".metadata.json").exists():
        try:
            existing = TransitionModels.load(model_path)
            reuse = (
                existing.trace_hash == inputs.trace.trace_hash
                and existing.calibration_workload_hash == inputs.calibration.hash
                and set(MARKOV_HORIZONS).issubset(set(existing.horizons))
            )
        except (ValueError, OSError):
            reuse = False
    if reuse:
        print("stage2: reusing the existing Stage 2 transition models", flush=True)
        metadata = read_json(model_path.with_suffix(".metadata.json"))
    else:
        print("stage2: fitting transition models on calibration only", flush=True)
        fitted = fit_stage2_transition_models(inputs.trace, inputs.calibration)
        metadata = fitted.save(model_path)
    models = TransitionModels.load(model_path)
    audit = verify_stage1_horizon_reuse(
        models,
        repository_root
        / inputs.preregistration["stage1_reference"]["transition_model_path"],
    )
    return models, metadata, audit


def run_calibration(
    repository_root: Path,
    preregistration_path: Path,
    output_dir: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    preregistration_path = preregistration_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_and_verify_stage2_inputs(repository_root, preregistration_path)
    prereg = inputs.preregistration
    capacities = tuple(int(value) for value in prereg["cache_capacities"])
    decision = tuple(int(value) for value in prereg["decision_capacities"])
    stride = int(prereg["static_weight_learning"]["subsample_stride"])
    eta_grid = tuple(float(value) for value in prereg["online_learning"]["eta_grid"])
    initializations = tuple(prereg["online_learning"]["initialization_grid"])

    model_path = output_dir / "transition_models.npz"
    models, model_metadata, horizon_audit = ensure_transition_models(
        inputs, repository_root, model_path
    )

    static_learning: dict[str, Any] = {}
    uniform_costs: dict[str, Any] = {}
    for pool in POOLS:
        print(f"stage2 calibration: collecting {pool}-pool examples", flush=True)
        collector = CalibrationExampleCollector(stride, decision)
        results = simulate_race_variant(
            inputs.trace,
            inputs.calibration,
            capacities,
            uniform_variant(pool),
            models,
            enable_diagnostics=False,
            example_collector=collector,
        )
        uniform_costs[pool] = {int(item.capacity): int(item.misses) for item in results}
        dataset = build_pair_dataset(collector.normalized, collector.distances)
        global_fit = learn_static_weights(dataset)
        per_layer_fits = []
        per_layer_weights = []
        for layer in range(inputs.trace.num_layers):
            chosen = [
                index for index, value in enumerate(collector.layer_of) if value == layer
            ]
            layer_dataset = build_pair_dataset(
                [collector.normalized[index] for index in chosen],
                [collector.distances[index] for index in chosen],
            )
            fit = learn_static_weights(layer_dataset)
            fit["layer"] = layer
            per_layer_fits.append(fit)
            per_layer_weights.append(fit["weights"])
        static_learning[pool] = {
            "adviser_order": list(pool_names(pool)),
            "collected_examples": len(collector),
            "global": global_fit,
            "per_layer": per_layer_fits,
            "per_layer_weights": per_layer_weights,
        }
        print(
            f"  {pool}: {len(collector)} examples, {dataset.pairs} pairs, "
            f"objective {global_fit['objective']:.5f} (uniform {global_fit['objective_at_uniform']:.5f})",
            flush=True,
        )

    base = {
        "repository_root": str(repository_root),
        "preregistration_path": str(preregistration_path),
        "model_path": str(model_path),
        "capacities": list(capacities),
        "decision_capacities": list(decision),
    }
    primary_static = np.asarray(static_learning["primary"]["global"]["weights"])
    extended_static = np.asarray(static_learning["extended"]["global"]["weights"])
    primary_per_layer = np.asarray(static_learning["primary"]["per_layer_weights"])

    grid_payloads = []
    for eta in eta_grid:
        for initialization in initializations:
            variant = online_variant(
                loss="rank",
                eta=eta,
                initialization=initialization,
                static_weights=primary_static,
            )
            grid_payloads.append(
                {
                    **base,
                    "label": f"online_rank_eta{eta}_init_{initialization}",
                    "variant": variant_to_spec(variant),
                }
            )
    grid_payloads.append(
        {**base, "label": "static_global", "variant": variant_to_spec(static_variant(primary_static))}
    )
    grid_payloads.append(
        {
            **base,
            "label": "static_per_layer",
            "variant": variant_to_spec(static_per_layer_variant(primary_per_layer)),
        }
    )
    grid_payloads.append(
        {
            **base,
            "label": "static_global_extended",
            "variant": variant_to_spec(static_variant(extended_static, pool="extended")),
        }
    )
    print(f"stage2 calibration: running {len(grid_payloads)} grid configurations", flush=True)
    grid_results = _run_jobs(grid_payloads, workers)
    by_label = {row["label"]: row for row in grid_results}

    ranked = sorted(
        (
            (
                by_label[f"online_rank_eta{eta}_init_{initialization}"]["selection_cost"],
                float(eta),
                0 if initialization == "uniform" else 1,
                float(eta),
                initialization,
            )
            for eta in eta_grid
            for initialization in initializations
        )
    )
    selected_cost, _eta_key, _init_key, selected_eta, selected_initialization = ranked[0]
    print(
        f"stage2 calibration: selected eta={selected_eta} init={selected_initialization} "
        f"(calibration cost {selected_cost})",
        flush=True,
    )

    final_payloads = [
        {
            **base,
            "label": "cost_selected",
            "variant": variant_to_spec(
                online_variant(
                    loss="cost",
                    eta=selected_eta,
                    initialization=selected_initialization,
                    static_weights=primary_static,
                )
            ),
        },
        {
            **base,
            "label": "online_rank_global_scope",
            "variant": variant_to_spec(
                online_variant(
                    loss="rank",
                    eta=selected_eta,
                    initialization=selected_initialization,
                    static_weights=primary_static,
                    scope="global",
                )
            ),
        },
        {
            **base,
            "label": "online_rank_extended",
            "variant": variant_to_spec(
                online_variant(
                    loss="rank",
                    eta=selected_eta,
                    initialization=selected_initialization,
                    static_weights=extended_static,
                    pool="extended",
                )
            ),
        },
        {
            **base,
            "label": "cost_extended",
            "variant": variant_to_spec(
                online_variant(
                    loss="cost",
                    eta=selected_eta,
                    initialization=selected_initialization,
                    static_weights=extended_static,
                    pool="extended",
                )
            ),
        },
    ]
    final_results = _run_jobs(final_payloads, workers)
    by_label.update({row["label"]: row for row in final_results})

    online_label = f"online_rank_eta{selected_eta}_init_{selected_initialization}"
    online_cost = by_label[online_label]["selection_cost"]
    cost_cost = by_label["cost_selected"]["selection_cost"]
    primary_loss = "rank" if online_cost <= cost_cost else "cost"
    primary_variant = online_variant(
        loss=primary_loss,
        eta=selected_eta,
        initialization=selected_initialization,
        static_weights=primary_static,
    )
    print(
        f"stage2 calibration: primary variant {primary_variant.name} "
        f"(rank {online_cost} vs cost {cost_cost})",
        flush=True,
    )

    uniform_selection_cost = {
        pool: int(sum(uniform_costs[pool][value] for value in decision)) for pool in POOLS
    }
    selection = {
        "schema_version": "race_stage2_calibration_selection_v1",
        "created_at_utc": utc_now(),
        "preregistration_hash": inputs.preregistration_hash,
        "trace_hash": inputs.trace.trace_hash,
        "calibration_workload_hash": inputs.calibration.hash,
        "calibration_sequence_ids": list(inputs.calibration.sequence_ids),
        "evaluation_sequence_ids": sorted(
            {
                sequence
                for workload in inputs.workloads
                for sequence in workload.sequence_ids
            }
        ),
        "calibration_evaluation_disjoint": not bool(
            set(inputs.calibration.sequence_ids)
            & {
                sequence
                for workload in inputs.workloads
                for sequence in workload.sequence_ids
            }
        ),
        "selection_capacities": list(decision),
        "selection_objective": "unit misses summed over capacities 12, 16, 24 and 32 on the frozen calibration path",
        "transition_model_hash": model_metadata["npz_sha256"],
        "stage1_horizon_reuse_audit": horizon_audit,
        "adviser_pools": {pool: adviser_parameters(pool) for pool in POOLS},
        "static_weight_learning": {
            "tau": TAU,
            "iterations": ITERATIONS,
            "initial_step": INITIAL_STEP,
            "armijo_c": ARMIJO_C,
            "max_backtracks": MAX_BACKTRACKS,
            "subsample_stride": stride,
            "source_policy": "RACE_UNIFORM of the same adviser pool on the calibration path",
            "results": static_learning,
        },
        "uniform_calibration_costs": uniform_costs,
        "uniform_selection_cost": uniform_selection_cost,
        "grid_results": [
            {
                "label": row["label"],
                "variant_id": row["variant_id"],
                "costs": row["costs"],
                "selection_cost": row["selection_cost"],
                "mean_weights": row["mean_weights"],
                "learning": row["learning"],
            }
            for row in grid_results + final_results
        ],
        "selected_eta": float(selected_eta),
        "selected_initialization": selected_initialization,
        "eta_selection_table": {
            f"eta{eta}_init_{initialization}": by_label[
                f"online_rank_eta{eta}_init_{initialization}"
            ]["selection_cost"]
            for eta in eta_grid
            for initialization in initializations
        },
        "primary_loss": primary_loss,
        "primary_variant_id": primary_variant.variant_id,
        "primary_variant_name": primary_variant.name,
        "online_rank_selection_cost": online_cost,
        "cost_sensitive_selection_cost": cost_cost,
        "weight_scope_primary": "per_layer",
        "causality_audit": {
            "fit_workload": inputs.calibration.name,
            "fit_workload_hash": inputs.calibration.hash,
            "future_evaluation_access": False,
            "evaluation_reselection": False,
            "delayed_update_offset_same_layer_events": 32,
        },
    }
    selection_path = output_dir / "selection.json"
    atomic_write_json(selection_path, selection)
    selection_hash = sha256_file(selection_path)
    _write_sidecar(selection_path, selection_hash)

    variants = _frozen_variant_table(
        primary_variant=primary_variant,
        primary_static=primary_static,
        primary_per_layer=primary_per_layer,
        extended_static=extended_static,
        selected_eta=selected_eta,
        selected_initialization=selected_initialization,
        primary_loss=primary_loss,
    )
    frozen = {
        "schema_version": "race_stage2_frozen_config_v1",
        "frozen_at_utc": utc_now(),
        "preregistration_hash": inputs.preregistration_hash,
        "preregistration_path": str(preregistration_path.relative_to(repository_root)),
        "stage0_trace_hash": inputs.trace.trace_hash,
        "stage0_archive_manifest_sha256": prereg["stage0_reference"][
            "final_archive_manifest_sha256"
        ],
        "stage1_archive_manifest_sha256": prereg["stage1_reference"][
            "final_archive_manifest_sha256"
        ],
        "stage1_frozen_config_file_sha256": inputs.stage1_frozen["file_sha256"],
        "stage1_winner_method_id": prereg["stage1_reference"]["winner_method_id"],
        "stage2_repository_head": resolve_git_commit(repository_root),
        "stage2_source_bundle_hash": stage2_source_bundle_hash(repository_root),
        "calibration_workload_hash": inputs.calibration.hash,
        "workload_hashes": {
            workload.name: workload.hash for workload in inputs.workloads
        },
        "cache_capacities": list(capacities),
        "decision_capacities": list(decision),
        "transition_model_path": str(model_path.relative_to(repository_root)),
        "transition_model_hash": model_metadata["npz_sha256"],
        "selection_path": str(selection_path.relative_to(repository_root)),
        "selection_file_sha256": selection_hash,
        "H_max": 32,
        "selected_eta": float(selected_eta),
        "selected_initialization": selected_initialization,
        "primary_loss": primary_loss,
        "primary_variant_id": primary_variant.variant_id,
        "primary_variant_name": primary_variant.name,
        "weight_scope_primary": "per_layer",
        "adviser_pools": {pool: adviser_parameters(pool) for pool in POOLS},
        "static_weights_global": {
            "primary": primary_static.tolist(),
            "extended": extended_static.tolist(),
        },
        "static_weights_per_layer_primary": primary_per_layer.tolist(),
        "variants": variants,
        "statistics": dict(prereg["statistics"]),
        "success_criteria": dict(prereg["success_criteria"]),
        "regression_criterion": dict(prereg["regression_criterion"]),
        "environment": environment_record(),
        "evaluation_started": False,
    }
    frozen_path = output_dir / "stage2_frozen_config.json"
    atomic_write_json(frozen_path, frozen)
    frozen_hash = sha256_file(frozen_path)
    _write_sidecar(frozen_path, frozen_hash)
    return {
        "selection": selection,
        "selection_file_sha256": selection_hash,
        "frozen_config": frozen,
        "frozen_config_file_sha256": frozen_hash,
    }


def _frozen_variant_table(
    *,
    primary_variant,
    primary_static: np.ndarray,
    primary_per_layer: np.ndarray,
    extended_static: np.ndarray,
    selected_eta: float,
    selected_initialization: str,
    primary_loss: str,
) -> list[dict[str, Any]]:
    entries = [
        ("RACE_UNIFORM", "primary", uniform_variant()),
        ("RACE_STATIC", "primary", static_variant(primary_static)),
        (
            "RACE_ONLINE",
            "primary",
            online_variant(
                loss="rank",
                eta=selected_eta,
                initialization=selected_initialization,
                static_weights=primary_static,
            ),
        ),
        (
            "RACE_COST",
            "primary",
            online_variant(
                loss="cost",
                eta=selected_eta,
                initialization=selected_initialization,
                static_weights=primary_static,
            ),
        ),
        ("RACE_STATIC_PERLAYER", "ablation", static_per_layer_variant(primary_per_layer)),
        (
            "RACE_ONLINE_GLOBAL",
            "ablation",
            online_variant(
                loss=primary_loss,
                eta=selected_eta,
                initialization=selected_initialization,
                static_weights=primary_static,
                scope="global",
            ),
        ),
        ("RACE_UNIFORM_EXTENDED", "ablation", uniform_variant("extended")),
        ("RACE_STATIC_EXTENDED", "ablation", static_variant(extended_static, pool="extended")),
        (
            "RACE_ONLINE_EXTENDED",
            "ablation",
            online_variant(
                loss="rank",
                eta=selected_eta,
                initialization=selected_initialization,
                static_weights=extended_static,
                pool="extended",
            ),
        ),
        (
            "RACE_COST_EXTENDED",
            "ablation",
            online_variant(
                loss="cost",
                eta=selected_eta,
                initialization=selected_initialization,
                static_weights=extended_static,
                pool="extended",
            ),
        ),
    ]
    table = []
    for label, role, variant in entries:
        table.append(
            {
                "label": label,
                "role": role,
                "is_primary": variant.variant_id == primary_variant.variant_id
                and role == "primary",
                "variant_id": variant.variant_id,
                "spec": variant_to_spec(variant),
            }
        )
    return table


def load_and_verify_stage2_frozen(path: Path) -> dict[str, Any]:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Stage 2 frozen config hash mismatch: {actual} != {expected}")
    value = read_json(path)
    value["file_sha256"] = actual
    return value


def _write_sidecar(path: Path, digest: str) -> None:
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
