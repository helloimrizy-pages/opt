"""Stage 3 calibration and configuration freeze.

Every fitted quantity — feature standardization, static popularity, the pairwise
ranking weights and the L2 choice — comes from the frozen Stage 0 calibration
workload path alone. No evaluation sequence is read.
"""

from __future__ import annotations

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

from .features import BASE_NAMES, ALL_NAMES, FeatureState, static_popularity
from .frozen import load_and_verify_stage3_inputs, stage3_source_bundle_hash
from .ranking import RankingModel, fit_ranking_model, pairwise_accuracy, group_slices
from .simulation import GroupCollector, simulate_stage3, stage1_winner_scorer


VARIANTS = (
    "STAGE3_RANKER",
    "STAGE3_RANKER_NO_REQUEST_SCOPE",
    "STAGE3_RANKER_POOLED",
    "STAGE3_RANKER_ROUND1_DATA",
)


def _collect(
    inputs: Any,
    models: TransitionModels,
    state: FeatureState,
    scorers: Mapping[int, Any],
    capacities: Sequence[int],
    stride: int,
    warmup: int,
) -> GroupCollector:
    collector = GroupCollector(capacities, stride, warmup)
    simulate_stage3(
        inputs.trace,
        inputs.calibration,
        capacities,
        scorers,
        state,
        variant="collection",
        collector=collector,
    )
    return collector


def _fit_per_capacity(
    collector: GroupCollector,
    capacities: Sequence[int],
    names: Sequence[str],
    l2_grid: Sequence[float],
) -> dict[int, RankingModel]:
    return {
        int(capacity): fit_ranking_model(
            *collector.dataset(int(capacity)), names, l2_grid=l2_grid
        )
        for capacity in capacities
    }


def run_calibration(
    repository_root: Path, preregistration_path: Path, output_dir: Path
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    preregistration_path = preregistration_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_and_verify_stage3_inputs(repository_root, preregistration_path)
    prereg = inputs.preregistration
    capacities = tuple(int(v) for v in prereg["decision_capacities"])
    training = prereg["training"]
    l2_grid = tuple(float(v) for v in training["regularization_grid"])
    stride = int(training["sampling_stride"])
    warmup = int(training["warmup_events_skipped"])

    models = TransitionModels.load(
        repository_root / prereg["stage2_reference"]["transition_model_path"]
    )
    popularity = static_popularity(inputs.trace, inputs.calibration)

    selection: dict[str, Any] = {
        "schema_version": "race_stage3_calibration_selection_v1",
        "created_at_utc": utc_now(),
        "preregistration_hash": inputs.preregistration_hash,
        "trace_hash": inputs.trace.trace_hash,
        "calibration_workload_hash": inputs.calibration.hash,
        "calibration_sequence_ids": list(inputs.calibration.sequence_ids),
        "calibration_evaluation_disjoint": not bool(
            set(inputs.calibration.sequence_ids)
            & {s for w in inputs.workloads for s in w.sequence_ids}
        ),
        "decision_capacities": list(capacities),
        "l2_grid": list(l2_grid),
        "sampling_stride": stride,
        "warmup_events_skipped": warmup,
        "rounds": {},
    }
    fitted: dict[str, Any] = {}

    for scope, names in (("all", ALL_NAMES), ("no_request_scope", BASE_NAMES)):
        include = scope == "all"
        state = FeatureState(
            models, inputs.trace.num_layers, inputs.trace.num_experts, popularity,
            include_request_scope=include,
        )
        print(f"stage3 calibration [{scope}]: round 1 collection under the Stage 1 winner",
              flush=True)
        seed = stage1_winner_scorer(models, state)
        round1 = _collect(
            inputs, models, state, {c: seed for c in capacities}, capacities, stride, warmup
        )
        pooled_features, pooled_targets, pooled_groups = round1.pooled()
        pooled = fit_ranking_model(
            pooled_features, pooled_targets, pooled_groups, names, l2_grid=l2_grid
        )
        round1_per_capacity = _fit_per_capacity(round1, capacities, names, l2_grid)
        print(f"  round 1 pooled holdout accuracy {100*pooled.holdout_accuracy:.2f}% "
              f"over {round1.groups:,} groups", flush=True)

        print(f"stage3 calibration [{scope}]: round 2 collection under the round-1 model",
              flush=True)
        round2 = _collect(
            inputs, models, state,
            {c: (lambda block, m=pooled: m.score(block)) for c in capacities},
            capacities, stride, warmup,
        )
        round2_per_capacity = _fit_per_capacity(round2, capacities, names, l2_grid)
        for capacity in capacities:
            print(f"  round 2 capacity {capacity:>2d} holdout accuracy "
                  f"{100*round2_per_capacity[capacity].holdout_accuracy:.2f}% "
                  f"(l2={round2_per_capacity[capacity].l2})", flush=True)
        fitted[scope] = {
            "pooled": pooled,
            "round1_per_capacity": round1_per_capacity,
            "round2_per_capacity": round2_per_capacity,
        }
        selection["rounds"][scope] = {
            "round1_groups": round1.groups,
            "round2_groups": round2.groups,
            "pooled": pooled.as_dict(),
            "round1_per_capacity": {
                str(c): m.as_dict() for c, m in round1_per_capacity.items()
            },
            "round2_per_capacity": {
                str(c): m.as_dict() for c, m in round2_per_capacity.items()
            },
        }

    variant_models = {
        "STAGE3_RANKER": {
            "scope": "all",
            "models": {str(c): m.as_dict() for c, m in fitted["all"]["round2_per_capacity"].items()},
        },
        "STAGE3_RANKER_NO_REQUEST_SCOPE": {
            "scope": "no_request_scope",
            "models": {
                str(c): m.as_dict()
                for c, m in fitted["no_request_scope"]["round2_per_capacity"].items()
            },
        },
        "STAGE3_RANKER_POOLED": {
            "scope": "all",
            "models": {str(c): fitted["all"]["pooled"].as_dict() for c in capacities},
        },
        "STAGE3_RANKER_ROUND1_DATA": {
            "scope": "all",
            "models": {
                str(c): m.as_dict() for c, m in fitted["all"]["round1_per_capacity"].items()
            },
        },
    }

    selection_path = output_dir / "selection.json"
    atomic_write_json(selection_path, selection)
    selection_hash = sha256_file(selection_path)
    _sidecar(selection_path, selection_hash)

    frozen = {
        "schema_version": "race_stage3_frozen_config_v1",
        "frozen_at_utc": utc_now(),
        "preregistration_hash": inputs.preregistration_hash,
        "preregistration_path": str(preregistration_path.relative_to(repository_root)),
        "stage0_trace_hash": inputs.trace.trace_hash,
        "stage0_archive_manifest_sha256": prereg["stage0_reference"]["final_archive_manifest_sha256"],
        "stage1_archive_manifest_sha256": prereg["stage1_reference"]["final_archive_manifest_sha256"],
        "stage2_archive_manifest_sha256": prereg["stage2_reference"]["final_archive_manifest_sha256"],
        "stage1_winner_method_id": prereg["stage1_reference"]["winner_method_id"],
        "stage3_repository_head": resolve_git_commit(repository_root),
        "stage3_source_bundle_hash": stage3_source_bundle_hash(repository_root),
        "calibration_workload_hash": inputs.calibration.hash,
        "workload_hashes": {w.name: w.hash for w in inputs.workloads},
        "cache_capacities": list(prereg["cache_capacities"]),
        "decision_capacities": list(capacities),
        "transition_model_path": prereg["stage2_reference"]["transition_model_path"],
        "transition_model_hash": prereg["stage2_reference"]["transition_model_sha256"],
        "static_popularity_source": "frozen Stage 0 calibration workload request counts",
        "feature_names": {"all": list(ALL_NAMES), "no_request_scope": list(BASE_NAMES)},
        "selection_path": str(selection_path.relative_to(repository_root)),
        "selection_file_sha256": selection_hash,
        "primary_variant": "STAGE3_RANKER",
        "variants": variant_models,
        "success_criteria": dict(prereg["success_criteria"]),
        "regression_criterion": dict(prereg["regression_criterion"]),
        "statistics": dict(prereg["statistics"]),
        "environment": environment_record(),
        "evaluation_started": False,
    }
    frozen_path = output_dir / "stage3_frozen_config.json"
    atomic_write_json(frozen_path, frozen)
    frozen_hash = sha256_file(frozen_path)
    _sidecar(frozen_path, frozen_hash)
    return {"selection": selection, "frozen_config": frozen, "frozen_config_file_sha256": frozen_hash}


def load_and_verify_stage3_frozen(path: Path) -> dict[str, Any]:
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Stage 3 frozen config hash mismatch: {actual} != {expected}")
    value = read_json(path)
    value["file_sha256"] = actual
    return value


def build_variant_scorers(
    frozen: Mapping[str, Any], variant: str
) -> tuple[dict[int, Any], bool, dict[int, RankingModel]]:
    """Deployable scorers for one frozen variant, plus its feature scope."""

    entry = frozen["variants"][variant]
    include = entry["scope"] == "all"
    models = {int(c): RankingModel.from_dict(v) for c, v in entry["models"].items()}
    scorers = {c: (lambda block, m=m: m.score(block)) for c, m in models.items()}
    for capacity in frozen["cache_capacities"]:
        scorers.setdefault(int(capacity), lambda block: np.zeros(block.shape[1]))
    return scorers, include, models


def _sidecar(path: Path, digest: str) -> None:
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
