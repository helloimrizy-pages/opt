"""Stage 3 pilot and mechanism audits, on calibration data only."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from race_stage1.models import TransitionModels
from race_stage1.simulation import simulate_causal_capacities
from residency_headroom.common import atomic_write_json, environment_record, utc_now
from residency_headroom.simulator import simulate_oracle

from .calibration import build_variant_scorers, load_and_verify_stage3_frozen
from .features import FeatureState, static_popularity
from .frozen import load_and_verify_stage3_inputs, truncated_workload
from .simulation import simulate_stage3, stage1_winner_scorer


def run_pilot_audit(
    repository_root: Path,
    preregistration_path: Path,
    output_dir: Path,
    *,
    sequences: int = 8,
    frozen_config_path: Path | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    inputs = load_and_verify_stage3_inputs(repository_root, preregistration_path)
    prereg = inputs.preregistration
    capacities = tuple(int(v) for v in prereg["cache_capacities"])
    models = TransitionModels.load(
        repository_root / prereg["stage2_reference"]["transition_model_path"]
    )
    popularity = static_popularity(inputs.trace, inputs.calibration)
    pilot = truncated_workload(inputs.calibration, sequences, "calibration_pilot")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    state = FeatureState(models, inputs.trace.num_layers, inputs.trace.num_experts, popularity)

    # The Stage 1 winner, expressed in Stage 3 features, must reproduce Stage 1 exactly.
    seed = stage1_winner_scorer(models, state)
    replay = simulate_stage3(
        inputs.trace, pilot, capacities, {c: seed for c in capacities}, state,
        variant="stage1_equivalence",
    )
    stage1_models = TransitionModels.load(
        repository_root / prereg["stage1_reference"]["transition_model_path"]
    )
    reference = simulate_causal_capacities(
        inputs.trace, pilot, capacities, dict(prereg["stage1_reference"]["winner_spec"]),
        stage1_models,
    )
    left = {int(r.capacity): int(r.misses) for r in reference}
    right = {int(r.capacity): int(r.misses) for r in replay}
    add("stage3_reproduces_stage1_winner_exactly", left == right, {"stage1": left, "stage3": right})

    perfect = simulate_stage3(
        inputs.trace, pilot, capacities, {c: (lambda b: np.zeros(b.shape[1])) for c in capacities},
        state, variant="perfect", perfect_score_override=True,
    )
    oracle = {c: int(simulate_oracle(inputs.trace, pilot, c).misses) for c in capacities}
    got = {int(r.capacity): int(r.misses) for r in perfect}
    add("perfect_score_equals_stage0_oracle", got == oracle, {"perfect": got, "oracle": oracle})

    started = time.perf_counter()
    first = simulate_stage3(
        inputs.trace, pilot, capacities, {c: seed for c in capacities}, state,
        variant="determinism", enable_diagnostics=True,
    )
    elapsed = time.perf_counter() - started
    second = simulate_stage3(
        inputs.trace, pilot, capacities, {c: seed for c in capacities}, state,
        variant="determinism", enable_diagnostics=False,
    )
    add("replay_is_deterministic_and_observer_free",
        [r.misses for r in first] == [r.misses for r in second])

    for item in first:
        if item.hits + item.misses != item.requests or item.misses != item.admissions:
            add("cache_accounting", False, {"capacity": item.capacity})
            break
    else:
        add("cache_accounting", True)
    add("capacity_invariants", all(r.maximum_occupancy <= r.capacity for r in first))
    add("oracle_dominance", all(r.misses >= oracle[r.capacity] for r in first))

    frozen_costs = None
    if frozen_config_path is not None and Path(frozen_config_path).exists():
        frozen = load_and_verify_stage3_frozen(Path(frozen_config_path))
        scorers, include, _m = build_variant_scorers(frozen, frozen["primary_variant"])
        fitted_state = FeatureState(
            models, inputs.trace.num_layers, inputs.trace.num_experts, popularity,
            include_request_scope=include,
        )
        primary = simulate_stage3(inputs.trace, pilot, capacities, scorers, fitted_state,
                                  variant=frozen["primary_variant"], enable_diagnostics=True)
        frozen_costs = {int(r.capacity): int(r.misses) for r in primary}
        add("frozen_primary_beats_stage1_on_pilot",
            all(frozen_costs[c] <= right[c] for c in capacities if c != inputs.trace.top_k),
            {"stage3": frozen_costs, "stage1": right})

    events = int(first[0].events)
    report = {
        "schema_version": "race_stage3_pilot_audit_v1",
        "created_at_utc": utc_now(),
        "pilot_workload": pilot.name,
        "pilot_sequences": len(pilot.sequences),
        "pilot_workload_hash": pilot.hash,
        "pilot_data_scope": "prefix of the frozen Stage 0 calibration workload; no evaluation sequence",
        "events": events,
        "runtime_seconds": elapsed,
        "microseconds_per_event": 1e6 * elapsed / events if events else None,
        "stage1_reference_pilot_costs": left,
        "stage3_replay_of_stage1_pilot_costs": right,
        "oracle_pilot_costs": oracle,
        "frozen_primary_pilot_costs": frozen_costs,
        "ranking_diagnostics": {int(r.capacity): r.ranking for r in first},
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "environment": environment_record(),
        "note": "Pilot results may not change any preregistered threshold or grid.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "pilot_audit.json", report)
    return report
