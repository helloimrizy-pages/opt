"""Stage 2 pilot and causality audits.

The pilot runs only on a prefix of the frozen calibration workload, so it can never
leak evaluation information, and it is forbidden from changing any preregistered
threshold or grid.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from race_stage1.models import TransitionModels
from race_stage1.simulation import simulate_causal_capacities
from residency_headroom.common import atomic_write_json, environment_record, utc_now
from residency_headroom.simulator import simulate_oracle

from . import H_MAX
from .advisers import PRIMARY_POOL, pool_size
from .frozen import (
    Stage2Inputs,
    load_and_verify_stage2_inputs,
    perfect_score_matches_oracle,
    truncated_workload,
)
from .policy import RaceVariant, online_variant, uniform_variant
from .simulation import simulate_perfect_score, simulate_race_variant


STAGE1_EQUIVALENCE_CASES = (
    ({"method": "markov_h", "horizon": 1}, "MARKOV_H1"),
    ({"method": "markov_h", "horizon": 2}, "MARKOV_H2"),
    ({"method": "markov_h", "horizon": 4}, "MARKOV_H4"),
    ({"method": "markov_h", "horizon": 8}, "MARKOV_H8"),
    ({"method": "markov_h", "horizon": 16}, "MARKOV_H16"),
    ({"method": "gate_ewma", "alpha": 0.95}, "GATE_EWMA"),
    ({"method": "persistence"}, "PERSISTENCE"),
    ({"method": "markov_plus_ewma", "horizon": 2, "beta": 0.5, "history_alpha": 0.95}, "STAGE1_HYBRID"),
)


def single_adviser_variant(adviser: str, pool: str = "primary") -> RaceVariant:
    from .advisers import pool_names

    names = pool_names(pool)
    weights = np.zeros(len(names), dtype=np.float64)
    weights[names.index(adviser)] = 1.0
    return RaceVariant(
        name=f"SINGLE_{adviser}",
        weight_source="static_global",
        static_weights=weights,
        pool=pool,
    )


def run_pilot_audit(
    repository_root: Path,
    preregistration_path: Path,
    model_path: Path,
    output_dir: Path,
    *,
    sequences: int | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    inputs = load_and_verify_stage2_inputs(repository_root, preregistration_path)
    prereg = inputs.preregistration
    if sequences is None:
        sequences = int(
            str(prereg["pilot"]["data"]).split("first ")[1].split(" ")[0]
        )
    capacities = tuple(int(value) for value in prereg["cache_capacities"])
    models = TransitionModels.load(model_path)
    stage1_models = TransitionModels.load(
        repository_root / prereg["stage1_reference"]["transition_model_path"]
    )
    pilot = truncated_workload(inputs.calibration, sequences, "calibration_pilot")
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, details: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})

    add(
        "stage1_perfect_score_equals_stage0_oracle_from_archive",
        perfect_score_matches_oracle(inputs)["passed"],
        perfect_score_matches_oracle(inputs),
    )

    # Stage 2 mechanism driven by exact next-use scores must reproduce the oracle.
    perfect = simulate_perfect_score(inputs.trace, pilot, capacities, models)
    oracle_costs = {
        capacity: int(simulate_oracle(inputs.trace, pilot, capacity).misses)
        for capacity in capacities
    }
    perfect_costs = {int(item.capacity): int(item.misses) for item in perfect}
    add(
        "stage2_perfect_score_equals_stage0_oracle_on_pilot",
        perfect_costs == oracle_costs,
        {"perfect": perfect_costs, "oracle": oracle_costs},
    )

    # A single-adviser Stage 2 variant must reproduce the frozen Stage 1 predictor.
    equivalence = []
    for spec, adviser in STAGE1_EQUIVALENCE_CASES:
        pool = "extended" if adviser == "STAGE1_HYBRID" else "primary"
        stage1 = simulate_causal_capacities(
            inputs.trace, pilot, capacities, spec, stage1_models
        )
        stage2 = simulate_race_variant(
            inputs.trace,
            pilot,
            capacities,
            single_adviser_variant(adviser, pool),
            models,
            enable_diagnostics=False,
        )
        left = {int(item.capacity): int(item.misses) for item in stage1}
        right = {int(item.capacity): int(item.misses) for item in stage2}
        equivalence.append(
            {"adviser": adviser, "stage1": left, "stage2": right, "equal": left == right}
        )
    add(
        "stage2_single_adviser_reproduces_stage1_predictor",
        all(item["equal"] for item in equivalence),
        equivalence,
    )

    variant = online_variant(loss="rank", eta=0.3, initialization="uniform")
    started = time.perf_counter()
    with_diagnostics = simulate_race_variant(
        inputs.trace,
        pilot,
        capacities,
        variant,
        models,
        enable_diagnostics=True,
        label_cross_check=True,
    )
    elapsed = time.perf_counter() - started
    without_diagnostics = simulate_race_variant(
        inputs.trace, pilot, capacities, variant, models, enable_diagnostics=False
    )
    add(
        "diagnostic_observer_does_not_change_any_cost",
        [item.misses for item in with_diagnostics]
        == [item.misses for item in without_diagnostics],
        {
            "with": [item.misses for item in with_diagnostics],
            "without": [item.misses for item in without_diagnostics],
        },
    )
    repeat = simulate_race_variant(
        inputs.trace, pilot, capacities, variant, models, enable_diagnostics=False
    )
    add(
        "replay_is_deterministic",
        [item.misses for item in repeat] == [item.misses for item in without_diagnostics],
    )

    violations = []
    cross_checks = 0
    for item in with_diagnostics:
        cross_checks += int(item.learning["label_cross_checks_passed"])
        offset = item.learning["minimum_update_minus_decision_offset"]
        if offset is not None and offset < H_MAX:
            violations.append({"capacity": item.capacity, "minimum_offset": offset})
        for sample in item.learning["causality_samples"]:
            if not (
                sample["weight_update_event_index"]
                >= sample["label_resolution_event_index"]
                >= sample["decision_event_index"] + H_MAX
            ):
                violations.append({"capacity": item.capacity, "sample": sample})
    add("delayed_update_never_precedes_its_label", not violations, violations[:10])
    add(
        "causal_capped_label_matches_offline_future_use",
        cross_checks > 0,
        {"cross_checks_passed": cross_checks},
    )

    unresolved = {
        int(item.capacity): {
            "generated": item.learning["examples_generated"],
            "resolved": item.learning["examples_resolved"],
            "unresolved_at_stream_end": item.learning["examples_unresolved_at_stream_end"],
            "unresolved_fraction": item.learning["unresolved_fraction"],
            "average_delay": item.learning["average_feedback_delay_same_layer_events"],
            "maximum_delay": item.learning["maximum_feedback_delay_same_layer_events"],
        }
        for item in with_diagnostics
    }
    trailing_ok = all(
        value["generated"] == value["resolved"] + value["unresolved_at_stream_end"]
        for value in unresolved.values()
    )
    add("delayed_example_accounting_balances", trailing_ok, unresolved)

    stability = []
    for item in with_diagnostics:
        for row in item.weight_rows:
            weights = np.asarray(row["end_weights"], dtype=np.float64)
            stability.append(
                {
                    "capacity": item.capacity,
                    "layer": row["layer"],
                    "sum": float(weights.sum()),
                    "min": float(weights.min()),
                    "finite": bool(np.isfinite(weights).all()),
                    "effective": row["end_effective_advisers"],
                }
            )
    add(
        "adviser_weights_stay_on_the_simplex",
        all(
            abs(item["sum"] - 1.0) < 1e-9 and item["min"] >= 0.0 and item["finite"]
            for item in stability
        ),
        stability[:8],
    )

    events = int(with_diagnostics[0].events)
    report = {
        "schema_version": "race_stage2_pilot_audit_v1",
        "created_at_utc": utc_now(),
        "pilot_workload": pilot.name,
        "pilot_sequences": len(pilot.sequences),
        "pilot_workload_hash": pilot.hash,
        "pilot_data_scope": "prefix of the frozen Stage 0 calibration workload; no evaluation sequence",
        "events": events,
        "capacities": list(capacities),
        "adviser_pool_size": pool_size("primary"),
        "adviser_order": list(PRIMARY_POOL),
        "runtime_seconds_with_diagnostics": elapsed,
        "microseconds_per_event": 1e6 * elapsed / events if events else None,
        "pilot_costs": {int(item.capacity): int(item.misses) for item in with_diagnostics},
        "delayed_feedback": unresolved,
        "ranking_diagnostics": {
            int(item.capacity): item.ranking for item in with_diagnostics
        },
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
        "environment": environment_record(),
        "note": "Pilot results may not change any preregistered threshold or hyperparameter grid.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "pilot_audit.json", report)
    return report
