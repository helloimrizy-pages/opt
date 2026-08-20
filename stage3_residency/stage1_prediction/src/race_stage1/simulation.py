from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from residency_headroom.common import sha256_json
from residency_headroom.simulator import expert_byte_matrix
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload

from .models import TransitionModels, same_layer_indices
from .policy import make_predictor


@dataclass(frozen=True)
class Stage1SequenceMetrics:
    sequence_position: int
    source_sequence_id: int
    domain: str
    segment_index: int
    events: int
    requests: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    bytes_requested: int
    bytes_transferred: int
    admission_bytes: int


@dataclass(frozen=True)
class Stage1SimulationResult:
    method: str
    method_id: str
    parameters: dict[str, Any]
    causal: bool
    diagnostic: bool
    capacity: int
    workload: str
    regime: str
    events: int
    requests: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    bytes_requested: int
    bytes_transferred: int
    admission_bytes: int
    maximum_occupancy: int
    per_sequence: tuple[Stage1SequenceMetrics, ...]

    @property
    def condition_id(self) -> str:
        return sha256_json(
            {
                "method_id": self.method_id,
                "capacity": self.capacity,
                "workload": self.workload,
            }
        )[:20]

    def result_record(
        self, *, trace_hash: str, preregistration_hash: str, model_hash: str
    ) -> dict[str, Any]:
        value = asdict(self)
        value.pop("per_sequence")
        value.update(
            {
                "schema_version": "race_stage1_result_row_v1",
                "cache_capacity": self.capacity,
                "condition_id": self.condition_id,
                "cost_model": "unit_miss",
                "lambda": 0.0,
                "total_cost": float(self.misses),
                "miss_cost": float(self.misses),
                "switch_cost": float(self.admissions),
                "normalized_total_cost": self.misses / self.requests,
                "miss_rate": self.misses / self.requests,
                "expert_transfers": self.misses,
                "cache_churn": self.admissions + self.evictions,
                "trace_hash": trace_hash,
                "preregistration_hash": preregistration_hash,
                "transition_model_hash": model_hash,
            }
        )
        return value

    def sequence_records(
        self, *, trace_hash: str, preregistration_hash: str, model_hash: str
    ) -> list[dict[str, Any]]:
        rows = []
        for item in self.per_sequence:
            row = asdict(item)
            row.update(
                {
                    "schema_version": "race_stage1_per_sequence_v1",
                    "method": self.method,
                    "method_id": self.method_id,
                    "parameters": self.parameters,
                    "causal": self.causal,
                    "diagnostic": self.diagnostic,
                    "capacity": self.capacity,
                    "cache_capacity": self.capacity,
                    "workload": self.workload,
                    "regime": self.regime,
                    "condition_id": self.condition_id,
                    "expert_transfers": item.misses,
                    "trace_hash": trace_hash,
                    "preregistration_hash": preregistration_hash,
                    "transition_model_hash": model_hash,
                }
            )
            rows.append(row)
        return rows


def method_id(spec: Mapping[str, Any]) -> str:
    method = str(spec["method"])
    if method == "gate_ewma":
        return f"gate_ewma_alpha{_number(float(spec['alpha']))}"
    if method == "markov_h":
        return f"markov_h_h{int(spec['horizon'])}"
    if method == "markov_plus_ewma":
        return (
            f"markov_plus_ewma_h{int(spec['horizon'])}"
            f"_beta{_number(float(spec['beta']))}"
            f"_alpha{_number(float(spec['history_alpha']))}"
        )
    return method


def causal_specs(preregistration: Mapping[str, Any]) -> list[dict[str, Any]]:
    hyper = preregistration["hyperparameters"]
    specs: list[dict[str, Any]] = [
        {"method": "persistence"},
        {"method": "last_gate"},
    ]
    specs.extend(
        {"method": "gate_ewma", "alpha": float(alpha)}
        for alpha in hyper["gate_ewma_alpha_grid"]
    )
    specs.append({"method": "markov_1", "horizon": 1})
    specs.extend(
        {"method": "markov_h", "horizon": int(horizon)}
        for horizon in hyper["markov_h_grid"]
    )
    return specs


def hybrid_specs(
    preregistration: Mapping[str, Any], selected_horizon: int
) -> list[dict[str, Any]]:
    hyper = preregistration["hyperparameters"]
    return [
        {
            "method": "markov_plus_ewma",
            "horizon": int(selected_horizon),
            "beta": float(beta),
            "history_alpha": float(hyper["hybrid_history_alpha"]),
        }
        for beta in hyper["hybrid_beta_grid"]
    ]


def simulate_causal_capacities(
    trace: RoutingTrace,
    workload: Workload,
    capacities: Sequence[int],
    spec: Mapping[str, Any],
    models: TransitionModels,
) -> list[Stage1SimulationResult]:
    capacities = tuple(map(int, capacities))
    _validate_capacities(capacities, trace.top_k, trace.num_experts)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    predictors = [
        make_predictor(
            spec,
            num_experts=trace.num_experts,
            layer_ordinal=ordinal,
            models=models,
        )
        for ordinal in range(trace.num_layers)
    ]
    residents = [
        [frozenset() for _layer in layers]
        for _capacity in capacities
    ]
    last_used = np.full((trace.num_layers, trace.num_experts), -1, dtype=np.int64)
    clocks = np.zeros(trace.num_layers, dtype=np.int64)
    byte_matrix = expert_byte_matrix(trace)
    totals = np.zeros((len(capacities), 9), dtype=np.int64)
    maximum = np.zeros(len(capacities), dtype=np.int64)
    per_capacity_sequences: list[list[Stage1SequenceMetrics]] = [
        [] for _capacity in capacities
    ]

    for sequence, view in workload.iter_slices(trace):
        sequence_totals = np.zeros((len(capacities), 9), dtype=np.int64)
        for index in range(view.start, view.stop):
            layer = int(trace.layer_index[index])
            ordinal = layer_to_ordinal[layer]
            request_array = trace.requested_expert_ids[index].astype(np.int64, copy=False)
            weights = trace.router_weights[index].astype(np.float64, copy=False)
            request = frozenset(map(int, request_array))
            gates = {
                int(expert): float(weight)
                for expert, weight in zip(request_array, weights)
            }
            scores = np.asarray(predictors[ordinal].step(request, gates), dtype=np.float64)
            if scores.shape != (trace.num_experts,) or not np.isfinite(scores).all():
                raise RuntimeError("Causal predictor returned invalid scores")
            clocks[ordinal] += 1
            last_used[ordinal, request_array] = clocks[ordinal]
            requested_bytes = int(byte_matrix[ordinal, request_array].sum(dtype=np.int64))
            for capacity_index, capacity in enumerate(capacities):
                before = residents[capacity_index][ordinal]
                hits = request & before
                misses = request - before
                spare = capacity - len(request)
                old = before - request
                ranked = sorted(
                    old,
                    key=lambda expert: (
                        -float(scores[expert]),
                        -int(last_used[ordinal, expert]),
                        expert,
                    ),
                )
                after = request | frozenset(ranked[:spare])
                admissions = after - before
                evictions = before - after
                if admissions != misses or not request.issubset(after) or len(after) > capacity:
                    raise RuntimeError("Causal replay violated frozen Stage 0 cache semantics")
                residents[capacity_index][ordinal] = after
                missing_array = np.fromiter(misses, dtype=np.int64)
                transfer_bytes = (
                    int(byte_matrix[ordinal, missing_array].sum(dtype=np.int64))
                    if missing_array.size
                    else 0
                )
                event = np.asarray(
                    [
                        1,
                        len(request),
                        len(hits),
                        len(misses),
                        len(admissions),
                        len(evictions),
                        requested_bytes,
                        transfer_bytes,
                        transfer_bytes,
                    ],
                    dtype=np.int64,
                )
                sequence_totals[capacity_index] += event
                maximum[capacity_index] = max(maximum[capacity_index], len(after))
        totals += sequence_totals
        for capacity_index in range(len(capacities)):
            per_capacity_sequences[capacity_index].append(
                _sequence_metrics(sequence, sequence_totals[capacity_index])
            )
    return _build_results(
        method=str(spec["method"]),
        identifier=method_id(spec),
        parameters={key: value for key, value in spec.items() if key != "method"},
        causal=True,
        diagnostic=False,
        capacities=capacities,
        workload=workload,
        totals=totals,
        maximum=maximum,
        per_capacity_sequences=per_capacity_sequences,
    )


def simulate_lookahead_capacities(
    trace: RoutingTrace,
    workload: Workload,
    capacities: Sequence[int],
    horizons: Sequence[int],
    *,
    include_perfect: bool = True,
) -> list[Stage1SimulationResult]:
    capacities = tuple(map(int, capacities))
    horizons_value: tuple[int | None, ...] = tuple(map(int, horizons)) + (
        (None,) if include_perfect else tuple()
    )
    _validate_capacities(capacities, trace.top_k, trace.num_experts)
    if any(item is not None and item < 1 for item in horizons_value):
        raise ValueError("Lookahead horizons must be positive")
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    indices_by_layer = same_layer_indices(trace, workload)
    future: list[list[deque[int]]] = [
        [deque() for _expert in range(trace.num_experts)] for _layer in layers
    ]
    for ordinal, indices in enumerate(indices_by_layer):
        for position, request in enumerate(trace.requested_expert_ids[indices]):
            for expert in request:
                future[ordinal][int(expert)].append(position)
    positions = np.zeros(trace.num_layers, dtype=np.int64)
    last_used = np.full((trace.num_layers, trace.num_experts), -1, dtype=np.int64)
    residents = [
        [
            [frozenset() for _layer in layers]
            for _capacity in capacities
        ]
        for _horizon in horizons_value
    ]
    byte_matrix = expert_byte_matrix(trace)
    totals = np.zeros((len(horizons_value), len(capacities), 9), dtype=np.int64)
    maximum = np.zeros((len(horizons_value), len(capacities)), dtype=np.int64)
    per_method_sequences: list[list[list[Stage1SequenceMetrics]]] = [
        [[] for _capacity in capacities] for _horizon in horizons_value
    ]

    for sequence, view in workload.iter_slices(trace):
        sequence_totals = np.zeros_like(totals)
        for index in range(view.start, view.stop):
            layer = int(trace.layer_index[index])
            ordinal = layer_to_ordinal[layer]
            position = int(positions[ordinal])
            request_array = trace.requested_expert_ids[index].astype(np.int64, copy=False)
            request = frozenset(map(int, request_array))
            for expert in request:
                queue = future[ordinal][expert]
                if not queue or queue[0] != position:
                    raise RuntimeError("Lookahead same-layer future accounting changed")
                queue.popleft()
                last_used[ordinal, expert] = position
            requested_bytes = int(byte_matrix[ordinal, request_array].sum(dtype=np.int64))
            infinity = int(indices_by_layer[ordinal].size + 1)
            for horizon_index, horizon in enumerate(horizons_value):
                cutoff = infinity if horizon is None else position + horizon
                for capacity_index, capacity in enumerate(capacities):
                    before = residents[horizon_index][capacity_index][ordinal]
                    hits = request & before
                    misses = request - before
                    spare = capacity - len(request)
                    old = before - request

                    def rank(expert: int) -> tuple[int, int, int]:
                        queue = future[ordinal][expert]
                        following = queue[0] if queue else infinity
                        visible = following if following <= cutoff else infinity
                        return visible, -int(last_used[ordinal, expert]), expert

                    after = request | frozenset(sorted(old, key=rank)[:spare])
                    admissions = after - before
                    evictions = before - after
                    if admissions != misses or not request.issubset(after) or len(after) > capacity:
                        raise RuntimeError("Lookahead replay violated frozen cache semantics")
                    residents[horizon_index][capacity_index][ordinal] = after
                    missing_array = np.fromiter(misses, dtype=np.int64)
                    transfer_bytes = (
                        int(byte_matrix[ordinal, missing_array].sum(dtype=np.int64))
                        if missing_array.size
                        else 0
                    )
                    sequence_totals[horizon_index, capacity_index] += np.asarray(
                        [
                            1,
                            len(request),
                            len(hits),
                            len(misses),
                            len(admissions),
                            len(evictions),
                            requested_bytes,
                            transfer_bytes,
                            transfer_bytes,
                        ],
                        dtype=np.int64,
                    )
                    maximum[horizon_index, capacity_index] = max(
                        maximum[horizon_index, capacity_index], len(after)
                    )
            positions[ordinal] += 1
        totals += sequence_totals
        for horizon_index in range(len(horizons_value)):
            for capacity_index in range(len(capacities)):
                per_method_sequences[horizon_index][capacity_index].append(
                    _sequence_metrics(
                        sequence, sequence_totals[horizon_index, capacity_index]
                    )
                )
    for ordinal, indices in enumerate(indices_by_layer):
        if positions[ordinal] != indices.size or any(future[ordinal][expert] for expert in range(trace.num_experts)):
            raise RuntimeError("Lookahead did not consume the complete same-layer future")

    results: list[Stage1SimulationResult] = []
    for horizon_index, horizon in enumerate(horizons_value):
        if horizon is None:
            name = "perfect_score_simple_policy"
            parameters: dict[str, Any] = {"horizon": "full_remaining_trace"}
        else:
            name = f"lookahead_oracle_h{horizon}"
            parameters = {"horizon": horizon}
        results.extend(
            _build_results(
                method=name,
                identifier=name,
                parameters=parameters,
                causal=False,
                diagnostic=True,
                capacities=capacities,
                workload=workload,
                totals=totals[horizon_index],
                maximum=maximum[horizon_index],
                per_capacity_sequences=per_method_sequences[horizon_index],
            )
        )
    return results


def _build_results(
    *,
    method: str,
    identifier: str,
    parameters: dict[str, Any],
    causal: bool,
    diagnostic: bool,
    capacities: Sequence[int],
    workload: Workload,
    totals: np.ndarray,
    maximum: np.ndarray,
    per_capacity_sequences: Sequence[Sequence[Stage1SequenceMetrics]],
) -> list[Stage1SimulationResult]:
    results = []
    for capacity_index, capacity in enumerate(capacities):
        values = totals[capacity_index]
        if values[2] + values[3] != values[1]:
            raise RuntimeError("hits + misses != expert requests")
        if values[3] != values[4]:
            raise RuntimeError("misses != mandatory admissions")
        results.append(
            Stage1SimulationResult(
                method=method,
                method_id=identifier,
                parameters=dict(parameters),
                causal=causal,
                diagnostic=diagnostic,
                capacity=int(capacity),
                workload=workload.name,
                regime=workload.regime,
                events=int(values[0]),
                requests=int(values[1]),
                hits=int(values[2]),
                misses=int(values[3]),
                admissions=int(values[4]),
                evictions=int(values[5]),
                bytes_requested=int(values[6]),
                bytes_transferred=int(values[7]),
                admission_bytes=int(values[8]),
                maximum_occupancy=int(maximum[capacity_index]),
                per_sequence=tuple(per_capacity_sequences[capacity_index]),
            )
        )
    return results


def _sequence_metrics(sequence: Any, values: np.ndarray) -> Stage1SequenceMetrics:
    return Stage1SequenceMetrics(
        sequence_position=int(sequence.position),
        source_sequence_id=int(sequence.source_sequence_id),
        domain=str(sequence.domain),
        segment_index=int(sequence.segment_index),
        events=int(values[0]),
        requests=int(values[1]),
        hits=int(values[2]),
        misses=int(values[3]),
        admissions=int(values[4]),
        evictions=int(values[5]),
        bytes_requested=int(values[6]),
        bytes_transferred=int(values[7]),
        admission_bytes=int(values[8]),
    )


def _validate_capacities(capacities: Sequence[int], top_k: int, num_experts: int) -> None:
    if not capacities or len(set(capacities)) != len(capacities):
        raise ValueError("Cache capacities must be nonempty and unique")
    if any(capacity < top_k or capacity > num_experts for capacity in capacities):
        raise ValueError("Stage 1 capacities must hold the complete atomic request")


def _number(value: float) -> str:
    return format(value, ".12g")
