"""Causal RACE replay over a frozen workload path.

The cache model, admission rule, retention slot count and deterministic
tie-breaking are the frozen Stage 1 mechanism. The only Stage 2 change is that the
retention score is an adviser-weight combination of rank-normalized adviser
opinions instead of a single fixed predictor score.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from race_stage1.models import TransitionModels, same_layer_indices
from residency_headroom.common import sha256_json
from residency_headroom.simulator import expert_byte_matrix
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload

from . import H_MAX, NOT_REUSED_WITHIN_HORIZON
from .advisers import AdviserBank, pool_names
from .diagnostics import (
    LearningAccumulator,
    RankingAccumulator,
    build_future_occurrences,
    horizon_weight_summary,
    weight_summary,
)
from .hedge import HedgeWeights
from .labels import LabelWindow, PendingExample
from .losses import pairwise_losses
from .policy import RaceVariant
from .ranking import combined_scores, midrank_normalize, retention_order


@dataclass(frozen=True)
class Stage2SequenceMetrics:
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
class Stage2SimulationResult:
    variant: str
    variant_id: str
    parameters: dict[str, Any]
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
    per_sequence: tuple[Stage2SequenceMetrics, ...]
    learning: dict[str, Any] = field(default_factory=dict)
    ranking: dict[str, Any] = field(default_factory=dict)
    weight_rows: tuple[dict[str, Any], ...] = ()
    adviser_mean_weights: dict[str, float] = field(default_factory=dict)
    trajectory: tuple[dict[str, Any], ...] = ()
    layer_misses: tuple[int, ...] = ()

    @property
    def condition_id(self) -> str:
        return sha256_json(
            {
                "variant_id": self.variant_id,
                "capacity": self.capacity,
                "workload": self.workload,
            }
        )[:20]

    def result_record(self, *, trace_hash: str, preregistration_hash: str, config_hash: str) -> dict[str, Any]:
        return {
            "schema_version": "race_stage2_result_row_v1",
            "variant": self.variant,
            "variant_id": self.variant_id,
            "method_id": self.variant_id,
            "parameters": dict(self.parameters),
            "causal": True,
            "capacity": self.capacity,
            "cache_capacity": self.capacity,
            "workload": self.workload,
            "regime": self.regime,
            "condition_id": self.condition_id,
            "events": self.events,
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "admissions": self.admissions,
            "evictions": self.evictions,
            "bytes_requested": self.bytes_requested,
            "bytes_transferred": self.bytes_transferred,
            "admission_bytes": self.admission_bytes,
            "maximum_occupancy": self.maximum_occupancy,
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
            "stage2_config_hash": config_hash,
        }

    def sequence_records(self, *, trace_hash: str, config_hash: str) -> list[dict[str, Any]]:
        rows = []
        for item in self.per_sequence:
            row = asdict(item)
            row.update(
                {
                    "schema_version": "race_stage2_per_sequence_v1",
                    "variant": self.variant,
                    "variant_id": self.variant_id,
                    "method_id": self.variant_id,
                    "capacity": self.capacity,
                    "cache_capacity": self.capacity,
                    "workload": self.workload,
                    "regime": self.regime,
                    "condition_id": self.condition_id,
                    "expert_transfers": item.misses,
                    "trace_hash": trace_hash,
                    "stage2_config_hash": config_hash,
                }
            )
            rows.append(row)
        return rows

    def diagnostic_record(self, *, trace_hash: str, config_hash: str) -> dict[str, Any]:
        return {
            "schema_version": "race_stage2_diagnostics_v1",
            "variant": self.variant,
            "variant_id": self.variant_id,
            "capacity": self.capacity,
            "workload": self.workload,
            "regime": self.regime,
            "condition_id": self.condition_id,
            "learning": self.learning,
            "ranking": self.ranking,
            "layer_misses": list(self.layer_misses),
            "weights_by_stream": list(self.weight_rows),
            "adviser_mean_weights": dict(self.adviser_mean_weights),
            "trace_hash": trace_hash,
            "stage2_config_hash": config_hash,
        }


class CalibrationExampleCollector:
    """Deterministic stride subsample of resolved calibration learning examples."""

    def __init__(self, stride: int, capacities: Sequence[int]) -> None:
        if stride < 1:
            raise ValueError("Example stride must be positive")
        self.stride = int(stride)
        self.capacities = tuple(int(value) for value in capacities)
        self._seen: dict[tuple[int, int], int] = {}
        self.normalized: list[np.ndarray] = []
        self.distances: list[np.ndarray] = []
        self.capacity_of: list[int] = []
        self.layer_of: list[int] = []

    def offer(
        self, capacity: int, layer: int, normalized: np.ndarray, distances: np.ndarray
    ) -> None:
        if capacity not in self.capacities:
            return
        key = (int(capacity), int(layer))
        index = self._seen.get(key, 0)
        self._seen[key] = index + 1
        if index % self.stride:
            return
        self.normalized.append(np.asarray(normalized, dtype=np.float64).copy())
        self.distances.append(np.asarray(distances, dtype=np.int64).copy())
        self.capacity_of.append(int(capacity))
        self.layer_of.append(int(layer))

    def __len__(self) -> int:
        return len(self.normalized)


def validate_capacities(capacities: Sequence[int], top_k: int, num_experts: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in capacities)
    if not values or len(set(values)) != len(values):
        raise ValueError("Cache capacities must be nonempty and unique")
    if any(value < top_k or value > num_experts for value in values):
        raise ValueError("Stage 2 capacities must hold the complete atomic request")
    return values


def _uniform_layer_bytes(byte_matrix: np.ndarray) -> np.ndarray | None:
    first = byte_matrix[:, :1]
    if np.all(byte_matrix == first):
        return byte_matrix[:, 0].astype(np.int64, copy=True)
    return None


def simulate_race_variant(
    trace: RoutingTrace,
    workload: Workload,
    capacities: Sequence[int],
    variant: RaceVariant,
    models: TransitionModels,
    *,
    enable_diagnostics: bool = True,
    example_collector: CalibrationExampleCollector | None = None,
    trajectory_stride: int | None = None,
    perfect_score_override: bool = False,
    label_cross_check: bool = False,
) -> list[Stage2SimulationResult]:
    """Replay one RACE variant causally over a frozen workload path."""

    capacities = validate_capacities(capacities, trace.top_k, trace.num_experts)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    num_layers = trace.num_layers
    num_experts = trace.num_experts
    num_capacities = len(capacities)

    requests_all = trace.requested_expert_ids.astype(np.int64, copy=False)
    sorted_requests_all = np.sort(requests_all, axis=1)
    gates_all = trace.router_weights.astype(np.float64, copy=False)
    byte_matrix = expert_byte_matrix(trace)
    uniform_bytes = _uniform_layer_bytes(byte_matrix)

    names = pool_names(variant.pool)
    size = len(names)
    bank = AdviserBank(models, num_layers, num_experts, pool=variant.pool)
    window = LabelWindow(num_layers, num_experts)
    resident = np.zeros((num_capacities, num_layers, num_experts), dtype=bool)
    last_used = np.full((num_layers, num_experts), -1, dtype=np.int64)
    position = np.zeros(num_layers, dtype=np.int64)
    request_mask = np.zeros(num_experts, dtype=bool)

    streams = num_layers if variant.scope == "per_layer" else 1
    initial_matrix = variant.initial_weight_matrix(streams)
    hedges = [HedgeWeights(streams, initial_matrix) for _ in capacities]
    initial_weights = np.stack([hedge.snapshot() for hedge in hedges])
    pending: list[list[deque[PendingExample]]] = [
        [deque() for _layer in range(num_layers)] for _capacity in capacities
    ]
    ranking = [RankingAccumulator() for _capacity in capacities]
    learning = [LearningAccumulator(names=names) for _capacity in capacities]
    weight_sum = np.zeros((num_capacities, streams, size), dtype=np.float64)
    weight_square = np.zeros((num_capacities, streams, size), dtype=np.float64)
    weight_count = np.zeros((num_capacities, streams), dtype=np.int64)
    trajectory: list[dict[str, Any]] = []

    layer_streams = same_layer_indices(trace, workload)
    future = (
        build_future_occurrences(trace, workload, layer_streams)
        if (enable_diagnostics or perfect_score_override)
        else None
    )

    totals = np.zeros((num_capacities, 9), dtype=np.int64)
    layer_misses = np.zeros((num_capacities, num_layers), dtype=np.int64)
    maximum = [0] * num_capacities
    per_capacity_sequences: list[list[Stage2SequenceMetrics]] = [
        [] for _capacity in capacities
    ]
    spares = [capacity - trace.top_k for capacity in capacities]
    want_cost_loss = True

    for sequence, view in workload.iter_slices(trace):
        sequence_totals = [[0] * 9 for _capacity in capacities]
        for index in range(view.start, view.stop):
            ordinal = layer_to_ordinal[int(trace.layer_index[index])]
            request = requests_all[index]
            gates = gates_all[index]
            step = int(position[ordinal])
            window.push(ordinal, step, request)
            scores = bank.step(ordinal, request, gates, sorted_requests_all[index])
            true_distance = (
                future.advance(ordinal, step, request) if future is not None else None
            )
            if perfect_score_override:
                scores = np.broadcast_to(
                    -true_distance.astype(np.float64), (size, num_experts)
                )

            if step >= H_MAX:
                target = step - H_MAX
                due = [
                    index_c
                    for index_c in range(num_capacities)
                    if pending[index_c][ordinal]
                    and pending[index_c][ordinal][0].decision_position == target
                ]
                if due:
                    capped = window.capped_distances(ordinal, step)
                    for index_c in due:
                        example = pending[index_c][ordinal].popleft()
                        if example.expected_capped is not None:
                            if not np.array_equal(
                                example.expected_capped, capped[example.candidates]
                            ):
                                raise RuntimeError(
                                    "Causal capped label disagrees with the offline future-use label"
                                )
                            learning[index_c].label_cross_checks += 1
                        stream = ordinal if variant.scope == "per_layer" else 0
                        losses = pairwise_losses(
                            example.normalized,
                            capped[example.candidates],
                            want_rank=True,
                            want_cost=want_cost_loss,
                        )
                        applied = False
                        if losses.usable:
                            learning[index_c].note_losses(
                                rank=losses.rank,
                                cost=losses.cost,
                                weights=example.deployed_weights,
                            )
                            if example_collector is not None:
                                example_collector.offer(
                                    capacities[index_c],
                                    ordinal,
                                    example.normalized,
                                    capped[example.candidates],
                                )
                            if variant.adaptive:
                                chosen = (
                                    losses.rank if variant.loss == "rank" else losses.cost
                                )
                                if chosen is not None:
                                    if step < example.decision_position + H_MAX:
                                        raise RuntimeError(
                                            "Stage 2 attempted an update before its label was observable"
                                        )
                                    hedges[index_c].update(stream, chosen, float(variant.eta))
                                    applied = True
                        learning[index_c].note_resolution(
                            layer=ordinal,
                            decision_position=example.decision_position,
                            resolution_position=step,
                            update_position=step,
                            usable=losses.usable,
                            applied=applied,
                        )

            request_mask[request] = True
            last_used[ordinal, request] = step
            if uniform_bytes is not None:
                unit = int(uniform_bytes[ordinal])
                requested_bytes = unit * int(request.size)
            else:
                unit = 0
                requested_bytes = int(byte_matrix[ordinal, request].sum(dtype=np.int64))

            for index_c in range(num_capacities):
                row = resident[index_c, ordinal]
                present = row[request]
                hits = int(present.sum())
                misses = int(request.size) - hits
                if uniform_bytes is not None:
                    transfer_bytes = unit * misses
                else:
                    missing = request[~present]
                    transfer_bytes = (
                        int(byte_matrix[ordinal, missing].sum(dtype=np.int64))
                        if missing.size
                        else 0
                    )
                candidates = np.flatnonzero(row & ~request_mask)
                spare = spares[index_c]
                if spare <= 0:
                    retained = 0
                    row.fill(False)
                    row[request] = True
                elif candidates.size > spare:
                    normalized = midrank_normalize(scores[:, candidates])
                    stream = ordinal if variant.scope == "per_layer" else 0
                    deployed = hedges[index_c].weights(stream)
                    combined = combined_scores(deployed, normalized)
                    order = retention_order(
                        candidates, combined, last_used[ordinal, candidates]
                    )
                    row.fill(False)
                    row[request] = True
                    row[candidates[order[:spare]]] = True
                    retained = spare
                    pending[index_c][ordinal].append(
                        PendingExample(
                            step,
                            candidates,
                            normalized,
                            (
                                np.minimum(
                                    true_distance[candidates], NOT_REUSED_WITHIN_HORIZON
                                )
                                if label_cross_check and true_distance is not None
                                else None
                            ),
                            deployed.copy(),
                        )
                    )
                    learning[index_c].generated += 1
                    if variant.adaptive:
                        weight_sum[index_c, stream] += deployed
                        weight_square[index_c, stream] += deployed * deployed
                        weight_count[index_c, stream] += 1
                    if enable_diagnostics:
                        ranking[index_c].record(
                            combined=combined,
                            true_distance=true_distance[candidates],
                            order=order,
                            spare=spare,
                        )
                else:
                    row[request] = True
                    retained = int(candidates.size)
                    if enable_diagnostics:
                        ranking[index_c].events_without_eviction += 1
                evictions = int(candidates.size) - retained
                occupancy = int(request.size) + retained
                if occupancy > capacities[index_c]:
                    raise RuntimeError("RACE replay exceeded the cache capacity")
                layer_misses[index_c, ordinal] += misses
                bucket = sequence_totals[index_c]
                bucket[0] += 1
                bucket[1] += int(request.size)
                bucket[2] += hits
                bucket[3] += misses
                bucket[4] += misses
                bucket[5] += evictions
                bucket[6] += requested_bytes
                bucket[7] += transfer_bytes
                bucket[8] += transfer_bytes
                if occupancy > maximum[index_c]:
                    maximum[index_c] = occupancy

            request_mask[request] = False
            if trajectory_stride and step % int(trajectory_stride) == 0:
                stream = ordinal if variant.scope == "per_layer" else 0
                for index_c in range(num_capacities):
                    trajectory.append(
                        {
                            "capacity": capacities[index_c],
                            "layer": ordinal if variant.scope == "per_layer" else -1,
                            "same_layer_position": step,
                            "weights": hedges[index_c].weights(stream).tolist(),
                        }
                    )
            position[ordinal] += 1
        for index_c in range(num_capacities):
            values = np.asarray(sequence_totals[index_c], dtype=np.int64)
            totals[index_c] += values
            per_capacity_sequences[index_c].append(_sequence_metrics(sequence, values))

    for index_c in range(num_capacities):
        for layer in range(num_layers):
            learning[index_c].unresolved_at_stream_end += len(pending[index_c][layer])
        hedges[index_c].validate()
    if future is not None:
        future.finish()

    results: list[Stage2SimulationResult] = []
    for index_c, capacity in enumerate(capacities):
        values = totals[index_c]
        if values[2] + values[3] != values[1]:
            raise RuntimeError("hits + misses != expert requests")
        if values[3] != values[4]:
            raise RuntimeError("misses != mandatory admissions")
        final_weights = hedges[index_c].snapshot()
        rows = weight_summary(
            initial=initial_weights[index_c],
            final=final_weights,
            deployed_sum=weight_sum[index_c],
            deployed_square_sum=weight_square[index_c],
            deployed_count=weight_count[index_c],
            names=names,
        )
        if variant.scope == "global":
            for row in rows:
                row["layer"] = -1
        results.append(
            Stage2SimulationResult(
                variant=variant.name,
                variant_id=variant.variant_id,
                parameters=variant.parameters(),
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
                maximum_occupancy=int(maximum[index_c]),
                per_sequence=tuple(per_capacity_sequences[index_c]),
                learning=learning[index_c].as_dict(),
                ranking=ranking[index_c].as_dict() if enable_diagnostics else {},
                weight_rows=tuple(rows),
                adviser_mean_weights=horizon_weight_summary(rows, names),
                trajectory=tuple(
                    item for item in trajectory if item["capacity"] == capacity
                ),
                layer_misses=tuple(int(value) for value in layer_misses[index_c]),
            )
        )
    return results


def simulate_perfect_score(
    trace: RoutingTrace,
    workload: Workload,
    capacities: Sequence[int],
    models: TransitionModels,
) -> list[Stage2SimulationResult]:
    """Drive the unchanged Stage 2 eviction mechanism with exact next-use scores."""

    from .policy import uniform_variant

    return simulate_race_variant(
        trace,
        workload,
        capacities,
        uniform_variant(),
        models,
        enable_diagnostics=False,
        perfect_score_override=True,
    )


def _sequence_metrics(sequence: Any, values: np.ndarray) -> Stage2SequenceMetrics:
    return Stage2SequenceMetrics(
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


def total_cost(results: Sequence[Stage2SimulationResult], capacities: Sequence[int]) -> int:
    """Unit-miss cost summed over the requested capacities."""

    wanted = set(int(value) for value in capacities)
    return int(sum(result.misses for result in results if result.capacity in wanted))


def variant_costs(results: Sequence[Stage2SimulationResult]) -> dict[int, int]:
    return {int(result.capacity): int(result.misses) for result in results}


ExampleOffer = Callable[[int, int, np.ndarray, np.ndarray], None]
