"""Causal Stage 3 replay over a frozen workload path.

The cache model, mandatory admission, retention slot count and deterministic
tie-breaking are the frozen Stage 1 mechanism, unchanged. Only the retention score
differs: one calibration-fitted linear ranking function per cache capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from race_stage1.models import TransitionModels, same_layer_indices
from race_stage2.diagnostics import RankingAccumulator, build_future_occurrences
from residency_headroom.common import sha256_json
from residency_headroom.simulator import expert_byte_matrix
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload

from . import CAP
from .features import FeatureState


Scorer = Callable[[np.ndarray], np.ndarray]
"""Maps a ``[feature, expert]`` block to a per-expert retention score.

A scorer may additionally expose ``observe(layer, request, gates, sorted_request,
position)``. When present it is called once per event, before scoring and before the
shared feature state absorbs the event, which lets an audit scorer reproduce a
reference policy's own update order exactly.
"""


@dataclass(frozen=True)
class Stage3SequenceMetrics:
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
class Stage3SimulationResult:
    variant: str
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
    per_sequence: tuple[Stage3SequenceMetrics, ...]
    ranking: dict[str, Any] = field(default_factory=dict)
    layer_misses: tuple[int, ...] = ()

    @property
    def condition_id(self) -> str:
        return sha256_json(
            {"variant": self.variant, "capacity": self.capacity, "workload": self.workload}
        )[:20]

    def result_record(self, *, trace_hash: str, config_hash: str) -> dict[str, Any]:
        return {
            "schema_version": "race_stage3_result_row_v1",
            "variant": self.variant,
            "method_id": self.variant,
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
            "normalized_total_cost": self.misses / self.requests,
            "miss_rate": self.misses / self.requests,
            "expert_transfers": self.misses,
            "trace_hash": trace_hash,
            "stage3_config_hash": config_hash,
        }

    def sequence_records(self, *, trace_hash: str, config_hash: str) -> list[dict[str, Any]]:
        rows = []
        for item in self.per_sequence:
            row = asdict(item)
            row.update(
                {
                    "schema_version": "race_stage3_per_sequence_v1",
                    "variant": self.variant,
                    "method_id": self.variant,
                    "capacity": self.capacity,
                    "cache_capacity": self.capacity,
                    "workload": self.workload,
                    "regime": self.regime,
                    "condition_id": self.condition_id,
                    "expert_transfers": item.misses,
                    "trace_hash": trace_hash,
                    "stage3_config_hash": config_hash,
                }
            )
            rows.append(row)
        return rows

    def diagnostic_record(self, *, trace_hash: str, config_hash: str) -> dict[str, Any]:
        return {
            "schema_version": "race_stage3_diagnostics_v1",
            "variant": self.variant,
            "capacity": self.capacity,
            "workload": self.workload,
            "regime": self.regime,
            "condition_id": self.condition_id,
            "ranking": self.ranking,
            "layer_misses": list(self.layer_misses),
            "trace_hash": trace_hash,
            "stage3_config_hash": config_hash,
        }


class GroupCollector:
    """Accumulates within-candidate-set training groups from a calibration replay."""

    def __init__(self, capacities: Sequence[int], stride: int, warmup: int) -> None:
        self.capacities = tuple(int(value) for value in capacities)
        self.stride = int(stride)
        self.warmup = int(warmup)
        self._features: dict[int, list[np.ndarray]] = {c: [] for c in self.capacities}
        self._targets: dict[int, list[np.ndarray]] = {c: [] for c in self.capacities}
        self._groups: dict[int, list[np.ndarray]] = {c: [] for c in self.capacities}
        self._next: dict[int, int] = {c: 0 for c in self.capacities}

    def offer(
        self,
        capacity: int,
        block: np.ndarray,
        candidates: np.ndarray,
        distances: np.ndarray,
    ) -> None:
        self._features[capacity].append(block[:, candidates].T.astype(np.float32))
        self._targets[capacity].append(distances.astype(np.float32))
        self._groups[capacity].append(
            np.full(candidates.size, self._next[capacity], dtype=np.int32)
        )
        self._next[capacity] += 1

    def dataset(self, capacity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._features.get(capacity):
            raise ValueError(
                f"No training groups were collected at capacity {capacity}; a capacity with zero "
                "spare residency offers no retention choice and cannot train a ranking model"
            )
        return (
            np.concatenate(self._features[capacity]).astype(np.float64),
            np.concatenate(self._targets[capacity]).astype(np.float64),
            np.concatenate(self._groups[capacity]),
        )

    def pooled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features, targets, groups = [], [], []
        offset = 0
        for capacity in self.capacities:
            block, target, group = self.dataset(capacity)
            features.append(block)
            targets.append(target)
            groups.append(group + offset)
            offset += int(group.max()) + 1
        return np.concatenate(features), np.concatenate(targets), np.concatenate(groups)

    @property
    def groups(self) -> int:
        return sum(self._next.values())


def validate_capacities(capacities: Sequence[int], top_k: int, num_experts: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in capacities)
    if not values or len(set(values)) != len(values):
        raise ValueError("Cache capacities must be nonempty and unique")
    if any(value < top_k or value > num_experts for value in values):
        raise ValueError("Stage 3 capacities must hold the complete atomic request")
    return values


def simulate_stage3(
    trace: RoutingTrace,
    workload: Workload,
    capacities: Sequence[int],
    scorers: Mapping[int, Scorer],
    state: FeatureState,
    *,
    variant: str = "stage3",
    enable_diagnostics: bool = False,
    collector: GroupCollector | None = None,
    perfect_score_override: bool = False,
) -> list[Stage3SimulationResult]:
    """Replay one Stage 3 scorer set causally over a frozen workload path."""

    capacities = validate_capacities(capacities, trace.top_k, trace.num_experts)
    layers = tuple(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: ordinal for ordinal, layer in enumerate(layers)}
    num_layers, num_experts = trace.num_layers, trace.num_experts
    count = len(capacities)

    requests_all = trace.requested_expert_ids.astype(np.int64, copy=False)
    sorted_all = np.sort(requests_all, axis=1)
    gates_all = trace.router_weights.astype(np.float64, copy=False)
    byte_matrix = expert_byte_matrix(trace)
    uniform_bytes = (
        byte_matrix[:, 0].astype(np.int64, copy=True)
        if np.all(byte_matrix == byte_matrix[:, :1])
        else None
    )

    resident = np.zeros((count, num_layers, num_experts), dtype=bool)
    last_used = np.full((num_layers, num_experts), -1, dtype=np.int64)
    position = np.zeros(num_layers, dtype=np.int64)
    request_mask = np.zeros(num_experts, dtype=bool)
    spares = [capacity - trace.top_k for capacity in capacities]

    need_future = enable_diagnostics or perfect_score_override or collector is not None
    future = (
        build_future_occurrences(trace, workload, same_layer_indices(trace, workload))
        if need_future
        else None
    )
    observers = []
    for capacity in capacities:
        scorer = scorers[capacity]
        if hasattr(scorer, "observe") and scorer not in observers:
            observers.append(scorer)
    ranking = [RankingAccumulator() for _ in capacities]
    totals = np.zeros((count, 9), dtype=np.int64)
    layer_misses = np.zeros((count, num_layers), dtype=np.int64)
    maximum = [0] * count
    per_capacity: list[list[Stage3SequenceMetrics]] = [[] for _ in capacities]
    state.reset()
    for scorer in observers:
        if hasattr(scorer, "reset"):
            scorer.reset()
    sampled = 0

    for sequence, view in workload.iter_slices(trace):
        state.begin_request()
        sequence_totals = [[0] * 9 for _ in capacities]
        for index in range(view.start, view.stop):
            ordinal = layer_to_ordinal[int(trace.layer_index[index])]
            request = requests_all[index]
            gates = gates_all[index]
            step = int(position[ordinal])
            true_distance = (
                future.advance(ordinal, step, request) if future is not None else None
            )
            block = state.features(ordinal, request, gates, sorted_all[index], step)
            for scorer in observers:
                scorer.observe(ordinal, request, gates, sorted_all[index], step)
            if perfect_score_override:
                scores = [-true_distance.astype(np.float64)] * count
            else:
                scores = [scorers[capacity](block) for capacity in capacities]
            state.absorb(ordinal, request, gates, step)

            take_sample = (
                collector is not None
                and step >= collector.warmup
                and sampled % collector.stride == 0
            )
            request_mask[request] = True
            last_used[ordinal, request] = step
            if uniform_bytes is not None:
                unit = int(uniform_bytes[ordinal])
                requested_bytes = unit * int(request.size)
            else:
                unit = 0
                requested_bytes = int(byte_matrix[ordinal, request].sum(dtype=np.int64))

            for slot, capacity in enumerate(capacities):
                row = resident[slot, ordinal]
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
                spare = spares[slot]
                if spare <= 0:
                    retained = 0
                    row.fill(False)
                    row[request] = True
                elif candidates.size > spare:
                    value = scores[slot][candidates]
                    order = np.lexsort((candidates, -last_used[ordinal, candidates], -value))
                    row.fill(False)
                    row[request] = True
                    row[candidates[order[:spare]]] = True
                    retained = spare
                    if take_sample and candidates.size >= 4:
                        collector.offer(
                            capacity,
                            block,
                            candidates,
                            np.minimum(true_distance[candidates], CAP),
                        )
                    if enable_diagnostics:
                        ranking[slot].record(
                            combined=value,
                            true_distance=true_distance[candidates],
                            order=order,
                            spare=spare,
                        )
                else:
                    row[request] = True
                    retained = int(candidates.size)
                    if enable_diagnostics:
                        ranking[slot].events_without_eviction += 1
                evictions = int(candidates.size) - retained
                occupancy = int(request.size) + retained
                if occupancy > capacity:
                    raise RuntimeError("Stage 3 replay exceeded the cache capacity")
                layer_misses[slot, ordinal] += misses
                bucket = sequence_totals[slot]
                bucket[0] += 1
                bucket[1] += int(request.size)
                bucket[2] += hits
                bucket[3] += misses
                bucket[4] += misses
                bucket[5] += evictions
                bucket[6] += requested_bytes
                bucket[7] += transfer_bytes
                bucket[8] += transfer_bytes
                if occupancy > maximum[slot]:
                    maximum[slot] = occupancy

            request_mask[request] = False
            position[ordinal] += 1
            sampled += 1
        for slot in range(count):
            values = np.asarray(sequence_totals[slot], dtype=np.int64)
            totals[slot] += values
            per_capacity[slot].append(_sequence_metrics(sequence, values))

    if future is not None:
        future.finish()

    results = []
    for slot, capacity in enumerate(capacities):
        values = totals[slot]
        if values[2] + values[3] != values[1]:
            raise RuntimeError("hits + misses != expert requests")
        if values[3] != values[4]:
            raise RuntimeError("misses != mandatory admissions")
        results.append(
            Stage3SimulationResult(
                variant=variant,
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
                maximum_occupancy=int(maximum[slot]),
                per_sequence=tuple(per_capacity[slot]),
                ranking=ranking[slot].as_dict() if enable_diagnostics else {},
                layer_misses=tuple(int(v) for v in layer_misses[slot]),
            )
        )
    return results


class Stage1WinnerScorer:
    """The frozen Stage 1 winner, reproduced with its own arithmetic and update order.

    Stage 1 computes ``0.5 * Markov-H2 + 0.5 * history`` where ``history`` is a
    request-indicator EWMA at alpha 0.95 that is updated with the current event
    *before* the retention decision. Reproducing that exactly needs Stage 1's own
    recurrence and its own summation order, so this scorer keeps its own state
    instead of reading the shared feature block. It exists to prove that the Stage 3
    eviction mechanism is the Stage 1 mechanism, and is never used as a policy.
    """

    def __init__(self, models: TransitionModels, num_layers: int, num_experts: int,
                 beta: float = 0.5, alpha: float = 0.95) -> None:
        self.matrix = np.stack([models.matrix(2, layer) for layer in range(num_layers)])
        self.matrix.flags.writeable = False
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.history = np.zeros((num_layers, num_experts), dtype=np.float64)
        self._score = np.zeros(num_experts, dtype=np.float64)

    def reset(self) -> None:
        self.history.fill(0.0)

    def observe(self, layer: int, request: np.ndarray, gates: np.ndarray,
                sorted_request: np.ndarray, position: int) -> None:
        conditional = self.matrix[layer][sorted_request].mean(axis=0)
        row = self.history[layer]
        row *= self.alpha
        row[request] += 1.0 - self.alpha
        self._score = self.beta * conditional + (1.0 - self.beta) * row

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return self._score


def stage1_winner_scorer(models: TransitionModels, state: FeatureState) -> Stage1WinnerScorer:
    """The frozen Stage 1 winner as an exact audit scorer."""

    return Stage1WinnerScorer(models, state.num_layers, state.num_experts)


def _sequence_metrics(sequence: Any, values: np.ndarray) -> Stage3SequenceMetrics:
    return Stage3SequenceMetrics(
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
