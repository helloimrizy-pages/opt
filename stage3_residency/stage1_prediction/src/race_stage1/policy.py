from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from residency_headroom.policies import CacheTransition

from .models import TransitionModels


class CausalPredictor(Protocol):
    name: str

    def step(self, request: frozenset[int], gate_weights: Mapping[int, float]) -> np.ndarray:
        """Return scores using calibration state and evaluation information through now."""


class PersistencePredictor:
    name = "persistence"

    def __init__(self, num_experts: int) -> None:
        self.num_experts = int(num_experts)
        self.previous = frozenset()

    def step(self, request: frozenset[int], gate_weights: Mapping[int, float]) -> np.ndarray:
        del gate_weights
        scores = np.zeros(self.num_experts, dtype=np.float64)
        if self.previous:
            scores[np.fromiter(self.previous, dtype=np.int64)] = 1.0
        self.previous = request
        return scores


class LastGatePredictor:
    name = "last_gate"

    def __init__(self, num_experts: int) -> None:
        self.values = np.zeros(num_experts, dtype=np.float64)

    def step(self, request: frozenset[int], gate_weights: Mapping[int, float]) -> np.ndarray:
        for expert in request:
            self.values[expert] = float(gate_weights[expert])
        return self.values.copy()


class GateEWMAPredictor:
    name = "gate_ewma"

    def __init__(self, num_experts: int, alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("Gate-EWMA alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.values = np.zeros(num_experts, dtype=np.float64)

    def step(self, request: frozenset[int], gate_weights: Mapping[int, float]) -> np.ndarray:
        self.values *= self.alpha
        scale = 1.0 - self.alpha
        for expert in request:
            self.values[expert] += scale * float(gate_weights[expert])
        return self.values.copy()


class MarkovPredictor:
    name = "markov"

    def __init__(self, matrix: np.ndarray) -> None:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("Markov matrix must be square")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Markov matrix must contain probabilities")
        self.matrix = values

    def step(self, request: frozenset[int], gate_weights: Mapping[int, float]) -> np.ndarray:
        del gate_weights
        current = np.fromiter(sorted(request), dtype=np.int64)
        return self.matrix[current].mean(axis=0)


class MarkovPlusEWMAPredictor:
    name = "markov_plus_ewma"

    def __init__(self, matrix: np.ndarray, num_experts: int, beta: float, alpha: float) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Hybrid beta must lie in [0, 1]")
        if not 0.0 < alpha < 1.0:
            raise ValueError("Hybrid history alpha must lie in (0, 1)")
        self.markov = MarkovPredictor(matrix)
        self.beta = float(beta)
        self.alpha = float(alpha)
        self.history = np.zeros(num_experts, dtype=np.float64)

    def step(self, request: frozenset[int], gate_weights: Mapping[int, float]) -> np.ndarray:
        conditional = self.markov.step(request, gate_weights)
        self.history *= self.alpha
        scale = 1.0 - self.alpha
        if request:
            self.history[np.fromiter(request, dtype=np.int64)] += scale
        return self.beta * conditional + (1.0 - self.beta) * self.history


@dataclass(frozen=True)
class PredictionStep:
    transition: CacheTransition
    scores: np.ndarray


class PredictionRetentionPolicy:
    """Frozen Stage 0 admission semantics with one shared score-to-retention rule."""

    def __init__(self, capacity: int, num_experts: int, predictor: CausalPredictor) -> None:
        if capacity < 0 or capacity > num_experts:
            raise ValueError("Invalid cache capacity")
        self.capacity = int(capacity)
        self.num_experts = int(num_experts)
        self.predictor = predictor
        self.resident: frozenset[int] = frozenset()
        self.last_used = np.full(num_experts, -1, dtype=np.int64)
        self.clock = 0

    def process(self, request: Any, gate_values: Any) -> PredictionStep:
        requested_array = np.asarray(request, dtype=np.int64)
        weights_array = np.asarray(gate_values, dtype=np.float64)
        if requested_array.ndim != 1 or requested_array.size == 0:
            raise ValueError("Atomic request must be a nonempty vector")
        if weights_array.shape != requested_array.shape:
            raise ValueError("Gate weights must align with the atomic request")
        requested = frozenset(map(int, requested_array))
        if len(requested) != requested_array.size:
            raise ValueError("Atomic request contains duplicate experts")
        if min(requested) < 0 or max(requested) >= self.num_experts:
            raise ValueError("Atomic request contains an out-of-range expert")
        if self.capacity > 0 and len(requested) > self.capacity:
            raise ValueError("Atomic request exceeds cache capacity")
        gates = {int(expert): float(weight) for expert, weight in zip(requested_array, weights_array)}
        scores = np.asarray(self.predictor.step(requested, gates), dtype=np.float64)
        if scores.shape != (self.num_experts,) or not np.isfinite(scores).all():
            raise RuntimeError("Predictor returned invalid expert scores")

        before = self.resident
        hits = requested & before
        misses = requested - before
        if self.capacity == 0:
            after = frozenset()
        else:
            self.clock += 1
            for expert in requested:
                self.last_used[expert] = self.clock
            spare = self.capacity - len(requested)
            old = before - requested
            ranked = sorted(
                old,
                key=lambda expert: (
                    -float(scores[expert]),
                    -int(self.last_used[expert]),
                    expert,
                ),
            )
            after = requested | frozenset(ranked[:spare])
        admissions = after - before
        evictions = before - after
        if self.capacity > 0 and admissions != misses:
            raise RuntimeError("Prediction policy violated mandatory admission")
        if len(after) > self.capacity:
            raise RuntimeError("Prediction policy exceeded cache capacity")
        if self.capacity > 0 and not requested.issubset(after):
            raise RuntimeError("Prediction policy dropped a current-request expert")
        self.resident = after
        return PredictionStep(
            transition=CacheTransition(
                request=requested,
                before=before,
                after=after,
                hits=hits,
                misses=misses,
                admissions=admissions,
                evictions=evictions,
            ),
            scores=scores,
        )


def make_predictor(
    spec: Mapping[str, Any],
    *,
    num_experts: int,
    layer_ordinal: int,
    models: TransitionModels,
) -> CausalPredictor:
    method = str(spec["method"])
    if method == "persistence":
        return PersistencePredictor(num_experts)
    if method == "last_gate":
        return LastGatePredictor(num_experts)
    if method == "gate_ewma":
        return GateEWMAPredictor(num_experts, float(spec["alpha"]))
    if method in {"markov_1", "markov_h"}:
        horizon = 1 if method == "markov_1" else int(spec["horizon"])
        return MarkovPredictor(models.matrix(horizon, layer_ordinal))
    if method == "markov_plus_ewma":
        return MarkovPlusEWMAPredictor(
            models.matrix(int(spec["horizon"]), layer_ordinal),
            num_experts,
            float(spec["beta"]),
            float(spec["history_alpha"]),
        )
    raise ValueError(f"Unknown causal predictor {method!r}")
