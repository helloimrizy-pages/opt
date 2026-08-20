"""The frozen Stage 2 adviser pool.

Every adviser is a Stage 0 or Stage 1 signal reused unchanged; Stage 2 adds no new
feature family and retunes nothing. Larger raw scores always mean "retain".

Two pools exist. ``primary`` is the preregistered nine-adviser pool that drives the
Stage 2 verdict. ``extended`` additionally exposes the frozen Stage 1 winner itself
as one adviser and is only ever used for the clearly labeled adviser-diversity
ablation, because the Stage 1 winner is a raw-scale blend that percentile-normalized
combination of its two ingredients cannot reproduce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from race_stage1.models import TransitionModels, fit_transition_models
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload


PRIMARY_POOL: tuple[str, ...] = (
    "MARKOV_H1",
    "MARKOV_H2",
    "MARKOV_H4",
    "MARKOV_H8",
    "MARKOV_H16",
    "MARKOV_H32",
    "GATE_EWMA",
    "LFU_DECAY",
    "PERSISTENCE",
)
STAGE1_HYBRID = "STAGE1_HYBRID"
EXTENDED_POOL: tuple[str, ...] = PRIMARY_POOL + (STAGE1_HYBRID,)
POOLS: dict[str, tuple[str, ...]] = {"primary": PRIMARY_POOL, "extended": EXTENDED_POOL}

ADVISER_NAMES = PRIMARY_POOL
NUM_ADVISERS = len(PRIMARY_POOL)

MARKOV_HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
STAGE1_INHERITED_HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16)
GATE_EWMA_ALPHA = 0.95
LFU_DECAY_ALPHA = 0.95
STAGE1_HYBRID_HORIZON = 2
STAGE1_HYBRID_BETA = 0.5
STAGE1_HYBRID_HISTORY_ALPHA = 0.95

MARKOV_SLICE = slice(0, len(MARKOV_HORIZONS))
GATE_EWMA_INDEX = PRIMARY_POOL.index("GATE_EWMA")
LFU_DECAY_INDEX = PRIMARY_POOL.index("LFU_DECAY")
PERSISTENCE_INDEX = PRIMARY_POOL.index("PERSISTENCE")
STAGE1_HYBRID_INDEX = EXTENDED_POOL.index(STAGE1_HYBRID)
HYBRID_MARKOV_ROW = MARKOV_HORIZONS.index(STAGE1_HYBRID_HORIZON)


def pool_names(pool: str = "primary") -> tuple[str, ...]:
    try:
        return POOLS[pool]
    except KeyError as error:
        raise ValueError(f"Unknown adviser pool {pool!r}") from error


def pool_size(pool: str = "primary") -> int:
    return len(pool_names(pool))


def fit_stage2_transition_models(
    trace: RoutingTrace, calibration: Workload
) -> TransitionModels:
    """Fit the Stage 2 Markov horizons with the unmodified Stage 1 fitting routine."""

    return fit_transition_models(trace, calibration, MARKOV_HORIZONS)


def verify_stage1_horizon_reuse(
    stage2: TransitionModels, stage1_path: Path
) -> dict[str, object]:
    """Prove that the Stage 2 model reproduces the frozen Stage 1 horizons exactly."""

    stage1 = TransitionModels.load(stage1_path)
    if stage1.trace_hash != stage2.trace_hash:
        raise ValueError("Stage 2 transition models reference a different trace")
    if stage1.calibration_workload_hash != stage2.calibration_workload_hash:
        raise ValueError("Stage 2 transition models reference a different calibration path")
    checked = []
    difference = 0.0
    for horizon in STAGE1_INHERITED_HORIZONS:
        if horizon not in stage1.horizons:
            raise ValueError(f"Frozen Stage 1 model lacks horizon {horizon}")
        for layer in range(stage2.num_layers):
            left = stage1.matrix(horizon, layer)
            right = stage2.matrix(horizon, layer)
            # Both archives persist float32 probabilities, so the archived-precision
            # comparison is the exact reproduction test.
            if not np.array_equal(left.astype(np.float32), right.astype(np.float32)):
                raise ValueError(
                    f"Stage 2 refit changed the frozen Stage 1 horizon {horizon} at layer {layer}"
                )
            difference = max(difference, float(np.abs(left - right).max()))
        checked.append(int(horizon))
    return {
        "passed": True,
        "stage1_horizons_reproduced_bitwise_at_archive_precision": checked,
        "maximum_absolute_float64_difference": difference,
        "stage2_horizons": list(map(int, stage2.horizons)),
        "new_horizons": [
            int(horizon)
            for horizon in stage2.horizons
            if horizon not in STAGE1_INHERITED_HORIZONS
        ],
        "smoothing": stage2.smoothing,
    }


class AdviserBank:
    """Per-layer adviser state producing one ``[adviser, expert]`` score block."""

    def __init__(
        self,
        models: TransitionModels,
        num_layers: int,
        num_experts: int,
        pool: str = "primary",
    ) -> None:
        missing = [h for h in MARKOV_HORIZONS if h not in models.horizons]
        if missing:
            raise ValueError(f"Transition models lack Stage 2 horizons {missing}")
        if models.num_layers != num_layers or models.num_experts != num_experts:
            raise ValueError("Transition models do not match the trace architecture")
        self.pool = pool
        self.names = pool_names(pool)
        self.size = len(self.names)
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.markov = np.stack(
            [
                np.stack([models.matrix(h, layer) for h in MARKOV_HORIZONS])
                for layer in range(num_layers)
            ]
        )
        self.markov.flags.writeable = False
        self._gate = np.zeros((num_layers, num_experts), dtype=np.float64)
        self._frequency = np.zeros((num_layers, num_experts), dtype=np.float64)
        self._history = np.zeros((num_layers, num_experts), dtype=np.float64)
        self._previous: list[np.ndarray | None] = [None] * num_layers
        self._buffer = np.zeros((self.size, num_experts), dtype=np.float64)
        self._include_hybrid = STAGE1_HYBRID in self.names

    def reset(self) -> None:
        """Clear every adaptive adviser state at a frozen workload-path boundary."""

        self._gate.fill(0.0)
        self._frequency.fill(0.0)
        self._history.fill(0.0)
        self._previous = [None] * self.num_layers
        self._buffer.fill(0.0)

    def step(
        self,
        layer_ordinal: int,
        request: np.ndarray,
        gates: np.ndarray,
        sorted_request: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance every adviser with the current same-layer event and score experts.

        ``sorted_request`` must be ``request`` in ascending expert order. The Markov
        row average is taken in that order so the floating-point result is bitwise
        identical to the frozen Stage 1 predictor, which sorts the atomic request
        before averaging. The returned buffer is reused across calls; downstream code
        copies whatever it needs to retain.
        """

        if sorted_request is None:
            sorted_request = np.sort(request)
        buffer = self._buffer
        np.mean(
            self.markov[layer_ordinal][:, sorted_request, :], axis=1, out=buffer[MARKOV_SLICE]
        )

        gate = self._gate[layer_ordinal]
        gate *= GATE_EWMA_ALPHA
        gate[request] += (1.0 - GATE_EWMA_ALPHA) * gates
        buffer[GATE_EWMA_INDEX] = gate

        frequency = self._frequency[layer_ordinal]
        frequency *= LFU_DECAY_ALPHA
        frequency[request] += 1.0
        buffer[LFU_DECAY_INDEX] = frequency

        persistence = buffer[PERSISTENCE_INDEX]
        persistence.fill(0.0)
        previous = self._previous[layer_ordinal]
        if previous is not None:
            persistence[previous] = 1.0
        self._previous[layer_ordinal] = request

        if self._include_hybrid:
            history = self._history[layer_ordinal]
            history *= STAGE1_HYBRID_HISTORY_ALPHA
            history[request] += 1.0 - STAGE1_HYBRID_HISTORY_ALPHA
            np.multiply(buffer[HYBRID_MARKOV_ROW], STAGE1_HYBRID_BETA, out=buffer[STAGE1_HYBRID_INDEX])
            buffer[STAGE1_HYBRID_INDEX] += (1.0 - STAGE1_HYBRID_BETA) * history

        return buffer


def adviser_parameters(pool: str = "primary") -> dict[str, object]:
    record: dict[str, object] = {
        "pool": pool,
        "order": list(pool_names(pool)),
        "markov_horizons": list(MARKOV_HORIZONS),
        "gate_ewma_alpha": GATE_EWMA_ALPHA,
        "lfu_decay_alpha": LFU_DECAY_ALPHA,
        "stage1_inherited_horizons": list(STAGE1_INHERITED_HORIZONS),
    }
    if STAGE1_HYBRID in pool_names(pool):
        record["stage1_hybrid"] = {
            "horizon": STAGE1_HYBRID_HORIZON,
            "beta": STAGE1_HYBRID_BETA,
            "history_alpha": STAGE1_HYBRID_HISTORY_ALPHA,
            "identifier": "markov_plus_ewma_h2_beta0.5_alpha0.95",
        }
    return record


def horizon_index(horizon: int) -> int:
    return MARKOV_HORIZONS.index(int(horizon))


def uniform_weights(count: int = NUM_ADVISERS) -> np.ndarray:
    return np.full(int(count), 1.0 / float(count), dtype=np.float64)


def validate_simplex(
    weights: Sequence[float] | np.ndarray,
    size: int = NUM_ADVISERS,
    tolerance: float = 1e-9,
) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size != int(size):
        raise ValueError(f"Adviser weights must be a vector of {size} values")
    if not np.isfinite(values).all():
        raise ValueError("Adviser weights must be finite")
    if np.any(values < -tolerance):
        raise ValueError("Adviser weights must be nonnegative")
    if abs(float(values.sum()) - 1.0) > tolerance:
        raise ValueError("Adviser weights must sum to one")
    return values
