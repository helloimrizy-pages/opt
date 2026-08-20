"""Capped future-reuse targets and the delayed-feedback pending queue.

The capped target ``d_tilde`` is fully determined once ``H_MAX`` further same-layer
events have been observed, which is exactly what makes the Stage 2 online updates
causal. Nothing in this module ever reads beyond the current same-layer event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import H_MAX, NOT_REUSED_WITHIN_HORIZON


_PHASE_ORDER = np.stack(
    [(np.arange(1, H_MAX + 1, dtype=np.int64) + phase) % H_MAX for phase in range(H_MAX)]
)
_PHASE_ORDER.flags.writeable = False


@dataclass
class PendingExample:
    """One delayed learning example awaiting its capped future-reuse label."""

    decision_position: int
    candidates: np.ndarray
    normalized: np.ndarray
    expected_capped: np.ndarray | None = None
    deployed_weights: np.ndarray | None = None

    @property
    def resolution_position(self) -> int:
        return self.decision_position + H_MAX


class LabelWindow:
    """Rolling per-layer window over the last ``H_MAX`` same-layer atomic requests."""

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self._window = np.zeros((num_layers, H_MAX, num_experts), dtype=bool)

    def reset(self) -> None:
        self._window.fill(False)

    def push(self, layer_ordinal: int, position: int, request: np.ndarray) -> None:
        """Record the atomic request observed at same-layer ``position``."""

        slot = self._window[layer_ordinal, int(position) % H_MAX]
        slot.fill(False)
        slot[request] = True

    def capped_distances(self, layer_ordinal: int, position: int) -> np.ndarray:
        """Capped next-use distances for the decision made ``H_MAX`` events ago.

        ``position`` is the current same-layer event index; the returned vector is
        ``d_tilde`` for decision position ``position - H_MAX`` and uses only the
        requests observed at positions ``position - H_MAX + 1 .. position``.
        """

        if position < H_MAX:
            raise ValueError("A capped label needs H_MAX observed follow-up events")
        block = self._window[layer_ordinal][_PHASE_ORDER[int(position) % H_MAX]]
        seen = block.any(axis=0)
        return np.where(seen, block.argmax(axis=0) + 1, NOT_REUSED_WITHIN_HORIZON)


def reference_capped_distances(
    requests: Sequence[Sequence[int]],
    decision_position: int,
    num_experts: int,
) -> np.ndarray:
    """Obviously correct capped-distance reference used only by the test suite."""

    distances = np.full(num_experts, NOT_REUSED_WITHIN_HORIZON, dtype=np.int64)
    for offset in range(1, H_MAX + 1):
        index = decision_position + offset
        if index >= len(requests):
            break
        for expert in requests[index]:
            if distances[int(expert)] == NOT_REUSED_WITHIN_HORIZON:
                distances[int(expert)] = offset
    return distances


def cap(distances: np.ndarray) -> np.ndarray:
    """Apply the Stage 2 cap to true (possibly unbounded) next-use distances."""

    return np.minimum(np.asarray(distances), NOT_REUSED_WITHIN_HORIZON)
