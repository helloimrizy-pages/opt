"""Delayed multiplicative-weights (Hedge) adviser adaptation.

The update is exactly ``w_j <- w_j * exp(-eta * loss_j)`` followed by simplex
renormalization. It is carried in natural-logarithm space with per-step maximum
subtraction, which is algebraically identical and cannot overflow or underflow the
internal state for bounded losses.
"""

from __future__ import annotations

import numpy as np


class HedgeWeights:
    """One weight simplex per independent learning stream."""

    def __init__(self, streams: int, initial: np.ndarray) -> None:
        start = np.asarray(initial, dtype=np.float64)
        if streams < 1:
            raise ValueError("At least one learning stream is required")
        if start.ndim == 1:
            start = np.tile(start, (int(streams), 1))
        elif start.ndim != 2 or start.shape[0] != int(streams):
            raise ValueError("Initial adviser weights must be [adviser] or [stream, adviser]")
        else:
            start = start.copy()
        if not np.isfinite(start).all() or np.any(start < 0):
            raise ValueError("Initial adviser weights must be finite and nonnegative")
        totals = start.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Initial adviser weights must have positive mass")
        start /= totals[:, None]
        self.streams = int(streams)
        self.size = int(start.shape[1])
        with np.errstate(divide="ignore"):
            self._log = np.log(start)
        self._weights = start
        self.updates = np.zeros(self.streams, dtype=np.int64)

    def weights(self, stream: int) -> np.ndarray:
        """Current normalized weights; the caller must not mutate the returned row."""

        return self._weights[stream]

    def snapshot(self) -> np.ndarray:
        return self._weights.copy()

    def update(self, stream: int, losses: np.ndarray, eta: float) -> None:
        values = np.asarray(losses, dtype=np.float64)
        if values.shape != (self.size,):
            raise ValueError("Loss vector does not match the adviser count")
        if not np.isfinite(values).all():
            raise ValueError("Adviser losses must be finite")
        if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
            raise ValueError("Adviser losses must lie in [0, 1]")
        row = self._log[stream]
        row -= float(eta) * values
        row -= row.max()
        weights = np.exp(row)
        weights /= weights.sum()
        self._weights[stream] = weights
        self.updates[stream] += 1

    def validate(self, tolerance: float = 1e-9) -> None:
        if not np.isfinite(self._weights).all():
            raise RuntimeError("Adviser weights left the finite range")
        if np.any(self._weights < -tolerance):
            raise RuntimeError("Adviser weights became negative")
        sums = self._weights.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > 1e-9):
            raise RuntimeError("Adviser weights left the probability simplex")


def entropy(weights: np.ndarray) -> float:
    """Shannon entropy in nats, with the standard ``0 log 0 = 0`` convention."""

    values = np.asarray(weights, dtype=np.float64)
    positive = values[values > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def effective_advisers(weights: np.ndarray) -> float:
    """Perplexity ``exp(H(w))``: how many advisers are effectively in play."""

    return float(np.exp(entropy(weights)))
