"""Bounded pairwise future-use ranking losses for the delayed adviser update."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairwiseLosses:
    """Per-adviser losses for one resolved learning example."""

    rank: np.ndarray | None
    cost: np.ndarray | None
    comparable_pairs: int
    cost_weight_sum: float

    @property
    def usable(self) -> bool:
        return self.comparable_pairs > 0


def pairwise_losses(
    normalized: np.ndarray,
    distances: np.ndarray,
    *,
    want_rank: bool = True,
    want_cost: bool = True,
) -> PairwiseLosses:
    """Unweighted and cost-sensitive pairwise inversion losses in ``[0, 1]``.

    A candidate pair ``(a, b)`` is comparable when ``d_tilde(a) < d_tilde(b)``; the
    correct ranking then places ``a`` above ``b``. ``I_j(a, b)`` is ``0`` when
    adviser ``j`` ranks ``a`` strictly above ``b``, ``1`` when it ranks ``b``
    strictly above ``a`` and ``0.5`` on an exact adviser-score tie.
    """

    scores = np.asarray(normalized, dtype=np.float64)
    values = np.asarray(distances, dtype=np.float64)
    if scores.ndim != 2 or values.ndim != 1 or scores.shape[1] != values.shape[0]:
        raise ValueError("Adviser scores and capped distances must align")
    if np.any(values <= 0):
        raise ValueError("Capped next-use distances must be positive")
    better = values[:, None] < values[None, :]
    comparable = int(better.sum())
    if comparable == 0:
        return PairwiseLosses(None, None, 0, 0.0)
    difference = scores[:, :, None] - scores[:, None, :]
    inverted = (difference < 0.0) & better
    tied = (difference == 0.0) & better
    rank_loss = None
    if want_rank:
        rank_loss = (
            inverted.sum(axis=(1, 2), dtype=np.float64)
            + 0.5 * tied.sum(axis=(1, 2), dtype=np.float64)
        ) / float(comparable)
    cost_loss = None
    weight_sum = 0.0
    if want_cost:
        potential = 1.0 / values
        magnitude = np.abs(potential[:, None] - potential[None, :])
        magnitude *= better
        weight_sum = float(magnitude.sum())
        if weight_sum > 0.0:
            cost_loss = (
                (inverted * magnitude).sum(axis=(1, 2))
                + 0.5 * (tied * magnitude).sum(axis=(1, 2))
            ) / weight_sum
    return PairwiseLosses(rank_loss, cost_loss, comparable, weight_sum)


@dataclass(frozen=True)
class RankingPairStats:
    comparable: int
    concordant: float
    discordant: float
    tied: float


def combined_ranking_stats(
    scores: np.ndarray, distances: np.ndarray
) -> RankingPairStats:
    """Concordance of one combined score vector against a next-use distance vector."""

    combined = np.asarray(scores, dtype=np.float64)
    values = np.asarray(distances, dtype=np.float64)
    if combined.shape != values.shape:
        raise ValueError("Combined scores and distances must align")
    better = values[:, None] < values[None, :]
    comparable = int(better.sum())
    if comparable == 0:
        return RankingPairStats(0, 0.0, 0.0, 0.0)
    difference = combined[:, None] - combined[None, :]
    concordant = float(((difference > 0.0) & better).sum())
    discordant = float(((difference < 0.0) & better).sum())
    tied = float(((difference == 0.0) & better).sum())
    return RankingPairStats(comparable, concordant, discordant, tied)
