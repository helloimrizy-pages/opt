"""Deterministic adviser rank normalization and the frozen Stage 1 eviction order."""

from __future__ import annotations

import numpy as np


def midrank_normalize(scores: np.ndarray) -> np.ndarray:
    """Percentile-normalize each adviser row over the shared candidate set.

    ``scores`` has shape ``[adviser, candidate]`` and larger raw values mean
    "retain more strongly". The result lies in ``[0, 1]`` with ``1.0`` for the
    adviser's strongest retention recommendation. Exact ties receive identical
    average ranks, which makes the transform a deterministic function of the
    candidate score multiset and keeps adviser indifference visible to the
    downstream combination and loss.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Adviser scores must have shape [adviser, candidate]")
    if not np.isfinite(values).all():
        raise ValueError("Adviser scores must be finite before normalization")
    advisers, count = values.shape
    if count == 0:
        return np.zeros((advisers, 0), dtype=np.float64)
    if count == 1:
        return np.zeros((advisers, 1), dtype=np.float64)
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    changed = ordered[:, 1:] != ordered[:, :-1]
    positions = np.arange(count, dtype=np.int64)
    first = np.empty((advisers, count), dtype=np.int64)
    first[:, 0] = 0
    np.multiply(changed, positions[1:], out=first[:, 1:])
    np.maximum.accumulate(first, axis=1, out=first)
    last = np.empty((advisers, count), dtype=np.int64)
    last[:, -1] = count - 1
    last[:, :-1] = np.where(changed, positions[:-1], count - 1)
    last = np.minimum.accumulate(last[:, ::-1], axis=1)[:, ::-1]
    midranks = (first + last) * 0.5
    normalized = np.empty((advisers, count), dtype=np.float64)
    np.put_along_axis(normalized, order, midranks, axis=1)
    normalized /= float(count - 1)
    return normalized


def combined_scores(weights: np.ndarray, normalized: np.ndarray) -> np.ndarray:
    """RACE retention score ``S_e = sum_j w_j z_{j,e}``."""

    weight_vector = np.asarray(weights, dtype=np.float64)
    if weight_vector.ndim != 1 or weight_vector.shape[0] != normalized.shape[0]:
        raise ValueError("Weight vector does not match the adviser dimension")
    return weight_vector @ normalized


def retention_order(
    candidates: np.ndarray, scores: np.ndarray, recency: np.ndarray
) -> np.ndarray:
    """Frozen Stage 1 candidate ordering: score desc, LRU recency desc, expert asc.

    Returns positions into ``candidates``; the first ``spare`` entries are retained.
    """

    identifiers = np.asarray(candidates, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    used = np.asarray(recency, dtype=np.int64)
    if identifiers.shape != values.shape or identifiers.shape != used.shape:
        raise ValueError("Candidate, score and recency vectors must align")
    return np.lexsort((identifiers, -used, -values))
