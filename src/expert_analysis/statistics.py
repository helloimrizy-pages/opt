from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

from .metrics import DomainStatistics


DOMAIN_PAIRS = tuple(combinations(("general", "math", "coding", "reasoning"), 2))

RAW_METRIC_ARRAYS = {
    "routing_frequency": "routing_counts",
    "gate_mass": "gate_sums",
    "functional_contribution": "contribution_sums",
    "gradient_attribution": "gradient_sums",
}

NORMALIZED_METRIC_KEYS = {
    "routing_frequency": "normalized_routing",
    "gate_mass": "normalized_gate",
    "functional_contribution": "normalized_contribution",
    "gradient_attribution": "normalized_gradient",
}


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """One-based descending ranks; tied values receive their average rank."""
    values = np.asarray(values, dtype=np.float64)
    return rankdata(-values, method="average")


def importance_percentiles(values: np.ndarray) -> np.ndarray:
    ranks = descending_ranks(values)
    if len(values) <= 1:
        return np.ones_like(ranks)
    return (len(values) - ranks) / (len(values) - 1)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else float("nan")


def safe_kendall(first: np.ndarray, second: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = kendalltau(first, second).statistic
    return float(value) if np.isfinite(value) else float("nan")


def topk_size(num_experts: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError("Top-k fraction must be in (0, 1]")
    return min(num_experts, max(1, int(math.ceil(num_experts * fraction))))


def topk_similarity(
    first: np.ndarray, second: np.ndarray, fraction: float
) -> tuple[int, int, float, float]:
    if len(first) != len(second):
        raise ValueError("Importance vectors must have the same length")
    k = topk_size(len(first), fraction)
    first_top = set(np.argsort(-np.asarray(first), kind="stable")[:k].tolist())
    second_top = set(np.argsort(-np.asarray(second), kind="stable")[:k].tolist())
    intersection = len(first_top & second_top)
    union = len(first_top | second_top)
    return k, intersection, intersection / k, intersection / union


def raw_example_values(statistics: DomainStatistics, metric: str) -> np.ndarray:
    if metric not in RAW_METRIC_ARRAYS:
        raise KeyError(metric)
    value = getattr(statistics, RAW_METRIC_ARRAYS[metric])
    if value is None:
        raise ValueError(f"Metric {metric!r} was not collected")
    return np.asarray(value)


def bootstrap_spearman_pair(
    first: DomainStatistics,
    second: DomainStatistics,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap domains independently and return [replicate, layer] correlations."""
    if replicates < 1:
        return np.empty((0, first.num_layers), dtype=np.float64)
    first_values = raw_example_values(first, metric)
    second_values = raw_example_values(second, metric)
    if first.num_layers != second.num_layers or first.num_experts != second.num_experts:
        raise ValueError("Domain statistic shapes do not match")
    output = np.full((replicates, first.num_layers), np.nan, dtype=np.float64)
    for replicate in range(replicates):
        first_indices = rng.integers(0, first.num_examples, size=first.num_examples)
        second_indices = rng.integers(0, second.num_examples, size=second.num_examples)
        first_sum = first_values[first_indices].sum(axis=0, dtype=np.float64)
        second_sum = second_values[second_indices].sum(axis=0, dtype=np.float64)
        for layer in range(first.num_layers):
            output[replicate, layer] = safe_spearman(first_sum[layer], second_sum[layer])
    return output


def confidence_interval(
    values: Iterable[float] | np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    return (
        float(finite.mean()),
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )


def nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")
