from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np

from .trace import RoutingTrace
from .workloads import Workload


def workload_diagnostics(trace: RoutingTrace, workload: Workload) -> dict[str, Any]:
    layers = list(map(int, trace.metadata["layer_indices"]))
    layer_to_ordinal = {layer: index for index, layer in enumerate(layers)}
    requests: list[list[np.ndarray]] = [[] for _ in layers]
    segments: dict[int, np.ndarray] = {}
    domains: dict[str, np.ndarray] = {}
    for sequence, view in workload.iter_slices(trace):
        segment_counts = segments.setdefault(
            sequence.segment_index,
            np.zeros((trace.num_layers, trace.num_experts), dtype=np.int64),
        )
        domain_counts = domains.setdefault(
            sequence.domain,
            np.zeros((trace.num_layers, trace.num_experts), dtype=np.int64),
        )
        for index in range(view.start, view.stop):
            ordinal = layer_to_ordinal[int(trace.layer_index[index])]
            request = trace.requested_expert_ids[index].astype(np.int64, copy=False)
            requests[ordinal].append(request)
            np.add.at(segment_counts[ordinal], request, 1)
            np.add.at(domain_counts[ordinal], request, 1)

    entropy_values: list[float] = []
    top10_values: list[float] = []
    top20_values: list[float] = []
    gini_values: list[float] = []
    jaccard_values: list[float] = []
    reuse_distances: list[int] = []
    autocorrelation_values: list[float] = []
    for layer_requests in requests:
        matrix = np.asarray(layer_requests, dtype=np.int64)
        counts = np.bincount(matrix.reshape(-1), minlength=trace.num_experts).astype(np.float64)
        probabilities = counts / counts.sum()
        positive = probabilities[probabilities > 0]
        entropy = float(-(positive * np.log2(positive)).sum() / np.log2(trace.num_experts))
        entropy_values.append(entropy)
        top10_values.append(float(np.sort(probabilities)[-10:].sum()))
        top20_values.append(float(np.sort(probabilities)[-20:].sum()))
        gini_values.append(_gini(counts))
        for first, second in zip(matrix, matrix[1:]):
            intersection = len(set(map(int, first)) & set(map(int, second)))
            union = len(set(map(int, first)) | set(map(int, second)))
            jaccard_values.append(intersection / union)
        last_seen = np.full(trace.num_experts, -1, dtype=np.int64)
        activity = np.zeros((len(matrix), trace.num_experts), dtype=np.int8)
        for position, request in enumerate(matrix):
            activity[position, request] = 1
            for expert in request:
                previous = int(last_seen[expert])
                if previous >= 0:
                    reuse_distances.append(position - previous)
                last_seen[expert] = position
        if len(activity) > 1:
            first = activity[:-1].reshape(-1).astype(np.float64)
            second = activity[1:].reshape(-1).astype(np.float64)
            if first.std() > 0 and second.std() > 0:
                autocorrelation_values.append(float(np.corrcoef(first, second)[0, 1]))

    adjacent_js: list[float] = []
    segment_ids = sorted(segments)
    for first, second in zip(segment_ids, segment_ids[1:]):
        adjacent_js.extend(_layer_js(segments[first], segments[second]))
    domain_js: list[float] = []
    for first, second in combinations(sorted(domains), 2):
        domain_js.extend(_layer_js(domains[first], domains[second]))
    distances = np.asarray(reuse_distances, dtype=np.float64)
    return {
        "workload": workload.name,
        "regime": workload.regime,
        "sequences": len(workload.sequences),
        "domains": "+".join(workload.domains),
        "events": int(sum(len(items) for items in requests)),
        "normalized_frequency_entropy_mean": float(np.mean(entropy_values)),
        "normalized_frequency_entropy_min": float(np.min(entropy_values)),
        "top_10_traffic_share_mean": float(np.mean(top10_values)),
        "top_20_traffic_share_mean": float(np.mean(top20_values)),
        "gini_mean": float(np.mean(gini_values)),
        "consecutive_jaccard_mean": _safe_mean(jaccard_values),
        "activity_lag1_autocorrelation_mean": _safe_mean(autocorrelation_values),
        "reuse_distance_mean_events": _safe_mean(distances),
        "reuse_distance_median_events": _safe_median(distances),
        "reuse_within_10_events": (
            float(np.mean(distances <= 10)) if distances.size else float("nan")
        ),
        "reuse_within_100_events": (
            float(np.mean(distances <= 100)) if distances.size else float("nan")
        ),
        "adjacent_segment_js_mean": _safe_mean(adjacent_js),
        "adjacent_segment_js_max": _safe_max(adjacent_js),
        "pairwise_domain_js_mean": _safe_mean(domain_js),
        "pairwise_domain_js_max": _safe_max(domain_js),
    }


def all_workload_diagnostics(
    trace: RoutingTrace, workloads: Iterable[Workload]
) -> list[dict[str, Any]]:
    return [workload_diagnostics(trace, workload) for workload in workloads]


def _layer_js(first: np.ndarray, second: np.ndarray) -> list[float]:
    values: list[float] = []
    for layer in range(first.shape[0]):
        left = first[layer].astype(np.float64)
        right = second[layer].astype(np.float64)
        if left.sum() == 0 or right.sum() == 0:
            continue
        left /= left.sum()
        right /= right.sum()
        midpoint = 0.5 * (left + right)
        values.append(0.5 * _kl(left, midpoint) + 0.5 * _kl(right, midpoint))
    return values


def _kl(first: np.ndarray, second: np.ndarray) -> float:
    positive = first > 0
    return float(np.sum(first[positive] * np.log2(first[positive] / second[positive])))


def _gini(values: np.ndarray) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0 or array.sum() == 0:
        return 0.0
    indices = np.arange(1, array.size + 1, dtype=np.float64)
    return float((2 * np.sum(indices * array) / (array.size * array.sum())) - (array.size + 1) / array.size)


def _safe_mean(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _safe_median(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _safe_max(values: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.max()) if finite.size else float("nan")
