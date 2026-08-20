from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload

from .models import TransitionModels, same_layer_indices
from .policy import make_predictor
from .simulation import method_id


def evaluate_prediction_quality(
    trace: RoutingTrace,
    workload: Workload,
    spec: Mapping[str, Any],
    models: TransitionModels,
    *,
    ranking_cutoffs: Sequence[int] = (8, 12, 16),
    additional_window_horizon: int | None = None,
) -> dict[str, Any]:
    cutoffs = tuple(map(int, ranking_cutoffs))
    if any(value < 1 or value > trace.num_experts for value in cutoffs):
        raise ValueError("Prediction-quality ranking cutoff is invalid")
    layer_indices = same_layer_indices(trace, workload)
    recall_sums = np.zeros(len(cutoffs), dtype=np.float64)
    precision_sums = np.zeros(len(cutoffs), dtype=np.float64)
    average_precision_sum = 0.0
    brier_sum = 0.0
    events = 0
    window_recall_sum = 0.0
    window_ap_sum = 0.0
    window_brier_sum = 0.0
    window_events = 0
    expert_ids = np.arange(trace.num_experts, dtype=np.int64)

    for ordinal, indices in enumerate(layer_indices):
        predictor = make_predictor(
            spec,
            num_experts=trace.num_experts,
            layer_ordinal=ordinal,
            models=models,
        )
        requests = np.asarray(trace.requested_expert_ids[indices], dtype=np.int64)
        gates = np.asarray(trace.router_weights[indices], dtype=np.float64)
        for position in range(requests.shape[0]):
            current = frozenset(map(int, requests[position]))
            gate_map = {
                int(expert): float(weight)
                for expert, weight in zip(requests[position], gates[position])
            }
            scores = np.asarray(predictor.step(current, gate_map), dtype=np.float64)
            if position + 1 >= requests.shape[0]:
                continue
            target = np.zeros(trace.num_experts, dtype=np.float64)
            target[requests[position + 1]] = 1.0
            order = np.lexsort((expert_ids, -scores))
            for cutoff_index, cutoff in enumerate(cutoffs):
                found = float(target[order[:cutoff]].sum())
                recall_sums[cutoff_index] += found / float(target.sum())
                precision_sums[cutoff_index] += found / cutoff
            average_precision_sum += _average_precision(order, target)
            brier_sum += float(np.square(np.clip(scores, 0.0, 1.0) - target).mean())
            events += 1

            if additional_window_horizon is not None:
                stop = min(requests.shape[0], position + additional_window_horizon + 1)
                window_target = np.zeros(trace.num_experts, dtype=np.float64)
                window_target[np.unique(requests[position + 1 : stop])] = 1.0
                if window_target.sum() > 0:
                    window_recall_sum += float(
                        window_target[order[: min(16, trace.num_experts)]].sum()
                        / window_target.sum()
                    )
                    window_ap_sum += _average_precision(order, window_target)
                    window_brier_sum += float(
                        np.square(np.clip(scores, 0.0, 1.0) - window_target).mean()
                    )
                    window_events += 1
    if events == 0:
        raise ValueError("Prediction-quality workload has no next-event targets")
    record: dict[str, Any] = {
        "schema_version": "race_stage1_prediction_quality_v1",
        "method": str(spec["method"]),
        "method_id": method_id(spec),
        "parameters": {key: value for key, value in spec.items() if key != "method"},
        "workload": workload.name,
        "regime": workload.regime,
        "target": "next_same_layer_atomic_request",
        "events": events,
        "average_precision": average_precision_sum / events,
        "brier_score": brier_sum / events,
    }
    for index, cutoff in enumerate(cutoffs):
        record[f"recall_at_{cutoff}"] = recall_sums[index] / events
        record[f"precision_at_{cutoff}"] = precision_sums[index] / events
    if additional_window_horizon is not None:
        record.update(
            {
                "additional_window_horizon": int(additional_window_horizon),
                "window_events": window_events,
                "window_recall_at_16": window_recall_sum / window_events,
                "window_average_precision": window_ap_sum / window_events,
                "window_brier_score": window_brier_sum / window_events,
            }
        )
    return record


def _average_precision(order: np.ndarray, target: np.ndarray) -> float:
    ordered = target[order]
    positives = float(ordered.sum())
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(ordered)
    ranks = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(((cumulative / ranks) * ordered).sum() / positives)
