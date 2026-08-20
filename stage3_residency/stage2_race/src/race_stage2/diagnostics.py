"""Offline ranking diagnostics.

`FutureUseObserver` holds the complete future of a frozen workload path. It is a
strict one-way sink: the simulator pushes copies of what it already decided and the
observer returns nothing that can re-enter policy state. That separation is what
keeps the causal Stage 2 decision path auditable while still allowing
oracle-referenced ranking measurements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import Workload

from . import NOT_REUSED_WITHIN_HORIZON
from .advisers import PRIMARY_POOL
from .hedge import effective_advisers, entropy
from .losses import combined_ranking_stats


class FutureOccurrences:
    """Per-layer next-use lookup built once from the frozen workload path."""

    def __init__(self, streams: Sequence[np.ndarray], num_experts: int) -> None:
        self.num_experts = int(num_experts)
        self._flat: list[np.ndarray] = []
        self._offsets: list[np.ndarray] = []
        self.sentinel: list[int] = []
        for requests in streams:
            values = np.asarray(requests, dtype=np.int64)
            length = int(values.shape[0])
            sentinel = length + 1
            experts = values.ravel()
            positions = np.repeat(np.arange(length, dtype=np.int64), values.shape[1])
            order = np.lexsort((positions, experts))
            sorted_experts = experts[order]
            sorted_positions = positions[order]
            counts = np.bincount(sorted_experts, minlength=self.num_experts).astype(np.int64)
            offsets = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(counts + 1))
            )
            flat = np.full(int(offsets[-1]), sentinel, dtype=np.int64)
            within = np.arange(sorted_positions.size, dtype=np.int64) - np.repeat(
                np.cumsum(counts) - counts, counts
            )
            flat[np.repeat(offsets[: self.num_experts], counts) + within] = sorted_positions
            self._flat.append(flat)
            self._offsets.append(offsets)
            self.sentinel.append(sentinel)
        self._cursor = [offsets[: self.num_experts].copy() for offsets in self._offsets]

    def reset(self) -> None:
        self._cursor = [offsets[: self.num_experts].copy() for offsets in self._offsets]

    def advance(self, layer_ordinal: int, position: int, request: np.ndarray) -> np.ndarray:
        """Consume the current event and return next-use distances for every expert."""

        flat = self._flat[layer_ordinal]
        cursor = self._cursor[layer_ordinal]
        if not np.all(flat[cursor[request]] == position):
            raise RuntimeError("Observer future-use accounting diverged from the trace")
        cursor[request] += 1
        return flat[cursor] - position

    def finish(self) -> None:
        for layer, cursor in enumerate(self._cursor):
            flat = self._flat[layer]
            if not np.all(flat[cursor] == self.sentinel[layer]):
                raise RuntimeError("Observer did not consume the complete same-layer future")


def build_future_occurrences(
    trace: RoutingTrace, workload: Workload, layer_streams: Sequence[np.ndarray]
) -> FutureOccurrences:
    return FutureOccurrences(
        [trace.requested_expert_ids[indices] for indices in layer_streams],
        trace.num_experts,
    )


@dataclass
class RankingAccumulator:
    """Aggregated ranking quality for one (variant, capacity, workload) condition."""

    eviction_events: int = 0
    events_without_eviction: int = 0
    comparable_capped: int = 0
    concordant_capped: float = 0.0
    discordant_capped: float = 0.0
    tied_capped: float = 0.0
    comparable_true: int = 0
    concordant_true: float = 0.0
    discordant_true: float = 0.0
    tied_true: float = 0.0
    oracle_consistent: int = 0
    oracle_optimal: int = 0
    evicted_experts: int = 0
    candidate_experts: int = 0

    def record(
        self,
        *,
        combined: np.ndarray,
        true_distance: np.ndarray,
        order: np.ndarray,
        spare: int,
    ) -> None:
        self.eviction_events += 1
        self.candidate_experts += int(true_distance.size)
        capped = np.minimum(true_distance, NOT_REUSED_WITHIN_HORIZON)
        stats = combined_ranking_stats(combined, capped)
        self.comparable_capped += stats.comparable
        self.concordant_capped += stats.concordant
        self.discordant_capped += stats.discordant
        self.tied_capped += stats.tied
        stats_true = combined_ranking_stats(combined, true_distance)
        self.comparable_true += stats_true.comparable
        self.concordant_true += stats_true.concordant
        self.discordant_true += stats_true.discordant
        self.tied_true += stats_true.tied
        evicted = true_distance[order[spare:]]
        self.evicted_experts += int(evicted.size)
        if evicted.size:
            ordered = np.sort(true_distance)
            if float(evicted.max()) == float(ordered[-1]):
                self.oracle_consistent += 1
            if np.array_equal(np.sort(evicted), ordered[-evicted.size :]):
                self.oracle_optimal += 1

    def as_dict(self) -> dict[str, Any]:
        def ratio(numerator: float, denominator: float) -> float | None:
            return float(numerator / denominator) if denominator else None

        capped_pairs = float(self.comparable_capped)
        true_pairs = float(self.comparable_true)
        return {
            "eviction_events": self.eviction_events,
            "events_without_eviction": self.events_without_eviction,
            "candidate_experts": self.candidate_experts,
            "evicted_experts": self.evicted_experts,
            "comparable_pairs_capped": self.comparable_capped,
            "concordant_capped": self.concordant_capped,
            "discordant_capped": self.discordant_capped,
            "tied_capped": self.tied_capped,
            "concordant_true": self.concordant_true,
            "discordant_true": self.discordant_true,
            "tied_true": self.tied_true,
            "oracle_consistent_events": self.oracle_consistent,
            "oracle_optimal_events": self.oracle_optimal,
            "pairwise_ordering_accuracy_capped": ratio(
                self.concordant_capped + 0.5 * self.tied_capped, capped_pairs
            ),
            "pairwise_concordance_capped": ratio(
                self.concordant_capped - self.discordant_capped,
                self.concordant_capped + self.discordant_capped,
            ),
            "comparable_pairs_true": self.comparable_true,
            "pairwise_ordering_accuracy_true": ratio(
                self.concordant_true + 0.5 * self.tied_true, true_pairs
            ),
            "pairwise_concordance_true": ratio(
                self.concordant_true - self.discordant_true,
                self.concordant_true + self.discordant_true,
            ),
            "oracle_consistent_eviction_rate": ratio(
                self.oracle_consistent, float(self.eviction_events)
            ),
            "oracle_optimal_eviction_rate": ratio(
                self.oracle_optimal, float(self.eviction_events)
            ),
        }


@dataclass
class LearningAccumulator:
    """Delayed-feedback bookkeeping for one (variant, capacity, workload) condition."""

    names: tuple[str, ...] = PRIMARY_POOL
    generated: int = 0
    resolved: int = 0
    skipped_no_comparable_pair: int = 0
    applied_updates: int = 0
    unresolved_at_stream_end: int = 0
    delay_sum: int = 0
    delay_max: int = 0
    minimum_update_offset: int | None = None
    updates_by_layer: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    rank_loss_sum: np.ndarray | None = None
    cost_loss_sum: np.ndarray | None = None
    mixture_rank_loss_sum: float = 0.0
    mixture_cost_loss_sum: float = 0.0
    loss_examples: int = 0
    label_cross_checks: int = 0
    causality_samples: list[dict[str, int]] = field(default_factory=list)
    causality_sample_limit: int = 256

    def __post_init__(self) -> None:
        if self.rank_loss_sum is None:
            self.rank_loss_sum = np.zeros(len(self.names), dtype=np.float64)
        if self.cost_loss_sum is None:
            self.cost_loss_sum = np.zeros(len(self.names), dtype=np.float64)

    def note_resolution(
        self,
        *,
        layer: int,
        decision_position: int,
        resolution_position: int,
        update_position: int,
        usable: bool,
        applied: bool,
    ) -> None:
        self.resolved += 1
        delay = int(update_position) - int(decision_position)
        self.delay_sum += delay
        self.delay_max = max(self.delay_max, delay)
        if self.minimum_update_offset is None or delay < self.minimum_update_offset:
            self.minimum_update_offset = delay
        if not usable:
            self.skipped_no_comparable_pair += 1
        if applied:
            self.applied_updates += 1
            self.updates_by_layer[int(layer)] += 1
        if len(self.causality_samples) < self.causality_sample_limit:
            self.causality_samples.append(
                {
                    "layer": int(layer),
                    "decision_event_index": int(decision_position),
                    "label_resolution_event_index": int(resolution_position),
                    "weight_update_event_index": int(update_position),
                    "usable": bool(usable),
                    "applied": bool(applied),
                }
            )

    def note_losses(
        self,
        *,
        rank: np.ndarray | None,
        cost: np.ndarray | None,
        weights: np.ndarray,
    ) -> None:
        self.loss_examples += 1
        if rank is not None:
            self.rank_loss_sum += rank
            self.mixture_rank_loss_sum += float(weights @ rank)
        if cost is not None:
            self.cost_loss_sum += cost
            self.mixture_cost_loss_sum += float(weights @ cost)

    def as_dict(self) -> dict[str, Any]:
        examples = float(self.loss_examples)
        rank_totals = self.rank_loss_sum.tolist()
        cost_totals = self.cost_loss_sum.tolist()
        best_rank = float(self.rank_loss_sum.min()) if examples else 0.0
        best_cost = float(self.cost_loss_sum.min()) if examples else 0.0
        return {
            "examples_generated": self.generated,
            "examples_resolved": self.resolved,
            "examples_unresolved_at_stream_end": self.unresolved_at_stream_end,
            "unresolved_fraction": (
                float(self.unresolved_at_stream_end / self.generated) if self.generated else 0.0
            ),
            "examples_skipped_no_comparable_pair": self.skipped_no_comparable_pair,
            "applied_updates": self.applied_updates,
            "label_cross_checks_passed": self.label_cross_checks,
            "average_feedback_delay_same_layer_events": (
                float(self.delay_sum / self.resolved) if self.resolved else None
            ),
            "maximum_feedback_delay_same_layer_events": self.delay_max,
            "minimum_update_minus_decision_offset": self.minimum_update_offset,
            "updates_by_layer": {str(k): int(v) for k, v in sorted(self.updates_by_layer.items())},
            "cumulative_adviser_rank_loss": rank_totals,
            "cumulative_adviser_cost_loss": cost_totals,
            "cumulative_mixture_rank_loss": float(self.mixture_rank_loss_sum),
            "cumulative_mixture_cost_loss": float(self.mixture_cost_loss_sum),
            "best_fixed_adviser_rank_loss": best_rank,
            "best_fixed_adviser_cost_loss": best_cost,
            "empirical_rank_regret": float(self.mixture_rank_loss_sum - best_rank),
            "empirical_cost_regret": float(self.mixture_cost_loss_sum - best_cost),
            "adviser_order": list(self.names),
            "best_fixed_adviser_by_rank_loss": (
                self.names[int(np.argmin(self.rank_loss_sum))] if examples else None
            ),
            "best_fixed_adviser_by_cost_loss": (
                self.names[int(np.argmin(self.cost_loss_sum))] if examples else None
            ),
            "mean_adviser_rank_loss": (
                (self.rank_loss_sum / examples).tolist() if examples else None
            ),
            "mean_adviser_cost_loss": (
                (self.cost_loss_sum / examples).tolist() if examples else None
            ),
            "causality_samples": self.causality_samples,
        }


def weight_summary(
    *,
    initial: np.ndarray,
    final: np.ndarray,
    deployed_sum: np.ndarray,
    deployed_square_sum: np.ndarray,
    deployed_count: np.ndarray,
    names: Sequence[str] = PRIMARY_POOL,
) -> list[dict[str, Any]]:
    """Per-layer starting, ending, mean, variance and concentration statistics."""

    rows: list[dict[str, Any]] = []
    for layer in range(initial.shape[0]):
        count = float(deployed_count[layer])
        if count > 0:
            mean = deployed_sum[layer] / count
            second = deployed_square_sum[layer] / count
            variance = np.maximum(second - mean * mean, 0.0)
        else:
            mean = final[layer].copy()
            variance = np.zeros_like(mean)
        rows.append(
            {
                "layer": int(layer),
                "decisions": int(count),
                "start_weights": initial[layer].tolist(),
                "end_weights": final[layer].tolist(),
                "mean_weights": mean.tolist(),
                "weight_variance": variance.tolist(),
                "dominant_adviser": names[int(np.argmax(mean))],
                "dominant_adviser_mean_weight": float(mean.max()),
                "end_entropy_nats": entropy(final[layer]),
                "end_effective_advisers": effective_advisers(final[layer]),
                "mean_entropy_nats": entropy(mean),
                "mean_effective_advisers": effective_advisers(mean),
            }
        )
    return rows


def horizon_weight_summary(
    rows: Sequence[Mapping[str, Any]], names: Sequence[str] = PRIMARY_POOL
) -> dict[str, float]:
    """Suite-level mean weight per adviser across layers."""

    if not rows:
        return {}
    stacked = np.asarray([row["mean_weights"] for row in rows], dtype=np.float64)
    counts = np.asarray([max(int(row["decisions"]), 0) for row in rows], dtype=np.float64)
    if counts.sum() > 0:
        mean = (stacked * counts[:, None]).sum(axis=0) / counts.sum()
    else:
        mean = stacked.mean(axis=0)
    return {name: float(value) for name, value in zip(names, mean)}
