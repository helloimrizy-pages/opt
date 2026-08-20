"""Controlled adaptation tests (Stage 2 specification section 38).

These are algorithm tests, not scientific evaluation evidence.
"""

from __future__ import annotations

import unittest

import numpy as np

from race_stage2.advisers import PRIMARY_POOL, fit_stage2_transition_models
from race_stage2.hedge import HedgeWeights, effective_advisers
from race_stage2.losses import pairwise_losses
from race_stage2.policy import RaceVariant, online_variant, static_variant, uniform_variant
from race_stage2.simulation import simulate_race_variant
from race_stage2.synthetic import (
    build_trace,
    regime_switch_examples,
    ring_request,
    uninformative_examples,
)
from residency_headroom.workloads import make_workload


TOP_K = 2
INDEX = {name: position for position, name in enumerate(PRIMARY_POOL)}
SHORT = ("MARKOV_H1", "MARKOV_H2", "MARKOV_H4")
LONG = ("MARKOV_H8", "MARKOV_H16", "MARKOV_H32")


def _split_workloads(trace, calibration_sequences: int = 4):
    identifiers = [int(item["sequence_id"]) for item in trace.metadata["sequences"]]
    calibration = make_workload(
        "cal",
        "calibration",
        [(item, 0, "c", "general") for item in identifiers[:calibration_sequences]],
    )
    evaluation = make_workload(
        "ev",
        "stationary",
        [(item, 0, "e", "general") for item in identifiers[calibration_sequences:]],
    )
    return calibration, evaluation


def _single(name: str) -> RaceVariant:
    weights = np.zeros(len(PRIMARY_POOL))
    weights[INDEX[name]] = 1.0
    return static_variant(weights)


def _run_ring(period: int, capacity: int, *, generation_length: int = 80, eta: float = 1.0):
    trace = build_trace(
        ring_request(period, TOP_K),
        prompts_per_domain=3,
        generation_length=generation_length,
        num_layers=1,
        num_experts=period * TOP_K,
        top_k=TOP_K,
    )
    calibration, evaluation = _split_workloads(trace)
    models = fit_stage2_transition_models(trace, calibration)
    online = simulate_race_variant(
        trace,
        evaluation,
        (capacity,),
        online_variant(loss="rank", eta=eta, initialization="uniform"),
        models,
        enable_diagnostics=False,
    )[0]
    uniform = simulate_race_variant(
        trace, evaluation, (capacity,), uniform_variant(), models, enable_diagnostics=False
    )[0]
    return online, uniform


class SyntheticHorizonTests(unittest.TestCase):
    def test_a_short_horizon_reuse_favours_short_horizon_advisers(self) -> None:
        online, uniform = _run_ring(period=4, capacity=6)
        weights = online.adviser_mean_weights
        short = sum(weights[name] for name in SHORT)
        long = sum(weights[name] for name in LONG)
        self.assertGreater(short, long)
        self.assertGreater(short, 0.75)
        self.assertIn(
            online.learning["best_fixed_adviser_by_rank_loss"], {"MARKOV_H1", "MARKOV_H2"}
        )
        self.assertLess(online.misses, uniform.misses)

    def test_b_long_horizon_reuse_favours_long_horizon_advisers(self) -> None:
        online, uniform = _run_ring(period=40, capacity=32)
        weights = online.adviser_mean_weights
        short = sum(weights[name] for name in SHORT)
        long = sum(weights[name] for name in LONG)
        self.assertGreater(long, short)
        self.assertGreater(weights["MARKOV_H32"], weights["MARKOV_H1"])
        self.assertEqual(online.learning["best_fixed_adviser_by_rank_loss"], "MARKOV_H32")
        losses = np.asarray(online.learning["mean_adviser_rank_loss"])
        markov = [losses[INDEX[f"MARKOV_H{h}"]] for h in (1, 2, 4, 8, 16, 32)]
        self.assertEqual(markov, sorted(markov, reverse=True))
        self.assertLess(online.misses, uniform.misses)

    def test_horizon_preference_moves_with_the_reuse_distance(self) -> None:
        short_run, _ = _run_ring(period=4, capacity=6)
        long_run, _ = _run_ring(period=40, capacity=32)
        self.assertGreater(
            sum(long_run.adviser_mean_weights[name] for name in LONG),
            sum(short_run.adviser_mean_weights[name] for name in LONG),
        )

    def test_recency_advisers_lose_weight_when_recency_is_inverted(self) -> None:
        # In a pure ring the oldest resident is the one reused soonest, so recency
        # signals rank the candidates backwards.
        online, _ = _run_ring(period=40, capacity=32)
        weights = online.adviser_mean_weights
        for name in ("PERSISTENCE", "LFU_DECAY", "GATE_EWMA"):
            self.assertLess(weights[name], 0.05, name)
        losses = np.asarray(online.learning["mean_adviser_rank_loss"])
        self.assertGreater(losses[INDEX["LFU_DECAY"]], 0.9)


class RegimeSwitchTests(unittest.TestCase):
    A_PERIOD = 12
    B_PERIOD = 20
    A_BASE = 0
    B_BASE = A_PERIOD * TOP_K
    NUM_EXPERTS = B_BASE + B_PERIOD * TOP_K
    CAPACITY = 16

    def _switch_trace(self, first: int, second: int):
        def request_for(sequence_id: int, layer: int, position: int) -> list[int]:
            del sequence_id, layer
            if position < first:
                slot = position % self.A_PERIOD
                return [self.A_BASE + slot * TOP_K + index for index in range(TOP_K)]
            slot = (position - first) % self.B_PERIOD
            return [self.B_BASE + slot * TOP_K + index for index in range(TOP_K)]

        trace = build_trace(
            request_for,
            prompts_per_domain=2,
            generation_length=first + second,
            num_layers=1,
            num_experts=self.NUM_EXPERTS,
            top_k=TOP_K,
        )
        identifiers = [int(item["sequence_id"]) for item in trace.metadata["sequences"]]
        calibration = make_workload(
            "cal", "calibration", [(item, 0, "c", "general") for item in identifiers[:4]]
        )
        evaluation = make_workload(
            "ev", "stationary", [(identifiers[4], 0, "e", "general")]
        )
        return trace, calibration, evaluation

    def test_c_online_weighting_follows_a_regime_switch(self) -> None:
        first, second = 64, 900
        trace, calibration, evaluation = self._switch_trace(first, second)
        models = fit_stage2_transition_models(trace, calibration)
        online = simulate_race_variant(
            trace,
            evaluation,
            (self.CAPACITY,),
            online_variant(loss="rank", eta=1.0, initialization="uniform"),
            models,
            enable_diagnostics=False,
            trajectory_stride=16,
        )[0]
        weights = np.asarray([item["weights"] for item in online.trajectory])
        positions = [int(item["same_layer_position"]) for item in online.trajectory]
        early = [k for k, p in enumerate(positions) if 32 <= p < first]
        late = [k for k, p in enumerate(positions) if p >= first + 400]
        short_index, long_index = INDEX["MARKOV_H8"], INDEX["MARKOV_H16"]
        self.assertGreater(weights[early, short_index].mean(), weights[early, long_index].mean())
        self.assertGreater(weights[late, long_index].mean(), weights[late, short_index].mean())
        self.assertGreater(weights[late, long_index].mean(), 0.8)
        # Adaptation beats every fixed weighting, including each regime's own optimum.
        for fixed in (
            uniform_variant(),
            _single("MARKOV_H8"),
            _single("MARKOV_H16"),
        ):
            reference = simulate_race_variant(
                trace, evaluation, (self.CAPACITY,), fixed, models, enable_diagnostics=False
            )[0]
            self.assertLess(online.misses, reference.misses, fixed.name)

    def test_plain_hedge_stops_tracking_after_a_deep_weight_collapse(self) -> None:
        # Characterization of a known limitation of plain multiplicative weights: once
        # one adviser has accumulated a large cumulative loss advantage its competitors
        # hold numerically negligible weight, so a late regime switch is not tracked
        # within a comparable number of rounds. Fixed-share is the standard remedy and
        # is deliberately out of scope for the preregistered Stage 2 algorithm.
        first, second = 400, 900
        trace, calibration, evaluation = self._switch_trace(first, second)
        models = fit_stage2_transition_models(trace, calibration)
        online = simulate_race_variant(
            trace,
            evaluation,
            (self.CAPACITY,),
            online_variant(loss="rank", eta=1.0, initialization="uniform"),
            models,
            enable_diagnostics=False,
            trajectory_stride=16,
        )[0]
        weights = np.asarray([item["weights"] for item in online.trajectory])
        positions = [int(item["same_layer_position"]) for item in online.trajectory]
        late = [k for k, p in enumerate(positions) if p >= first + 400]
        self.assertGreater(
            weights[late, INDEX["MARKOV_H8"]].mean(), weights[late, INDEX["MARKOV_H16"]].mean()
        )
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))


class UninformativeAdviserTests(unittest.TestCase):
    def _apply(self, blocks, targets, *, eta: float, advisers: int) -> HedgeWeights:
        weights = HedgeWeights(1, np.full(advisers, 1.0 / advisers))
        for block, distances in zip(blocks, targets):
            losses = pairwise_losses(block, distances, want_cost=False)
            if losses.usable:
                weights.update(0, losses.rank, eta)
        return weights

    def test_d_no_useful_adviser_produces_no_pathological_weights(self) -> None:
        advisers = len(PRIMARY_POOL)
        blocks, targets = uninformative_examples(rounds=1500, advisers=advisers)
        losses = np.zeros(advisers)
        used = 0
        for block, distances in zip(blocks, targets):
            value = pairwise_losses(block, distances, want_cost=False)
            if value.usable:
                losses += value.rank
                used += 1
        mean_loss = losses / used
        # Every adviser is equally worthless, so the loss spread is tiny and any
        # weight movement is bounded random drift rather than learned structure.
        self.assertLess(float(mean_loss.max() - mean_loss.min()), 0.05)
        for eta in (0.1, 1.0):
            weights = self._apply(blocks, targets, eta=eta, advisers=advisers)
            weights.validate()
            current = weights.weights(0)
            self.assertTrue(np.isfinite(current).all())
            self.assertAlmostEqual(float(current.sum()), 1.0, places=12)
            self.assertGreaterEqual(float(current.min()), 0.0)
        gentle = self._apply(blocks, targets, eta=0.1, advisers=advisers).weights(0)
        self.assertGreater(effective_advisers(gentle), 5.0)

    def test_d_extreme_learning_rate_does_not_explode(self) -> None:
        advisers = len(PRIMARY_POOL)
        blocks, targets = uninformative_examples(rounds=2000, advisers=advisers, seed=99)
        for eta in (0.1, 1.0, 50.0):
            weights = self._apply(blocks, targets, eta=eta, advisers=advisers)
            weights.validate()
            self.assertTrue(np.isfinite(weights.weights(0)).all())

    def test_regime_switch_at_the_loss_level_is_tracked(self) -> None:
        advisers = len(PRIMARY_POOL)
        blocks, targets = regime_switch_examples(
            rounds_per_regime=120, advisers=advisers, first_good=1, second_good=4
        )
        weights = HedgeWeights(1, np.full(advisers, 1.0 / advisers))
        halfway = None
        for index, (block, distances) in enumerate(zip(blocks, targets)):
            losses = pairwise_losses(block, distances, want_cost=False)
            if losses.usable:
                weights.update(0, losses.rank, 0.3)
            if index == 119:
                halfway = weights.weights(0).copy()
        final = weights.weights(0)
        self.assertGreater(float(halfway[1]), float(halfway[4]))
        self.assertGreater(float(final[4]), float(final[1]))
        weights.validate()


if __name__ == "__main__":
    unittest.main()
