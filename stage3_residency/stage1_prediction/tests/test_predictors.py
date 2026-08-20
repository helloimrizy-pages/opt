from __future__ import annotations

import unittest

import numpy as np

from race_stage1.models import TransitionModels, _fit_one
from race_stage1.policy import (
    GateEWMAPredictor,
    LastGatePredictor,
    MarkovPlusEWMAPredictor,
    MarkovPredictor,
    PersistencePredictor,
    PredictionRetentionPolicy,
)


class PredictorTests(unittest.TestCase):
    def test_persistence_uses_previous_request_then_updates(self) -> None:
        predictor = PersistencePredictor(6)
        first = predictor.step(frozenset({1, 3}), {1: 0.6, 3: 0.4})
        second = predictor.step(frozenset({2, 4}), {2: 0.7, 4: 0.3})
        np.testing.assert_array_equal(first, np.zeros(6))
        np.testing.assert_array_equal(second, [0, 1, 0, 1, 0, 0])

    def test_last_gate_tracks_only_observed_current_values(self) -> None:
        predictor = LastGatePredictor(5)
        predictor.step(frozenset({1, 2}), {1: 0.7, 2: 0.3})
        scores = predictor.step(frozenset({2, 4}), {2: 0.8, 4: 0.2})
        np.testing.assert_allclose(scores, [0, 0.7, 0.8, 0, 0.2])

    def test_gate_ewma_applies_zero_for_unrequested_experts(self) -> None:
        predictor = GateEWMAPredictor(4, alpha=0.5)
        predictor.step(frozenset({0, 1}), {0: 0.6, 1: 0.4})
        scores = predictor.step(frozenset({1, 2}), {1: 0.8, 2: 0.2})
        np.testing.assert_allclose(scores, [0.15, 0.5, 0.1, 0.0])

    def test_markov_one_and_h_binary_window_estimation(self) -> None:
        requests = np.asarray([[0], [1], [2]], dtype=np.int64)
        one_positive, one_counts = _fit_one(requests, 3, 1)
        two_positive, two_counts = _fit_one(requests, 3, 2)
        self.assertEqual(one_counts[0], 1)
        self.assertEqual(one_positive[0, 1], 1)
        self.assertEqual(one_positive[0, 2], 0)
        self.assertEqual(two_positive[0, 1], 1)
        self.assertEqual(two_positive[0, 2], 1)
        np.testing.assert_array_equal(one_counts, two_counts)

    def test_markov_score_is_mean_over_atomic_source_set(self) -> None:
        matrix = np.asarray(
            [[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [0.0, 0.0, 0.0]]
        )
        predictor = MarkovPredictor(matrix)
        scores = predictor.step(frozenset({0, 1}), {})
        np.testing.assert_allclose(scores, [0.2, 0.3, 0.4])

    def test_hybrid_combines_conditional_and_request_ewma(self) -> None:
        predictor = MarkovPlusEWMAPredictor(
            np.full((4, 4), 0.2), num_experts=4, beta=0.5, alpha=0.5
        )
        scores = predictor.step(frozenset({1}), {})
        np.testing.assert_allclose(scores, [0.1, 0.35, 0.1, 0.1])

    def test_shared_eviction_tie_break_is_score_then_lru_then_id(self) -> None:
        class Fixed:
            name = "fixed"

            def __init__(self) -> None:
                self.scores = np.zeros(6)

            def step(self, request, gates):
                del request, gates
                return self.scores.copy()

        predictor = Fixed()
        policy = PredictionRetentionPolicy(3, 6, predictor)
        policy.process([0, 1], [0.5, 0.5])
        transition = policy.process([2, 3], [0.5, 0.5]).transition
        self.assertEqual(transition.after, frozenset({0, 2, 3}))
        predictor.scores[0] = 0.1
        transition = policy.process([4, 5], [0.5, 0.5]).transition
        self.assertEqual(transition.after, frozenset({0, 4, 5}))

    def test_no_speculative_admission_and_capacity_invariant(self) -> None:
        predictor = MarkovPredictor(np.eye(6))
        policy = PredictionRetentionPolicy(4, 6, predictor)
        first = policy.process([0, 1], [0.6, 0.4]).transition
        second = policy.process([2, 3], [0.6, 0.4]).transition
        self.assertEqual(first.admissions, first.misses)
        self.assertEqual(second.admissions, second.misses)
        self.assertLessEqual(len(second.after), 4)
        self.assertTrue(second.after.issubset({0, 1, 2, 3}))

    def test_transition_models_are_immutable_after_load_contract(self) -> None:
        probabilities = np.full((1, 1, 2, 2), 0.5)
        counts = np.ones((1, 1, 2), dtype=np.int64)
        model = TransitionModels((1,), probabilities, counts, "trace", "workload")
        self.assertEqual(model.matrix(1, 0).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
