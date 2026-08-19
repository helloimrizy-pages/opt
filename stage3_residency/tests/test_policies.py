from __future__ import annotations

import unittest

import numpy as np

from residency_headroom.policies import (
    DecayedLFUPolicy,
    LFUPolicy,
    LRUPolicy,
    RandomPolicy,
    StaticHotsetPolicy,
)


class PolicyTests(unittest.TestCase):
    def test_atomic_multi_expert_request_has_no_intra_event_hit(self) -> None:
        policy = LRUPolicy(capacity=2, num_experts=5)
        transition = policy.process([1, 2])
        self.assertEqual(transition.misses, frozenset({1, 2}))
        self.assertEqual(transition.hits, frozenset())
        self.assertEqual(transition.after, frozenset({1, 2}))
        self.assertEqual(transition.admissions, transition.misses)

    def test_lru_atomic_ties_are_deterministic(self) -> None:
        policy = LRUPolicy(capacity=3, num_experts=6)
        policy.process([0, 1])
        second = policy.process([2, 3])
        self.assertEqual(second.after, frozenset({0, 2, 3}))
        third = policy.process([1, 4])
        self.assertEqual(third.after, frozenset({1, 2, 4}))

    def test_lfu_keeps_highest_cumulative_frequency(self) -> None:
        policy = LFUPolicy(capacity=3, num_experts=6)
        policy.process([0, 1])
        policy.process([0, 2])
        transition = policy.process([3, 4])
        self.assertEqual(transition.after, frozenset({0, 3, 4}))

    def test_lfu_decay_adapts_and_is_deterministic(self) -> None:
        first = DecayedLFUPolicy(capacity=3, num_experts=6, alpha=0.5)
        second = DecayedLFUPolicy(capacity=3, num_experts=6, alpha=0.5)
        requests = ([0, 1], [0, 2], [3, 4], [1, 5])
        states_one = [first.process(value).after for value in requests]
        states_two = [second.process(value).after for value in requests]
        self.assertEqual(states_one, states_two)
        self.assertTrue(np.all(first.frequency >= 0))

    def test_static_hotset_uses_calibration_scores_only_for_old_candidates(self) -> None:
        policy = StaticHotsetPolicy(3, 6, np.asarray([9, 8, 1, 0, 0, 0]))
        policy.process([0, 2])
        transition = policy.process([3, 4])
        self.assertEqual(transition.after, frozenset({0, 3, 4}))
        self.assertNotIn(1, transition.after, "Static policy must not prefetch unseen expert 1")

    def test_capacity_invariant_and_invalid_small_positive_capacity(self) -> None:
        policy = LRUPolicy(capacity=2, num_experts=5)
        with self.assertRaises(ValueError):
            policy.process([0, 1, 2])
        zero = LRUPolicy(capacity=0, num_experts=5).process([0, 1, 2])
        self.assertEqual(zero.after, frozenset())
        self.assertEqual(len(zero.misses), 3)

    def test_random_policy_reproducible_by_seed_and_layer(self) -> None:
        requests = ([0, 1], [2, 3], [4, 5], [0, 2], [1, 4])
        first = RandomPolicy(4, 6, seed=77, layer=3)
        second = RandomPolicy(4, 6, seed=77, layer=3)
        self.assertEqual(
            [first.process(value).after for value in requests],
            [second.process(value).after for value in requests],
        )


if __name__ == "__main__":
    unittest.main()
