from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import rankdata

from race_stage2 import H_MAX, NOT_REUSED_WITHIN_HORIZON
from race_stage2.advisers import (
    EXTENDED_POOL,
    PRIMARY_POOL,
    pool_names,
    pool_size,
    uniform_weights,
    validate_simplex,
)
from race_stage2.hedge import HedgeWeights, effective_advisers, entropy
from race_stage2.labels import (
    LabelWindow,
    PendingExample,
    cap,
    reference_capped_distances,
)
from race_stage2.losses import combined_ranking_stats, pairwise_losses
from race_stage2.ranking import combined_scores, midrank_normalize, retention_order
from race_stage2.static_weights import (
    build_pair_dataset,
    learn_static_weights,
    logistic_objective,
    project_to_simplex,
    zero_one_pairwise_loss,
)


class RankNormalizationTests(unittest.TestCase):
    def test_matches_average_rank_percentiles(self) -> None:
        rng = np.random.default_rng(3)
        for _case in range(200):
            count = int(rng.integers(1, 33))
            block = rng.integers(0, 5, size=(9, count)).astype(np.float64)
            actual = midrank_normalize(block)
            expected = np.stack(
                [
                    (rankdata(row, method="average") - 1.0) / max(count - 1, 1)
                    for row in block
                ]
            )
            self.assertTrue(np.allclose(actual, expected))

    def test_bounds_orientation_and_ties(self) -> None:
        block = np.asarray([[0.1, 0.9, 0.5, 0.5], [2.0, 2.0, 2.0, 2.0]])
        normalized = midrank_normalize(block)
        self.assertTrue(np.all(normalized >= 0.0) and np.all(normalized <= 1.0))
        # Higher raw score means retain more strongly.
        self.assertEqual(float(normalized[0, 1]), 1.0)
        self.assertEqual(float(normalized[0, 0]), 0.0)
        # Exact ties share one midrank.
        self.assertEqual(float(normalized[0, 2]), float(normalized[0, 3]))
        # A fully indifferent adviser contributes the same value everywhere.
        self.assertTrue(np.allclose(normalized[1], 0.5))

    def test_monotone_transform_preserves_order(self) -> None:
        rng = np.random.default_rng(5)
        raw = rng.random((1, 20))
        left = midrank_normalize(raw)
        right = midrank_normalize(np.exp(3.0 * raw))
        self.assertTrue(np.array_equal(np.argsort(left, axis=1), np.argsort(right, axis=1)))

    def test_degenerate_candidate_counts(self) -> None:
        self.assertEqual(midrank_normalize(np.zeros((9, 0))).shape, (9, 0))
        self.assertTrue(np.array_equal(midrank_normalize(np.ones((9, 1))), np.zeros((9, 1))))

    def test_rejects_nonfinite_scores(self) -> None:
        with self.assertRaises(ValueError):
            midrank_normalize(np.asarray([[np.nan, 1.0]]))


class CombinedScoreTests(unittest.TestCase):
    def test_convex_combination(self) -> None:
        normalized = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        weights = np.asarray([0.25, 0.75])
        self.assertTrue(
            np.allclose(combined_scores(weights, normalized), np.asarray([0.25, 0.75]))
        )

    def test_rejects_mismatched_weight_vector(self) -> None:
        with self.assertRaises(ValueError):
            combined_scores(np.asarray([1.0]), np.zeros((3, 4)))

    def test_deterministic_tie_break_is_lru_then_expert_id(self) -> None:
        candidates = np.asarray([7, 3, 11, 5], dtype=np.int64)
        scores = np.asarray([0.5, 0.5, 0.5, 0.9])
        recency = np.asarray([10, 40, 40, 1], dtype=np.int64)
        order = retention_order(candidates, scores, recency)
        # 5 wins on score; then recency 40 (3 before 11 by expert id); then 7.
        self.assertEqual(list(candidates[order]), [5, 3, 11, 7])

    def test_tie_break_is_stable_under_permutation(self) -> None:
        rng = np.random.default_rng(11)
        candidates = np.asarray([2, 9, 4, 6, 1], dtype=np.int64)
        scores = np.asarray([0.3, 0.3, 0.3, 0.3, 0.3])
        recency = np.asarray([5, 5, 5, 5, 5], dtype=np.int64)
        base = list(candidates[retention_order(candidates, scores, recency)])
        for _case in range(20):
            permutation = rng.permutation(candidates.size)
            shuffled = candidates[permutation]
            order = retention_order(shuffled, scores[permutation], recency[permutation])
            self.assertEqual(list(shuffled[order]), base)


class SimplexAndHedgeTests(unittest.TestCase):
    def test_pool_sizes_and_simplex_validation(self) -> None:
        self.assertEqual(pool_size("primary"), 9)
        self.assertEqual(pool_size("extended"), 10)
        self.assertEqual(pool_names("primary"), PRIMARY_POOL)
        self.assertEqual(pool_names("extended"), EXTENDED_POOL)
        validate_simplex(uniform_weights(9), 9)
        with self.assertRaises(ValueError):
            validate_simplex(np.asarray([0.5, 0.6] + [0.0] * 7), 9)
        with self.assertRaises(ValueError):
            validate_simplex(np.asarray([-0.1, 1.1] + [0.0] * 7), 9)

    def test_multiplicative_update_matches_the_definition(self) -> None:
        weights = HedgeWeights(1, uniform_weights(4))
        losses = np.asarray([0.0, 0.25, 0.75, 1.0])
        eta = 0.7
        weights.update(0, losses, eta)
        expected = np.exp(-eta * losses) / np.exp(-eta * losses).sum()
        self.assertTrue(np.allclose(weights.weights(0), expected))
        weights.update(0, losses, eta)
        expected = np.exp(-2 * eta * losses) / np.exp(-2 * eta * losses).sum()
        self.assertTrue(np.allclose(weights.weights(0), expected))
        weights.validate()

    def test_zero_loss_update_leaves_weights_unchanged(self) -> None:
        start = np.asarray([0.1, 0.2, 0.3, 0.4])
        weights = HedgeWeights(2, start)
        weights.update(0, np.zeros(4), 1.0)
        weights.update(1, np.zeros(4), 0.1)
        self.assertTrue(np.allclose(weights.weights(0), start))
        self.assertTrue(np.allclose(weights.weights(1), start))

    def test_equal_losses_leave_weights_unchanged(self) -> None:
        weights = HedgeWeights(1, uniform_weights(5))
        weights.update(0, np.full(5, 0.42), 1.0)
        self.assertTrue(np.allclose(weights.weights(0), uniform_weights(5)))

    def test_extreme_losses_stay_finite_and_on_the_simplex(self) -> None:
        weights = HedgeWeights(1, uniform_weights(9))
        pattern = np.zeros(9)
        pattern[0] = 1.0
        for _step in range(5000):
            weights.update(0, pattern, 1.0)
        current = weights.weights(0)
        weights.validate()
        self.assertTrue(np.isfinite(current).all())
        self.assertAlmostEqual(float(current.sum()), 1.0, places=12)
        self.assertGreaterEqual(float(current.min()), 0.0)
        self.assertAlmostEqual(float(current[0]), 0.0, places=12)
        # The log-space state stays finite, so a collapsed adviser can still recover.
        recovery = np.ones(9)
        recovery[0] = 0.0
        for _step in range(6000):
            weights.update(0, recovery, 1.0)
        self.assertGreater(float(weights.weights(0)[0]), 0.99)

    def test_rejects_out_of_range_losses(self) -> None:
        weights = HedgeWeights(1, uniform_weights(3))
        with self.assertRaises(ValueError):
            weights.update(0, np.asarray([0.0, 0.0, 1.5]), 0.1)
        with self.assertRaises(ValueError):
            weights.update(0, np.asarray([0.0, 0.0, np.nan]), 0.1)

    def test_streams_are_independent(self) -> None:
        weights = HedgeWeights(3, uniform_weights(4))
        weights.update(1, np.asarray([1.0, 0.0, 0.0, 0.0]), 1.0)
        self.assertTrue(np.allclose(weights.weights(0), uniform_weights(4)))
        self.assertTrue(np.allclose(weights.weights(2), uniform_weights(4)))
        self.assertFalse(np.allclose(weights.weights(1), uniform_weights(4)))
        self.assertEqual(list(weights.updates), [0, 1, 0])

    def test_entropy_and_effective_adviser_count(self) -> None:
        self.assertAlmostEqual(entropy(uniform_weights(4)), float(np.log(4)))
        self.assertAlmostEqual(effective_advisers(uniform_weights(4)), 4.0)
        vertex = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(entropy(vertex), 0.0)
        self.assertAlmostEqual(effective_advisers(vertex), 1.0)


class LabelTests(unittest.TestCase):
    def test_capped_distance_matches_the_reference_implementation(self) -> None:
        rng = np.random.default_rng(17)
        num_experts = 24
        requests = [
            sorted(map(int, rng.choice(num_experts, size=4, replace=False)))
            for _position in range(400)
        ]
        window = LabelWindow(1, num_experts)
        for position, request in enumerate(requests):
            window.push(0, position, np.asarray(request, dtype=np.int64))
            if position < H_MAX:
                continue
            actual = window.capped_distances(0, position)
            expected = reference_capped_distances(requests, position - H_MAX, num_experts)
            self.assertTrue(np.array_equal(actual, expected), position)

    def test_encoding_boundaries(self) -> None:
        num_experts = 6
        window = LabelWindow(1, num_experts)
        requests = [[0]] + [[1]] * H_MAX
        for position, request in enumerate(requests):
            window.push(0, position, np.asarray(request, dtype=np.int64))
        distances = window.capped_distances(0, H_MAX)
        self.assertEqual(int(distances[1]), 1)
        self.assertEqual(int(distances[0]), NOT_REUSED_WITHIN_HORIZON)
        self.assertEqual(NOT_REUSED_WITHIN_HORIZON, 33)

    def test_thirty_second_event_is_still_inside_the_window(self) -> None:
        num_experts = 4
        window = LabelWindow(1, num_experts)
        for position in range(H_MAX):
            window.push(0, position, np.asarray([0], dtype=np.int64))
        window.push(0, H_MAX, np.asarray([3], dtype=np.int64))
        distances = window.capped_distances(0, H_MAX)
        self.assertEqual(int(distances[3]), H_MAX)

    def test_label_needs_a_full_window(self) -> None:
        window = LabelWindow(1, 4)
        with self.assertRaises(ValueError):
            window.capped_distances(0, H_MAX - 1)

    def test_cap_helper_and_pending_resolution_offset(self) -> None:
        self.assertTrue(
            np.array_equal(cap(np.asarray([1, 32, 33, 900])), np.asarray([1, 32, 33, 33]))
        )
        example = PendingExample(11, np.asarray([1, 2]), np.zeros((9, 2)))
        self.assertEqual(example.resolution_position, 11 + H_MAX)


class PairwiseLossTests(unittest.TestCase):
    def test_perfect_and_inverted_advisers(self) -> None:
        distances = np.asarray([1, 2, 3, 4])
        perfect = np.asarray([[1.0, 0.75, 0.5, 0.0]])
        inverted = np.asarray([[0.0, 0.5, 0.75, 1.0]])
        indifferent = np.asarray([[0.3, 0.3, 0.3, 0.3]])
        block = np.concatenate([perfect, inverted, indifferent])
        losses = pairwise_losses(block, distances)
        self.assertEqual(losses.comparable_pairs, 6)
        self.assertAlmostEqual(float(losses.rank[0]), 0.0)
        self.assertAlmostEqual(float(losses.rank[1]), 1.0)
        self.assertAlmostEqual(float(losses.rank[2]), 0.5)
        self.assertAlmostEqual(float(losses.cost[0]), 0.0)
        self.assertAlmostEqual(float(losses.cost[1]), 1.0)
        self.assertAlmostEqual(float(losses.cost[2]), 0.5)

    def test_losses_are_bounded(self) -> None:
        rng = np.random.default_rng(23)
        for _case in range(200):
            count = int(rng.integers(2, 20))
            block = rng.random((9, count))
            distances = rng.integers(1, NOT_REUSED_WITHIN_HORIZON + 1, count)
            losses = pairwise_losses(block, distances)
            if not losses.usable:
                continue
            for vector in (losses.rank, losses.cost):
                self.assertIsNotNone(vector)
                self.assertGreaterEqual(float(vector.min()), -1e-12)
                self.assertLessEqual(float(vector.max()), 1.0 + 1e-12)

    def test_cost_weighting_prioritises_near_term_inversions(self) -> None:
        # True retention order is candidate 0 (d=1), 2 (d=10), 3 (d=11), 1 (d=30).
        # Each adviser inverts exactly one adjacent pair, so their unweighted rank
        # losses are identical; only the cost-sensitive loss separates them.
        distances = np.asarray([1, 30, 10, 11])
        near_inverter = np.asarray([2 / 3, 0.0, 1.0, 1 / 3])  # swaps d=1 against d=10
        far_inverter = np.asarray([1.0, 0.0, 1 / 3, 2 / 3])  # swaps d=10 against d=11
        block = np.stack([near_inverter, far_inverter])
        losses = pairwise_losses(block, distances)
        self.assertAlmostEqual(float(losses.rank[0]), 1 / 6)
        self.assertAlmostEqual(float(losses.rank[1]), 1 / 6)
        self.assertGreater(float(losses.cost[0]), 30.0 * float(losses.cost[1]))

    def test_no_comparable_pair_is_skipped(self) -> None:
        losses = pairwise_losses(np.random.default_rng(1).random((9, 5)), np.full(5, 33))
        self.assertFalse(losses.usable)
        self.assertIsNone(losses.rank)
        self.assertIsNone(losses.cost)
        self.assertEqual(losses.comparable_pairs, 0)

    def test_rejects_nonpositive_distances(self) -> None:
        with self.assertRaises(ValueError):
            pairwise_losses(np.zeros((9, 2)), np.asarray([0, 1]))

    def test_combined_ranking_stats(self) -> None:
        distances = np.asarray([1, 2, 3])
        stats = combined_ranking_stats(np.asarray([3.0, 2.0, 1.0]), distances)
        self.assertEqual((stats.comparable, stats.concordant, stats.discordant, stats.tied), (3, 3.0, 0.0, 0.0))
        stats = combined_ranking_stats(np.asarray([1.0, 1.0, 1.0]), distances)
        self.assertEqual((stats.comparable, stats.tied), (3, 3.0))


class StaticWeightTests(unittest.TestCase):
    def test_simplex_projection(self) -> None:
        rng = np.random.default_rng(29)
        for _case in range(200):
            vector = rng.normal(size=9) * 3.0
            projected = project_to_simplex(vector)
            self.assertAlmostEqual(float(projected.sum()), 1.0, places=12)
            self.assertGreaterEqual(float(projected.min()), 0.0)
        already = uniform_weights(9)
        self.assertTrue(np.allclose(project_to_simplex(already), already))

    def test_learns_the_informative_adviser(self) -> None:
        rng = np.random.default_rng(31)
        blocks, targets = [], []
        for _round in range(200):
            count = int(rng.integers(4, 16))
            distances = rng.integers(1, 34, count)
            block = rng.random((9, count))
            block[4] = np.argsort(np.argsort(-distances)) / max(count - 1, 1)
            blocks.append(block)
            targets.append(distances)
        dataset = build_pair_dataset(blocks, targets)
        result = learn_static_weights(dataset)
        weights = np.asarray(result["weights"])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=9)
        self.assertEqual(int(np.argmax(weights)), 4)
        self.assertLess(result["objective"], result["objective_at_uniform"])
        self.assertLess(
            result["zero_one_pairwise_loss"], result["zero_one_pairwise_loss_at_uniform"]
        )

    def test_learns_a_mixture_when_no_adviser_is_sufficient(self) -> None:
        rng = np.random.default_rng(37)
        blocks, targets = [], []
        for _round in range(400):
            count = 12
            distances = rng.permutation(np.arange(1, count + 1))
            perfect = np.argsort(np.argsort(-distances)) / (count - 1)
            block = rng.random((9, count))
            block[2] = np.clip(0.5 * perfect + 0.5 * rng.random(count), 0.0, 1.0)
            block[6] = np.clip(0.5 * perfect + 0.5 * rng.random(count), 0.0, 1.0)
            blocks.append(block)
            targets.append(distances)
        dataset = build_pair_dataset(blocks, targets)
        result = learn_static_weights(dataset)
        weights = np.asarray(result["weights"])
        self.assertGreater(float(weights[[2, 6]].sum()), 0.9)
        self.assertGreater(float(weights[2]), 0.2)
        self.assertGreater(float(weights[6]), 0.2)

    def test_optimum_satisfies_first_order_conditions(self) -> None:
        rng = np.random.default_rng(41)
        blocks, targets = [], []
        for _round in range(150):
            count = 10
            distances = rng.integers(1, 34, count)
            blocks.append(rng.random((9, count)))
            targets.append(distances)
        dataset = build_pair_dataset(blocks, targets)
        result = learn_static_weights(dataset)
        weights = np.asarray(result["weights"])
        _value, gradient = logistic_objective(weights, dataset)
        support = weights > 1e-9
        self.assertLess(float(gradient[support].max() - gradient[support].min()), 1e-4)
        if (~support).any():
            self.assertGreaterEqual(
                float(gradient[~support].min()), float(gradient[support].max()) - 1e-4
            )
        self.assertLessEqual(zero_one_pairwise_loss(weights, dataset), 1.0)

    def test_rejects_inconsistent_examples(self) -> None:
        with self.assertRaises(ValueError):
            build_pair_dataset([np.zeros((9, 3)), np.zeros((10, 3))], [np.asarray([1, 2, 3])] * 2)
        with self.assertRaises(ValueError):
            build_pair_dataset([np.zeros((9, 3))], [np.full(3, 33)])


if __name__ == "__main__":
    unittest.main()
