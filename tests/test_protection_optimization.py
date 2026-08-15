from __future__ import annotations

import unittest

import numpy as np

from expert_analysis.protection_optimization import (
    bits_matrix_for_allocation,
    build_expert_memory_matrix,
    random_allocation,
    solve_max_min_coverage,
    solve_weighted_selection,
    uniform_bits_matrix,
)

OLMOE_EXPERT_SHAPES = [(2048, 1024), (2048, 2048)]


class MemoryMatrixTests(unittest.TestCase):
    def test_olmoe_expert_accounting_matches_stage1_records(self) -> None:
        memory = build_expert_memory_matrix(
            [OLMOE_EXPERT_SHAPES] * 16, group_size=128, num_experts=64
        )
        self.assertEqual(int(memory.weight_count[0, 0]), 6_291_456)
        self.assertEqual(int(memory.group_count[0, 0]), 49_152)
        self.assertEqual(int(memory.bytes_by_bits[4][0, 0]), 3_244_032)
        self.assertEqual(int(memory.bytes_by_bits[8][0, 0]), 6_389_760)
        self.assertEqual(int(memory.bytes_by_bits[3][0, 0]), 2_457_600)
        self.assertEqual(int(memory.bytes_by_bits[16][0, 0]), 12_582_912)
        self.assertEqual(
            memory.total_increment_bytes(4), 1024 * (6_389_760 - 3_244_032)
        )
        self.assertEqual(
            memory.total_increment_bytes(3), 1024 * (6_389_760 - 2_457_600)
        )
        self.assertAlmostEqual(
            memory.effective_bits_per_weight(uniform_bits_matrix(4)), 4.125
        )

    def test_budget_is_floor_of_fraction(self) -> None:
        memory = build_expert_memory_matrix([[(4, 4)]], group_size=2, num_experts=3)
        total = memory.total_increment_bytes(4)
        self.assertEqual(
            memory.protection_budget_bytes(4, 0.2), int(np.floor(0.2 * total))
        )


def small_problem() -> tuple[np.ndarray, np.ndarray]:
    # Two layers x three experts; expert (0,0) is the only General specialist,
    # (1,2) the only Math specialist; others cover Coding/Reasoning.
    scores = np.zeros((2, 3, 4))
    scores[0, 0] = [1.0, 0.0, 0.0, 0.0]
    scores[1, 2] = [0.0, 1.0, 0.0, 0.0]
    scores[0, 1] = [0.0, 0.0, 0.7, 0.0]
    scores[1, 0] = [0.0, 0.0, 0.3, 0.6]
    scores[0, 2] = [0.0, 0.0, 0.0, 0.4]
    delta = np.full((2, 3), 10, dtype=np.int64)
    return scores, delta


class MaxMinMilpTests(unittest.TestCase):
    def test_max_min_picks_every_domain_specialist(self) -> None:
        scores, delta = small_problem()
        result = solve_max_min_coverage(scores, delta, budget_bytes=40)
        x = result.protected
        # With four affordable experts the optimum must cover all four domains.
        self.assertEqual(int(x.sum()), 4)
        self.assertEqual(x[0, 0], 1)
        self.assertEqual(x[1, 2], 1)
        self.assertEqual(x[0, 1], 1)
        self.assertEqual(x[1, 0], 1)
        self.assertAlmostEqual(result.objective_value, 0.6, places=6)
        self.assertEqual(result.metadata["solver_status"], 0)
        self.assertLessEqual(result.metadata["used_protection_bytes"], 40)

    def test_max_min_respects_memory_budget(self) -> None:
        scores, delta = small_problem()
        result = solve_max_min_coverage(scores, delta, budget_bytes=25)
        self.assertLessEqual(int(result.protected.sum()), 2)
        self.assertLessEqual(result.metadata["used_protection_bytes"], 25)

    def test_max_min_constraints_hold_for_every_domain(self) -> None:
        rng = np.random.default_rng(9)
        scores = rng.random((3, 5, 4))
        scores /= scores.sum(axis=(0, 1))
        delta = rng.integers(5, 20, size=(3, 5)).astype(np.int64)
        result = solve_max_min_coverage(scores, delta, budget_bytes=60)
        coverage = np.einsum("led,le->d", scores, result.protected.astype(float))
        self.assertGreaterEqual(coverage.min(), result.objective_value - 1e-8)


class WeightedSelectionTests(unittest.TestCase):
    def test_average_specialization_objective_maximizes_mean(self) -> None:
        scores, delta = small_problem()
        weights = scores.mean(axis=2)
        result = solve_weighted_selection(
            weights, delta, budget_bytes=20, problem="average_specialization"
        )
        # The two highest mean-specialization experts are (0,0) and (1,2).
        self.assertEqual(int(result.protected.sum()), 2)
        self.assertEqual(result.protected[0, 0], 1)
        self.assertEqual(result.protected[1, 2], 1)

    def test_global_importance_objective(self) -> None:
        weights = np.asarray([[0.5, 0.1], [0.2, 0.4]])
        delta = np.full((2, 2), 7, dtype=np.int64)
        result = solve_weighted_selection(
            weights, delta, budget_bytes=14, problem="global_importance"
        )
        self.assertEqual(result.protected[0, 0], 1)
        self.assertEqual(result.protected[1, 1], 1)
        self.assertAlmostEqual(result.objective_value, 0.9, places=9)

    def test_single_domain_objective_ignores_other_domains(self) -> None:
        scores, delta = small_problem()
        general_only = scores[:, :, 0]
        result = solve_weighted_selection(
            general_only, delta, budget_bytes=10, problem="single_domain_general"
        )
        self.assertEqual(result.protected[0, 0], 1)
        self.assertEqual(int(result.protected.sum()), 1)

    def test_exact_memory_constraint_excludes_unaffordable_items(self) -> None:
        weights = np.asarray([[10.0, 1.0]])
        delta = np.asarray([[100, 5]], dtype=np.int64)
        result = solve_weighted_selection(
            weights, delta, budget_bytes=50, problem="knapsack"
        )
        self.assertEqual(result.protected[0, 0], 0)
        self.assertEqual(result.protected[0, 1], 1)


class RandomAllocationTests(unittest.TestCase):
    def test_random_allocations_are_deterministic_and_feasible(self) -> None:
        delta = np.full((4, 8), 3, dtype=np.int64)
        first = random_allocation(delta, budget_bytes=20, seed=1001)
        second = random_allocation(delta, budget_bytes=20, seed=1001)
        np.testing.assert_array_equal(first.protected, second.protected)
        self.assertLessEqual(first.metadata["used_protection_bytes"], 20)
        self.assertEqual(int(first.protected.sum()), 6)
        other = random_allocation(delta, budget_bytes=20, seed=1002)
        self.assertFalse(np.array_equal(first.protected, other.protected))
        self.assertTrue(first.metadata["score_independent"])


class BitsMatrixTests(unittest.TestCase):
    def test_four_to_eight_assignment(self) -> None:
        x = np.asarray([[1, 0], [0, 1]])
        bits = bits_matrix_for_allocation(x, base_bits=4)
        np.testing.assert_array_equal(bits, [[8, 4], [4, 8]])

    def test_three_to_eight_assignment(self) -> None:
        x = np.asarray([[0, 1], [1, 0]])
        bits = bits_matrix_for_allocation(x, base_bits=3)
        np.testing.assert_array_equal(bits, [[3, 8], [8, 3]])

    def test_no_other_precision_is_allowed(self) -> None:
        with self.assertRaises(ValueError):
            bits_matrix_for_allocation(np.asarray([[1]]), base_bits=5)
        with self.assertRaises(ValueError):
            bits_matrix_for_allocation(
                np.asarray([[1]]), base_bits=4, protected_bits=16
            )


if __name__ == "__main__":
    unittest.main()
