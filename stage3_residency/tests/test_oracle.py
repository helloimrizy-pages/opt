from __future__ import annotations

import unittest

import numpy as np

from residency_headroom.exact_solver import solve_exact
from residency_headroom.oracle import oracle_solution, validate_oracle


class OracleTests(unittest.TestCase):
    def test_exact_small_state_optimum_and_switch_objective(self) -> None:
        requests = [{0, 1}, {1, 2}, {2, 3}, {0, 3}]
        miss_only = solve_exact(requests, 3, 4)
        movement = solve_exact(requests, 3, 4, switch_lambda=0.5)
        self.assertEqual(miss_only.misses, 4)
        self.assertEqual(miss_only.admissions, 4)
        self.assertAlmostEqual(movement.total_cost, 1.5 * miss_only.total_cost)
        for request, state in zip(requests, miss_only.states):
            self.assertTrue(set(request).issubset(state))
            self.assertLessEqual(len(state), 3)

    def test_exact_solver_supports_heterogeneous_tiny_costs(self) -> None:
        requests = [{0}, {1}, {2}, {0}, {1}]
        weighted = solve_exact(
            requests,
            2,
            3,
            miss_costs=np.asarray([10.0, 1.0, 1.0]),
            admission_costs=np.asarray([10.0, 1.0, 1.0]),
        )
        unit = solve_exact(requests, 2, 3)
        self.assertGreater(weighted.miss_cost, unit.miss_cost)
        self.assertIn(0, weighted.states[2])

    def test_scalable_oracle_matches_exact_for_known_set_trace(self) -> None:
        requests = [{0, 1}, {1, 2}, {3, 4}, {0, 4}, {1, 3}]
        misses, admissions, states = oracle_solution(requests, 3, 5)
        exact = solve_exact(requests, 3, 5)
        self.assertEqual(misses, exact.misses)
        self.assertEqual(admissions, exact.admissions)
        self.assertEqual(len(states), len(requests))

    def test_scalable_oracle_exhaustive_and_random_validation(self) -> None:
        result = validate_oracle(
            random_cases=50, seed=9, exhaustive_max_events=3, lambda_values=(0, 0.5, 1)
        )
        self.assertTrue(result.passed)
        self.assertGreater(result.exhaustive_cases, 100)
        self.assertEqual(result.random_cases, 50)
        self.assertEqual(result.maximum_cost_difference, 0.0)


if __name__ == "__main__":
    unittest.main()
