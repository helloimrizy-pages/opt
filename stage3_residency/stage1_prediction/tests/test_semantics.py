from __future__ import annotations

import unittest

import numpy as np

from helpers import make_trace, workload_config
from race_stage1.analysis import _cluster_multiplicities, _stage1_decision
from race_stage1.lookahead import LimitedLookaheadOracle, _optimal_cost_from_initial
from race_stage1.metrics import (
    baseline_improvement,
    oracle_gap_closed,
    residual_headroom,
)
from race_stage1.models import fit_transition_models, same_layer_indices
from race_stage1.policy import MarkovPredictor, PredictionRetentionPolicy
from race_stage1.simulation import (
    simulate_causal_capacities,
    simulate_lookahead_capacities,
)
from residency_headroom.exact_solver import solve_exact
from residency_headroom.workloads import (
    build_calibration_workload,
    build_workloads,
    split_sequences,
)


class Stage1SemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = make_trace(prompts_per_domain=3, generation_length=4)
        self.config = workload_config(prompts_per_domain=3)
        self.split = split_sequences(self.trace, self.config["calibration_fraction"])
        self.calibration = build_calibration_workload(self.trace, self.split, seed=99)
        self.workloads = build_workloads(self.trace, self.split, self.config)
        self.models = fit_transition_models(self.trace, self.calibration, [1, 2, 4])

    def test_calibration_and_evaluation_are_disjoint(self) -> None:
        calibration = set(self.calibration.sequence_ids)
        evaluation = {
            sequence
            for workload in self.workloads
            for sequence in workload.sequence_ids
        }
        self.assertFalse(calibration & evaluation)
        self.assertEqual(self.models.calibration_workload_hash, self.calibration.hash)

    def test_same_layer_stream_skips_other_layer_events(self) -> None:
        workload = self.workloads[0]
        streams = same_layer_indices(self.trace, workload)
        self.assertEqual(len(streams), self.trace.num_layers)
        for ordinal, indices in enumerate(streams):
            self.assertTrue(np.all(self.trace.layer_index[indices] == ordinal))
            self.assertTrue(np.all(np.diff(indices) >= self.trace.num_layers))

    def test_atomic_replay_accounting_capacity_and_determinism(self) -> None:
        workload = self.workloads[0]
        spec = {"method": "markov_h", "horizon": 2}
        first = simulate_causal_capacities(
            self.trace, workload, [2, 3, 4], spec, self.models
        )
        second = simulate_causal_capacities(
            self.trace, workload, [2, 3, 4], spec, self.models
        )
        self.assertEqual(first, second)
        for result in first:
            self.assertEqual(result.hits + result.misses, result.requests)
            self.assertEqual(result.misses, result.admissions)
            self.assertLessEqual(result.maximum_occupancy, result.capacity)

    def test_future_trace_mutation_cannot_change_first_causal_action(self) -> None:
        matrix = np.full((6, 6), 0.5)
        first = PredictionRetentionPolicy(3, 6, MarkovPredictor(matrix))
        second = PredictionRetentionPolicy(3, 6, MarkovPredictor(matrix))
        prefix = [([0, 1], [0.6, 0.4]), ([2, 3], [0.6, 0.4])]
        first_states = [first.process(request, gates).transition.after for request, gates in prefix]
        second_states = [second.process(request, gates).transition.after for request, gates in prefix]
        # The hypothetical suffix is deliberately different, but no future object is an
        # argument to a causal policy; identical observed prefixes give identical actions.
        future_one = [[0, 4], [1, 5]]
        future_two = [[4, 5], [0, 1]]
        self.assertNotEqual(future_one, future_two)
        self.assertEqual(first_states, second_states)

    def test_limited_lookahead_first_action_matches_finite_horizon_dp(self) -> None:
        requests = (
            frozenset({0, 1}),
            frozenset({2, 3}),
            frozenset({0, 4}),
            frozenset({1, 5}),
        )
        policy = LimitedLookaheadOracle(requests, 3, 6, horizon=2)
        for position, request in enumerate(requests):
            before = policy.resident
            transition = policy.process(request)
            visible = requests[position : position + 3]
            optimal = _optimal_cost_from_initial(visible, before, 3)
            forced = len(request - before) + _optimal_cost_from_initial(
                visible[1:], transition.after, 3
            )
            self.assertEqual(forced, optimal)
        policy.finish()

    def test_full_lookahead_reproduces_stage0_exact_optimum(self) -> None:
        requests = (
            frozenset({0, 1}),
            frozenset({2, 3}),
            frozenset({0, 4}),
            frozenset({1, 5}),
            frozenset({2, 4}),
        )
        policy = LimitedLookaheadOracle(requests, 3, 6, horizon=None)
        misses = sum(len(policy.process(request).misses) for request in requests)
        policy.finish()
        self.assertEqual(misses, solve_exact(requests, 3, 6).misses)

    def test_multicapacity_lookahead_is_deterministic_and_accounted(self) -> None:
        workload = self.workloads[0]
        first = simulate_lookahead_capacities(
            self.trace, workload, [2, 3, 4], [1, 2, 4]
        )
        second = simulate_lookahead_capacities(
            self.trace, workload, [2, 3, 4], [1, 2, 4]
        )
        self.assertEqual(first, second)
        for result in first:
            self.assertEqual(result.hits + result.misses, result.requests)
            self.assertEqual(result.misses, result.admissions)
            self.assertLessEqual(result.maximum_occupancy, result.capacity)

    def test_gap_closed_and_zero_denominator(self) -> None:
        self.assertAlmostEqual(baseline_improvement(100, 80), 0.2)
        self.assertAlmostEqual(oracle_gap_closed(100, 80, 60), 0.5)
        self.assertAlmostEqual(residual_headroom(100, 80, 60), 0.2)
        self.assertIsNone(oracle_gap_closed(100, 100, 100))

    def test_bootstrap_clusters_are_deterministic_and_domain_stratified(self) -> None:
        domains = {0: "general", 1: "general", 2: "coding", 3: "coding"}
        first = _cluster_multiplicities(domains, 50, 77)
        second = _cluster_multiplicities(domains, 50, 77)
        for sequence in domains:
            np.testing.assert_array_equal(first[sequence], second[sequence])
        np.testing.assert_array_equal(first[0] + first[1], np.full(50, 2))
        np.testing.assert_array_equal(first[2] + first[3], np.full(50, 2))

    def test_frozen_decision_rule_excludes_capacity_eight(self) -> None:
        rows = [
            {"capacity": 8, "oracle_gap_closed": None, "residual_headroom": 0.0},
            {"capacity": 12, "oracle_gap_closed": 0.2, "residual_headroom": 0.12},
            {"capacity": 16, "oracle_gap_closed": 0.3, "residual_headroom": 0.15},
            {"capacity": 24, "oracle_gap_closed": 0.4, "residual_headroom": 0.18},
            {"capacity": 32, "oracle_gap_closed": 0.6, "residual_headroom": 0.08},
        ]
        rule = {
            "strong_go": {
                "gap_closed_strictly_below": 0.5,
                "gap_closed_required_capacities": 3,
                "residual_headroom_at_least": 0.1,
                "residual_required_capacities": 3,
            },
            "no_go": {
                "gap_closed_at_least": 0.75,
                "gap_closed_required_capacities": 3,
                "or_residual_headroom_strictly_below": 0.05,
                "residual_required_capacities": 3,
            },
        }
        self.assertEqual(_stage1_decision(rows, rule)["verdict"], "RACE_STAGE1_STRONG_GO")


if __name__ == "__main__":
    unittest.main()
