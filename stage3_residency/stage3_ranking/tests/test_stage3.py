from __future__ import annotations

import unittest

import numpy as np

from helpers import make_trace, workload_config
from race_stage1.models import fit_transition_models
from race_stage1.simulation import simulate_causal_capacities
from race_stage3 import CAP
from race_stage3.features import (
    ALL_NAMES,
    BASE_NAMES,
    FeatureState,
    static_popularity,
)
from race_stage3.ranking import (
    RankingModel,
    build_pairs,
    fit_pairwise_logistic,
    fit_ranking_model,
    group_slices,
    pairwise_accuracy,
    standardize,
)
from race_stage3.simulation import (
    GroupCollector,
    simulate_stage3,
    stage1_winner_scorer,
)
from residency_headroom.simulator import simulate_oracle
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import (
    build_calibration_workload,
    build_workloads,
    split_sequences,
)


HORIZONS = (1, 2, 4, 8, 16, 32)


class RankingFitTests(unittest.TestCase):
    def test_group_slices(self) -> None:
        groups = np.asarray([0, 0, 1, 1, 1, 2])
        self.assertEqual(group_slices(groups), [(0, 2), (2, 5), (5, 6)])
        self.assertEqual(group_slices(np.asarray([], dtype=np.int64)), [])

    def test_pairs_are_ordered_and_group_balanced(self) -> None:
        features = np.asarray([[1.0], [2.0], [3.0], [10.0], [20.0]])
        targets = np.asarray([1.0, 2.0, 3.0, 5.0, 5.0])
        groups = np.asarray([0, 0, 0, 1, 1])
        differences, weights = build_pairs(features, targets, groups)
        # Group 1 has no comparable pair (equal targets) and contributes nothing.
        self.assertEqual(differences.shape[0], 3)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        # Every difference is (better candidate) minus (worse candidate).
        self.assertTrue(np.all(differences < 0))

    def test_recovers_the_informative_feature(self) -> None:
        rng = np.random.default_rng(3)
        blocks, targets, groups = [], [], []
        for index in range(400):
            count = int(rng.integers(4, 16))
            distance = rng.integers(1, CAP + 1, count).astype(np.float64)
            row = rng.random((count, 6))
            row[:, 2] = -distance + rng.normal(0, 0.2, count)
            blocks.append(row)
            targets.append(distance)
            groups.append(np.full(count, index))
        features = np.concatenate(blocks)
        target = np.concatenate(targets)
        group = np.concatenate(groups)
        model = fit_ranking_model(
            features, target, group, [f"f{i}" for i in range(6)], l2_grid=(0.01, 0.003)
        )
        self.assertEqual(int(np.argmax(np.abs(model.weights))), 2)
        self.assertGreater(model.weights[2], 0.0)
        self.assertGreater(model.holdout_accuracy, 0.9)

    def test_objective_is_convex_and_optimum_is_stationary(self) -> None:
        rng = np.random.default_rng(5)
        differences = rng.normal(size=(4000, 5))
        weights = np.full(4000, 1.0 / 4000)
        theta, value, info = fit_pairwise_logistic(differences, weights, 1e-3)
        self.assertTrue(info["converged"])
        for _ in range(20):
            perturbed = theta + rng.normal(scale=0.05, size=theta.shape)
            margin = differences @ perturbed
            other = float(weights @ np.logaddexp(0.0, -margin)) / weights.sum() \
                + 0.5e-3 * float(perturbed @ perturbed)
            self.assertGreaterEqual(other, value - 1e-9)

    def test_pairwise_accuracy_bounds(self) -> None:
        targets = np.asarray([1.0, 2.0, 3.0])
        groups = np.zeros(3)
        self.assertAlmostEqual(pairwise_accuracy(np.asarray([3.0, 2.0, 1.0]), targets, groups), 1.0)
        self.assertAlmostEqual(pairwise_accuracy(np.asarray([1.0, 2.0, 3.0]), targets, groups), 0.0)
        self.assertAlmostEqual(pairwise_accuracy(np.zeros(3), targets, groups), 0.5)

    def test_model_round_trips(self) -> None:
        model = RankingModel(np.asarray([1.0, -2.0]), ("a", "b"), 0.01, 0.7, {"iterations": 3})
        restored = RankingModel.from_dict(model.as_dict())
        self.assertTrue(np.array_equal(model.weights, restored.weights))
        self.assertEqual(model.feature_names, restored.feature_names)

    def test_standardize_handles_constant_columns(self) -> None:
        mean, scale = standardize(np.asarray([[1.0, 5.0], [3.0, 5.0]]))
        self.assertTrue(np.all(scale > 0))


class Stage3SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trace = make_trace(
            prompts_per_domain=4, generation_length=48, num_layers=2, num_experts=12, top_k=3
        )
        cls.config = workload_config(prompts_per_domain=4)
        cls.split = split_sequences(cls.trace, cls.config["calibration_fraction"])
        cls.calibration = build_calibration_workload(cls.trace, cls.split, seed=99)
        cls.workloads = build_workloads(cls.trace, cls.split, cls.config)
        cls.models = fit_transition_models(cls.trace, cls.calibration, HORIZONS)
        cls.popularity = static_popularity(cls.trace, cls.calibration)
        cls.capacities = (3, 5, 7, 9)
        cls.workload = cls.workloads[0]

    def _state(self, include: bool = True) -> FeatureState:
        return FeatureState(
            self.models, self.trace.num_layers, self.trace.num_experts, self.popularity,
            include_request_scope=include,
        )

    def test_feature_block_shape_and_finiteness(self) -> None:
        for include, names in ((True, ALL_NAMES), (False, BASE_NAMES)):
            state = self._state(include)
            request = np.asarray([0, 1, 2], dtype=np.int64)
            block = state.features(0, request, np.asarray([0.5, 0.3, 0.2]), np.sort(request), 0)
            self.assertEqual(block.shape, (len(names), self.trace.num_experts))
            self.assertTrue(np.isfinite(block).all())

    def test_stage3_reproduces_the_frozen_stage1_winner_exactly(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        replay = simulate_stage3(
            self.trace, self.workload, self.capacities,
            {c: scorer for c in self.capacities}, state, variant="equiv",
        )
        reference = simulate_causal_capacities(
            self.trace, self.workload, self.capacities,
            {"method": "markov_plus_ewma", "horizon": 2, "beta": 0.5, "history_alpha": 0.95},
            self.models,
        )
        self.assertEqual(
            [r.misses for r in reference], [r.misses for r in replay]
        )

    def test_perfect_score_reproduces_the_stage0_oracle(self) -> None:
        for workload in self.workloads[:4]:
            state = self._state()
            perfect = simulate_stage3(
                self.trace, workload, self.capacities,
                {c: (lambda b: np.zeros(b.shape[1])) for c in self.capacities},
                state, variant="perfect", perfect_score_override=True,
            )
            for item in perfect:
                self.assertEqual(
                    item.misses, simulate_oracle(self.trace, workload, item.capacity).misses,
                    (workload.name, item.capacity),
                )

    def test_cache_semantics_are_unchanged(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        results = simulate_stage3(
            self.trace, self.workload, self.capacities,
            {c: scorer for c in self.capacities}, state, enable_diagnostics=True,
        )
        oracle = {c: simulate_oracle(self.trace, self.workload, c).misses for c in self.capacities}
        for item in results:
            self.assertEqual(item.hits + item.misses, item.requests)
            self.assertEqual(item.misses, item.admissions)
            self.assertLessEqual(item.maximum_occupancy, item.capacity)
            self.assertEqual(item.misses, sum(r.misses for r in item.per_sequence))
            self.assertEqual(item.misses, sum(item.layer_misses))
            self.assertGreaterEqual(item.misses, oracle[item.capacity])

    def test_capacity_equal_to_top_k_is_degenerate(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        results = simulate_stage3(
            self.trace, self.workload, (self.trace.top_k,),
            {self.trace.top_k: scorer}, state,
        )
        self.assertEqual(results[0].misses, results[0].requests)

    def test_replay_is_deterministic_and_observer_free(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        first = simulate_stage3(self.trace, self.workload, self.capacities,
                                {c: scorer for c in self.capacities}, state,
                                enable_diagnostics=True)
        second = simulate_stage3(self.trace, self.workload, self.capacities,
                                 {c: scorer for c in self.capacities}, state,
                                 enable_diagnostics=False)
        self.assertEqual([r.misses for r in first], [r.misses for r in second])

    def test_state_reset_restores_identical_features(self) -> None:
        state = self._state()
        request = np.asarray([0, 1, 2], dtype=np.int64)
        gates = np.asarray([0.5, 0.3, 0.2])
        first = state.features(0, request, gates, np.sort(request), 0).copy()
        state.absorb(0, request, gates, 0)
        state.absorb(0, request, gates, 1)
        state.reset()
        again = state.features(0, request, gates, np.sort(request), 0)
        self.assertTrue(np.allclose(first, again))

    def test_request_scope_features_reset_at_request_boundaries(self) -> None:
        state = self._state()
        request = np.asarray([0, 1, 2], dtype=np.int64)
        gates = np.asarray([0.5, 0.3, 0.2])
        for step in range(5):
            state.absorb(0, request, gates, step)
        index = ALL_NAMES.index("request_count")
        before = state.features(0, request, gates, np.sort(request), 5)[index]
        self.assertGreater(float(before.max()), 0.0)
        state.begin_request()
        after = state.features(0, request, gates, np.sort(request), 5)[index]
        self.assertEqual(float(after.max()), 0.0)

    def test_future_mutation_cannot_change_earlier_actions(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        baseline = simulate_stage3(self.trace, self.workload, (7,), {7: scorer}, state)
        arrays = self.trace.arrays()
        requests = arrays["requested_expert_ids"].copy()
        last = self.workload.sequences[-1].source_sequence_id
        view = self.trace.sequence_slices()[last]
        rotated = (requests[view.start:view.stop] + 5) % self.trace.num_experts
        for row in range(rotated.shape[0]):
            if np.unique(rotated[row]).size != rotated.shape[1]:
                rotated[row] = np.arange(rotated.shape[1])
        requests[view.start:view.stop] = rotated
        arrays["requested_expert_ids"] = requests
        metadata = dict(self.trace.metadata)
        metadata.pop("trace_hash", None)
        mutated = RoutingTrace.from_mapping(arrays, metadata, validate=False)
        state2 = self._state()
        changed = simulate_stage3(mutated, self.workload, (7,),
                                  {7: stage1_winner_scorer(self.models, state2)}, state2)
        self.assertEqual(
            [r.misses for r in baseline[0].per_sequence[:-1]],
            [r.misses for r in changed[0].per_sequence[:-1]],
        )

    def test_collector_gathers_valid_groups(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        decision = tuple(c for c in self.capacities if c > self.trace.top_k)
        collector = GroupCollector(decision, stride=3, warmup=4)
        simulate_stage3(self.trace, self.workload, self.capacities,
                        {c: scorer for c in self.capacities}, state, collector=collector)
        self.assertGreater(collector.groups, 0)
        for capacity in decision:
            features, targets, groups = collector.dataset(capacity)
            self.assertEqual(features.shape[0], targets.shape[0])
            self.assertEqual(features.shape[1], len(ALL_NAMES))
            self.assertTrue(np.all(targets >= 1))
            self.assertTrue(np.all(targets <= CAP))
        pooled = collector.pooled()
        self.assertEqual(pooled[0].shape[0], pooled[1].shape[0])

    def test_learned_model_beats_the_stage1_winner_on_calibration(self) -> None:
        state = self._state()
        scorer = stage1_winner_scorer(self.models, state)
        decision = tuple(c for c in self.capacities if c > self.trace.top_k)
        collector = GroupCollector(decision, stride=2, warmup=4)
        reference = simulate_stage3(
            self.trace, self.calibration, decision,
            {c: scorer for c in decision}, state, collector=collector,
        )
        models = {
            c: fit_ranking_model(*collector.dataset(c), ALL_NAMES, l2_grid=(0.01, 0.003))
            for c in decision
        }
        learned = simulate_stage3(
            self.trace, self.calibration, decision,
            {c: (lambda b, m=models[c]: m.score(b)) for c in decision}, state,
        )
        before = sum(r.misses for r in reference)
        after = sum(r.misses for r in learned)
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
