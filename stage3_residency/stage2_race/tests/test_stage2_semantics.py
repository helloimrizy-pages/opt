from __future__ import annotations

import unittest

import numpy as np

from helpers import make_trace, workload_config
from race_stage1.models import fit_transition_models
from race_stage1.simulation import simulate_causal_capacities
from race_stage2 import H_MAX
from race_stage2.advisers import (
    AdviserBank,
    PRIMARY_POOL,
    fit_stage2_transition_models,
    verify_stage1_horizon_reuse,
)
from race_stage2.policy import (
    RaceVariant,
    online_variant,
    static_per_layer_variant,
    static_variant,
    uniform_variant,
    variant_from_spec,
    variant_to_spec,
)
from race_stage2.simulation import (
    CalibrationExampleCollector,
    simulate_perfect_score,
    simulate_race_variant,
)
from race_stage2.synthetic import build_trace
from residency_headroom.simulator import simulate_oracle
from residency_headroom.trace import RoutingTrace
from residency_headroom.workloads import (
    build_calibration_workload,
    build_workloads,
    make_workload,
    split_sequences,
)


def _single_adviser(name: str, pool: str = "primary") -> RaceVariant:
    from race_stage2.advisers import pool_names

    names = pool_names(pool)
    weights = np.zeros(len(names), dtype=np.float64)
    weights[names.index(name)] = 1.0
    return RaceVariant(
        name=f"SINGLE_{name}",
        weight_source="static_global",
        static_weights=weights,
        pool=pool,
    )


class Stage2SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trace = make_trace(
            prompts_per_domain=4, generation_length=48, num_layers=2, num_experts=12, top_k=3
        )
        cls.config = workload_config(prompts_per_domain=4)
        cls.split = split_sequences(cls.trace, cls.config["calibration_fraction"])
        cls.calibration = build_calibration_workload(cls.trace, cls.split, seed=99)
        cls.workloads = build_workloads(cls.trace, cls.split, cls.config)
        cls.models = fit_stage2_transition_models(cls.trace, cls.calibration)
        cls.capacities = (3, 5, 7, 9)
        cls.workload = cls.workloads[0]

    def test_stage2_markov_fit_reuses_stage1_horizons(self) -> None:
        stage1 = fit_transition_models(self.trace, self.calibration, [1, 2, 4, 8, 16])
        for horizon in (1, 2, 4, 8, 16):
            for layer in range(self.trace.num_layers):
                self.assertTrue(
                    np.allclose(
                        stage1.matrix(horizon, layer), self.models.matrix(horizon, layer)
                    )
                )
        self.assertIn(32, self.models.horizons)

    def test_single_adviser_reproduces_the_frozen_stage1_predictor(self) -> None:
        cases = [
            ({"method": "markov_h", "horizon": 1}, "MARKOV_H1"),
            ({"method": "markov_h", "horizon": 2}, "MARKOV_H2"),
            ({"method": "markov_h", "horizon": 8}, "MARKOV_H8"),
            ({"method": "gate_ewma", "alpha": 0.95}, "GATE_EWMA"),
            ({"method": "persistence"}, "PERSISTENCE"),
            (
                {"method": "markov_plus_ewma", "horizon": 2, "beta": 0.5, "history_alpha": 0.95},
                "STAGE1_HYBRID",
            ),
        ]
        stage1_models = fit_transition_models(self.trace, self.calibration, [1, 2, 4, 8, 16])
        for spec, adviser in cases:
            pool = "extended" if adviser == "STAGE1_HYBRID" else "primary"
            stage1 = simulate_causal_capacities(
                self.trace, self.workload, self.capacities, spec, stage1_models
            )
            stage2 = simulate_race_variant(
                self.trace,
                self.workload,
                self.capacities,
                _single_adviser(adviser, pool),
                self.models,
                enable_diagnostics=False,
            )
            self.assertEqual(
                [item.misses for item in stage1],
                [item.misses for item in stage2],
                adviser,
            )

    def test_perfect_score_reproduces_the_stage0_oracle(self) -> None:
        for workload in self.workloads[:4]:
            perfect = simulate_perfect_score(
                self.trace, workload, self.capacities, self.models
            )
            for item in perfect:
                oracle = simulate_oracle(self.trace, workload, item.capacity)
                self.assertEqual(item.misses, oracle.misses, (workload.name, item.capacity))

    def test_cache_semantics_are_unchanged(self) -> None:
        results = simulate_race_variant(
            self.trace,
            self.workload,
            self.capacities,
            online_variant(loss="rank", eta=0.3, initialization="uniform"),
            self.models,
        )
        for item in results:
            self.assertEqual(item.hits + item.misses, item.requests)
            self.assertEqual(item.misses, item.admissions)
            self.assertLessEqual(item.maximum_occupancy, item.capacity)
            self.assertEqual(
                item.misses, sum(row.misses for row in item.per_sequence)
            )
            self.assertEqual(item.misses, sum(item.layer_misses))
        oracle = {
            capacity: simulate_oracle(self.trace, self.workload, capacity).misses
            for capacity in self.capacities
        }
        for item in results:
            self.assertGreaterEqual(item.misses, oracle[item.capacity])

    def test_capacity_equal_to_top_k_is_degenerate(self) -> None:
        results = simulate_race_variant(
            self.trace,
            self.workload,
            (self.trace.top_k,),
            online_variant(loss="rank", eta=1.0, initialization="uniform"),
            self.models,
        )
        item = results[0]
        self.assertEqual(item.misses, item.requests)
        self.assertEqual(item.learning["examples_generated"], 0)
        self.assertEqual(item.learning["applied_updates"], 0)

    def test_delayed_feedback_queue_resolves_exactly_at_h_max(self) -> None:
        results = simulate_race_variant(
            self.trace,
            self.workload,
            self.capacities,
            online_variant(loss="rank", eta=0.3, initialization="uniform"),
            self.models,
            label_cross_check=True,
        )
        for item in results:
            learning = item.learning
            self.assertEqual(
                learning["examples_generated"],
                learning["examples_resolved"] + learning["examples_unresolved_at_stream_end"],
            )
            if learning["examples_resolved"]:
                self.assertEqual(learning["minimum_update_minus_decision_offset"], H_MAX)
                self.assertEqual(
                    learning["average_feedback_delay_same_layer_events"], float(H_MAX)
                )
                self.assertEqual(learning["maximum_feedback_delay_same_layer_events"], H_MAX)
                self.assertGreater(learning["label_cross_checks_passed"], 0)
            for sample in learning["causality_samples"]:
                self.assertGreaterEqual(
                    sample["label_resolution_event_index"],
                    sample["decision_event_index"] + H_MAX,
                )
                self.assertGreaterEqual(
                    sample["weight_update_event_index"],
                    sample["label_resolution_event_index"],
                )

    def test_trailing_examples_are_left_unresolved(self) -> None:
        results = simulate_race_variant(
            self.trace,
            self.workload,
            (7,),
            online_variant(loss="rank", eta=0.3, initialization="uniform"),
            self.models,
        )
        learning = results[0].learning
        self.assertGreater(learning["examples_unresolved_at_stream_end"], 0)
        self.assertLessEqual(
            learning["examples_unresolved_at_stream_end"],
            H_MAX * self.trace.num_layers,
        )
        self.assertGreater(learning["unresolved_fraction"], 0.0)
        self.assertLess(learning["unresolved_fraction"], 1.0)

    def test_per_layer_weight_streams_are_independent(self) -> None:
        # Layer 0 reuses experts every 12 events and layer 1 every 20, over disjoint
        # expert blocks, so the two layers must learn different adviser weights.
        top_k, first_period, second_period = 2, 12, 20
        block = first_period * top_k

        def request_for(sequence_id: int, layer: int, position: int) -> list[int]:
            del sequence_id
            period = first_period if layer == 0 else second_period
            base = 0 if layer == 0 else block
            slot = position % period
            return [base + slot * top_k + index for index in range(top_k)]

        trace = build_trace(
            request_for,
            prompts_per_domain=2,
            generation_length=480,
            num_layers=2,
            num_experts=block + second_period * top_k,
            top_k=top_k,
        )
        identifiers = [int(item["sequence_id"]) for item in trace.metadata["sequences"]]
        calibration = make_workload(
            "cal", "calibration", [(item, 0, "c", "general") for item in identifiers[:4]]
        )
        evaluation = make_workload(
            "ev", "stationary", [(item, 0, "e", "general") for item in identifiers[4:]]
        )
        models = fit_stage2_transition_models(trace, calibration)
        results = simulate_race_variant(
            trace,
            evaluation,
            (16,),
            online_variant(loss="rank", eta=1.0, initialization="uniform"),
            models,
            enable_diagnostics=False,
        )
        rows = results[0].weight_rows
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["layer"] for row in rows], [0, 1])
        self.assertEqual(rows[0]["dominant_adviser"], "MARKOV_H8")
        self.assertEqual(rows[1]["dominant_adviser"], "MARKOV_H16")
        self.assertFalse(
            np.allclose(np.asarray(rows[0]["end_weights"]), np.asarray(rows[1]["end_weights"]))
        )
        updates = {
            int(layer): value
            for layer, value in results[0].learning["updates_by_layer"].items()
        }
        self.assertEqual(sorted(updates), [0, 1])
        self.assertGreater(min(updates.values()), 0)

    def test_global_weight_scope_uses_one_stream(self) -> None:
        results = simulate_race_variant(
            self.trace,
            self.workload,
            (7,),
            online_variant(
                loss="rank", eta=1.0, initialization="uniform", scope="global"
            ),
            self.models,
        )
        rows = results[0].weight_rows
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["layer"], -1)

    def test_capacities_do_not_share_online_state(self) -> None:
        variant = online_variant(loss="rank", eta=1.0, initialization="uniform")
        together = simulate_race_variant(
            self.trace, self.workload, self.capacities, variant, self.models,
            enable_diagnostics=False,
        )
        for capacity in self.capacities:
            alone = simulate_race_variant(
                self.trace, self.workload, (capacity,), variant, self.models,
                enable_diagnostics=False,
            )
            matched = next(item for item in together if item.capacity == capacity)
            self.assertEqual(alone[0].misses, matched.misses)
            self.assertEqual(
                alone[0].weight_rows[0]["end_weights"],
                matched.weight_rows[0]["end_weights"],
            )

    def test_reset_semantics_at_workload_boundaries(self) -> None:
        variant = online_variant(loss="rank", eta=1.0, initialization="uniform")
        first = simulate_race_variant(
            self.trace, self.workloads[1], (7,), variant, self.models, enable_diagnostics=False
        )
        simulate_race_variant(
            self.trace, self.workloads[0], (7,), variant, self.models, enable_diagnostics=False
        )
        again = simulate_race_variant(
            self.trace, self.workloads[1], (7,), variant, self.models, enable_diagnostics=False
        )
        self.assertEqual(first[0].misses, again[0].misses)
        self.assertEqual(
            first[0].weight_rows[0]["end_weights"], again[0].weight_rows[0]["end_weights"]
        )
        self.assertEqual(
            first[0].weight_rows[0]["start_weights"],
            [1.0 / len(PRIMARY_POOL)] * len(PRIMARY_POOL),
        )

    def test_adviser_bank_reset_clears_adaptive_state(self) -> None:
        bank = AdviserBank(self.models, self.trace.num_layers, self.trace.num_experts)
        request = np.asarray([0, 1, 2], dtype=np.int64)
        gates = np.asarray([0.5, 0.3, 0.2])
        first = bank.step(0, request, gates).copy()
        bank.step(0, request, gates)
        bank.reset()
        again = bank.step(0, request, gates).copy()
        self.assertTrue(np.allclose(first, again))

    def test_diagnostic_observer_never_changes_a_decision(self) -> None:
        variant = online_variant(loss="rank", eta=0.5, initialization="uniform")
        with_observer = simulate_race_variant(
            self.trace, self.workload, self.capacities, variant, self.models,
            enable_diagnostics=True,
        )
        without_observer = simulate_race_variant(
            self.trace, self.workload, self.capacities, variant, self.models,
            enable_diagnostics=False,
        )
        self.assertEqual(
            [item.misses for item in with_observer],
            [item.misses for item in without_observer],
        )
        self.assertEqual(
            [item.weight_rows for item in with_observer],
            [item.weight_rows for item in without_observer],
        )

    def test_future_mutation_cannot_change_earlier_causal_actions(self) -> None:
        variant = online_variant(loss="rank", eta=1.0, initialization="uniform")
        baseline = simulate_race_variant(
            self.trace, self.workload, (7,), variant, self.models, enable_diagnostics=False
        )
        arrays = self.trace.arrays()
        requests = arrays["requested_expert_ids"].copy()
        last_sequence = self.workload.sequences[-1].source_sequence_id
        view = self.trace.sequence_slices()[last_sequence]
        rotated = (requests[view.start : view.stop] + 5) % self.trace.num_experts
        for row in range(rotated.shape[0]):
            if np.unique(rotated[row]).size != rotated.shape[1]:
                rotated[row] = np.arange(rotated.shape[1])
        requests[view.start : view.stop] = rotated
        arrays["requested_expert_ids"] = requests
        metadata = dict(self.trace.metadata)
        metadata.pop("trace_hash", None)
        mutated = RoutingTrace.from_mapping(arrays, metadata, validate=False)
        changed = simulate_race_variant(
            mutated, self.workload, (7,), variant, self.models, enable_diagnostics=False
        )
        before = [row.misses for row in baseline[0].per_sequence[:-1]]
        after = [row.misses for row in changed[0].per_sequence[:-1]]
        self.assertEqual(before, after)
        self.assertNotEqual(
            baseline[0].per_sequence[-1].misses, changed[0].per_sequence[-1].misses
        )

    def test_uniform_static_and_online_are_deterministic(self) -> None:
        weights = np.full(len(PRIMARY_POOL), 1.0 / len(PRIMARY_POOL))
        for variant in (
            uniform_variant(),
            static_variant(weights),
            static_per_layer_variant(np.tile(weights, (self.trace.num_layers, 1))),
            online_variant(loss="cost", eta=0.3, initialization="static", static_weights=weights),
        ):
            first = simulate_race_variant(
                self.trace, self.workload, self.capacities, variant, self.models,
                enable_diagnostics=False,
            )
            second = simulate_race_variant(
                self.trace, self.workload, self.capacities, variant, self.models,
                enable_diagnostics=False,
            )
            self.assertEqual([x.misses for x in first], [x.misses for x in second])

    def test_uniform_static_and_per_layer_static_agree_when_weights_match(self) -> None:
        weights = np.full(len(PRIMARY_POOL), 1.0 / len(PRIMARY_POOL))
        uniform = simulate_race_variant(
            self.trace, self.workload, self.capacities, uniform_variant(), self.models,
            enable_diagnostics=False,
        )
        static = simulate_race_variant(
            self.trace, self.workload, self.capacities, static_variant(weights), self.models,
            enable_diagnostics=False,
        )
        per_layer = simulate_race_variant(
            self.trace,
            self.workload,
            self.capacities,
            static_per_layer_variant(np.tile(weights, (self.trace.num_layers, 1))),
            self.models,
            enable_diagnostics=False,
        )
        self.assertEqual([x.misses for x in uniform], [x.misses for x in static])
        self.assertEqual([x.misses for x in uniform], [x.misses for x in per_layer])

    def test_calibration_collector_subsamples_deterministically(self) -> None:
        collector = CalibrationExampleCollector(4, self.capacities)
        simulate_race_variant(
            self.trace,
            self.workload,
            self.capacities,
            uniform_variant(),
            self.models,
            enable_diagnostics=False,
            example_collector=collector,
        )
        self.assertGreater(len(collector), 0)
        for block, distances in zip(collector.normalized, collector.distances):
            self.assertEqual(block.shape[0], len(PRIMARY_POOL))
            self.assertEqual(block.shape[1], distances.shape[0])
            self.assertTrue(np.all(distances >= 1))
            self.assertTrue(np.all(distances <= H_MAX + 1))
        repeat = CalibrationExampleCollector(4, self.capacities)
        simulate_race_variant(
            self.trace,
            self.workload,
            self.capacities,
            uniform_variant(),
            self.models,
            enable_diagnostics=False,
            example_collector=repeat,
        )
        self.assertEqual(len(collector), len(repeat))
        self.assertTrue(np.array_equal(collector.normalized[0], repeat.normalized[0]))

    def test_variant_specifications_round_trip(self) -> None:
        weights = np.full(len(PRIMARY_POOL), 1.0 / len(PRIMARY_POOL))
        for variant in (
            uniform_variant(),
            static_variant(weights),
            online_variant(loss="cost", eta=1.0, initialization="static", static_weights=weights),
            uniform_variant("extended"),
        ):
            restored = variant_from_spec(variant_to_spec(variant))
            self.assertEqual(restored.variant_id, variant.variant_id)
            self.assertEqual(restored.parameters(), variant.parameters())

    def test_invalid_variants_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RaceVariant(name="bad", adaptive=True, loss="rank", eta=0.0)
        with self.assertRaises(ValueError):
            RaceVariant(name="bad", weight_source="static_global", static_weights=np.ones(9))
        with self.assertRaises(ValueError):
            RaceVariant(name="bad", weight_source="uniform", loss="rank")
        with self.assertRaises(ValueError):
            simulate_race_variant(
                self.trace, self.workload, (2,), uniform_variant(), self.models
            )


if __name__ == "__main__":
    unittest.main()
