from __future__ import annotations

import itertools
import unittest

import numpy as np

from expert_analysis.fragility import Stage2BScoreArtifacts
from expert_analysis.measured_damage import predicted_domain_delta_nll
from expert_analysis.measured_damage_optimization import (
    build_measured_allocation_record,
    regime_damage_slices,
    solve_measured_damage_robust,
)
from expert_analysis.protection_allocations import verify_allocation_record
from expert_analysis.protection_optimization import (
    bits_matrix_for_allocation,
    build_expert_memory_matrix,
)
from expert_analysis.specialist_preservation import (
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
)

SHAPE = (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS), 3)
# Interesting experts: everything else has exactly zero measured damage, so an
# optimal allocation only ever protects a subset of these.
INTERESTING = ((0, 0), (1, 3), (2, 7), (5, 11), (9, 40))


def _delta() -> np.ndarray:
    rng = np.random.default_rng(29)
    delta = np.zeros(SHAPE, dtype=np.float64)
    for layer, expert in INTERESTING:
        base = np.abs(rng.normal(0.05, 0.03, size=4))
        delta[layer, expert, :, 0] = base * 3.0  # 3-bit
        delta[layer, expert, :, 1] = base  # 4-bit
        delta[layer, expert, :, 2] = base * 0.02  # 8-bit
    return delta


def _brute_force_optimum(delta: np.ndarray, regime: str, budget_count: int) -> float:
    best = np.inf
    for size in range(budget_count + 1):
        for subset in itertools.combinations(INTERESTING, size):
            bits = np.full((NUM_MOE_LAYERS, NUM_EXPERTS), 4 if regime == "4to8" else 3)
            for layer, expert in subset:
                bits[layer, expert] = 8
            predicted = predicted_domain_delta_nll(bits, delta)
            best = min(best, float(predicted.max()))
    return best


class SolverTests(unittest.TestCase):
    def test_matches_brute_force_optimum(self) -> None:
        delta = _delta()
        delta_bytes = np.ones((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.int64)
        for regime in ("4to8", "3to8"):
            for budget in (1, 2, 3):
                solution = solve_measured_damage_robust(
                    delta, regime, delta_bytes, budget
                )
                expected = _brute_force_optimum(delta, regime, budget)
                self.assertAlmostEqual(
                    solution.objective_value, expected, places=9,
                    msg=f"{regime} budget {budget}",
                )
                self.assertLessEqual(int(solution.protected.sum()), budget)

    def test_negative_benefits_are_used_exactly(self) -> None:
        delta = _delta()
        # Make protecting (0, 0) actively harmful in one domain: the 8-bit
        # damage exceeds the 4-bit damage there.
        delta[0, 0, 1, 2] = delta[0, 0, 1, 1] + 0.5
        delta_bytes = np.ones((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.int64)
        solution = solve_measured_damage_robust(delta, "4to8", delta_bytes, 5)
        self.assertEqual(int(solution.protected[0, 0]), 0)
        self.assertGreater(solution.metadata["negative_benefit_cells"], 0)

    def test_predicted_deltas_recorded_consistently(self) -> None:
        delta = _delta()
        delta_bytes = np.ones((NUM_MOE_LAYERS, NUM_EXPERTS), dtype=np.int64)
        solution = solve_measured_damage_robust(delta, "4to8", delta_bytes, 3)
        bits = bits_matrix_for_allocation(solution.protected, 4)
        predicted = predicted_domain_delta_nll(bits, delta)
        for index, domain in enumerate(STAGE2B_DOMAINS):
            self.assertAlmostEqual(
                solution.metadata["predicted_delta_nll_by_domain"][domain],
                float(predicted[index]),
                places=12,
            )

    def test_regime_slices(self) -> None:
        delta = _delta()
        base, protected, benefit = regime_damage_slices(delta, "3to8")
        np.testing.assert_array_equal(base, delta[:, :, :, 0])
        np.testing.assert_array_equal(protected, delta[:, :, :, 2])
        np.testing.assert_array_equal(benefit, base - protected)
        with self.assertRaises(ValueError):
            regime_damage_slices(delta, "2to8")


class AllocationRecordTests(unittest.TestCase):
    def _scores(self) -> Stage2BScoreArtifacts:
        rng = np.random.default_rng(5)
        specialization = rng.random((NUM_MOE_LAYERS, NUM_EXPERTS, 4))
        specialization /= specialization.sum(axis=(0, 1))
        return Stage2BScoreArtifacts(
            functional=specialization.copy(),
            functional_specialization=specialization,
            routing_specialization=specialization.copy(),
            single_domain=specialization.copy(),
            global_importance=specialization.mean(axis=2),
            calibration_indices={d: list(range(25)) for d in STAGE2B_DOMAINS},
            metadata={
                "calibration_fingerprint": "test-fingerprint",
                "score_hashes": {"functional_importance_sha256": "test"},
            },
        )

    def test_record_hash_verifies(self) -> None:
        delta = _delta()
        memory = build_expert_memory_matrix(
            [[(8, 256), (16, 128)]] * NUM_MOE_LAYERS, group_size=128
        )
        budget = memory.protection_budget_bytes(4, 0.20)
        solution = solve_measured_damage_robust(
            delta, "4to8", memory.delta_protection_bytes(4), budget
        )
        damage_record = {
            "damage_sha256": "d" * 64,
            "bf16_nll": {d: 2.0 for d in STAGE2B_DOMAINS},
        }
        record = build_measured_allocation_record(
            "4to8", 0.20, solution, self._scores(), memory, damage_record, delta
        )
        verify_allocation_record(record)
        self.assertEqual(record["method"], "measured_damage_robust")
        self.assertEqual(record["base_bits"], 4)
        self.assertLessEqual(record["used_protection_bytes"], record["budget_bytes"])
        bits = np.asarray(record["expert_bits"], dtype=np.int64)
        self.assertEqual(
            int((bits == 8).sum()), record["protected_expert_count"]
        )


if __name__ == "__main__":
    unittest.main()
