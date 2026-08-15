from __future__ import annotations

import unittest

import numpy as np

from expert_analysis.protection_statistics import (
    MethodStatistics,
    build_replicate_indices,
    compute_method_statistics,
    development_decision,
    development_gates,
    final_decision,
    final_regime_assessment,
    paired_comparison,
    random_baseline_statistics,
)
from expert_analysis.specialist_preservation import STAGE2B_DOMAINS


def record(method: str, kind: str = "deterministic_milp") -> dict:
    return {
        "method": method,
        "method_label": method,
        "method_kind": kind,
        "regime": "4to8",
        "budget_fraction": 0.20,
    }


def constant_nll(value: float, count: int = 8) -> np.ndarray:
    return np.full(count, value, dtype=np.float64)


def make_statistics(
    method: str,
    relative_by_domain: dict[str, float],
    base_uplift: float = 0.05,
    count: int = 8,
) -> MethodStatistics:
    baseline = {d: constant_nll(1.0, count) for d in STAGE2B_DOMAINS}
    allocation = {
        d: constant_nll(1.0 + relative_by_domain[d], count) for d in STAGE2B_DOMAINS
    }
    base_uniform = {d: constant_nll(1.0 + base_uplift, count) for d in STAGE2B_DOMAINS}
    indices = build_replicate_indices(
        {d: count for d in STAGE2B_DOMAINS}, replicates=200, seed=7
    )
    return compute_method_statistics(
        record(method), allocation, baseline, base_uniform, indices
    )


class MetricTests(unittest.TestCase):
    def test_relative_delta_and_recovery_formulas(self) -> None:
        baseline = {d: constant_nll(2.0) for d in STAGE2B_DOMAINS}
        allocation = {d: constant_nll(2.2) for d in STAGE2B_DOMAINS}
        base_uniform = {d: constant_nll(2.5) for d in STAGE2B_DOMAINS}
        indices = build_replicate_indices(
            {d: 8 for d in STAGE2B_DOMAINS}, replicates=50, seed=3
        )
        statistics = compute_method_statistics(
            record("m"), allocation, baseline, base_uniform, indices
        )
        for domain in STAGE2B_DOMAINS:
            self.assertAlmostEqual(statistics.relative_delta[domain], 0.1, places=12)
            self.assertAlmostEqual(statistics.delta_nll[domain], 0.2, places=12)
            self.assertAlmostEqual(statistics.recovery[domain], 0.3, places=12)
        self.assertAlmostEqual(statistics.worst_relative_delta, 0.1, places=12)
        self.assertAlmostEqual(statistics.mean_relative_delta, 0.1, places=12)

    def test_worst_domain_is_recomputed_inside_each_replicate(self) -> None:
        rng = np.random.default_rng(0)
        count = 12
        baseline = {d: constant_nll(1.0, count) for d in STAGE2B_DOMAINS}
        # Two domains have identical means but different example-level noise, so
        # the replicate-level worst domain flips between them.
        allocation = {
            "general": 1.05 + 0.2 * rng.standard_normal(count),
            "math": 1.05 + 0.2 * rng.standard_normal(count),
            "coding": constant_nll(1.01, count),
            "reasoning": constant_nll(1.01, count),
        }
        base_uniform = {d: constant_nll(1.2, count) for d in STAGE2B_DOMAINS}
        indices = build_replicate_indices(
            {d: count for d in STAGE2B_DOMAINS}, replicates=400, seed=11
        )
        statistics = compute_method_statistics(
            record("m"), allocation, baseline, base_uniform, indices
        )
        np.testing.assert_allclose(
            statistics.replicate_worst,
            statistics.replicate_relative.max(axis=1),
        )
        general_wins = np.sum(
            statistics.replicate_relative[:, 0] > statistics.replicate_relative[:, 1]
        )
        self.assertGreater(int(general_wins), 0)
        self.assertLess(int(general_wins), 400)

    def test_replicate_indices_are_deterministic_and_paired(self) -> None:
        first = build_replicate_indices({d: 5 for d in STAGE2B_DOMAINS}, 10, 42)
        second = build_replicate_indices({d: 5 for d in STAGE2B_DOMAINS}, 10, 42)
        for domain in STAGE2B_DOMAINS:
            np.testing.assert_array_equal(first[domain], second[domain])

    def test_paired_comparison_sign_convention(self) -> None:
        better = make_statistics("better", {d: 0.01 for d in STAGE2B_DOMAINS})
        worse = make_statistics("worse", {d: 0.03 for d in STAGE2B_DOMAINS})
        comparison = paired_comparison(better, worse, "worst_relative_delta")
        self.assertLess(comparison["difference"], 0)
        self.assertTrue(comparison["favors_first"])
        self.assertTrue(comparison["ci_favors_first"])


class GateTests(unittest.TestCase):
    def build_panel(self, robust_relative: dict[str, float]):
        robust = make_statistics("robust_functional", robust_relative)
        randoms = [
            make_statistics(f"random_seed{seed}", {d: 0.04 for d in STAGE2B_DOMAINS})
            for seed in (1001, 1002, 1003, 1004, 1005)
        ]
        global_importance = make_statistics(
            "global_importance", {d: 0.03 for d in STAGE2B_DOMAINS}
        )
        average = make_statistics(
            "average_specialization",
            {"general": 0.025, "math": 0.012, "coding": 0.012, "reasoning": 0.012},
        )
        return robust, randoms, global_importance, average

    def test_all_gates_pass_for_a_dominating_robust_method(self) -> None:
        robust, randoms, global_importance, average = self.build_panel(
            {d: 0.01 for d in STAGE2B_DOMAINS}
        )
        gates = development_gates(robust, randoms, global_importance, average)
        for name in ("gate_a", "gate_b", "gate_c", "gate_d"):
            self.assertTrue(gates[name]["passed"], name)
        self.assertTrue(gates["all_passed"])
        decision = development_decision({"4to8": gates})
        self.assertEqual(decision["decision"], "FULL_EVALUATION_GO")

    def test_gate_a_and_b_fail_when_robust_is_worse(self) -> None:
        robust, randoms, global_importance, average = self.build_panel(
            {d: 0.05 for d in STAGE2B_DOMAINS}
        )
        gates = development_gates(robust, randoms, global_importance, average)
        self.assertFalse(gates["gate_a"]["passed"])
        self.assertFalse(gates["gate_b"]["passed"])
        self.assertFalse(gates["all_passed"])
        decision = development_decision({"4to8": gates, "3to8": gates})
        self.assertEqual(decision["decision"], "ROBUST_PRESERVATION_NO_GO")

    def test_gate_c_tolerance_uses_absolute_floor_near_zero(self) -> None:
        robust, randoms, global_importance, _ = self.build_panel(
            {d: 0.01 for d in STAGE2B_DOMAINS}
        )
        near_zero_average = make_statistics(
            "average_specialization", {d: 0.0 for d in STAGE2B_DOMAINS}
        )
        gates = development_gates(robust, randoms, global_importance, near_zero_average)
        self.assertTrue(gates["gate_c"]["absolute_floor_applied"])
        self.assertFalse(gates["gate_c"]["passed"])

    def test_gate_d_requires_three_positive_recovery_domains(self) -> None:
        robust = make_statistics(
            "robust_functional",
            {"general": 0.08, "math": 0.08, "coding": 0.01, "reasoning": 0.01},
            base_uplift=0.05,
        )
        _, randoms, global_importance, average = self.build_panel(
            {d: 0.01 for d in STAGE2B_DOMAINS}
        )
        gates = development_gates(robust, randoms, global_importance, average)
        self.assertFalse(gates["gate_d"]["passed"])


class FinalDecisionTests(unittest.TestCase):
    def build_final_panel(self, robust_level: float):
        statistics = {}
        comparisons = {}
        for budget in (0.05, 0.10, 0.20, 0.30):
            methods = {
                "robust_functional": make_statistics(
                    "robust_functional", {d: robust_level for d in STAGE2B_DOMAINS}
                ),
                "average_specialization": make_statistics(
                    "average_specialization", {d: 0.03 for d in STAGE2B_DOMAINS}
                ),
                "global_importance": make_statistics(
                    "global_importance", {d: 0.03 for d in STAGE2B_DOMAINS}
                ),
            }
            for seed in (1001, 1002, 1003, 1004, 1005):
                methods[f"random_seed{seed}"] = make_statistics(
                    f"random_seed{seed}", {d: 0.04 for d in STAGE2B_DOMAINS}
                )
            statistics[("4to8", budget)] = methods
            comparisons[budget] = paired_comparison(
                methods["robust_functional"],
                methods["average_specialization"],
                "worst_relative_delta",
            )
        return statistics, comparisons

    def test_strong_success_when_robust_dominates(self) -> None:
        statistics, comparisons = self.build_final_panel(robust_level=0.01)
        assessment = final_regime_assessment(
            statistics, comparisons, "4to8", (0.05, 0.10, 0.20, 0.30)
        )
        self.assertTrue(assessment["strong_success"])
        decision = final_decision({"4to8": assessment})
        self.assertEqual(decision["decision"], "STRONG SUCCESS")

    def test_negative_result_when_robust_never_wins(self) -> None:
        statistics, comparisons = self.build_final_panel(robust_level=0.06)
        assessment = final_regime_assessment(
            statistics, comparisons, "4to8", (0.05, 0.10, 0.20, 0.30)
        )
        self.assertFalse(assessment["strong_success"])
        self.assertFalse(assessment["qualified_success"])
        decision = final_decision({"4to8": assessment})
        self.assertEqual(decision["decision"], "NEGATIVE RESULT")

    def test_random_summary_reports_all_five_seeds(self) -> None:
        randoms = [
            make_statistics(f"random_seed{seed}", {d: 0.02 for d in STAGE2B_DOMAINS})
            for seed in (1001, 1002, 1003, 1004, 1005)
        ]
        summary = random_baseline_statistics(randoms)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(len(summary["individual_worst_relative_delta"]), 5)


if __name__ == "__main__":
    unittest.main()
