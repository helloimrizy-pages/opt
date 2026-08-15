from __future__ import annotations

import unittest
import zlib

import numpy as np

from expert_analysis.fragility_statistics import (
    fragility_transfer_check,
    protection_shift_analysis,
    stage2c_development_decision,
    stage2c_development_gates,
    stage2c_final_decision,
    stage2c_final_regime_assessment,
)
from expert_analysis.protection_statistics import (
    build_replicate_indices,
    compute_method_statistics,
)

DOMAINS = ("general", "math", "coding", "reasoning")
COUNT = 8
REPLICATES = 200


def _indices():
    return build_replicate_indices(
        {d: COUNT for d in DOMAINS}, replicates=REPLICATES, seed=123
    )


def _record(method: str, regime: str = "4to8", budget: float = 0.20) -> dict:
    return {
        "method": method,
        "method_label": method,
        "method_kind": "deterministic_milp",
        "regime": regime,
        "budget_fraction": budget,
    }


def _stats(
    method: str,
    levels: dict[str, float],
    bf16_level: float = 1.0,
    base_level: float = 1.5,
    indices=None,
    jitter: float = 0.01,
    regime: str = "4to8",
    budget: float = 0.20,
):
    rng = np.random.default_rng(zlib.crc32(method.encode("utf-8")))
    bf16 = {d: np.full(COUNT, bf16_level) + rng.normal(0, 1e-6, COUNT) for d in DOMAINS}
    base = {d: np.full(COUNT, base_level) + rng.normal(0, 1e-6, COUNT) for d in DOMAINS}
    alloc = {
        d: np.full(COUNT, levels[d]) + rng.normal(0, jitter, COUNT) for d in DOMAINS
    }
    return compute_method_statistics(
        _record(method, regime, budget), alloc, bf16, base,
        indices if indices is not None else _indices(),
    )


def _standard_panel(fr_levels, indices=None, regime="4to8", budget=0.20):
    """Fragility-Robust plus every comparator the gates need."""

    indices = indices if indices is not None else _indices()
    methods = {
        "fragility_robust": _stats(
            "fragility_robust", fr_levels, indices=indices, regime=regime,
            budget=budget,
        ),
        "robust_functional": _stats(
            "robust_functional",
            {d: 1.20 for d in DOMAINS}, indices=indices, regime=regime, budget=budget,
        ),
        "global_importance": _stats(
            "global_importance",
            {"general": 1.18, "math": 1.10, "coding": 1.08, "reasoning": 1.09},
            indices=indices, regime=regime, budget=budget,
        ),
        "average_specialization": _stats(
            "average_specialization",
            {"general": 1.16, "math": 1.12, "coding": 1.10, "reasoning": 1.08},
            indices=indices, regime=regime, budget=budget,
        ),
        "robust_routing": _stats(
            "robust_routing",
            {d: 1.22 for d in DOMAINS}, indices=indices, regime=regime, budget=budget,
        ),
    }
    for seed in (1001, 1002, 1003, 1004, 1005):
        methods[f"random_seed{seed}"] = _stats(
            f"random_seed{seed}",
            {d: 1.25 for d in DOMAINS}, indices=indices, regime=regime, budget=budget,
        )
    return methods


class BootstrapWorstRecomputationTests(unittest.TestCase):
    def test_worst_is_recomputed_inside_each_replicate(self) -> None:
        indices = _indices()
        rng = np.random.default_rng(0)
        bf16 = {d: np.full(COUNT, 1.0) for d in DOMAINS}
        base = {d: np.full(COUNT, 1.5) for d in DOMAINS}
        # Two domains alternate which one is worst example by example.
        alloc = {
            "general": np.where(np.arange(COUNT) % 2 == 0, 1.4, 1.0),
            "math": np.where(np.arange(COUNT) % 2 == 0, 1.0, 1.4),
            "coding": np.full(COUNT, 1.05),
            "reasoning": np.full(COUNT, 1.05),
        }
        alloc = {d: v + rng.normal(0, 1e-9, COUNT) for d, v in alloc.items()}
        stats = compute_method_statistics(
            _record("flip"), alloc, bf16, base, indices
        )
        expected = stats.replicate_relative.max(axis=1)
        np.testing.assert_allclose(stats.replicate_worst, expected)
        # The per-replicate worst must not always come from the frozen
        # point-estimate worst domain.
        frozen_index = DOMAINS.index(stats.worst_domain)
        frozen_only = stats.replicate_relative[:, frozen_index]
        self.assertTrue(np.any(stats.replicate_worst > frozen_only + 1e-12))


class DevelopmentGateTests(unittest.TestCase):
    def test_all_gates_pass_for_dominant_fragility_robust(self) -> None:
        methods = _standard_panel(
            {"general": 1.05, "math": 1.04, "coding": 1.05, "reasoning": 1.04}
        )
        gates = stage2c_development_gates(
            methods["fragility_robust"],
            methods["robust_functional"],
            [methods[f"random_seed{s}"] for s in (1001, 1002, 1003, 1004, 1005)],
            methods["global_importance"],
            methods["average_specialization"],
        )
        for key in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"):
            self.assertTrue(gates[key]["passed"], key)
        self.assertTrue(gates["all_passed"])

    def test_gate_a_fails_when_worse_than_robust_functional(self) -> None:
        methods = _standard_panel({d: 1.30 for d in DOMAINS})
        gates = stage2c_development_gates(
            methods["fragility_robust"],
            methods["robust_functional"],
            [methods[f"random_seed{s}"] for s in (1001, 1002, 1003, 1004, 1005)],
            methods["global_importance"],
            methods["average_specialization"],
        )
        self.assertFalse(gates["gate_a"]["passed"])
        self.assertFalse(gates["all_passed"])

    def test_gate_d_fails_without_broad_recovery(self) -> None:
        # Above the uniform base level (1.5) in two domains: recovery negative.
        methods = _standard_panel(
            {"general": 1.6, "math": 1.7, "coding": 1.05, "reasoning": 1.04}
        )
        gates = stage2c_development_gates(
            methods["fragility_robust"],
            methods["robust_functional"],
            [methods[f"random_seed{s}"] for s in (1001, 1002, 1003, 1004, 1005)],
            methods["global_importance"],
            methods["average_specialization"],
        )
        self.assertFalse(gates["gate_d"]["passed"])

    def test_gate_e_uses_lower_comparator_and_relative_tolerance(self) -> None:
        methods = _standard_panel(
            {"general": 1.155, "math": 1.155, "coding": 1.155, "reasoning": 1.155}
        )
        gates = stage2c_development_gates(
            methods["fragility_robust"],
            methods["robust_functional"],
            [methods[f"random_seed{s}"] for s in (1001, 1002, 1003, 1004, 1005)],
            methods["global_importance"],
            methods["average_specialization"],
        )
        comparator = gates["gate_e"]["comparator_mean_relative_delta"]
        self.assertAlmostEqual(
            comparator,
            min(
                methods["global_importance"].mean_relative_delta,
                methods["average_specialization"].mean_relative_delta,
            ),
        )
        # Mean ~0.155 vs comparator ~0.1125: more than 10% relatively worse.
        self.assertFalse(gates["gate_e"]["passed"])

    def test_gate_e_epsilon_guards_near_zero_comparator(self) -> None:
        indices = _indices()
        methods = _standard_panel(
            {d: 1.001 for d in DOMAINS}, indices=indices
        )
        # A comparator that essentially matches BF16: mean near zero.
        methods["global_importance"] = _stats(
            "global_importance", {d: 1.0 for d in DOMAINS}, indices=indices,
            jitter=0.0,
        )
        gates = stage2c_development_gates(
            methods["fragility_robust"],
            methods["robust_functional"],
            [methods[f"random_seed{s}"] for s in (1001, 1002, 1003, 1004, 1005)],
            methods["global_importance"],
            methods["average_specialization"],
        )
        self.assertFalse(gates["gate_e"]["passed"])
        self.assertGreater(gates["gate_e"]["relative_worseness"], 0.10)


class DevelopmentDecisionTests(unittest.TestCase):
    def test_go_requires_at_least_one_passing_regime(self) -> None:
        decision = stage2c_development_decision(
            {"4to8": {"all_passed": False}, "3to8": {"all_passed": True}}
        )
        self.assertEqual(decision["decision"], "FINAL_CONFIRMATION_GO")
        self.assertEqual(decision["authorized_regimes"], ["3to8"])
        self.assertEqual(decision["regimes_evaluated"]["4to8"], "FAIL")

    def test_no_go_when_both_fail(self) -> None:
        decision = stage2c_development_decision(
            {"4to8": {"all_passed": False}, "3to8": {"all_passed": False}}
        )
        self.assertEqual(decision["decision"], "FRAGILITY_ROBUST_NO_GO")
        self.assertEqual(decision["authorized_regimes"], [])


class FinalAssessmentTests(unittest.TestCase):
    def _build(self, fr_levels_by_budget, ci_wins_by_budget=None):
        indices = _indices()
        statistics = {}
        comparisons_average = {}
        comparisons_global = {}
        for budget in (0.05, 0.10, 0.20, 0.30):
            methods = _standard_panel(
                fr_levels_by_budget[budget], indices=indices, budget=budget
            )
            statistics[("4to8", budget)] = methods
            fr = methods["fragility_robust"]
            average = methods["average_specialization"]
            globl = methods["global_importance"]
            forced_ci = (
                ci_wins_by_budget[budget] if ci_wins_by_budget is not None else None
            )
            ci_favors = (
                forced_ci
                if forced_ci is not None
                else bool(
                    np.quantile(fr.replicate_worst - average.replicate_worst, 0.975)
                    < 0
                )
            )
            comparisons_average[budget] = {
                "difference": fr.worst_relative_delta - average.worst_relative_delta,
                "ci_favors_first": ci_favors,
            }
            comparisons_global[budget] = {
                "difference": fr.worst_relative_delta - globl.worst_relative_delta,
            }
        return statistics, comparisons_average, comparisons_global

    def test_strong_success_when_dominant_everywhere(self) -> None:
        levels = {d: 1.02 for d in DOMAINS}
        statistics, vs_average, vs_global = self._build(
            {b: levels for b in (0.05, 0.10, 0.20, 0.30)}
        )
        assessment = stage2c_final_regime_assessment(
            statistics, vs_average, vs_global, "4to8", (0.05, 0.10, 0.20, 0.30)
        )
        self.assertTrue(assessment["strong_success"])
        self.assertEqual(assessment["budgets_with_all_four_point_wins"], 4)
        decision = stage2c_final_decision({"4to8": assessment})
        self.assertEqual(decision["decision"], "STRONG SUCCESS")

    def test_qualified_when_only_two_budget_wins(self) -> None:
        good = {d: 1.02 for d in DOMAINS}
        bad = {"general": 1.17, "math": 1.02, "coding": 1.02, "reasoning": 1.02}
        statistics, vs_average, vs_global = self._build(
            {0.05: good, 0.10: good, 0.20: bad, 0.30: bad},
            ci_wins_by_budget={0.05: True, 0.10: False, 0.20: False, 0.30: False},
        )
        assessment = stage2c_final_regime_assessment(
            statistics, vs_average, vs_global, "4to8", (0.05, 0.10, 0.20, 0.30)
        )
        self.assertFalse(assessment["strong_success"])
        self.assertTrue(assessment["qualified_success"])
        decision = stage2c_final_decision({"4to8": assessment})
        self.assertEqual(decision["decision"], "SUCCESS WITH QUALIFICATIONS")

    def test_negative_when_never_beating_baselines(self) -> None:
        bad = {d: 1.30 for d in DOMAINS}
        statistics, vs_average, vs_global = self._build(
            {b: bad for b in (0.05, 0.10, 0.20, 0.30)}
        )
        assessment = stage2c_final_regime_assessment(
            statistics, vs_average, vs_global, "4to8", (0.05, 0.10, 0.20, 0.30)
        )
        self.assertFalse(assessment["strong_success"])
        self.assertFalse(assessment["qualified_success"])
        decision = stage2c_final_decision({"4to8": assessment})
        self.assertEqual(decision["decision"], "NEGATIVE RESULT")

    def test_systematic_negative_recovery_blocks_success(self) -> None:
        # General sits above the uniform base (1.5) at three of four budgets.
        good = {d: 1.02 for d in DOMAINS}
        harmed = {"general": 1.6, "math": 1.02, "coding": 1.02, "reasoning": 1.02}
        statistics, vs_average, vs_global = self._build(
            {0.05: harmed, 0.10: harmed, 0.20: harmed, 0.30: good}
        )
        assessment = stage2c_final_regime_assessment(
            statistics, vs_average, vs_global, "4to8", (0.05, 0.10, 0.20, 0.30)
        )
        self.assertEqual(
            assessment["systematic_negative_recovery_domains"], ["general"]
        )
        self.assertFalse(assessment["strong_success"])
        self.assertFalse(assessment["qualified_success"])


class MechanismAnalysisTests(unittest.TestCase):
    def test_protection_shift_spearman_perfect_alignment(self) -> None:
        q_norm = {
            "4to8": {"general": 2.0, "math": 1.0, "coding": 0.6, "reasoning": 0.4}
        }
        fragile = {
            "4to8": {"general": 0.8, "math": 0.6, "coding": 0.5, "reasoning": 0.4}
        }
        robust = {
            "4to8": {"general": 0.5, "math": 0.5, "coding": 0.5, "reasoning": 0.5}
        }
        result = protection_shift_analysis(fragile, robust, q_norm)
        self.assertAlmostEqual(
            result["regimes"]["4to8"]["spearman_fragility_vs_shift"], 1.0
        )
        self.assertTrue(result["descriptive_only"])

    def test_fragility_transfer_identifies_most_fragile_domain(self) -> None:
        q_norm = {
            "4to8": {"general": 2.0, "math": 1.0, "coding": 0.6, "reasoning": 0.4}
        }
        observed = {
            "4to8": {"general": 0.09, "math": 0.05, "coding": 0.03, "reasoning": 0.02}
        }
        result = fragility_transfer_check(q_norm, observed)
        entry = result["regimes"]["4to8"]
        self.assertAlmostEqual(entry["spearman"], 1.0)
        self.assertEqual(entry["most_fragile_domain_calibration"], "general")
        self.assertEqual(entry["most_degraded_domain_final"], "general")
        self.assertTrue(entry["most_fragile_domain_matches"])


if __name__ == "__main__":
    unittest.main()
