from __future__ import annotations

import unittest
import zlib

import numpy as np

from expert_analysis.measured_damage_statistics import (
    prediction_transfer_check,
    stage3_development_decision,
    stage3_development_gates,
    stage3_final_decision,
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


def _record(method: str) -> dict:
    return {
        "method": method,
        "method_label": method,
        "method_kind": "deterministic_milp",
        "regime": "4to8",
        "budget_fraction": 0.20,
    }


def _stats(method: str, levels: dict[str, float], indices):
    rng = np.random.default_rng(zlib.crc32(method.encode("utf-8")))
    bf16 = {d: np.full(COUNT, 1.0) + rng.normal(0, 1e-6, COUNT) for d in DOMAINS}
    base = {d: np.full(COUNT, 1.5) + rng.normal(0, 1e-6, COUNT) for d in DOMAINS}
    alloc = {d: np.full(COUNT, levels[d]) + rng.normal(0, 0.01, COUNT) for d in DOMAINS}
    return compute_method_statistics(_record(method), alloc, bf16, base, indices)


def _panel(primary_levels: dict[str, float]):
    indices = _indices()
    methods = {
        "measured_damage_robust": _stats(
            "measured_damage_robust", primary_levels, indices
        ),
        "robust_functional": _stats(
            "robust_functional", {d: 1.20 for d in DOMAINS}, indices
        ),
        "fragility_robust": _stats(
            "fragility_robust", {d: 1.18 for d in DOMAINS}, indices
        ),
        "global_importance": _stats(
            "global_importance",
            {"general": 1.16, "math": 1.10, "coding": 1.08, "reasoning": 1.09},
            indices,
        ),
        "average_specialization": _stats(
            "average_specialization",
            {"general": 1.10, "math": 1.12, "coding": 1.15, "reasoning": 1.10},
            indices,
        ),
    }
    for seed in (1001, 1002, 1003, 1004, 1005):
        methods[f"random_seed{seed}"] = _stats(
            f"random_seed{seed}", {d: 1.25 for d in DOMAINS}, indices
        )
    return methods


def _gates(methods):
    return stage3_development_gates(
        methods["measured_damage_robust"],
        methods["robust_functional"],
        methods["fragility_robust"],
        [methods[f"random_seed{seed}"] for seed in (1001, 1002, 1003, 1004, 1005)],
        methods["global_importance"],
        methods["average_specialization"],
    )


class DevelopmentGateTests(unittest.TestCase):
    def test_all_gates_pass_when_primary_dominates(self) -> None:
        gates = _gates(_panel({d: 1.05 for d in DOMAINS}))
        for key in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"):
            self.assertTrue(gates[key]["passed"], key)
        self.assertTrue(gates["all_passed"])

    def test_gate_a_requires_beating_both_prior_robust_methods(self) -> None:
        # Worse than Fragility-Robust (1.18) but better than everything else
        # in the worst domain.
        gates = _gates(
            _panel({"general": 1.19, "math": 1.05, "coding": 1.05, "reasoning": 1.05})
        )
        self.assertFalse(gates["gate_a"]["passed"])

    def test_gate_e_fails_on_large_mean_tradeoff(self) -> None:
        # Excellent worst domain, terrible mean relative to the better simple
        # baseline mean.
        gates = _gates(
            _panel({"general": 1.13, "math": 1.13, "coding": 1.13,
                    "reasoning": 1.13})
        )
        self.assertTrue(gates["gate_c"]["passed"])
        self.assertFalse(gates["gate_e"]["passed"])

    def test_decision_requires_a_passing_regime(self) -> None:
        passing = _gates(_panel({d: 1.05 for d in DOMAINS}))
        failing = _gates(
            _panel({"general": 1.30, "math": 1.05, "coding": 1.05, "reasoning": 1.05})
        )
        decision = stage3_development_decision(
            {"4to8": passing, "3to8": failing}, ["4to8", "3to8"]
        )
        self.assertEqual(decision["decision"], "FINAL_CONFIRMATION_GO")
        self.assertEqual(decision["authorized_regimes"], ["4to8"])
        decision = stage3_development_decision({"3to8": failing}, ["3to8"])
        self.assertEqual(decision["decision"], "MEASURED_DAMAGE_NO_GO")
        self.assertEqual(decision["authorized_regimes"], [])

    def test_final_decision_labels(self) -> None:
        strong = {"strong_success": True, "qualified_success": False}
        negative = {"strong_success": False, "qualified_success": False}
        self.assertEqual(
            stage3_final_decision({"4to8": strong, "3to8": negative})["decision"],
            "STRONG SUCCESS",
        )
        self.assertEqual(
            stage3_final_decision({"4to8": negative})["decision"],
            "NEGATIVE RESULT",
        )


class TransferCheckTests(unittest.TestCase):
    def test_perfect_rank_transfer(self) -> None:
        predicted = {
            f"alloc{i}": {d: 0.01 * (i + 1) * (j + 1) for j, d in enumerate(DOMAINS)}
            for i in range(5)
        }
        realized = {
            slug: {d: value * 0.9 for d, value in values.items()}
            for slug, values in predicted.items()
        }
        report = prediction_transfer_check(predicted, realized)
        self.assertAlmostEqual(report["worst_delta_spearman"], 1.0)
        for domain in DOMAINS:
            self.assertAlmostEqual(report["spearman_by_domain"][domain], 1.0)
        self.assertAlmostEqual(
            report["total_predicted_over_realized_ratio"], 1.0 / 0.9, places=9
        )

    def test_requires_three_allocations(self) -> None:
        values = {"a": {d: 1.0 for d in DOMAINS}, "b": {d: 2.0 for d in DOMAINS}}
        with self.assertRaises(ValueError):
            prediction_transfer_check(values, values)


if __name__ == "__main__":
    unittest.main()
