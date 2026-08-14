from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from expert_analysis.surrogate_validation import (
    PilotValidationData,
    analyze_fixed_surrogates,
    decision_from_analyses,
    create_surrogate_figures,
    grouped_bootstrap_indices,
    specificity_contrast,
    top_domain_accuracy,
    write_pilot_raw_npz,
    write_surrogate_summary,
    write_validation_tables,
)


DOMAINS = ("general", "math", "coding", "reasoning")


class SurrogateValidationTests(unittest.TestCase):
    def data(self) -> tuple[PilotValidationData, np.ndarray]:
        expert = np.arange(16, dtype=np.float64)[:, None]
        # Domain slopes vary with expert, so target-minus-other specificity is not
        # constant while every domain retains a positive across-expert association.
        aod = (
            0.1
            + expert * np.asarray([[0.012, 0.009, 0.015, 0.006]])
            + np.asarray([[0.03, 0.01, 0.04, 0.02]])
        )
        actual = aod * 0.02 - 0.002
        functional = np.roll(aod[::-1], shift=1, axis=1) + 0.01
        routing = np.roll(functional, shift=1, axis=1)
        distortion = np.linspace(0.01, 0.02, 16)
        per_example = np.repeat(actual[:, :, None], 100, axis=2)
        targets = tuple(DOMAINS[index % 4] for index in range(16))
        value = PilotValidationData(
            intervention_ids=tuple(f"expert_{index}" for index in range(16)),
            pair_ids=tuple(f"pair_{index // 2}" for index in range(16)),
            roles=tuple("specialist" if index % 2 == 0 else "control" for index in range(16)),
            target_domains=targets,
            layers=np.arange(16, dtype=np.int64),
            expert_ids=np.arange(16, dtype=np.int64),
            domains=DOMAINS,
            actual_delta_nll=actual,
            per_example_delta_nll=per_example,
            functional_importance=functional,
            routing_importance=routing,
            quantization_distortion=distortion,
            weight_risk_functional=functional * distortion[:, None],
            weight_risk_routing=routing * distortion[:, None],
            stage1_metadata={},
        )
        value.validate()
        return value, aod

    def test_grouped_bootstrap_is_deterministic_and_keeps_four_domains(self) -> None:
        first = grouped_bootstrap_indices(16, replicates=1000, seed=42)
        second = grouped_bootstrap_indices(16, replicates=1000, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (1000, 16))
        data, _ = self.data()
        self.assertEqual(data.actual_delta_nll[first[0]].shape, (16, 4))

    def test_specificity_is_target_minus_arithmetic_mean_other(self) -> None:
        values = np.asarray([[4.0, 1.0, 2.0, 3.0], [2.0, 5.0, 8.0, 11.0]])
        observed = specificity_contrast(values, np.asarray([0, 2]))
        np.testing.assert_allclose(observed, [2.0, 2.0])

    def test_top_domain_accuracy(self) -> None:
        predicted = np.asarray([[1, 4, 2, 3], [9, 1, 2, 3]])
        actual = np.asarray([[0, 5, 2, 1], [1, 2, 8, 3]])
        self.assertEqual(top_domain_accuracy(predicted, actual), 0.5)

    def test_identical_aod_passes_all_preregistered_gates(self) -> None:
        data, aod = self.data()
        analysis = analyze_fixed_surrogates(
            data,
            {
                "uod": aod * 2,
                "reod": aod * 3,
                "apd": aod * 4,
                "aod": aod,
            },
            primary_name="aod",
        )
        self.assertAlmostEqual(analysis["surrogates"]["aod"]["overall"]["spearman"], 1.0)
        self.assertTrue(analysis["primary_passed"])
        for name in ("gate_a", "gate_b", "gate_c", "gate_d", "gate_e"):
            self.assertTrue(analysis["primary_gates"][name]["passed"], name)
        self.assertEqual(
            analysis["surrogates"]["aod"]["within_expert_domain_ranking"][
                "top_domain_accuracy"
            ],
            1.0,
        )

    def test_gate_thresholds_are_strict(self) -> None:
        # Exercise decision logic directly at the preregistered boundaries.
        base_gates = {
            "gate_a": {"passed": False},
            "gate_b": {"passed": True},
            "gate_c": {"passed": True},
            "gate_d": {"passed": True},
            "gate_e": {"passed": True},
            "all_required_gates_passed": False,
        }
        aod = {
            "primary_surrogate": "aod",
            "primary_passed": False,
            "primary_gates": base_gates,
        }
        gqs = {
            "primary_surrogate": "gqs",
            "primary_passed": True,
            "primary_gates": {
                **base_gates,
                "gate_a": {"passed": True},
                "all_required_gates_passed": True,
            },
        }
        decision = decision_from_analyses(aod, gqs, audit_passed=True)
        self.assertEqual(decision["decision"], "SURROGATE_GO_GRADIENT")
        self.assertTrue(decision["gradient_fallback_triggered"])

    def test_audit_failure_blocks_go(self) -> None:
        gates = {
            **{f"gate_{name}": {"passed": True} for name in "abcde"},
            "all_required_gates_passed": True,
        }
        aod = {
            "primary_surrogate": "aod",
            "primary_passed": True,
            "primary_gates": gates,
        }
        decision = decision_from_analyses(aod, None, audit_passed=False)
        self.assertEqual(decision["decision"], "SURROGATE_NO_GO")
        self.assertFalse(decision["full_cost_matrix_authorized"])

    def test_report_tables_raw_npz_and_four_figure_pairs(self) -> None:
        data, aod = self.data()
        activation = {
            "gated_delta_squared": aod[None],
            "gated_baseline_squared": np.ones((1, 16, 4)),
            "ungated_delta_squared": np.ones((1, 16, 4)),
            "route_counts": np.ones((1, 16, 4), dtype=np.int64),
            "layer_energy": np.ones((1, 16, 4)),
            "domain_token_count": np.full((1, 16, 4), 6400),
            "aod": aod[None],
            "reod": aod[None],
            "apd": (aod / 6400)[None],
            "uod": np.ones((1, 16, 4)),
            "unobserved": np.zeros((1, 16, 4), dtype=bool),
        }
        scores = {
            "weight_risk_functional": data.weight_risk_functional,
            "weight_risk_routing": data.weight_risk_routing,
            "functional_importance": data.functional_importance,
            "routing_importance": data.routing_importance,
            "uod": activation["uod"][0],
            "reod": activation["reod"][0],
            "apd": activation["apd"][0],
            "aod": aod,
        }
        analysis = analyze_fixed_surrogates(
            data,
            {name: scores[name] for name in ("uod", "reod", "apd", "aod")},
            primary_name="aod",
        )
        decision = decision_from_analyses(analysis, None, audit_passed=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_validation_tables(root, data, scores, analysis)
            write_pilot_raw_npz(root / "pilot_surrogate_raw.npz", data, activation)
            figures = create_surrogate_figures(
                root, data, scores, analysis, primary_name="aod"
            )
            write_surrogate_summary(
                root / "SUMMARY.md",
                {"surrogate_decision": decision, "aod_analysis": analysis},
            )
            expected = [
                "pilot_surrogate_values.csv",
                "surrogate_comparison.csv",
                "surrogate_specificity.csv",
                "within_expert_domain_rankings.csv",
                "domain_specific_correlations.csv",
                "pilot_surrogate_raw.npz",
                "SUMMARY.md",
            ]
            self.assertTrue(all((root / name).stat().st_size > 0 for name in expected))
            self.assertEqual(len(figures), 8)
            self.assertTrue(all(path.stat().st_size > 0 for path in figures))


if __name__ == "__main__":
    unittest.main()
