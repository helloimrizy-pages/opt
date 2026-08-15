from __future__ import annotations

import unittest

import numpy as np

from expert_analysis.specialist_preservation import (
    CALIBRATION_EXAMPLES_PER_DOMAIN,
    CALIBRATION_SEED,
    STAGE2B_DOMAINS,
    CalibrationSelection,
    build_importance_tensor,
    layer_normalized_importance,
    normalized_specialization,
    select_calibration_indices,
    specialist_coverage,
    specialization_margins,
)


class CalibrationSelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_sorted_and_within_budget(self) -> None:
        first = select_calibration_indices()
        second = select_calibration_indices()
        total = 0
        for domain in STAGE2B_DOMAINS:
            np.testing.assert_array_equal(first.indices[domain], second.indices[domain])
            values = first.indices[domain]
            self.assertEqual(len(values), CALIBRATION_EXAMPLES_PER_DOMAIN)
            self.assertEqual(len(np.unique(values)), len(values))
            np.testing.assert_array_equal(values, np.sort(values))
            self.assertTrue(np.all((values >= 0) & (values < 100)))
            total += len(values)
        self.assertEqual(total, 100)
        self.assertEqual(first.seed, CALIBRATION_SEED)

    def test_selection_differs_across_domains(self) -> None:
        selection = select_calibration_indices()
        sets = [tuple(selection.indices[d].tolist()) for d in STAGE2B_DOMAINS]
        self.assertEqual(len(set(sets)), len(sets))

    def test_equal_calibration_budget_is_enforced(self) -> None:
        selection = select_calibration_indices()
        broken = CalibrationSelection(
            seed=selection.seed,
            per_domain=selection.per_domain,
            indices={
                domain: values[:-1] if domain == "math" else values
                for domain, values in selection.indices.items()
            },
        )
        with self.assertRaises(ValueError):
            broken.validate(100)


class NormalizationTests(unittest.TestCase):
    def test_layer_normalization_gives_equal_layer_mass(self) -> None:
        raw = np.asarray([[1.0, 3.0], [10.0, 30.0]])
        normalized = layer_normalized_importance(raw)
        np.testing.assert_allclose(normalized.sum(axis=1), [0.5, 0.5])
        np.testing.assert_allclose(normalized.sum(), 1.0)
        np.testing.assert_allclose(normalized[0], normalized[1])

    def test_layer_normalization_rejects_empty_layer(self) -> None:
        with self.assertRaises(ValueError):
            layer_normalized_importance(np.asarray([[0.0, 0.0], [1.0, 1.0]]))

    def test_importance_tensor_sums_to_one_per_domain(self) -> None:
        rng = np.random.default_rng(3)
        raw = {domain: rng.random((4, 5)) + 0.1 for domain in STAGE2B_DOMAINS}
        tensor = build_importance_tensor(raw)
        np.testing.assert_allclose(tensor.sum(axis=(0, 1)), np.ones(4), atol=1e-12)


class SpecializationTests(unittest.TestCase):
    def test_margin_formula_matches_hand_computation(self) -> None:
        importance = np.zeros((1, 1, 4))
        importance[0, 0] = [0.5, 0.2, 0.9, 0.1]
        margins = specialization_margins(importance)
        np.testing.assert_allclose(
            margins[0, 0], [0.5 - 0.9, 0.2 - 0.9, 0.9 - 0.5, 0.1 - 0.9]
        )

    def test_positive_part_and_domain_normalization(self) -> None:
        rng = np.random.default_rng(11)
        importance = rng.random((3, 4, 4))
        margins = specialization_margins(importance)
        positive, normalized = normalized_specialization(margins)
        self.assertTrue(np.all(positive >= 0))
        np.testing.assert_allclose(
            positive, np.clip(margins, 0.0, None), atol=0
        )
        np.testing.assert_allclose(
            normalized.sum(axis=(0, 1)), np.ones(4), atol=1e-12
        )

    def test_zero_specialist_mass_aborts(self) -> None:
        importance = np.full((2, 2, 4), 0.25 / 4)
        margins = specialization_margins(importance)
        with self.assertRaises(RuntimeError):
            normalized_specialization(margins)

    def test_routing_specialization_uses_identical_pipeline(self) -> None:
        rng = np.random.default_rng(5)
        raw = {domain: rng.integers(1, 50, size=(2, 3)).astype(float)
               for domain in STAGE2B_DOMAINS}
        tensor = build_importance_tensor(raw)
        _, normalized = normalized_specialization(specialization_margins(tensor))
        np.testing.assert_allclose(normalized.sum(axis=(0, 1)), np.ones(4), atol=1e-12)


class CoverageTests(unittest.TestCase):
    def test_coverage_computation(self) -> None:
        scores = np.zeros((2, 2, 4))
        scores[0, 0] = [0.6, 0.0, 0.0, 0.0]
        scores[1, 1] = [0.4, 1.0, 1.0, 1.0]
        x = np.asarray([[1, 0], [0, 0]])
        np.testing.assert_allclose(
            specialist_coverage(scores, x), [0.6, 0.0, 0.0, 0.0]
        )
        x_all = np.ones((2, 2), dtype=int)
        np.testing.assert_allclose(
            specialist_coverage(scores, x_all), [1.0, 1.0, 1.0, 1.0]
        )

    def test_coverage_rejects_non_binary(self) -> None:
        with self.assertRaises(ValueError):
            specialist_coverage(np.zeros((1, 1, 4)), np.asarray([[0.5]]))


if __name__ == "__main__":
    unittest.main()
