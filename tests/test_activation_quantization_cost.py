from __future__ import annotations

import unittest

import numpy as np
import torch

from expert_analysis.activation_quantization_cost import (
    ACTIVATION_COST_EPSILON,
    calculate_perturbation_sums,
    finalize_activation_surrogates,
    selected_routes,
)
from expert_analysis.expert_replay import ReplayCapture
from expert_analysis.gradient_quantization_cost import calculate_gqs


class ActivationQuantizationCostTests(unittest.TestCase):
    def test_gate_multiplication_and_aod_numerator_are_exact(self) -> None:
        baseline = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        quantized = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
        gates = torch.tensor([0.5, 0.25])
        sums = calculate_perturbation_sums(baseline, quantized, gates)
        delta = (quantized.double() - baseline.double()) * gates.double()[:, None]
        self.assertAlmostEqual(
            sums.gated_delta_squared, float(delta.square().sum().item())
        )
        self.assertAlmostEqual(
            sums.gated_baseline_squared,
            float((baseline.double() * gates.double()[:, None]).square().sum().item()),
        )
        self.assertEqual(sums.route_count, 2)

    def test_aod_reod_apd_and_uod_formulas(self) -> None:
        baseline = np.asarray([[1.0, 0.0], [0.0, 2.0]])
        quantized = np.asarray([[2.0, 0.0], [0.0, 0.0]])
        gates = np.asarray([0.5, 0.25])
        sums = calculate_perturbation_sums(baseline, quantized, gates)
        values = finalize_activation_surrogates(
            sums, layer_energy=10.0, domain_token_count=8
        )
        self.assertAlmostEqual(
            values.aod,
            sums.gated_delta_squared / (10.0 + ACTIVATION_COST_EPSILON),
        )
        self.assertAlmostEqual(
            values.reod,
            sums.gated_delta_squared
            / (sums.gated_baseline_squared + ACTIVATION_COST_EPSILON),
        )
        self.assertAlmostEqual(values.apd, sums.gated_delta_squared / 8.0)
        self.assertAlmostEqual(values.uod, sums.ungated_delta_squared / 2.0)
        self.assertFalse(values.unobserved)

    def test_zero_route_handling_is_explicit(self) -> None:
        empty = torch.empty((0, 4))
        sums = calculate_perturbation_sums(empty, empty, torch.empty(0))
        values = finalize_activation_surrogates(
            sums, layer_energy=2.0, domain_token_count=10
        )
        self.assertEqual(values.route_count, 0)
        self.assertTrue(values.unobserved)
        self.assertEqual((values.aod, values.reod, values.apd, values.uod), (0, 0, 0, 0))

    def test_selected_route_filtering_preserves_gate_and_example_alignment(self) -> None:
        capture = ReplayCapture(
            domain="general",
            model_layer_index=0,
            hidden_states=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            selected_expert_ids=torch.tensor([[0, 2], [1, 2], [0, 1]]),
            selected_gate_weights=torch.tensor(
                [[0.6, 0.2], [0.4, 0.3], [0.5, 0.1]], dtype=torch.float32
            ),
            example_indices=np.asarray([0, 0, 1]),
            token_positions=np.asarray([1, 2, 1]),
            layer_energy_by_example=np.asarray([1.0, 1.0]),
        )
        hidden, gates, examples = selected_routes(capture, 2)
        torch.testing.assert_close(hidden, capture.hidden_states[:2])
        torch.testing.assert_close(gates, torch.tensor([0.2, 0.3]))
        np.testing.assert_array_equal(examples, [0, 0])

    def test_gqs_and_gqs2_use_per_example_signed_sums(self) -> None:
        baseline = torch.zeros(3, 2)
        quantized = torch.tensor([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
        gates = torch.tensor([1.0, 0.5, 1.0])
        gradients = torch.tensor([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        examples = np.asarray([0, 0, 1])
        gqs, gqs2, signed = calculate_gqs(
            baseline,
            quantized,
            gates,
            gradients,
            examples,
            num_examples=3,
        )
        np.testing.assert_allclose(signed, [2.0, -2.0, 0.0])
        self.assertAlmostEqual(gqs, 4.0 / 3.0)
        self.assertAlmostEqual(gqs2, 8.0 / 3.0)

    def test_surrogate_calculation_is_deterministic(self) -> None:
        generator = torch.Generator().manual_seed(123)
        baseline = torch.randn(17, 5, generator=generator)
        quantized = baseline + torch.randn(17, 5, generator=generator) * 0.01
        gates = torch.rand(17, generator=generator)
        first = calculate_perturbation_sums(baseline, quantized, gates)
        second = calculate_perturbation_sums(baseline, quantized, gates)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
