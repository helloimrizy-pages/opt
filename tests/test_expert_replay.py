from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from expert_analysis.expert_replay import (
    ReplayCapture,
    ReplayCaptureSession,
    validate_replay_captures,
)
from expert_analysis.gradient_quantization_cost import MoeOutputGradientSession
from expert_analysis.modeling import discover_moe_layers


class Router(nn.Module):
    def __init__(self, hidden: int, experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(experts, hidden) * 0.1)

    def forward(self, hidden: torch.Tensor):
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), self.weight)
        probabilities = logits.float().softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        return logits, weights.to(logits.dtype), indices


class Experts(nn.Module):
    def __init__(self, hidden: int, intermediate: int, count: int) -> None:
        super().__init__()
        self.num_experts = count
        self.gate_up_proj = nn.Parameter(
            torch.randn(count, intermediate * 2, hidden) * 0.1
        )
        self.down_proj = nn.Parameter(torch.randn(count, hidden, intermediate) * 0.1)
        self.act_fn = F.silu

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        for expert_id in range(self.num_experts):
            rows, positions = torch.where(indices == expert_id)
            if not rows.numel():
                continue
            gate_up = F.linear(hidden[rows], self.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            values = F.linear(F.silu(gate) * up, self.down_proj[expert_id])
            output.index_add_(0, rows, values * weights[rows, positions, None])
        return output


class Moe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = Router(6, 4, 2)
        self.experts = Experts(6, 8, 4)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, hidden.shape[-1])
        _, weights, indices = self.gate(flat)
        return self.experts(flat, indices, weights).reshape(shape)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = Moe()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden)


class ExpertReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.model = Model().eval()
        self.specs = discover_moe_layers(self.model)
        self.hidden = torch.randn(2, 4, 6)
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)

    def capture(self) -> ReplayCapture:
        session = ReplayCaptureSession(
            self.specs,
            num_examples=2,
            sample_locations={0: [(0, 0, 0), (1, 1, 1)]},
        )
        with session:
            with session.batch([0, 1], self.mask):
                with torch.no_grad():
                    self.model(self.hidden)
        self.assertEqual(session.registered_hook_count, 0)
        return session.finalize("general")[0]

    def test_capture_filters_measured_positions_and_preserves_topk(self) -> None:
        capture = self.capture()
        self.assertEqual(capture.hidden_states.shape, (5, 6))
        self.assertEqual(capture.selected_expert_ids.shape, (5, 2))
        np.testing.assert_array_equal(capture.example_indices, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(capture.token_positions, [0, 1, 2, 0, 1])
        self.assertEqual(int(capture.selected_expert_ids.numel()), 10)
        self.assertEqual(len(capture.sample_row_indices), 2)

    def test_layer_energy_uses_actual_moe_output_before_residual(self) -> None:
        capture = self.capture()
        with torch.no_grad():
            actual = self.model.layers[0].mlp(self.hidden).float()
        expected = np.zeros(2, dtype=np.float64)
        for example in range(2):
            values = actual[example][self.mask[example]]
            expected[example] = float(values.double().square().sum().item())
        np.testing.assert_allclose(capture.layer_energy_by_example, expected, rtol=1e-6)

    def test_exact_bfloat16_storage_round_trip(self) -> None:
        capture = self.capture()
        capture.hidden_states = capture.hidden_states.to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "general" / "layer_00.npz"
            capture.save(path, {"capture_fingerprint": "test"})
            restored = ReplayCapture.load(
                path, expected_metadata={"capture_fingerprint": "test"}
            )
        self.assertEqual(restored.hidden_states.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(restored.hidden_states, capture.hidden_states))
        np.testing.assert_array_equal(restored.example_indices, capture.example_indices)

    def test_offline_replay_matches_live_contribution_and_moe_output(self) -> None:
        capture = self.capture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "general" / "layer_00.npz"
            capture.save(path, {"capture_fingerprint": "fingerprint"})
            result = validate_replay_captures(
                self.specs,
                root,
                ["general"],
                capture_fingerprint="fingerprint",
                validation_layer_indices=[0],
                atol=1e-6,
                rtol=1e-5,
            )
        self.assertTrue(result["passed"])
        self.assertEqual(result["sample_count"], 2)
        self.assertTrue(result["validates_tensorized_expert_indexing"])
        self.assertTrue(result["validates_gate_handling"])

    def test_gradient_hook_retains_aligned_moe_output_gradient(self) -> None:
        session = MoeOutputGradientSession(self.specs)
        hidden = self.hidden.clone().requires_grad_(True)
        versions = [parameter._version for parameter in self.model.parameters()]
        with session:
            with session.batch([0, 1], self.mask):
                output = self.model(hidden)
                output.square().mean().backward()
                session.collect_after_backward()
        capture = session.finalize("general")[0]
        self.assertEqual(capture.gradients.shape, (5, 6))
        np.testing.assert_array_equal(capture.example_indices, [0, 0, 0, 1, 1])
        self.assertTrue(torch.isfinite(capture.gradients).all())
        self.assertEqual(
            versions, [parameter._version for parameter in self.model.parameters()]
        )
        self.assertEqual(session.registered_hook_count, 0)


if __name__ == "__main__":
    unittest.main()
