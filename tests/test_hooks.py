from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from expert_analysis.hooks import ExpertInstrumentation
from expert_analysis.metrics import DomainStatistics
from expert_analysis.modeling import discover_moe_layers


class ToyRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.2)

    def forward(self, hidden: torch.Tensor):
        hidden = hidden.reshape(-1, hidden.shape[-1])
        logits = F.linear(hidden, self.weight)
        probabilities = logits.float().softmax(dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        return logits, weights.to(logits.dtype), indices


class ToyTensorExperts(nn.Module):
    def __init__(self, hidden_size: int, intermediate: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden_size) * 0.1
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate) * 0.1
        )
        self.act_fn = F.silu

    def forward(
        self, hidden: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        for expert_id in range(self.num_experts):
            rows, positions = torch.where(indices == expert_id)
            if rows.numel() == 0:
                continue
            gate_up = F.linear(hidden[rows], self.gate_up_proj[expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up, self.down_proj[expert_id]
            )
            output.index_add_(
                0, rows, expert_output * weights[rows, positions, None]
            )
        return output


class ToyMoe(nn.Module):
    def __init__(self, hidden_size: int = 6, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.gate = ToyRouter(hidden_size, num_experts, top_k)
        self.experts = ToyTensorExperts(hidden_size, 8, num_experts)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        _, weights, indices = self.gate(flat)
        return self.experts(flat, indices, weights).reshape(shape)


class ToyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = ToyMoe()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyLayer()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class LegacyExpert(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden))


class LegacyMoe(nn.Module):
    def __init__(self, hidden_size: int = 6, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [LegacyExpert(hidden_size) for _ in range(num_experts)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        logits = self.gate(flat)
        weights, indices = logits.float().softmax(-1).topk(self.top_k, -1)
        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            rows, positions = torch.where(indices == expert_id)
            if rows.numel():
                contribution = expert(flat[rows]) * weights[
                    rows, positions, None
                ].to(flat.dtype)
                output.index_add_(0, rows, contribution)
        return output.reshape(shape)


class LegacyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([LegacyMoe()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden)


class HookTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.hidden = torch.randn(2, 4, 6)
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

    def test_tensorized_collection_excludes_padding_and_removes_hooks(self) -> None:
        model = ToyModel()
        specs = discover_moe_layers(model)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].contribution_backend, "tensorized_gate_up")
        stats = DomainStatistics.zeros(2, 1, 4, [specs[0].block_name])
        instrumentation = ExpertInstrumentation(specs, stats)
        with instrumentation:
            with instrumentation.batch([0, 1], self.mask):
                with torch.inference_mode():
                    model(self.hidden)
        self.assertEqual(instrumentation.registered_hook_count, 0)
        np.testing.assert_array_equal(stats.token_counts, [4, 2])
        np.testing.assert_array_equal(
            stats.routing_counts.sum(axis=2).reshape(-1), [8, 4]
        )
        self.assertTrue(np.all(stats.gate_sums.sum(axis=2) > 0))
        self.assertTrue(np.all(stats.contribution_sums.sum(axis=2) > 0))
        diagnostic = instrumentation.diagnostic_report()[0]
        self.assertAlmostEqual(diagnostic["router_probability_sum_mean"], 1.0, places=5)
        self.assertEqual(diagnostic["selected_weight_reference"], "full_softmax")

    def test_module_list_compatibility(self) -> None:
        model = LegacyModel()
        specs = discover_moe_layers(model)
        self.assertEqual(specs[0].contribution_backend, "module_list")
        self.assertEqual(specs[0].capture_point, "block_post")
        stats = DomainStatistics.zeros(2, 1, 4, [specs[0].block_name])
        with ExpertInstrumentation(specs, stats) as instrumentation:
            with instrumentation.batch([0, 1], self.mask):
                with torch.inference_mode():
                    model(self.hidden)
        self.assertEqual(int(stats.routing_counts.sum()), 12)
        self.assertGreater(float(stats.contribution_sums.sum()), 0)

    def test_gradient_gate_attribution(self) -> None:
        model = ToyModel()
        specs = discover_moe_layers(model)
        stats = DomainStatistics.zeros(
            2, 1, 4, [specs[0].block_name], compute_gradient=True
        )
        hidden = self.hidden.clone().requires_grad_(True)
        with ExpertInstrumentation(
            specs, stats, compute_gradient_attribution=True
        ) as instrumentation:
            with instrumentation.batch([0, 1], self.mask):
                output = model(hidden)
                output.pow(2).sum().backward()
                instrumentation.finalize_gradients()
        self.assertIsNotNone(stats.gradient_sums)
        self.assertGreater(float(stats.gradient_sums.sum()), 0)


if __name__ == "__main__":
    unittest.main()
