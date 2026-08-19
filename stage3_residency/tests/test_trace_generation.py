from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from residency_headroom.trace_generation import DecodeRoutingCapture
from expert_analysis.modeling import discover_moe_layers


class _Router(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.norm_topk_prob = False
        self.weight = nn.Parameter(torch.randn(4, 5))

    def forward(self, hidden: torch.Tensor):
        logits = F.linear(hidden.reshape(-1, 5), self.weight)
        weights, indices = logits.softmax(-1).topk(2, dim=-1)
        return logits, weights, indices


class _Experts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.randn(4, 6, 5))
        self.down_proj = nn.Parameter(torch.randn(4, 5, 3))
        self.act_fn = F.silu

    def forward(self, hidden, indices, weights):
        del indices, weights
        return torch.zeros_like(hidden)


class _Moe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _Router()
        self.experts = _Experts()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        _logits, weights, indices = self.gate(hidden)
        return self.experts(hidden.reshape(-1, 5), indices, weights).reshape_as(hidden)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Moe()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(hidden)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TraceGenerationTests(unittest.TestCase):
    def test_decode_capture_ignores_prefill_and_records_one_atomic_set_per_layer(self) -> None:
        torch.manual_seed(4)
        model = _Model()
        specs = discover_moe_layers(model)
        before = sum(len(module._forward_hooks) for module in model.modules())
        capture = DecodeRoutingCapture(specs)
        with capture:
            model(torch.randn(1, 3, 5))  # inactive prefill
            capture.begin_token()
            model(torch.randn(1, 1, 5))
            records = capture.finish_token()
            self.assertEqual(len(records), 2)
            self.assertEqual([record.layer_index for record in records], [0, 1])
            for record in records:
                self.assertEqual(record.expert_ids.shape, (2,))
                self.assertEqual(len(set(map(int, record.expert_ids))), 2)
                self.assertEqual(record.router_weights.shape, (2,))
        after = sum(len(module._forward_hooks) for module in model.modules())
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
