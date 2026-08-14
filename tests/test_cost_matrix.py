from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from expert_analysis.cost_matrix import (
    CostChunk,
    build_memory_matrix,
    compute_activation_layer_costs,
    load_cost_chunk,
    save_cost_chunk,
    validate_full_cost_matrix,
    verify_pilot_reproduction,
)
from expert_analysis.expert_replay import ReplayCapture
from expert_analysis.modeling import MoeLayerSpec
from expert_analysis.quantization import (
    ExpertWeightLayout,
    ReversibleExpertQuantization,
)


class TinyExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(8)
        self.num_experts = 3
        self.gate_up_proj = nn.Parameter(torch.randn(3, 10, 4, generator=generator))
        self.down_proj = nn.Parameter(torch.randn(3, 4, 5, generator=generator))
        self.act_fn = F.silu


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = TinyExperts()


def tiny_spec(block: TinyBlock) -> MoeLayerSpec:
    return MoeLayerSpec(
        ordinal=0,
        model_layer_index=0,
        block_name="layers.0.mlp",
        router_name="layers.0.mlp.gate",
        experts_name="layers.0.mlp.experts",
        num_experts=3,
        top_k=1,
        contribution_backend="tensorized_gate_up",
        capture_point="experts_pre",
        block=block,
        router=block.gate,
        experts=block.experts,
    )


def tiny_capture() -> ReplayCapture:
    return ReplayCapture(
        domain="general",
        model_layer_index=0,
        hidden_states=torch.tensor(
            [[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, -1.0, 2.0], [1.0, 1.0, 0.0, 0.0]]
        ),
        selected_expert_ids=torch.tensor([[0], [1], [0]]),
        selected_gate_weights=torch.tensor([[0.6], [0.4], [0.5]]),
        example_indices=np.asarray([0, 0, 1]),
        token_positions=np.asarray([0, 1, 0]),
        layer_energy_by_example=np.asarray([4.0, 2.0]),
    )


class CostMatrixTests(unittest.TestCase):
    def test_cost_chunk_resume_validates_fingerprint(self) -> None:
        chunk = CostChunk(
            cost=np.asarray([1.0, 2.0, 0.0]),
            route_counts=np.asarray([2, 1, 0]),
            unobserved=np.asarray([False, False, True]),
            diagnostics={"uod": np.asarray([3.0, 4.0, 0.0])},
            metadata={},
        )
        expected = {"matrix_fingerprint": "abc", "bit_width": 4}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "general" / "bit_4.npz"
            save_cost_chunk(path, chunk, expected)
            restored = load_cost_chunk(
                path, expected_metadata=expected, num_experts=3
            )
            np.testing.assert_array_equal(restored.cost, chunk.cost)
            with self.assertRaises(RuntimeError):
                load_cost_chunk(
                    path,
                    expected_metadata={"matrix_fingerprint": "other", "bit_width": 4},
                    num_experts=3,
                )

    def test_layer_costs_filter_routes_handle_zero_routes_and_restore_weights(self) -> None:
        block = TinyBlock()
        spec = tiny_spec(block)
        layout = ExpertWeightLayout(spec)
        before = layout.all_fingerprints()
        chunks, diagnostics = compute_activation_layer_costs(
            spec,
            layout,
            {"general": tiny_capture()},
            ["general"],
            [4, 16],
            group_size=2,
            chunk_size=2,
        )
        self.assertEqual(layout.all_fingerprints(), before)
        four = chunks[("general", 4)]
        sixteen = chunks[("general", 16)]
        np.testing.assert_array_equal(four.route_counts, [2, 1, 0])
        np.testing.assert_array_equal(four.unobserved, [False, False, True])
        self.assertEqual(four.cost[2], 0.0)
        np.testing.assert_array_equal(sixteen.cost, np.zeros(3))
        self.assertTrue(all(row["exact_restoration_verified"] for row in diagnostics))

    def test_layer_costs_compute_only_requested_resume_chunks(self) -> None:
        block = TinyBlock()
        spec = tiny_spec(block)
        layout = ExpertWeightLayout(spec)
        chunks, diagnostics = compute_activation_layer_costs(
            spec,
            layout,
            {"general": tiny_capture()},
            ["general"],
            [3, 4, 16],
            group_size=2,
            chunk_size=2,
            requested_keys={("general", 4)},
        )
        self.assertEqual(set(chunks), {("general", 4)})
        self.assertEqual({row["bit_width"] for row in diagnostics}, {4})
        np.testing.assert_array_equal(
            chunks[("general", 4)].route_counts, np.asarray([2, 1, 0])
        )

    def test_qdq_fingerprint_reuse_is_exact(self) -> None:
        block = TinyBlock()
        layout = ExpertWeightLayout(tiny_spec(block))
        with ReversibleExpertQuantization(
            layout, 1, bits=4, group_size=2, verify_unrelated_experts=False
        ) as first:
            expected_original = first.original_fingerprint
            expected_quantized = first.quantized_fingerprint
        with ReversibleExpertQuantization(
            layout, 1, bits=4, group_size=2, verify_unrelated_experts=False
        ) as second:
            self.assertEqual(second.original_fingerprint, expected_original)
            self.assertEqual(second.quantized_fingerprint, expected_quantized)

    def test_full_matrix_shape_and_sixteen_bit_zero_behavior(self) -> None:
        cost = np.ones((2, 3, 4, 4), dtype=np.float64)
        cost[..., 0] = 4
        cost[..., 1] = 3
        cost[..., 2] = 2
        cost[..., 3] = 0
        route_counts = np.ones((2, 3, 4), dtype=np.int64)
        route_counts[0, 2, 1] = 0
        unobserved = route_counts == 0
        result = validate_full_cost_matrix(
            cost, route_counts, unobserved, [3, 4, 8, 16]
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["shape"], [2, 3, 4, 4])
        self.assertEqual(result["monotonicity_violation_count"], 0)

    def test_nonmonotonic_costs_are_reported_not_modified(self) -> None:
        cost = np.zeros((1, 1, 1, 4), dtype=np.float64)
        cost[0, 0, 0] = [1.0, 2.0, 0.5, 0.0]
        before = cost.copy()
        result = validate_full_cost_matrix(
            cost,
            np.ones((1, 1, 1), dtype=np.int64),
            np.zeros((1, 1, 1), dtype=bool),
            [3, 4, 8, 16],
        )
        self.assertEqual(result["monotonicity_violation_count"], 1)
        np.testing.assert_array_equal(cost, before)

    def test_exact_memory_accounting_matrix(self) -> None:
        block = TinyBlock()
        spec = tiny_spec(block)
        layout = ExpertWeightLayout(spec)
        memory = build_memory_matrix(
            [spec], {0: layout}, [3, 4, 8, 16], group_size=2
        )
        self.assertEqual(memory["projected_bytes"].shape, (1, 3, 4))
        self.assertTrue(np.all(memory["effective_bits_per_weight"][..., 3] == 16.0))
        # 60 weights: per-tensor 3-bit payloads use 15 + 8 bytes, plus 32 FP16 scales.
        self.assertTrue(np.all(memory["weight_count"] == 60))
        self.assertTrue(np.all(memory["number_of_groups"][..., 0] == 32))
        self.assertTrue(np.all(memory["projected_bytes"][..., 0] == 87))

    def test_pilot_extraction_reproduction(self) -> None:
        cost = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
        pilot = np.stack([cost[1, 2, :, 1], cost[0, 1, :, 1]])
        result = verify_pilot_reproduction(
            cost,
            np.asarray([0, 1]),
            np.asarray([0, 1, 2]),
            np.asarray(["general", "math", "coding", "reasoning"]),
            np.asarray([3, 4, 8, 16]),
            np.asarray([1, 0]),
            np.asarray([2, 1]),
            pilot,
            atol=0,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["exact_array_equal"])


if __name__ == "__main__":
    unittest.main()
