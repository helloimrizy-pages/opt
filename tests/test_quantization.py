from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from expert_analysis.masking import LossStatistics
from expert_analysis.modeling import MoeLayerSpec
from expert_analysis.quantization import (
    ExpertWeightLayout,
    ReversibleExpertQuantization,
    load_or_compute_loss_checkpoint,
    module_hook_count,
    projected_expert_storage,
    symmetric_groupwise_qdq,
)
from expert_analysis.modeling import discover_moe_layers


class TinyTensorizedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.num_experts = 4
        self.gate_up_proj = nn.Parameter(torch.randn(4, 6, 8, generator=generator))
        self.down_proj = nn.Parameter(torch.randn(4, 8, 3, generator=generator))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.experts = TinyTensorizedExperts()


def tiny_spec(block: TinyBlock) -> MoeLayerSpec:
    return MoeLayerSpec(
        ordinal=0,
        model_layer_index=3,
        block_name="layers.3.mlp",
        router_name="layers.3.mlp.gate",
        experts_name="layers.3.mlp.experts",
        num_experts=4,
        top_k=2,
        contribution_backend="tensorized_gate_up",
        capture_point="experts_pre",
        block=block,
        router=block.gate,
        experts=block.experts,
    )


class QuantizationTests(unittest.TestCase):
    def test_sixteen_bit_identity_and_nonnegative_distortion(self) -> None:
        weight = torch.randn(5, 11, generator=torch.Generator().manual_seed(2))
        result = symmetric_groupwise_qdq(weight, bits=16, group_size=4)
        self.assertTrue(torch.equal(result.dequantized, weight))
        self.assertEqual(result.number_of_groups, 0)
        self.assertEqual(result.squared_error, 0.0)
        self.assertEqual(result.relative_squared_error, 0.0)

    def test_quantization_is_deterministic_and_zero_groups_are_safe(self) -> None:
        weight = torch.tensor(
            [[-1.0, -0.5, 0.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        first = symmetric_groupwise_qdq(weight, bits=4, group_size=3)
        second = symmetric_groupwise_qdq(weight, bits=4, group_size=3)
        self.assertTrue(torch.equal(first.dequantized, second.dequantized))
        self.assertTrue(torch.equal(first.scales, second.scales))
        self.assertTrue(torch.equal(first.dequantized[1], weight[1]))
        self.assertGreaterEqual(first.relative_squared_error, 0.0)

    def test_groupwise_scale_and_qdq_are_correct_along_input_features(self) -> None:
        weight = torch.tensor([[-1.0, 0.5, 1.0, 2.0, -2.0]], dtype=torch.float32)
        result = symmetric_groupwise_qdq(weight, bits=3, group_size=3)
        expected_scales = torch.tensor([[1.0 / 3.0, 2.0 / 3.0]], dtype=torch.float16)
        self.assertTrue(torch.equal(result.scales.cpu(), expected_scales))
        scales = expected_scales.float()
        expected = torch.cat(
            [
                torch.round(weight[:, :3] / scales[:, :1]).clamp(-3, 3)
                * scales[:, :1],
                torch.round(weight[:, 3:] / scales[:, 1:2]).clamp(-3, 3)
                * scales[:, 1:2],
            ],
            dim=1,
        )
        torch.testing.assert_close(result.dequantized, expected, rtol=0, atol=0)
        self.assertEqual(result.number_of_groups, 2)

    def test_three_bit_distortion_is_not_below_four_bit_for_normal_weights(self) -> None:
        weight = torch.randn(23, 37, generator=torch.Generator().manual_seed(123))
        four = symmetric_groupwise_qdq(weight, bits=4, group_size=8)
        three = symmetric_groupwise_qdq(weight, bits=3, group_size=8)
        self.assertGreaterEqual(three.relative_squared_error, four.relative_squared_error)

    def test_effective_memory_accounting_is_exact(self) -> None:
        result = projected_expert_storage([(2, 5), (3, 4)], bits=4, group_size=3)
        self.assertEqual(result["weight_count"], 22)
        self.assertEqual(result["number_of_groups"], 10)
        self.assertEqual(result["quantized_weight_payload_bits"], 88)
        self.assertEqual(result["quantized_weight_packed_bytes"], 11)
        self.assertEqual(result["weight_packing_padding_bits"], 0)
        self.assertEqual(result["scale_storage_bytes"], 20)
        self.assertEqual(result["projected_bytes"], 31)
        self.assertAlmostEqual(result["effective_bits_per_weight"], 248 / 22)
        self.assertAlmostEqual(result["compression_ratio_vs_bf16"], 44 / 31)
        bf16 = projected_expert_storage([(2, 5), (3, 4)], bits=16, group_size=3)
        self.assertEqual(bf16["projected_bytes"], 44)
        self.assertEqual(bf16["effective_bits_per_weight"], 16.0)
        self.assertEqual(bf16["compression_ratio_vs_bf16"], 1.0)

    def test_expert_isolation_exact_restoration_and_no_hook_leakage(self) -> None:
        block = TinyBlock()
        layout = ExpertWeightLayout(tiny_spec(block))
        all_before = layout.all_fingerprints()
        router_before = block.gate.weight.detach().clone()
        hook = block.register_forward_hook(lambda *_: None)
        hooks_before = module_hook_count(block)
        try:
            context = ReversibleExpertQuantization(
                layout, expert_id=2, bits=4, group_size=4
            )
            with context:
                all_during = layout.all_fingerprints()
                self.assertNotEqual(all_during[2], all_before[2])
                for expert_id in (0, 1, 3):
                    self.assertEqual(all_during[expert_id], all_before[expert_id])
                self.assertTrue(torch.equal(block.gate.weight, router_before))
                self.assertEqual(module_hook_count(block), hooks_before)
            self.assertTrue(context.restoration_verified)
            self.assertEqual(layout.all_fingerprints(), all_before)
            self.assertTrue(torch.equal(block.gate.weight, router_before))
            self.assertEqual(module_hook_count(block), hooks_before)
            diagnostic = context.diagnostics()
            self.assertTrue(diagnostic["exact_restoration_verified"])
            self.assertTrue(diagnostic["unrelated_experts_verified_unchanged"])
            self.assertGreaterEqual(diagnostic["quantization_distortion"], 0.0)
        finally:
            hook.remove()
        self.assertEqual(module_hook_count(block), 0)

    def test_sixteen_bit_reversible_context_is_a_noop(self) -> None:
        block = TinyBlock()
        layout = ExpertWeightLayout(tiny_spec(block))
        before = layout.all_fingerprints()
        context = ReversibleExpertQuantization(
            layout, expert_id=1, bits=16, group_size=4
        )
        with context:
            self.assertEqual(layout.all_fingerprints(), before)
            self.assertEqual(context.distortion, 0.0)
        self.assertTrue(context.restoration_verified)
        self.assertEqual(layout.all_fingerprints(), before)

    def test_current_transformers_olmoe_tensorized_expert_axis(self) -> None:
        try:
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError:
            self.skipTest("Installed Transformers does not include OLMoE")
        config = OlmoeConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
            pad_token_id=0,
            eos_token_id=2,
        )
        model = OlmoeForCausalLM(config).eval()
        spec = discover_moe_layers(model)[0]
        layout = ExpertWeightLayout(spec)
        layout_metadata = layout.metadata()
        self.assertEqual(layout_metadata["tensor_count_per_expert"], 2)
        self.assertTrue(
            all(row["expert_axis"] == 0 for row in layout_metadata["tensors"])
        )
        self.assertEqual(
            {tuple(row["expert_slice_shape"]) for row in layout_metadata["tensors"]},
            {(32, 32), (32, 16)},
        )
        before = layout.all_fingerprints()
        context = ReversibleExpertQuantization(layout, 1, bits=4, group_size=8)
        with context:
            during = layout.all_fingerprints()
            self.assertNotEqual(during[1], before[1])
            self.assertEqual(
                {key: value for key, value in during.items() if key != 1},
                {key: value for key, value in before.items() if key != 1},
            )
        self.assertEqual(layout.all_fingerprints(), before)

    def test_intervention_checkpoint_resume_skips_compute_and_validates_metadata(self) -> None:
        calls = 0

        def compute() -> tuple[LossStatistics, dict[str, object]]:
            nonlocal calls
            calls += 1
            return (
                LossStatistics(
                    loss_sums=np.asarray([8.0, 9.0]),
                    token_counts=np.asarray([4, 4], dtype=np.uint32),
                ),
                {"hooks_before": 0, "hooks_after": 0},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bit_4" / "general.npz"
            expected = {"inference_fingerprint": "abc", "domain": "general"}
            first = load_or_compute_loss_checkpoint(
                path,
                expected,
                np.asarray([4, 4], dtype=np.uint32),
                compute,
                resume=True,
            )
            second = load_or_compute_loss_checkpoint(
                path,
                expected,
                np.asarray([4, 4], dtype=np.uint32),
                compute,
                resume=True,
            )
            self.assertFalse(first.resumed)
            self.assertTrue(second.resumed)
            self.assertEqual(calls, 1)
            np.testing.assert_array_equal(first.statistics.loss_sums, second.statistics.loss_sums)
            with self.assertRaises(RuntimeError):
                load_or_compute_loss_checkpoint(
                    path,
                    {"inference_fingerprint": "different", "domain": "general"},
                    np.asarray([4, 4], dtype=np.uint32),
                    compute,
                    resume=True,
                )

    def test_incomplete_checkpoint_is_recomputed(self) -> None:
        calls = 0

        def compute() -> tuple[LossStatistics, dict[str, object]]:
            nonlocal calls
            calls += 1
            return (
                LossStatistics(
                    loss_sums=np.asarray([4.0]),
                    token_counts=np.asarray([2], dtype=np.uint32),
                ),
                {},
            )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                loss_sums=np.asarray([999.0]),
                token_counts=np.asarray([2], dtype=np.uint32),
            )
            result = load_or_compute_loss_checkpoint(
                path,
                {"fingerprint": "valid"},
                np.asarray([2], dtype=np.uint32),
                compute,
                resume=True,
            )
            self.assertFalse(result.resumed)
            self.assertEqual(calls, 1)
            np.testing.assert_array_equal(result.statistics.loss_sums, [4.0])


if __name__ == "__main__":
    unittest.main()
