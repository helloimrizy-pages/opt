from __future__ import annotations

import unittest

import numpy as np
import torch

from expert_analysis.hooks import ExpertInstrumentation
from expert_analysis.metrics import DomainStatistics
from expert_analysis.modeling import discover_moe_layers


class TransformersOlmoeTests(unittest.TestCase):
    def test_current_huggingface_olmoe_instrumentation(self) -> None:
        try:
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError:
            self.skipTest("Installed Transformers does not include OLMoE")
        config = OlmoeConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=64,
            pad_token_id=0,
            eos_token_id=2,
        )
        model = OlmoeForCausalLM(config).eval()
        specs = discover_moe_layers(model)
        self.assertEqual(len(specs), 2)
        self.assertTrue(
            all(spec.contribution_backend == "tensorized_gate_up" for spec in specs)
        )
        stats = DomainStatistics.zeros(
            2,
            len(specs),
            4,
            [spec.block_name for spec in specs],
        )
        input_ids = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 0, 0]])
        attention_mask = (input_ids != 0).long()
        instrumentation = ExpertInstrumentation(specs, stats)
        with instrumentation:
            with instrumentation.batch([0, 1], attention_mask):
                with torch.inference_mode():
                    model.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
        np.testing.assert_array_equal(stats.token_counts, [5, 3])
        np.testing.assert_array_equal(
            stats.routing_counts.sum(axis=2),
            [[10, 10], [6, 6]],
        )
        self.assertTrue(np.all(stats.contribution_sums.sum(axis=(0, 2)) > 0))
        self.assertEqual(instrumentation.registered_hook_count, 0)
        for diagnostic in instrumentation.diagnostic_report():
            self.assertAlmostEqual(
                diagnostic["router_probability_sum_mean"], 1.0, places=5
            )
            self.assertEqual(
                diagnostic["selected_weight_reference"], "full_softmax"
            )


if __name__ == "__main__":
    unittest.main()
