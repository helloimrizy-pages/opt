from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from expert_analysis.collection import collect_prepared_domain
from expert_analysis.controlled import PreparedDomainExamples
from expert_analysis.hardware import RuntimeDevice
from expert_analysis.masking import (
    MaskTarget,
    evaluate_next_token_loss,
    run_masking_validation,
    validate_masking_mechanism,
)
from expert_analysis.modeling import ModelBundle, discover_moe_layers


class MaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from transformers import OlmoeConfig, OlmoeForCausalLM
        except ImportError:
            self.skipTest("Installed Transformers does not include OLMoE")
        torch.manual_seed(17)
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
        runtime = RuntimeDevice(torch.device("cpu"), torch.float32, "test CPU")
        self.bundle = ModelBundle(model, model.model, None, "toy", None, None, runtime)
        self.specs = discover_moe_layers(model)

    @staticmethod
    def examples(domain: str, offset: int = 0) -> PreparedDomainExamples:
        input_ids = np.asarray(
            [
                [3 + offset, 4, 5, 6, 7, 8],
                [9 + offset, 10, 11, 12, 13, 14],
                [15 + offset, 16, 17, 18, 19, 20],
                [21 + offset, 22, 23, 24, 25, 26],
            ],
            dtype=np.int32,
        )
        measurement = np.asarray([[0, 1, 1, 1, 1, 0]] * 4, dtype=np.uint8)
        return PreparedDomainExamples(
            domain,
            input_ids,
            np.ones_like(input_ids, dtype=np.uint8),
            measurement,
            {"repository": f"test/{domain}"},
        )

    def test_mask_target_parses_preregistered_contrast(self) -> None:
        target = MaskTarget.parse("11:27:coding:reasoning")
        self.assertEqual(target.model_layer_index, 11)
        self.assertEqual(target.expert_id, 27)
        self.assertEqual(target.expected_high_domain, "coding")
        self.assertEqual(target.expected_low_domain, "reasoning")

    def test_mask_changes_loss_without_weights_or_hook_leaks(self) -> None:
        examples = self.examples("general")
        smoke = validate_masking_mechanism(self.bundle, self.specs[0], examples)
        self.assertTrue(smoke["passed"])
        self.assertGreater(smoke["zeroed_routes"], 0)
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.bundle.model.named_parameters()
        }
        collected = collect_prepared_domain(
            self.bundle, self.specs, examples, batch_size=2
        ).statistics
        expert_id = int(
            np.argmax(collected.routing_counts[:, 0, :].sum(axis=0))
        )
        baseline, _ = evaluate_next_token_loss(self.bundle, examples, batch_size=2)
        masked, diagnostics = evaluate_next_token_loss(
            self.bundle,
            examples,
            batch_size=2,
            mask_spec=self.specs[0],
            expert_id=expert_id,
        )
        np.testing.assert_array_equal(
            masked.route_counts, collected.routing_counts[:, 0, expert_id]
        )
        self.assertGreater(int(masked.route_counts.sum()), 0)
        self.assertTrue(np.any(np.abs(masked.per_token_nll - baseline.per_token_nll) > 0))
        self.assertEqual(diagnostics["hooks_before"], diagnostics["hooks_after"])
        for name, parameter in self.bundle.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_end_to_end_masking_artifacts(self) -> None:
        domains = ("general", "math", "coding", "reasoning")
        examples = {
            domain: self.examples(domain, index)
            for index, domain in enumerate(domains)
        }
        statistics = {
            domain: collect_prepared_domain(
                self.bundle, self.specs, item, batch_size=4
            ).statistics
            for domain, item in examples.items()
        }
        expert_id = int(
            np.argmax(statistics["general"].routing_counts[:, 0, :].sum(axis=0))
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_masking_validation(
                self.bundle,
                self.specs,
                examples,
                statistics,
                [MaskTarget(0, expert_id, "coding", "general")],
                output_dir=Path(temporary),
                collection_fingerprint="test-fingerprint",
                batch_size=4,
                bootstrap_replicates=10,
                seed=31,
            )
            self.assertEqual(len(result["loss_rows"]), 4)
            self.assertEqual(len(result["domain_contrasts"]), 1)
            contrast = result["domain_contrasts"][0]
            self.assertEqual(contrast["contrast_high_domain"], "coding")
            self.assertEqual(contrast["contrast_low_domain"], "general")
            self.assertEqual(contrast["contrast_source"], "pre_registered_prompt_only_run")
            self.assertIn("causal_specialization_supported", contrast)
            self.assertTrue((Path(temporary) / "expert_masking_loss.csv").exists())
            self.assertTrue((Path(temporary) / "masking_results.json").exists())


if __name__ == "__main__":
    unittest.main()
