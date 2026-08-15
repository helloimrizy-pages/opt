from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from expert_analysis import quantization
from expert_analysis.controlled import PreparedDomainExamples
from expert_analysis.hardware import RuntimeDevice
from expert_analysis.modeling import ModelBundle, discover_moe_layers
from expert_analysis.protection_evaluation import (
    STAGE1_QDQ_FUNCTION,
    MixedPrecisionExpertManager,
    allocation_is_complete,
    allocation_slug,
    configure_strict_determinism,
    evaluate_allocation_records,
    run_repeated_baseline_check,
)
from expert_analysis.specialist_preservation import STAGE2B_DOMAINS


def tiny_olmoe_bundle() -> tuple[ModelBundle, list]:
    try:
        from transformers import OlmoeConfig, OlmoeForCausalLM
    except ImportError:
        raise unittest.SkipTest("Installed Transformers does not include OLMoE")
    config = OlmoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
    )
    torch.manual_seed(1234)
    model = OlmoeForCausalLM(config).eval()
    runtime = RuntimeDevice(torch.device("cpu"), torch.float32, "cpu-test")
    bundle = ModelBundle(
        model=model,
        backbone=model.model,
        tokenizer=None,
        checkpoint="tiny-olmoe-test",
        requested_revision=None,
        resolved_revision="test-revision",
        runtime=runtime,
    )
    return bundle, discover_moe_layers(model)


def synthetic_examples(num_examples: int = 2, sequence: int = 12) -> dict[str, PreparedDomainExamples]:
    rng = np.random.default_rng(7)
    output = {}
    for domain in STAGE2B_DOMAINS:
        input_ids = rng.integers(3, 60, size=(num_examples, sequence)).astype(np.int32)
        attention = np.ones_like(input_ids, dtype=np.uint8)
        measurement = np.zeros_like(input_ids, dtype=np.uint8)
        measurement[:, 1:-1] = 1
        output[domain] = PreparedDomainExamples(
            domain=domain,
            input_ids=input_ids,
            attention_mask=attention,
            measurement_mask=measurement,
            metadata={},
        )
        output[domain].validate()
    return output


def make_record(
    method: str,
    kind: str,
    bits: np.ndarray,
    regime: str | None,
    budget: float | None,
    group_size: int,
) -> dict:
    payload = f"{method}|{regime}|{budget}|{bits.tolist()}"
    return {
        "method": method,
        "method_label": method,
        "method_kind": kind,
        "regime": regime,
        "budget_fraction": budget,
        "group_size": group_size,
        "expert_bits": bits.tolist(),
        "allocation_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


class ManagerTests(unittest.TestCase):
    def test_stage1_qdq_function_is_reused_exactly(self) -> None:
        self.assertIs(STAGE1_QDQ_FUNCTION, quantization.symmetric_groupwise_qdq)

    def test_apply_verify_and_restore_mixed_precision(self) -> None:
        bundle, specs = tiny_olmoe_bundle()
        manager = MixedPrecisionExpertManager(bundle, specs, group_size=8)
        clean = manager.clean_fingerprints()
        bits = np.full((2, 4), 4, dtype=np.int64)
        bits[0, 1] = 8
        bits[1, 3] = 8
        report = manager.apply_allocation(
            bits, verification_seed=99, verification_samples_per_layer=4
        )
        self.assertEqual(report["quantized_experts"], 8)
        self.assertEqual(report["bits_histogram"], {"4": 6, "8": 2})
        self.assertTrue(report["assignment_verification"]["verified_exact_qdq"])
        eight_bit = report["mean_relative_distortion_by_bits"]["8"]
        four_bit = report["mean_relative_distortion_by_bits"]["4"]
        self.assertLess(eight_bit, four_bit)
        with self.assertRaises(RuntimeError):
            manager.apply_allocation(bits, verification_seed=99)
        restoration = manager.restore_clean()
        self.assertTrue(restoration["restoration_verified_bitwise"])
        self.assertEqual(manager.clean_fingerprints(), clean)

    def test_three_and_four_bit_assignments_install_exact_qdq_weights(self) -> None:
        bundle, specs = tiny_olmoe_bundle()
        manager = MixedPrecisionExpertManager(bundle, specs, group_size=8)
        layout = manager._layers[0].layout
        snapshot = {
            expert: layout.references(expert)[0].view().detach().clone()
            for expert in range(4)
        }
        for base in (3, 4):
            bits = np.full((2, 4), base, dtype=np.int64)
            bits[0, 0] = 8
            manager.apply_allocation(
                bits, verification_seed=1, verification_samples_per_layer=4
            )
            reference = layout.references(0)[0]
            expected_protected = quantization.symmetric_groupwise_qdq(
                snapshot[0], bits=8, group_size=8
            ).dequantized
            self.assertTrue(torch.equal(reference.view(), expected_protected))
            base_reference = layout.references(1)[0]
            expected_base = quantization.symmetric_groupwise_qdq(
                snapshot[1], bits=base, group_size=8
            ).dequantized
            self.assertTrue(torch.equal(base_reference.view(), expected_base))
            manager.restore_clean()
            self.assertTrue(torch.equal(reference.view(), snapshot[0]))

    def test_non_expert_mutation_is_detected(self) -> None:
        bundle, specs = tiny_olmoe_bundle()
        manager = MixedPrecisionExpertManager(bundle, specs, group_size=8)
        router = specs[0].router
        with torch.no_grad():
            router.weight[0, 0] += 1.0
        with self.assertRaises(RuntimeError):
            manager.verify_non_expert_integrity()

    def test_expert_mutation_fails_clean_verification(self) -> None:
        bundle, specs = tiny_olmoe_bundle()
        manager = MixedPrecisionExpertManager(bundle, specs, group_size=8)
        with torch.no_grad():
            manager._layers[1].parameters[0][0, 0, 0] += 0.5
        with self.assertRaises(RuntimeError):
            manager.verify_clean()


class DeterminismConfigTests(unittest.TestCase):
    def test_cpu_configuration_records_settings(self) -> None:
        settings = configure_strict_determinism("cpu", warn_only=True)
        self.assertTrue(settings["use_deterministic_algorithms"])
        self.assertTrue(settings["deterministic_warn_only"])
        self.assertEqual(settings["attn_implementation_requested"], "eager")
        torch.use_deterministic_algorithms(False)

    def test_cuda_configuration_requires_workspace_env(self) -> None:
        import os

        previous = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            with self.assertRaises(RuntimeError):
                configure_strict_determinism("cuda", warn_only=True)
        finally:
            if previous is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous
            torch.use_deterministic_algorithms(False)


class EvaluationLoopTests(unittest.TestCase):
    def test_evaluation_checkpoints_resume_and_restore(self) -> None:
        bundle, specs = tiny_olmoe_bundle()
        manager = MixedPrecisionExpertManager(bundle, specs, group_size=8)
        splits = synthetic_examples()
        split_hashes = {domain: "h" * 64 for domain in STAGE2B_DOMAINS}
        bf16_bits = np.full((2, 4), 16, dtype=np.int64)
        allocation_bits = np.full((2, 4), 4, dtype=np.int64)
        allocation_bits[0, 2] = 8
        records = [
            make_record("bf16_reference", "uniform_reference", bf16_bits, None, None, 8),
            make_record(
                "robust_functional", "deterministic_milp", allocation_bits,
                "4to8", 0.20, 8,
            ),
        ]
        clean = manager.clean_fingerprints()
        with tempfile.TemporaryDirectory() as temporary:
            losses_dir = Path(temporary) / "losses"
            first = evaluate_allocation_records(
                bundle, manager, records, splits, split_hashes,
                losses_dir, "run-fingerprint", batch_size=1,
            )
            self.assertFalse(any(item.get("resumed") for item in first))
            self.assertEqual(manager.clean_fingerprints(), clean)
            for record in records:
                self.assertTrue(
                    allocation_is_complete(losses_dir, record, "run-fingerprint")
                )
                slug = allocation_slug(record)
                for domain in STAGE2B_DOMAINS:
                    self.assertTrue((losses_dir / slug / f"{domain}.npz").is_file())
            second = evaluate_allocation_records(
                bundle, manager, records, splits, split_hashes,
                losses_dir, "run-fingerprint", batch_size=1,
            )
            self.assertTrue(all(item["resumed"] for item in second))
            self.assertFalse(
                allocation_is_complete(losses_dir, records[1], "other-run")
            )

    def test_quantized_allocation_changes_losses(self) -> None:
        bundle, specs = tiny_olmoe_bundle()
        manager = MixedPrecisionExpertManager(bundle, specs, group_size=8)
        splits = synthetic_examples()
        from expert_analysis.masking import evaluate_next_token_loss

        domain = STAGE2B_DOMAINS[0]
        baseline, _ = evaluate_next_token_loss(bundle, splits[domain], batch_size=1)
        bits = np.full((2, 4), 3, dtype=np.int64)
        manager.apply_allocation(bits, verification_seed=5)
        quantized, _ = evaluate_next_token_loss(bundle, splits[domain], batch_size=1)
        manager.restore_clean()
        restored, _ = evaluate_next_token_loss(bundle, splits[domain], batch_size=1)
        self.assertFalse(np.array_equal(baseline.loss_sums, quantized.loss_sums))
        np.testing.assert_array_equal(baseline.loss_sums, restored.loss_sums)

    def test_repeated_baseline_check_passes_on_cpu(self) -> None:
        bundle, _ = tiny_olmoe_bundle()
        splits = synthetic_examples(num_examples=1)
        report = run_repeated_baseline_check(bundle, splits, batch_size=1)
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
