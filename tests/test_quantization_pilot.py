from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.balanced import BALANCED_DOMAINS
from expert_analysis.masking import LossStatistics
from expert_analysis.quantization import projected_expert_storage
from expert_analysis.quantization_pilot import (
    analyze_quantization_pilot,
    build_pilot_preregistration,
    create_quantization_figures,
    pilot_intervention_panel,
    validate_pilot_preregistration,
    write_quantization_outputs,
    write_quantization_summary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BALANCED_DIR = REPOSITORY_ROOT / "results" / "expert_domain_balanced_causal_validation"


class QuantizationPilotTests(unittest.TestCase):
    def preregistration(self) -> dict[str, object]:
        return build_pilot_preregistration(
            BALANCED_DIR / "selected_experts_preregistered.json",
            BALANCED_DIR / "matched_controls.csv",
        )

    def test_pilot_selection_is_deterministic_and_uses_frozen_functional_score(self) -> None:
        first = self.preregistration()
        second = self.preregistration()
        self.assertEqual(first["pilot_panel_fingerprint"], second["pilot_panel_fingerprint"])
        self.assertEqual(first["pairs"], second["pairs"])
        validate_pilot_preregistration(first)
        expected = {
            "general": [(13, 52, 13, 36), (12, 40, 12, 2)],
            "math": [(8, 11, 8, 54), (12, 63, 12, 43)],
            "coding": [(13, 2, 13, 61), (11, 27, 11, 7)],
            "reasoning": [(13, 20, 13, 63), (11, 48, 11, 47)],
        }
        for domain in BALANCED_DOMAINS:
            pairs = [row for row in first["pairs"] if row["target_domain"] == domain]
            observed = [
                (
                    row["specialist"]["layer"],
                    row["specialist"]["expert_id"],
                    row["matched_control"]["layer"],
                    row["matched_control"]["expert_id"],
                )
                for row in pairs
            ]
            self.assertEqual(observed, expected[domain])
            margins = [row["specialist"]["specialization_margin"] for row in pairs]
            self.assertGreaterEqual(margins[0], margins[1])
            self.assertEqual(
                [row["pilot_functional_specialization_rank"] for row in pairs], [1, 2]
            )
        self.assertFalse(first["selection"]["masking_effect_sizes_used"])
        self.assertFalse(first["selection"]["quantization_results_used"])

    def test_pilot_selection_does_not_depend_on_intervention_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg_path = root / "selected_experts_preregistered.json"
            controls_path = root / "matched_controls.csv"
            shutil.copyfile(
                BALANCED_DIR / "selected_experts_preregistered.json", prereg_path
            )
            shutil.copyfile(BALANCED_DIR / "matched_controls.csv", controls_path)
            before = build_pilot_preregistration(prereg_path, controls_path)
            (root / "results.json").write_text(
                json.dumps({"adversarial_quantization_outcome": 1e9}), encoding="utf-8"
            )
            (root / "masking_results.csv").write_text(
                "layer,expert_id,effect\n13,52,-999\n", encoding="utf-8"
            )
            after = build_pilot_preregistration(prereg_path, controls_path)
            self.assertEqual(before["pilot_panel_fingerprint"], after["pilot_panel_fingerprint"])
            self.assertEqual(before["pairs"], after["pairs"])

    def test_bootstrap_analysis_is_deterministic_and_writes_all_outputs(self) -> None:
        prereg, baselines, quantized, metadata, masking = self.synthetic_inputs()
        first, first_arrays = analyze_quantization_pilot(
            prereg,
            baselines,
            quantized,
            metadata,
            masking,
            [4],
            bootstrap_replicates=1000,
            seed=42,
        )
        second, second_arrays = analyze_quantization_pilot(
            prereg,
            baselines,
            quantized,
            metadata,
            masking,
            [4],
            bootstrap_replicates=1000,
            seed=42,
        )
        self.assertEqual(first["quantization_pilot_results"], second["quantization_pilot_results"])
        self.assertEqual(first["specialist_vs_control"], second["specialist_vs_control"])
        self.assertEqual(first["stage1_decision"], second["stage1_decision"])
        np.testing.assert_array_equal(
            first_arrays["per_example_loss_changes"],
            second_arrays["per_example_loss_changes"],
        )
        self.assertEqual(first_arrays["per_example_loss_changes"].shape, (1, 16, 4, 100))
        self.assertEqual(len(first["quantization_pilot_results"]), 64)
        self.assertEqual(len(first["quantization_pilot_pairwise"]), 48)
        self.assertEqual(len(first["specialist_vs_control"]), 8)
        self.assertEqual(len(first["quantization_distortion"]), 16)
        self.assertEqual(first["stage1_decision"]["decision"], "GO")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_quantization_outputs(first, first_arrays, output)
            results = {
                "run_config": {
                    "model": "allenai/OLMoE-1B-7B-0924",
                    "model_revision": "revision",
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "group_size": 128,
                    "runtime_description": "synthetic",
                },
                "pilot_panel_preregistration": prereg,
                "quantization_analysis": first,
            }
            write_quantization_summary(results, output / "SUMMARY.md")
            figures = create_quantization_figures(results, output)
            self.assertEqual(len(figures), 10)
            required = {
                "quantization_pilot_results.csv",
                "quantization_pilot_pairwise.csv",
                "specialist_vs_control.csv",
                "quantization_vs_masking.csv",
                "quantization_distortion.csv",
                "per_example_quantization_losses.npz",
                "stage1_decision.json",
                "SUMMARY.md",
            }
            self.assertTrue(all((output / name).stat().st_size > 0 for name in required))
            self.assertTrue(all(path.stat().st_size > 0 for path in figures))

    def test_fallback_triggers_only_for_too_small_primary_effects(self) -> None:
        prereg, baselines, quantized, metadata, masking = self.synthetic_inputs()
        panel = pilot_intervention_panel(prereg)
        for intervention in panel:
            for domain in BALANCED_DOMAINS:
                key = (4, intervention["layer"], intervention["expert_id"], domain)
                quantized[key] = LossStatistics(
                    loss_sums=baselines[domain].loss_sums.copy(),
                    token_counts=baselines[domain].token_counts.copy(),
                )
        too_small, _ = analyze_quantization_pilot(
            prereg, baselines, quantized, metadata, masking, [4]
        )
        self.assertEqual(too_small["stage1_decision"]["decision"], "PENDING_FALLBACK")

        # A clearly negative specialist outcome is scientifically unattractive, not
        # merely unmeasurable, and therefore must not trigger the 3-bit fallback.
        for intervention in panel:
            for domain in BALANCED_DOMAINS:
                delta = (
                    -0.01
                    if intervention["role"] == "specialist"
                    and domain == intervention["target_domain"]
                    else 0.0
                )
                key = (4, intervention["layer"], intervention["expert_id"], domain)
                quantized[key] = LossStatistics(
                    loss_sums=baselines[domain].loss_sums + delta * 64,
                    token_counts=baselines[domain].token_counts.copy(),
                )
        unattractive, _ = analyze_quantization_pilot(
            prereg, baselines, quantized, metadata, masking, [4]
        )
        self.assertEqual(unattractive["stage1_decision"]["decision"], "NO_GO")

        # Once the frozen fallback has been evaluated, a failure of both settings is
        # final rather than another pending state.
        for intervention in panel:
            meta = dict(metadata[(4, intervention["layer"], intervention["expert_id"])])
            memory = projected_expert_storage([(6, 8), (8, 3)], bits=3, group_size=4)
            meta["memory_accounting"] = memory
            metadata[(3, intervention["layer"], intervention["expert_id"])] = meta
            for domain in BALANCED_DOMAINS:
                for bit_width in (4, 3):
                    key = (
                        bit_width,
                        intervention["layer"],
                        intervention["expert_id"],
                        domain,
                    )
                    quantized[key] = LossStatistics(
                        loss_sums=baselines[domain].loss_sums.copy(),
                        token_counts=baselines[domain].token_counts.copy(),
                    )
        both_small, _ = analyze_quantization_pilot(
            prereg, baselines, quantized, metadata, masking, [4, 3]
        )
        self.assertEqual(both_small["stage1_decision"]["decision"], "NO_GO")

    def synthetic_inputs(self) -> tuple[dict, dict, dict, dict, dict]:
        prereg = self.preregistration()
        panel = pilot_intervention_panel(prereg)
        baselines = {
            domain: LossStatistics(
                loss_sums=np.full(100, 128.0, dtype=np.float64),
                token_counts=np.full(100, 64, dtype=np.uint32),
            )
            for domain in BALANCED_DOMAINS
        }
        quantized = {}
        metadata = {}
        noise = np.linspace(-0.0002, 0.0002, 100)
        memory = projected_expert_storage([(6, 8), (8, 3)], bits=4, group_size=4)
        masking_rows = []
        for intervention_index, intervention in enumerate(panel):
            target_effect = (
                0.006 + intervention_index * 0.00005
                if intervention["role"] == "specialist"
                else 0.001 + intervention_index * 0.00001
            )
            non_target_effect = 0.0002
            distortion = 0.005 + intervention_index * 0.0001
            metadata[(4, intervention["layer"], intervention["expert_id"])] = {
                "quantization_distortion": distortion,
                "memory_accounting": memory,
                "original_expert_fingerprint": f"original-{intervention_index}",
                "quantized_expert_fingerprint": f"quantized-{intervention_index}",
                "exact_restoration_verified": True,
                "unrelated_experts_verified_unchanged": True,
            }
            for domain in BALANCED_DOMAINS:
                delta = (
                    target_effect
                    if domain == intervention["target_domain"]
                    else non_target_effect
                ) + noise
                quantized[(4, intervention["layer"], intervention["expert_id"], domain)] = (
                    LossStatistics(
                        loss_sums=baselines[domain].loss_sums + delta * 64,
                        token_counts=baselines[domain].token_counts.copy(),
                    )
                )
            masking_rows.append(
                {
                    "role": (
                        "specialized"
                        if intervention["role"] == "specialist"
                        else "control"
                    ),
                    "layer": intervention["layer"],
                    "expert_id": intervention["expert_id"],
                    "target_minus_mean_other_contrast": target_effect - non_target_effect,
                }
            )
        masking = {
            "integrity_validation": {"passed": True},
            "preregistration": {
                "preregistration_fingerprint": prereg["source"][
                    "balanced_preregistration_fingerprint"
                ]
            },
            "balanced_analysis": {"intervention_contrasts": masking_rows},
        }
        return prereg, baselines, quantized, metadata, masking


if __name__ == "__main__":
    unittest.main()
