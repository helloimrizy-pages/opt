from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.io_utils import atomic_write_json
from expert_analysis.masking import LossStatistics
from expert_analysis.protection_allocations import generate_all_allocations
from expert_analysis.protection_optimization import build_expert_memory_matrix
from expert_analysis.protection_reporting import (
    analyze_phase,
    attach_protected_counts,
    create_phase_figures,
    render_main_table,
    write_development_decision,
    write_phase_outputs,
)
from expert_analysis.specialist_preservation import STAGE2B_DOMAINS
from test_protection_allocations import synthetic_scores

RUN_FINGERPRINT = "test-run-fingerprint"
NUM_EXAMPLES = 6


def synthetic_losses(rng: np.ndarray, bias: float) -> np.ndarray:
    return 1.0 + bias + 0.05 * rng


class ReportingPipelineTests(unittest.TestCase):
    """Synthetic end-to-end run of the development analysis pipeline."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scores = synthetic_scores()
        memory = build_expert_memory_matrix(
            [[(4, 4)], [(4, 4)]], group_size=2, num_experts=3
        )
        self.allocations_dir = self.root / "allocations"
        self.registry = generate_all_allocations(scores, memory, self.allocations_dir)
        self.losses_dir = self.root / "development" / "losses"
        rng = np.random.default_rng(2)
        bias_by_method = {
            "bf16_reference": 0.0,
            "uniform_8bit_reference": 0.005,
            "uniform_4bit_reference": 0.05,
            "uniform_3bit_reference": 0.12,
            "robust_functional": 0.010,
            "robust_routing": 0.015,
            "average_specialization": 0.020,
            "global_importance": 0.030,
            "general_only": 0.035,
            "math_only": 0.035,
            "coding_only": 0.035,
            "reasoning_only": 0.035,
        }
        for entry in self.registry["entries"]:
            record = json.loads((self.allocations_dir / entry["file"]).read_text())
            if (
                record["method_kind"] != "uniform_reference"
                and record["budget_fraction"] != 0.20
            ):
                continue
            slug = entry["file"][: -len(".json")] if record[
                "method_kind"
            ] != "uniform_reference" else record["method"]
            bias = bias_by_method.get(record["method"], 0.040)
            for domain in STAGE2B_DOMAINS:
                noise = rng.standard_normal(NUM_EXAMPLES)
                nll = synthetic_losses(noise, bias)
                statistics = LossStatistics(
                    loss_sums=nll * 64.0,
                    token_counts=np.full(NUM_EXAMPLES, 64, dtype=np.uint32),
                )
                path = self.losses_dir / slug / f"{domain}.npz"
                statistics.save(path)
                atomic_write_json(
                    path.with_suffix(".metadata.json"),
                    {
                        "run_fingerprint": RUN_FINGERPRINT,
                        "allocation_sha256": record["allocation_sha256"],
                        "domain": domain,
                    },
                )

    def test_development_analysis_outputs_and_decision(self) -> None:
        analysis = analyze_phase(
            "development",
            self.allocations_dir,
            self.losses_dir,
            RUN_FINGERPRINT,
            replicates=100,
            seed=5,
        )
        self.assertIn("development_gates", analysis)
        decision = analysis["development_decision"]["decision"]
        self.assertIn(
            decision, ("FULL_EVALUATION_GO", "ROBUST_PRESERVATION_NO_GO")
        )
        method_rows = analysis["method_rows"]
        self.assertEqual(len(method_rows), 13 * 2)
        robust_rows = [
            row for row in method_rows if row["method"] == "robust_functional"
        ]
        self.assertEqual(len(robust_rows), 2)
        for row in robust_rows:
            self.assertLess(row["worst_relative_delta"], 0.1)
        comparisons = analysis["comparisons"]
        self.assertTrue(
            any(c["second"] == "random_mean" for c in comparisons)
        )
        self.assertTrue(
            any(c["second"] == "robust_routing" for c in comparisons)
        )

        attach_protected_counts(self.allocations_dir)
        phase_dir = self.root / "development"
        paths = write_phase_outputs(analysis, phase_dir)
        for path in paths.values():
            self.assertTrue(path.is_file())
        decision_path = write_development_decision(analysis, self.root)
        payload = json.loads(decision_path.read_text())
        self.assertEqual(payload["decision"], decision)
        self.assertTrue(payload["method_never_modified_by_gate"])

        table = render_main_table(
            {"method_rows": analysis["method_rows"]}, "relative"
        )
        self.assertTrue(any("Robust-Functional" in line for line in table))

        figures = create_phase_figures(
            analysis, self.allocations_dir, self.root / "figures"
        )
        self.assertGreater(len(figures), 0)
        for path in figures:
            self.assertTrue(path.is_file())

    def test_bf16_biased_methods_have_positive_relative_delta(self) -> None:
        analysis = analyze_phase(
            "development",
            self.allocations_dir,
            self.losses_dir,
            RUN_FINGERPRINT,
            replicates=50,
            seed=5,
        )
        for row in analysis["method_rows"]:
            if row["method"] == "robust_functional":
                for domain in STAGE2B_DOMAINS:
                    self.assertGreater(row[f"recovery_{domain}"], 0.0)


if __name__ == "__main__":
    unittest.main()
