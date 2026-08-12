from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.analysis import analyze_results
from expert_analysis.io_utils import atomic_write_json
from expert_analysis.metrics import DomainStatistics
from expert_analysis.report import write_summary


class AnalysisTests(unittest.TestCase):
    def test_end_to_end_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            domains = ["general", "math", "coding", "reasoning"]
            atomic_write_json(
                output / "collection_config.json",
                {
                    "domains": domains,
                    "model": "toy/model",
                    "model_revision": None,
                    "resolved_model_revision": "abc",
                    "seed": 42,
                    "max_length": 32,
                    "num_examples": 6,
                    "quick": True,
                    "include_reference_answers": True,
                    "compute_gradient_attribution": False,
                    "device": "cpu",
                    "device_description": "test CPU",
                    "dtype": "float32",
                    "package_versions": {},
                },
            )
            layers = [
                {
                    "model_layer_index": index,
                    "block_name": f"layers.{index}.mlp",
                    "num_experts": 4,
                    "top_k": 2,
                }
                for index in range(2)
            ]
            atomic_write_json(
                output / "architecture.json",
                {
                    "model_class": "Toy",
                    "num_moe_layers": 2,
                    "num_experts": 4,
                    "layers": layers,
                },
            )
            domain_dir = output / "domains"
            rng = np.random.default_rng(10)
            for domain_index, domain in enumerate(domains):
                stats = DomainStatistics.zeros(
                    6, 2, 4, [item["block_name"] for item in layers]
                )
                stats.routing_counts[:] = rng.integers(1, 8, size=(6, 2, 4))
                stats.gate_sums[:] = rng.random((6, 2, 4)) + domain_index * 0.03
                stats.contribution_sums[:] = (
                    rng.random((6, 2, 4)) + domain_index * np.arange(4) * 0.02
                )
                stats.token_counts[:] = rng.integers(10, 25, size=6)
                stats.save(domain_dir / f"{domain}.npz")
                atomic_write_json(
                    domain_dir / f"{domain}.metadata.json",
                    {
                        "repository": f"test/{domain}",
                        "config": "main",
                        "split": "test",
                        "substituted": False,
                    },
                )
            results = analyze_results(
                output, bootstrap_replicates=5, specialized_per_layer=2
            )
            write_summary(results, output / "SUMMARY.md")
            expected = [
                "expert_importance_by_domain.csv",
                "cross_domain_correlations.csv",
                "topk_overlap.csv",
                "routing_vs_functional_correlation.csv",
                "domain_specialized_experts.csv",
                "same_domain_split_half.csv",
                "results.json",
                "SUMMARY.md",
            ]
            for filename in expected:
                self.assertTrue((output / filename).exists(), filename)
            self.assertEqual(len(results["expert_importance"]), 4 * 2 * 4)
            self.assertEqual(len(results["same_domain_split_half"]), 4 * 3 * 3)
            self.assertIn("# Go / No-Go Assessment", (output / "SUMMARY.md").read_text())


if __name__ == "__main__":
    unittest.main()
