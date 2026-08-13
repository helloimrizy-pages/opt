from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.balanced import (
    BALANCED_DOMAINS,
    match_routing_controls,
    rank_candidate_experts,
    select_specialized_experts,
)
from expert_analysis.balanced_analysis import (
    analyze_balanced_interventions,
    create_balanced_figures,
    intervention_panel,
    write_balanced_outputs,
)
from expert_analysis.masking import LossStatistics
from expert_analysis.metrics import DomainStatistics


class BalancedCausalValidationTests(unittest.TestCase):
    @staticmethod
    def statistics() -> dict[str, DomainStatistics]:
        examples = 100
        layers = 4
        experts = 64
        specialist_ids = {
            domain: [(0, domain_index), (1, 8 + domain_index), (2, 16 + domain_index)]
            for domain_index, domain in enumerate(BALANCED_DOMAINS)
        }
        output = {}
        for domain_index, domain in enumerate(BALANCED_DOMAINS):
            stats = DomainStatistics.zeros(
                examples,
                layers,
                experts,
                [f"layers.{layer}.mlp" for layer in range(layers)],
            )
            stats.token_counts[:] = 64
            stats.routing_counts[:] = 1
            stats.gate_sums[:] = 0.1
            base = np.linspace(0.5, 1.5, experts)
            stats.contribution_sums[:] = base[None, None, :]
            for specialist_domain, identities in specialist_ids.items():
                for layer, expert_id in identities:
                    if specialist_domain == domain:
                        stats.contribution_sums[:, layer, expert_id] = 100.0
                        stats.routing_counts[:, layer, expert_id] = 32
                    else:
                        stats.contribution_sums[:, layer, expert_id] = 1e-4
            output[domain] = stats
        return output

    def test_selection_and_control_matching_are_deterministic(self) -> None:
        statistics = self.statistics()
        first_candidates = rank_candidate_experts(statistics)
        second_candidates = rank_candidate_experts(statistics)
        self.assertEqual(first_candidates, second_candidates)
        selected, tiers = select_specialized_experts(first_candidates)
        controls, control_tiers = match_routing_controls(first_candidates, selected)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len(controls), 12)
        self.assertEqual(set(tiers.values()), {"strict"})
        self.assertTrue(all(not row["fallback_used"] for row in controls))
        for domain in BALANCED_DOMAINS:
            rows = [row for row in selected if row["target_domain"] == domain]
            self.assertEqual(len(rows), 3)
            self.assertEqual(len({row["layer"] for row in rows}), 3)
            self.assertTrue(all(row["eligible_strict"] for row in rows))
        control_ids = {
            (pair["control"]["layer"], pair["control"]["expert_id"])
            for pair in controls
        }
        self.assertEqual(len(control_ids), 12)
        for pair in controls:
            self.assertTrue(pair["same_layer"])
            self.assertEqual(pair["specialized"]["layer"], pair["control"]["layer"])
            self.assertLessEqual(
                pair["control"]["specialization_margin"],
                0.25 * pair["specialized"]["specialization_margin"] + 1e-15,
            )
        self.assertTrue(all(value["same_layer"] for value in control_tiers.values()))

    def test_balanced_analysis_outputs_and_figures(self) -> None:
        statistics = self.statistics()
        candidates = rank_candidate_experts(statistics)
        selected, _ = select_specialized_experts(candidates)
        controls, _ = match_routing_controls(candidates, selected)
        preregistration = {"matched_controls": controls}
        panel = intervention_panel(preregistration)
        baselines = {
            domain: LossStatistics(
                loss_sums=np.full(100, 192.0, dtype=np.float64),
                token_counts=np.full(100, 64, dtype=np.uint32),
            )
            for domain in BALANCED_DOMAINS
        }
        masked = {}
        provenance = {}
        example_noise = np.linspace(-0.001, 0.001, 100)
        for intervention in panel:
            identity = (intervention["layer"], intervention["expert_id"])
            provenance[identity] = {
                "source": "new_inference",
                "hooks_before": 0,
                "hooks_after": 0,
            }
            target_effect = 0.010 if intervention["role"] == "specialized" else 0.003
            non_target_effect = 0.001
            for domain in BALANCED_DOMAINS:
                delta = (
                    target_effect
                    if domain == intervention["target_domain"]
                    else non_target_effect
                ) + example_noise
                route_counts = statistics[domain].routing_counts[
                    :, intervention["layer"], intervention["expert_id"]
                ]
                masked[(identity[0], identity[1], domain)] = LossStatistics(
                    loss_sums=baselines[domain].loss_sums
                    + delta * baselines[domain].token_counts,
                    token_counts=baselines[domain].token_counts.copy(),
                    route_counts=route_counts.copy(),
                    zeroed_gate_mass=np.ones(100, dtype=np.float64),
                )
        analysis, arrays = analyze_balanced_interventions(
            preregistration,
            baselines,
            masked,
            provenance,
            bootstrap_replicates=10,
            seed=42,
        )
        self.assertEqual(len(analysis["masking_results"]), 96)
        self.assertEqual(len(analysis["pairwise_domain_contrasts"]), 72)
        self.assertEqual(len(analysis["specialized_vs_control"]), 12)
        self.assertEqual(len(analysis["aggregate_results"]), 5)
        self.assertEqual(arrays["per_example_loss_changes"].shape, (24, 4, 100))
        self.assertEqual(analysis["decision"]["label"], "STRONG GO")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_balanced_outputs(analysis, arrays, output)
            results = {
                "preregistration": preregistration,
                "balanced_analysis": analysis,
            }
            figure_paths = create_balanced_figures(results, output)
            self.assertEqual(len(figure_paths), 10)
            self.assertTrue(all(path.stat().st_size > 0 for path in figure_paths))


if __name__ == "__main__":
    unittest.main()
