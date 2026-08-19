from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import make_trace, workload_config
from residency_headroom.common import read_json, read_jsonl
from residency_headroom.diagnostics import all_workload_diagnostics
from residency_headroom.evaluation import run_evaluation
from residency_headroom.freeze import freeze_evaluation
from residency_headroom.reporting import write_analysis_outputs
from residency_headroom.simulator import (
    expected_unlimited_misses,
    per_sequence_rows,
    policy_specs,
    result_rows,
    simulate_oracle,
    simulate_policy,
)
from residency_headroom.statistics import analyze_headroom
from residency_headroom.workloads import (
    build_calibration_workload,
    build_workloads,
    calibration_frequency_scores,
    split_sequences,
)


class SimulatorStatisticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = make_trace(generation_length=2)
        self.config = workload_config()
        self.config["bootstrap_replicates"] = 20
        self.split = split_sequences(self.trace, self.config["calibration_fraction"])
        self.workloads = build_workloads(self.trace, self.split, self.config)
        calibration = build_calibration_workload(self.trace, self.split, 99)
        self.static_scores = calibration_frequency_scores(self.trace, calibration)

    def test_metrics_byte_accounting_and_cost_rows(self) -> None:
        workload = self.workloads[0]
        result = simulate_policy(self.trace, workload, 4, "lru")
        self.assertEqual(result.hits + result.misses, result.requests)
        self.assertEqual(result.admissions, result.misses)
        self.assertEqual(result.bytes_transferred, result.misses * 1024)
        rows = result_rows(
            result,
            lambda_values=[0, 0.5],
            cost_models=["unit_miss", "expert_bytes"],
            trace_hash=self.trace.trace_hash,
            config_hash="config",
            domain_label="general",
            selected_decay_alpha=0.95,
        )
        self.assertEqual(len(rows), 4)
        unit = next(row for row in rows if row["cost_model"] == "unit_miss" and row["lambda"] == 0.5)
        self.assertEqual(unit["total_cost"], 1.5 * result.misses)
        self.assertEqual(unit["requests"], result.requests)
        self.assertEqual(
            sum(row["misses"] for row in per_sequence_rows(result, trace_hash="x", config_hash="y")),
            result.misses,
        )

    def test_oracle_dominance_monotonicity_and_unlimited_sanity(self) -> None:
        workload = next(item for item in self.workloads if item.name == "general_to_coding")
        oracle_costs = []
        for capacity in self.config["cache_capacities"]:
            oracle = simulate_oracle(self.trace, workload, capacity)
            lru = simulate_policy(self.trace, workload, capacity, "lru")
            lfu = simulate_policy(self.trace, workload, capacity, "lfu")
            self.assertLessEqual(oracle.misses, lru.misses)
            self.assertLessEqual(oracle.misses, lfu.misses)
            oracle_costs.append(oracle.misses)
        self.assertTrue(all(after <= before for before, after in zip(oracle_costs, oracle_costs[1:])))
        unlimited = simulate_oracle(self.trace, workload, self.trace.num_experts)
        self.assertEqual(unlimited.misses, expected_unlimited_misses(self.trace, workload))

    def test_bootstrap_aggregation_is_deterministic_and_reporting_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = freeze_evaluation(
                self.trace, self.config, root / "frozen", oracle_random_cases=5
            )
            evaluation_dir = root / "evaluation"
            manifest = run_evaluation(
                self.trace, frozen, root / "frozen", evaluation_dir
            )
            self.assertTrue(manifest["sanity_passed"])
            self.assertTrue(read_json(evaluation_dir / "sanity_checks.json")["passed"])
            all_rows = read_jsonl(evaluation_dir / "results.jsonl")
            all_sequences = read_jsonl(evaluation_dir / "per_sequence_results.jsonl")
            diagnostics = all_workload_diagnostics(self.trace, self.workloads)
            first = analyze_headroom(all_rows, all_sequences, diagnostics, frozen)
            second = analyze_headroom(all_rows, all_sequences, diagnostics, frozen)
            self.assertEqual(first["decision"], second["decision"])
            self.assertEqual(first["regime_headroom"], second["regime_headroom"])
            self.assertEqual(first["decision"]["decision"], "PILOT_ONLY_NO_STAGE0_DECISION")
            self.assertEqual(
                first["primary_definition"]["regime_bootstrap_cluster"],
                "source_sequence_id, stratified by domain across workload components",
            )
            self.assertEqual(len(first["regime_headroom"]), 20)
            for row in first["regime_headroom"]:
                self.assertLessEqual(row["oracle_cost"], row["best_simple_cost"])
                self.assertGreaterEqual(row["headroom"], 0)
            outputs = write_analysis_outputs(first, frozen, root / "report")
            self.assertTrue(Path(outputs["report"]).exists())
            self.assertTrue((root / "report" / "pilot_audit_report.md").exists())
            self.assertTrue((root / "report" / "figures" / "figure2_oracle_headroom.png").exists())


if __name__ == "__main__":
    unittest.main()
