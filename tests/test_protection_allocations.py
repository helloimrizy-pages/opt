from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.protection_allocations import (
    generate_all_allocations,
    load_frozen_registry,
    verify_allocation_record,
    write_allocation_summaries,
)
from expert_analysis.protection_optimization import build_expert_memory_matrix
from expert_analysis.specialist_preservation import (
    STAGE2B_DOMAINS,
    SpecialistScores,
    build_importance_tensor,
    normalized_specialization,
    select_calibration_indices,
    specialization_margins,
)


def synthetic_scores(num_layers: int = 2, num_experts: int = 3) -> SpecialistScores:
    rng = np.random.default_rng(17)
    raw = {
        domain: rng.random((num_layers, num_experts)) + 0.05
        for domain in STAGE2B_DOMAINS
    }
    functional = build_importance_tensor(raw)
    margins = specialization_margins(functional)
    positive, spec = normalized_specialization(margins)
    routing_raw = {
        domain: rng.integers(1, 30, size=(num_layers, num_experts)).astype(float)
        for domain in STAGE2B_DOMAINS
    }
    routing = build_importance_tensor(routing_raw)
    routing_margins = specialization_margins(routing)
    routing_pos, routing_spec = normalized_specialization(routing_margins)
    metadata = {
        "calibration_fingerprint": "f" * 64,
        "score_hashes": {"functional_importance_sha256": "a" * 64},
        "domains_detail": {
            domain: {"single_domain_input_ids_sha256": "b" * 64}
            for domain in STAGE2B_DOMAINS
        },
    }
    stacked_raw = np.stack([raw[d] for d in STAGE2B_DOMAINS], axis=-1)
    return SpecialistScores(
        selection=select_calibration_indices(),
        functional_raw=stacked_raw,
        functional=functional,
        functional_specialization_raw=margins,
        functional_specialization_pos=positive,
        functional_specialization=spec,
        routing_raw=np.stack([routing_raw[d] for d in STAGE2B_DOMAINS], axis=-1),
        routing=routing,
        routing_specialization_raw=routing_margins,
        routing_specialization_pos=routing_pos,
        routing_specialization=routing_spec,
        single_domain_raw=stacked_raw,
        single_domain=functional,
        global_importance=functional.mean(axis=2),
        metadata=metadata,
    )


class AllocationFreezingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scores = synthetic_scores()
        self.memory = build_expert_memory_matrix(
            [[(4, 4)], [(4, 4)]], group_size=2, num_experts=3
        )
        self.registry = generate_all_allocations(
            self.scores, self.memory, self.root / "allocations"
        )

    def test_registry_contains_every_preregistered_allocation(self) -> None:
        entries = self.registry["entries"]
        deterministic = [e for e in entries if e["method_kind"] == "deterministic_milp"]
        randoms = [e for e in entries if e["method_kind"] == "random"]
        references = [e for e in entries if e["method_kind"] == "uniform_reference"]
        self.assertEqual(len(deterministic), 8 * 2 * 4)
        self.assertEqual(len(randoms), 5 * 2 * 4)
        self.assertEqual(len(references), 4)
        self.assertTrue(self.registry["frozen"])

    def test_frozen_registry_verifies_and_detects_tampering(self) -> None:
        allocations_dir = self.root / "allocations"
        load_frozen_registry(allocations_dir)
        target = allocations_dir / "robust_functional_4to8_budget20.json"
        record = json.loads(target.read_text())
        record["protected_expert_count"] = 999
        target.write_text(json.dumps(record))
        with self.assertRaises(RuntimeError):
            load_frozen_registry(allocations_dir)

    def test_allocation_sha_detects_bit_edit(self) -> None:
        allocations_dir = self.root / "allocations"
        record = json.loads(
            (allocations_dir / "global_importance_3to8_budget10.json").read_text()
        )
        verify_allocation_record(record)
        record["expert_bits"][0][0] = 8 if record["expert_bits"][0][0] != 8 else 3
        with self.assertRaises(RuntimeError):
            verify_allocation_record(record)

    def test_every_allocation_respects_its_budget_and_regime_bits(self) -> None:
        allocations_dir = self.root / "allocations"
        for entry in self.registry["entries"]:
            record = json.loads((allocations_dir / entry["file"]).read_text())
            bits = np.asarray(record["expert_bits"])
            if record["method_kind"] == "uniform_reference":
                self.assertFalse(record["matched_budget_competitor"])
                self.assertTrue(np.all(bits == record["base_bits"]))
                continue
            base = record["base_bits"]
            self.assertIn(base, (3, 4))
            self.assertTrue(np.all(np.isin(bits, (base, 8))))
            self.assertLessEqual(
                record["used_protection_bytes"], record["budget_bytes"]
            )
            protected = int((bits == 8).sum())
            self.assertEqual(protected, record["protected_expert_count"])

    def test_random_allocations_never_exceed_max_min_optimum(self) -> None:
        allocations_dir = self.root / "allocations"
        by_key: dict[tuple[str, float], dict[str, float]] = {}
        optimum: dict[tuple[str, float], float] = {}
        for entry in self.registry["entries"]:
            record = json.loads((allocations_dir / entry["file"]).read_text())
            if record["method_kind"] == "uniform_reference":
                continue
            key = (record["regime"], record["budget_fraction"])
            by_key.setdefault(key, {})[record["method"]] = record[
                "functional_specialist_coverage_min"
            ]
            if record["method"] == "robust_functional":
                optimum[key] = record["solver_metadata"]["objective_z"]
        for key, methods in by_key.items():
            for method, value in methods.items():
                self.assertLessEqual(value, optimum[key] + 1e-8, (key, method))

    def test_summaries_are_written(self) -> None:
        coverage_path, summary_path = write_allocation_summaries(
            self.root / "allocations", self.root
        )
        self.assertTrue(coverage_path.is_file())
        self.assertTrue(summary_path.is_file())
        header = coverage_path.read_text().splitlines()[0]
        for domain in STAGE2B_DOMAINS:
            self.assertIn(f"coverage_{domain}", header)


if __name__ == "__main__":
    unittest.main()
