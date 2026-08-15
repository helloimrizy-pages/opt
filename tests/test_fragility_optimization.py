from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.balanced import canonical_sha256, file_sha256
from expert_analysis.fragility import (
    STAGE2C_REGIMES,
    Stage2BScoreArtifacts,
    build_calibration_fragility_record,
    compute_regime_fragility,
)
from expert_analysis.fragility_optimization import (
    build_fragility_allocation_record,
    generate_fragility_robust_allocations,
    load_frozen_stage2c_registry,
    predicted_residual_risk,
    solve_fragility_robust,
)
from expert_analysis.protection_optimization import (
    BASE_BITS_BY_REGIME,
    PROTECTION_FRACTIONS,
    RANDOM_ALLOCATION_SEEDS,
    build_expert_memory_matrix,
)

DOMAINS = ("general", "math", "coding", "reasoning")


def small_specialization(num_layers: int = 2, num_experts: int = 4) -> np.ndarray:
    """Each domain's specialists live in a distinct expert column pattern."""

    rng = np.random.default_rng(7)
    raw = rng.random((num_layers, num_experts, 4)) * 0.05
    for domain_index in range(4):
        raw[:, domain_index, domain_index] += 1.0
    return raw / raw.sum(axis=(0, 1))[None, None, :]


def small_memory():
    return build_expert_memory_matrix(
        [[(8, 4), (8, 8)]] * 2, group_size=4, num_experts=4
    )


def fake_fragility_record(uniform4=(2.6, 2.1, 2.05, 2.2), uniform3=(3.0, 2.4, 2.2, 2.6)):
    bf16 = dict(zip(DOMAINS, (2.0, 2.0, 2.0, 2.0), strict=True))
    base = {
        4: dict(zip(DOMAINS, uniform4, strict=True)),
        3: dict(zip(DOMAINS, uniform3, strict=True)),
    }
    regime_results = {
        regime: compute_regime_fragility(bf16, base[bits], bits)
        for regime, bits in STAGE2C_REGIMES.items()
    }
    return build_calibration_fragility_record(
        regime_results=regime_results,
        calibration_subset_hashes={d: {"input_ids_sha256": "0" * 64} for d in DOMAINS},
        model_info={"model": "test"},
        qdq_config={"group_size": 128},
        environment={},
        reproduction={"all_reproduced": True},
    )


def fake_scores(specialization: np.ndarray) -> Stage2BScoreArtifacts:
    return Stage2BScoreArtifacts(
        functional=specialization.copy(),
        functional_specialization=specialization,
        routing_specialization=specialization.copy(),
        single_domain=specialization.copy(),
        global_importance=specialization.mean(axis=2),
        calibration_indices={d: list(range(25)) for d in DOMAINS},
        metadata={
            "calibration_fingerprint": "f" * 64,
            "score_hashes": {"functional_specialization_sha256": "a" * 64},
        },
    )


class ResidualRiskTests(unittest.TestCase):
    def test_residual_risk_formula(self) -> None:
        q = np.asarray([2.0, 1.0, 0.5, 0.5])
        coverage = np.asarray([0.5, 0.0, 1.0, 0.25])
        risk = predicted_residual_risk(q, coverage)
        np.testing.assert_allclose(risk, [1.0, 1.0, 0.0, 0.375])

    def test_residual_risk_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            predicted_residual_risk(np.asarray([1.0, 1.0]), np.asarray([0.0, 0.0]))
        with self.assertRaises(ValueError):
            predicted_residual_risk(
                np.asarray([-1.0, 1.0, 1.0, 1.0]), np.zeros(4)
            )
        with self.assertRaises(ValueError):
            predicted_residual_risk(np.ones(4), np.asarray([0.0, 0.0, 0.0, 1.5]))


class FragilityRobustMilpTests(unittest.TestCase):
    def test_solution_is_brute_force_optimal(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(4)
        budget = int(delta.sum() * 0.5)
        q = np.asarray([1.8, 1.0, 0.6, 0.6])
        result = solve_fragility_robust(scores, q, delta, budget)
        best = np.inf
        flat_delta = delta.reshape(-1)
        flat_scores = scores.reshape(-1, 4)
        for assignment in itertools.product((0, 1), repeat=delta.size):
            x = np.asarray(assignment)
            if int((flat_delta * x).sum()) > budget:
                continue
            coverage = flat_scores.T @ x
            best = min(best, float((q * (1 - coverage)).max()))
        self.assertAlmostEqual(result.objective_value, best, places=9)

    def test_fragile_domain_receives_more_coverage(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(4)
        budget = int(delta.sum() * 0.3)
        uniform = solve_fragility_robust(
            scores, np.ones(4), delta, budget
        )
        skewed = solve_fragility_robust(
            scores, np.asarray([2.5, 0.5, 0.5, 0.5]), delta, budget
        )
        self.assertGreaterEqual(
            skewed.metadata["coverage_by_domain"]["general"],
            uniform.metadata["coverage_by_domain"]["general"],
        )

    def test_memory_budget_is_respected_exactly(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(3)
        budget = int(delta.reshape(-1)[0])
        result = solve_fragility_robust(scores, np.ones(4), delta, budget)
        self.assertLessEqual(
            result.metadata["used_protection_bytes"], budget
        )
        self.assertLessEqual(result.metadata["protected_expert_count"], 1)

    def test_zero_budget_protects_nothing(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(4)
        q = np.asarray([1.5, 1.0, 0.75, 0.75])
        result = solve_fragility_robust(scores, q, delta, 0)
        self.assertEqual(result.metadata["protected_expert_count"], 0)
        self.assertAlmostEqual(result.objective_value, float(q.max()))

    def test_all_zero_fragility_is_rejected(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(4)
        with self.assertRaises(ValueError):
            solve_fragility_robust(scores, np.zeros(4), delta, int(delta.sum()))

    def test_zero_fragility_domain_is_deprioritized(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(4)
        budget = int(delta.reshape(-1)[0]) * 2
        q = np.asarray([2.0, 2.0, 0.0, 0.0])
        result = solve_fragility_robust(scores, q, delta, budget)
        risk = result.metadata["residual_risk_by_domain"]
        self.assertEqual(risk["coding"], 0.0)
        self.assertEqual(risk["reasoning"], 0.0)

    def test_deterministic_allocation(self) -> None:
        scores = small_specialization()
        memory = small_memory()
        delta = memory.delta_protection_bytes(4)
        budget = int(delta.sum() * 0.4)
        q = np.asarray([1.2, 1.1, 0.9, 0.8])
        first = solve_fragility_robust(scores, q, delta, budget)
        second = solve_fragility_robust(scores, q, delta, budget)
        np.testing.assert_array_equal(first.protected, second.protected)


class AllocationRecordTests(unittest.TestCase):
    def test_record_is_consistent_and_hash_verified(self) -> None:
        from expert_analysis.protection_allocations import verify_allocation_record

        scores_array = small_specialization()
        scores = fake_scores(scores_array)
        memory = small_memory()
        fragility = fake_fragility_record()
        delta = memory.delta_protection_bytes(4)
        budget = memory.protection_budget_bytes(4, 0.20)
        q = np.asarray(
            [
                fragility["regimes"]["4to8"]["domains"][d]["normalized_fragility"]
                for d in DOMAINS
            ]
        )
        solution = solve_fragility_robust(scores_array, q, delta, budget)
        record = build_fragility_allocation_record(
            "4to8", 0.20, solution, scores, memory, fragility
        )
        verify_allocation_record(record)
        self.assertEqual(record["method"], "fragility_robust")
        self.assertEqual(record["base_bits"], 4)
        bits = np.asarray(record["expert_bits"])
        self.assertTrue(np.all(np.isin(bits, (4, 8))))
        coverage = np.asarray(
            [record["functional_specialist_coverage"][d] for d in DOMAINS]
        )
        risk = np.asarray([record["predicted_residual_risk"][d] for d in DOMAINS])
        np.testing.assert_allclose(risk, q * (1 - coverage), atol=1e-12)
        self.assertAlmostEqual(
            record["predicted_max_residual_risk"], float(risk.max())
        )
        self.assertEqual(record["fragility_sha256"], fragility["fragility_sha256"])


def _fake_stage2b_comparators(directory: Path) -> dict:
    """Fabricate a minimal frozen Stage 2B registry with zero-coverage records."""

    directory.mkdir(parents=True, exist_ok=True)
    methods = [
        "robust_functional", "robust_routing", "average_specialization",
        "global_importance", "general_only", "math_only", "coding_only",
        "reasoning_only",
        *[f"random_seed{seed}" for seed in RANDOM_ALLOCATION_SEEDS],
    ]
    entries = []
    for regime in BASE_BITS_BY_REGIME:
        for fraction in PROTECTION_FRACTIONS:
            for method in methods:
                content = {
                    "schema": "stage2b_allocation_v1",
                    "method": method,
                    "method_label": method,
                    "method_kind": "deterministic_milp",
                    "regime": regime,
                    "budget_fraction": fraction,
                    "expert_bits": [[BASE_BITS_BY_REGIME[regime]] * 4] * 2,
                    "functional_specialist_coverage": {d: 0.0 for d in DOMAINS},
                }
                record = dict(content)
                record["allocation_sha256"] = canonical_sha256(content)
                record["created_at_utc"] = "frozen"
                name = f"{method}_{regime}_budget{int(round(fraction * 100))}.json"
                path = directory / name
                path.write_text(json.dumps(record), encoding="utf-8")
                entries.append(
                    {
                        "file": name,
                        "method": method,
                        "method_label": method,
                        "method_kind": "deterministic_milp",
                        "regime": regime,
                        "budget_fraction": fraction,
                        "allocation_sha256": record["allocation_sha256"],
                        "file_sha256": file_sha256(path),
                    }
                )
    for name, bits in (
        ("bf16_reference", 16), ("uniform_8bit_reference", 8),
        ("uniform_4bit_reference", 4), ("uniform_3bit_reference", 3),
    ):
        content = {
            "schema": "stage2b_allocation_v1",
            "method": name,
            "method_label": name,
            "method_kind": "uniform_reference",
            "regime": None,
            "budget_fraction": None,
            "expert_bits": [[bits] * 4] * 2,
        }
        record = dict(content)
        record["allocation_sha256"] = canonical_sha256(content)
        record["created_at_utc"] = "frozen"
        path = directory / f"{name}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        entries.append(
            {
                "file": f"{name}.json",
                "method": name,
                "method_label": name,
                "method_kind": "uniform_reference",
                "regime": None,
                "budget_fraction": None,
                "allocation_sha256": record["allocation_sha256"],
                "file_sha256": file_sha256(path),
            }
        )
    return {"registry_sha256": "e" * 64, "entries": entries}


class RegistryGenerationTests(unittest.TestCase):
    def test_generate_freezes_all_eight_and_reuses_comparators(self) -> None:
        scores_array = small_specialization()
        scores = fake_scores(scores_array)
        memory = small_memory()
        fragility = fake_fragility_record()
        with tempfile.TemporaryDirectory() as tmp:
            stage2b_dir = Path(tmp) / "stage2b_allocations"
            stage2c_dir = Path(tmp) / "stage2c_allocations"
            stage2b_registry = _fake_stage2b_comparators(stage2b_dir)
            registry = generate_fragility_robust_allocations(
                scores, memory, fragility, stage2c_dir, stage2b_registry, stage2b_dir
            )
            self.assertEqual(len(registry["new_entries"]), 8)
            self.assertEqual(
                len(registry["reused_entries"]), len(stage2b_registry["entries"])
            )
            self.assertEqual(registry["valid_regimes"], ["4to8", "3to8"])
            self.assertEqual(registry["development_seed"], 45)
            self.assertEqual(registry["final_seed"], 44)
            reloaded = load_frozen_stage2c_registry(stage2c_dir, stage2b_dir)
            self.assertEqual(
                reloaded["registry_sha256"], registry["registry_sha256"]
            )
            for entry in registry["new_entries"]:
                self.assertTrue((stage2c_dir / entry["file"]).is_file())

    def test_reused_comparator_tampering_is_detected(self) -> None:
        scores_array = small_specialization()
        scores = fake_scores(scores_array)
        memory = small_memory()
        fragility = fake_fragility_record()
        with tempfile.TemporaryDirectory() as tmp:
            stage2b_dir = Path(tmp) / "stage2b_allocations"
            stage2c_dir = Path(tmp) / "stage2c_allocations"
            stage2b_registry = _fake_stage2b_comparators(stage2b_dir)
            generate_fragility_robust_allocations(
                scores, memory, fragility, stage2c_dir, stage2b_registry, stage2b_dir
            )
            target = stage2b_dir / "global_importance_4to8_budget20.json"
            record = json.loads(target.read_text())
            record["functional_specialist_coverage"]["general"] = 0.9
            target.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_frozen_stage2c_registry(stage2c_dir, stage2b_dir)

    def test_invalid_regime_is_skipped(self) -> None:
        scores_array = small_specialization()
        scores = fake_scores(scores_array)
        memory = small_memory()
        # 3-bit base improves every domain: the 3to8 regime is invalid.
        fragility = fake_fragility_record(uniform3=(1.9, 1.9, 1.9, 1.9))
        with tempfile.TemporaryDirectory() as tmp:
            stage2b_dir = Path(tmp) / "stage2b_allocations"
            stage2c_dir = Path(tmp) / "stage2c_allocations"
            stage2b_registry = _fake_stage2b_comparators(stage2b_dir)
            registry = generate_fragility_robust_allocations(
                scores, memory, fragility, stage2c_dir, stage2b_registry, stage2b_dir
            )
            self.assertEqual(registry["valid_regimes"], ["4to8"])
            self.assertEqual(registry["invalid_regimes"], ["3to8"])
            self.assertEqual(len(registry["new_entries"]), 4)


if __name__ == "__main__":
    unittest.main()
