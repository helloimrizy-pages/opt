from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from expert_analysis.balanced import canonical_sha256, file_sha256
from expert_analysis.fragility_evaluation import (
    stage2c_phase_records,
    stage2c_run_fingerprint,
    verify_preregistration_unchanged,
    write_stage2c_preregistration,
)

DOMAINS = ("general", "math", "coding", "reasoning")


def _payload(marker: str = "a") -> dict:
    return {
        "schema": "stage2c_preregistration_v1",
        "marker": marker,
        "development_seed": 45,
        "final_seed": 44,
        "created_at_utc": "now",
    }


class PreregistrationImmutabilityTests(unittest.TestCase):
    def test_write_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            path, digest = write_stage2c_preregistration(results_dir, _payload())
            self.assertTrue(path.is_file())
            self.assertEqual(digest, file_sha256(path))
            self.assertEqual(verify_preregistration_unchanged(results_dir), digest)

    def test_modified_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            path, _ = write_stage2c_preregistration(results_dir, _payload())
            data = json.loads(path.read_text())
            data["development_seed"] = 43
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_preregistration_unchanged(results_dir)

    def test_rewrite_with_different_content_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            write_stage2c_preregistration(results_dir, _payload("a"))
            with self.assertRaises(RuntimeError):
                write_stage2c_preregistration(results_dir, _payload("b"))
            # Identical content (up to timestamp) is idempotent.
            payload = _payload("a")
            payload["created_at_utc"] = "later"
            write_stage2c_preregistration(results_dir, payload)

    def test_missing_preregistration_blocks_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                verify_preregistration_unchanged(Path(tmp))


class RunFingerprintTests(unittest.TestCase):
    def _bundle(self):
        return types.SimpleNamespace(
            checkpoint="allenai/OLMoE-1B-7B-0924",
            resolved_revision="rev",
            runtime=types.SimpleNamespace(dtype="torch.bfloat16"),
        )

    def test_fingerprint_depends_on_preregistration(self) -> None:
        registry = {"registry_sha256": "r" * 64}
        hashes = {d: "h" * 64 for d in DOMAINS}
        determinism = {"use_deterministic_algorithms": True}
        first = stage2c_run_fingerprint(
            self._bundle(), registry, "p1", hashes, "development", 1, determinism
        )
        second = stage2c_run_fingerprint(
            self._bundle(), registry, "p2", hashes, "development", 1, determinism
        )
        third = stage2c_run_fingerprint(
            self._bundle(), registry, "p1", hashes, "final", 1, determinism
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(
            first,
            stage2c_run_fingerprint(
                self._bundle(), registry, "p1", hashes, "development", 1, determinism
            ),
        )


def _write_record(directory: Path, content: dict) -> dict:
    name = content.pop("_file_name")
    record = dict(content)
    record["allocation_sha256"] = canonical_sha256(content)
    record["created_at_utc"] = "frozen"
    path = directory / name
    path.write_text(json.dumps(record), encoding="utf-8")
    return {
        "file": path.name,
        "method": record["method"],
        "method_label": record["method"],
        "method_kind": record["method_kind"],
        "regime": record["regime"],
        "budget_fraction": record["budget_fraction"],
        "allocation_sha256": record["allocation_sha256"],
        "file_sha256": file_sha256(path),
    }


class PhaseRecordSelectionTests(unittest.TestCase):
    def _registry(self, tmp: Path) -> tuple[dict, Path, Path]:
        stage2c_dir = tmp / "stage2c"
        stage2b_dir = tmp / "stage2b"
        stage2c_dir.mkdir()
        stage2b_dir.mkdir()
        new_entries = []
        reused_entries = []
        for regime in ("4to8", "3to8"):
            for fraction in (0.05, 0.10, 0.20, 0.30):
                name = f"fragility_robust_{regime}_budget{int(round(fraction*100))}.json"
                content = {
                    "_file_name": name,
                    "method": "fragility_robust",
                    "method_kind": "deterministic_milp",
                    "regime": regime,
                    "budget_fraction": fraction,
                }
                entry = _write_record(stage2c_dir, content)
                entry["source"] = "stage2c_new"
                new_entries.append(entry)
                name = f"robust_functional_{regime}_budget{int(round(fraction*100))}.json"
                content = {
                    "_file_name": name,
                    "method": "robust_functional",
                    "method_kind": "deterministic_milp",
                    "regime": regime,
                    "budget_fraction": fraction,
                }
                entry = _write_record(stage2b_dir, content)
                entry["source"] = "stage2b_frozen"
                reused_entries.append(entry)
        for reference in ("bf16_reference", "uniform_4bit_reference"):
            content = {
                "_file_name": f"{reference}.json",
                "method": reference,
                "method_kind": "uniform_reference",
                "regime": None,
                "budget_fraction": None,
            }
            entry = _write_record(stage2b_dir, content)
            entry["source"] = "stage2b_frozen"
            reused_entries.append(entry)
        registry = {
            "valid_regimes": ["4to8", "3to8"],
            "new_entries": new_entries,
            "reused_entries": reused_entries,
        }
        return registry, stage2c_dir, stage2b_dir

    def test_development_selects_only_twenty_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, stage2c_dir, stage2b_dir = self._registry(Path(tmp))
            references, competitors = stage2c_phase_records(
                registry, stage2c_dir, stage2b_dir, "development"
            )
            self.assertEqual(len(references), 2)
            self.assertEqual(len(competitors), 4)
            self.assertTrue(
                all(r["budget_fraction"] == 0.20 for r in competitors)
            )

    def test_final_requires_authorized_regimes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, stage2c_dir, stage2b_dir = self._registry(Path(tmp))
            with self.assertRaises(ValueError):
                stage2c_phase_records(registry, stage2c_dir, stage2b_dir, "final")
            references, competitors = stage2c_phase_records(
                registry, stage2c_dir, stage2b_dir, "final", ["3to8"]
            )
            self.assertEqual(len(competitors), 8)
            self.assertTrue(all(r["regime"] == "3to8" for r in competitors))

    def test_final_rejects_invalid_regime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, stage2c_dir, stage2b_dir = self._registry(Path(tmp))
            registry["valid_regimes"] = ["4to8"]
            with self.assertRaises(RuntimeError):
                stage2c_phase_records(
                    registry, stage2c_dir, stage2b_dir, "final", ["3to8"]
                )

    def test_invalid_regime_competitors_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry, stage2c_dir, stage2b_dir = self._registry(Path(tmp))
            registry["valid_regimes"] = ["4to8"]
            _, competitors = stage2c_phase_records(
                registry, stage2c_dir, stage2b_dir, "development"
            )
            self.assertTrue(all(r["regime"] == "4to8" for r in competitors))


if __name__ == "__main__":
    unittest.main()
