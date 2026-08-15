from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from expert_analysis.stage2c_preflight import (
    verify_seed44_untouched,
    verify_stage2c_upstream,
)

RESULTS_ROOT = Path("results")
STAGE2B_DIR = RESULTS_ROOT / "robust_specialist_preservation"


@unittest.skipUnless(STAGE2B_DIR.is_dir(), "frozen results not present")
class Stage2cUpstreamTests(unittest.TestCase):
    def test_frozen_prior_state_passes(self) -> None:
        report = verify_stage2c_upstream(RESULTS_ROOT)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["stage2b"]["decision"], "ROBUST_PRESERVATION_NO_GO"
        )
        self.assertTrue(report["stage2b"]["historical_motivation_only"])

    def _tampered_root(self, tmp: Path, mutate) -> Path:
        """Symlink the real results tree, replacing the Stage 2B decision."""

        root = tmp / "results"
        root.mkdir()
        for name in (
            "expert_domain_balanced_causal_validation",
            "expert_quantization_pilot",
            "quantization_cost_surrogate",
        ):
            os.symlink((RESULTS_ROOT / name).resolve(), root / name)
        stage2b = root / "robust_specialist_preservation"
        stage2b.mkdir()
        os.symlink(
            (STAGE2B_DIR / "allocations").resolve(), stage2b / "allocations"
        )
        os.symlink((STAGE2B_DIR / "splits").resolve(), stage2b / "splits")
        decision = json.loads(
            (STAGE2B_DIR / "stage2b_decision.json").read_text()
        )
        mutate(decision)
        (stage2b / "stage2b_decision.json").write_text(
            json.dumps(decision), encoding="utf-8"
        )
        return root

    def test_altered_stage2b_gate_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def mutate(decision):
                decision["development_gates"]["4to8"]["gate_a"][
                    "robust_worst_relative_delta"
                ] = 0.001

            root = self._tampered_root(Path(tmp), mutate)
            with self.assertRaises(RuntimeError):
                verify_stage2c_upstream(root)

    def test_non_no_go_stage2b_decision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def mutate(decision):
                decision["decision"] = "FULL_EVALUATION_GO"
                decision["development_decision"]["passing_regimes"] = ["3to8"]

            root = self._tampered_root(Path(tmp), mutate)
            with self.assertRaises(RuntimeError):
                verify_stage2c_upstream(root)


@unittest.skipUnless(STAGE2B_DIR.is_dir(), "frozen results not present")
class Seed44IsolationTests(unittest.TestCase):
    def test_untouched_seed44_passes(self) -> None:
        report = verify_seed44_untouched(RESULTS_ROOT)
        self.assertTrue(report["passed"])
        self.assertTrue(report["verified_without_model_evaluation"])
        for domain in ("general", "math", "coding", "reasoning"):
            self.assertEqual(report["domains"][domain]["num_examples"], 100)

    def test_stage2c_final_outputs_block_when_unauthorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage2c_dir = Path(tmp) / "fragility_robust_preservation"
            losses = stage2c_dir / "final_seed44" / "losses" / "x"
            losses.mkdir(parents=True)
            (losses / "general.npz").write_bytes(b"data")
            with self.assertRaises(RuntimeError):
                verify_seed44_untouched(RESULTS_ROOT, stage2c_dir)
            report = verify_seed44_untouched(
                RESULTS_ROOT, stage2c_dir, allow_authorized_final=True
            )
            self.assertTrue(report["stage2c_final_outputs_exist"])


if __name__ == "__main__":
    unittest.main()
