from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from expert_analysis.stage3_preflight import (
    verify_seed44_untouched_stage3,
    verify_stage3_upstream,
)

RESULTS_ROOT = Path("results")
STAGE2C_DIR = RESULTS_ROOT / "fragility_robust_preservation"


@unittest.skipUnless(STAGE2C_DIR.is_dir(), "frozen results not present")
class Stage3UpstreamTests(unittest.TestCase):
    def test_frozen_prior_state_passes(self) -> None:
        report = verify_stage3_upstream(RESULTS_ROOT)
        self.assertTrue(report["passed"])
        self.assertEqual(report["stage2c"]["decision"], "FRAGILITY_ROBUST_NO_GO")
        self.assertTrue(report["stage2c"]["historical_motivation_only"])
        self.assertTrue(
            report["measured_not_estimated"]["stage2c_no_reweighting_respected"]
        )

    def _tampered_root(self, tmp: Path, mutate) -> Path:
        """Symlink the real results tree, replacing the Stage 2C decision."""

        root = tmp / "results"
        root.mkdir()
        for name in (
            "expert_domain_balanced_causal_validation",
            "expert_quantization_pilot",
            "quantization_cost_surrogate",
            "robust_specialist_preservation",
        ):
            os.symlink((RESULTS_ROOT / name).resolve(), root / name)
        stage2c = root / "fragility_robust_preservation"
        stage2c.mkdir()
        for name in ("allocations", "splits", "calibration", "audits"):
            source = STAGE2C_DIR / name
            if source.exists():
                os.symlink(source.resolve(), stage2c / name)
        decision = json.loads((STAGE2C_DIR / "stage2c_decision.json").read_text())
        mutate(decision)
        (stage2c / "stage2c_decision.json").write_text(
            json.dumps(decision), encoding="utf-8"
        )
        return root

    def test_altered_stage2c_gate_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def mutate(decision):
                decision["development_gates"]["4to8"]["gate_a"][
                    "fragility_robust_worst_relative_delta"
                ] = 0.001

            root = self._tampered_root(Path(tmp), mutate)
            with self.assertRaises(RuntimeError):
                verify_stage3_upstream(root)

    def test_non_no_go_stage2c_decision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:

            def mutate(decision):
                decision["decision"] = "FINAL_CONFIRMATION_GO"
                decision["development_decision"]["authorized_regimes"] = ["3to8"]

            root = self._tampered_root(Path(tmp), mutate)
            with self.assertRaises(RuntimeError):
                verify_stage3_upstream(root)


@unittest.skipUnless(STAGE2C_DIR.is_dir(), "frozen results not present")
class Seed44IsolationTests(unittest.TestCase):
    def test_untouched_seed44_passes(self) -> None:
        report = verify_seed44_untouched_stage3(RESULTS_ROOT)
        self.assertTrue(report["passed"])
        self.assertFalse(report["stage3_final_outputs_exist"])

    def test_stage3_final_outputs_block_when_unauthorized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stage3_dir = Path(tmp) / "measured_damage_preservation"
            losses = stage3_dir / "final_seed44" / "losses" / "x"
            losses.mkdir(parents=True)
            (losses / "general.npz").write_bytes(b"data")
            with self.assertRaises(RuntimeError):
                verify_seed44_untouched_stage3(RESULTS_ROOT, stage3_dir)
            report = verify_seed44_untouched_stage3(
                RESULTS_ROOT, stage3_dir, allow_authorized_final=True
            )
            self.assertTrue(report["stage3_final_outputs_exist"])


if __name__ == "__main__":
    unittest.main()
