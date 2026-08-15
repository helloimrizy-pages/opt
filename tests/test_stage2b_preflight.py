from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from expert_analysis.balanced import load_controlled_source
from expert_analysis.specialist_preservation import (
    STAGE2B_DOMAINS,
    build_specialist_scores,
)
from expert_analysis.stage2b_preflight import verify_frozen_upstream_decisions

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPOSITORY_ROOT / "results"
SOURCE_DIR = RESULTS_ROOT / "expert_domain_causal_validation"


@unittest.skipUnless(
    (RESULTS_ROOT / "quantization_cost_surrogate" / "surrogate_decision.json").is_file(),
    "frozen upstream artifacts are unavailable",
)
class FrozenStateTests(unittest.TestCase):
    def test_upstream_decisions_are_frozen_and_verified(self) -> None:
        report = verify_frozen_upstream_decisions(RESULTS_ROOT)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["stage2a_surrogate"]["decision"], "SURROGATE_NO_GO"
        )
        self.assertEqual(report["stage1_quantization_pilot"]["decision"], "GO")

    def test_real_calibration_scores_are_normalized_and_hashed(self) -> None:
        source = load_controlled_source(SOURCE_DIR)
        scores = build_specialist_scores(source)
        np.testing.assert_allclose(
            scores.functional.sum(axis=(0, 1)), np.ones(4), atol=1e-9
        )
        np.testing.assert_allclose(
            scores.functional_specialization.sum(axis=(0, 1)), np.ones(4), atol=1e-9
        )
        np.testing.assert_allclose(
            scores.routing_specialization.sum(axis=(0, 1)), np.ones(4), atol=1e-9
        )
        for domain in STAGE2B_DOMAINS:
            detail = scores.metadata["domains_detail"][domain]
            self.assertEqual(
                len(detail["calibration_indices_into_frozen_set"]), 25
            )
            self.assertEqual(len(detail["calibration_input_row_sha256"]), 25)
        self.assertEqual(len(scores.metadata["calibration_fingerprint"]), 64)
        second = build_specialist_scores(source)
        self.assertEqual(
            scores.metadata["calibration_fingerprint"],
            second.metadata["calibration_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
