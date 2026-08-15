from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.fragility import (
    STAGE2C_REGIMES,
    build_calibration_fragility_record,
    clipped_fragility,
    compute_regime_fragility,
    fragility_vector,
    load_frozen_fragility,
    load_frozen_stage2b_scores,
    mean_calibration_nll,
    normalized_fragility,
    raw_relative_fragility,
    save_calibration_fragility,
    verify_fragility_record,
)
from expert_analysis.masking import LossStatistics

STAGE2B_DIR = Path("results/robust_specialist_preservation")
DOMAINS = ("general", "math", "coding", "reasoning")


def _fake_record(
    bf16=(2.0, 3.0, 2.5, 4.0),
    uniform4=(2.1, 3.3, 2.5, 4.2),
    uniform3=(2.4, 3.9, 2.8, 4.4),
):
    bf16_nll = dict(zip(DOMAINS, bf16, strict=True))
    base_nll = {
        4: dict(zip(DOMAINS, uniform4, strict=True)),
        3: dict(zip(DOMAINS, uniform3, strict=True)),
    }
    regime_results = {
        regime: compute_regime_fragility(bf16_nll, base_nll[bits], bits)
        for regime, bits in STAGE2C_REGIMES.items()
    }
    return build_calibration_fragility_record(
        regime_results=regime_results,
        calibration_subset_hashes={
            domain: {"input_ids_sha256": "0" * 64} for domain in DOMAINS
        },
        model_info={"model": "test", "resolved_model_revision": "abc"},
        qdq_config={"group_size": 128},
        environment={"package_versions": {}},
        reproduction={"all_reproduced": True},
    )


class FragilityFormulaTests(unittest.TestCase):
    def test_raw_relative_fragility(self) -> None:
        self.assertAlmostEqual(raw_relative_fragility(2.0, 2.2), 0.1)
        self.assertAlmostEqual(raw_relative_fragility(2.0, 1.8), -0.1)

    def test_raw_fragility_rejects_invalid_nll(self) -> None:
        with self.assertRaises(ValueError):
            raw_relative_fragility(0.0, 1.0)
        with self.assertRaises(ValueError):
            raw_relative_fragility(float("nan"), 1.0)

    def test_clipping_is_nonnegative_not_absolute(self) -> None:
        self.assertEqual(clipped_fragility(0.25), 0.25)
        self.assertEqual(clipped_fragility(-0.25), 0.0)
        self.assertNotEqual(clipped_fragility(-0.25), 0.25)

    def test_normalization_has_mean_one(self) -> None:
        values = {"general": 0.4, "math": 0.2, "coding": 0.1, "reasoning": 0.1}
        normalized, valid = normalized_fragility(values)
        self.assertTrue(valid)
        self.assertAlmostEqual(
            float(np.mean([normalized[d] for d in DOMAINS])), 1.0
        )
        self.assertAlmostEqual(normalized["general"], 0.4 / 0.2)

    def test_all_zero_fragility_marks_regime_invalid(self) -> None:
        values = {domain: 0.0 for domain in DOMAINS}
        normalized, valid = normalized_fragility(values)
        self.assertFalse(valid)
        self.assertIsNone(normalized)

    def test_normalization_requires_domain_order(self) -> None:
        with self.assertRaises(ValueError):
            normalized_fragility({"math": 0.1, "general": 0.1})

    def test_compute_regime_fragility_clips_improvement(self) -> None:
        bf16 = {d: 2.0 for d in DOMAINS}
        base = {"general": 2.2, "math": 1.9, "coding": 2.1, "reasoning": 2.0}
        result = compute_regime_fragility(bf16, base, 4)
        self.assertTrue(result["regime_valid"])
        self.assertEqual(result["domains"]["math"]["clipped_fragility"], 0.0)
        self.assertLess(result["domains"]["math"]["relative_delta"], 0.0)
        self.assertAlmostEqual(
            result["domains"]["general"]["clipped_fragility"], 0.1
        )

    def test_compute_regime_fragility_invalid_when_all_improve(self) -> None:
        bf16 = {d: 2.0 for d in DOMAINS}
        base = {d: 1.9 for d in DOMAINS}
        result = compute_regime_fragility(bf16, base, 3)
        self.assertFalse(result["regime_valid"])
        for domain in DOMAINS:
            self.assertIsNone(result["domains"][domain]["normalized_fragility"])

    def test_mean_calibration_nll(self) -> None:
        statistics = LossStatistics(
            loss_sums=np.asarray([64.0, 128.0]),
            token_counts=np.asarray([64, 64], dtype=np.uint32),
        )
        self.assertAlmostEqual(mean_calibration_nll(statistics), 1.5)


class FragilityRecordTests(unittest.TestCase):
    def test_record_integrity_hash_round_trip(self) -> None:
        record = _fake_record()
        verify_fragility_record(record)
        with tempfile.TemporaryDirectory() as tmp:
            calibration_dir = Path(tmp)
            json_path, csv_path = save_calibration_fragility(record, calibration_dir)
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            loaded = load_frozen_fragility(calibration_dir)
            self.assertEqual(loaded["fragility_sha256"], record["fragility_sha256"])

    def test_tampered_record_fails_verification(self) -> None:
        record = _fake_record()
        record["regimes"]["4to8"]["domains"]["general"]["clipped_fragility"] = 9.9
        with self.assertRaises(RuntimeError):
            verify_fragility_record(record)

    def test_refreeze_with_different_values_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration_dir = Path(tmp)
            save_calibration_fragility(_fake_record(), calibration_dir)
            different = _fake_record(uniform4=(2.3, 3.3, 2.5, 4.2))
            with self.assertRaises(RuntimeError):
                save_calibration_fragility(different, calibration_dir)

    def test_fragility_vector_order_and_invalid_regime(self) -> None:
        record = _fake_record()
        vector = record["regimes"]["4to8"]
        values = fragility_vector(record, "4to8")
        for index, domain in enumerate(DOMAINS):
            self.assertAlmostEqual(
                float(values[index]),
                vector["domains"][domain]["normalized_fragility"],
            )
        invalid = _fake_record(uniform4=(1.9, 2.9, 2.4, 3.9))
        with self.assertRaises(RuntimeError):
            fragility_vector(invalid, "4to8")
        with self.assertRaises(ValueError):
            fragility_vector(record, "5to8")


@unittest.skipUnless(STAGE2B_DIR.is_dir(), "frozen Stage 2B artifacts not present")
class FrozenStage2BScoreReuseTests(unittest.TestCase):
    def test_frozen_scores_load_and_verify(self) -> None:
        scores = load_frozen_stage2b_scores(STAGE2B_DIR)
        self.assertEqual(scores.functional_specialization.shape, (16, 64, 4))
        self.assertTrue(
            np.allclose(scores.functional_specialization.sum(axis=(0, 1)), 1.0)
        )
        self.assertEqual(len(scores.calibration_indices["general"]), 25)

    def test_tampered_score_array_is_rejected(self) -> None:
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "stage2b"
            (fake / "calibration").mkdir(parents=True)
            for name in (
                "calibration_metadata.json",
                "functional_importance.npz",
                "routing_specialization.npz",
            ):
                shutil.copy(
                    STAGE2B_DIR / "calibration" / name, fake / "calibration" / name
                )
            with np.load(
                STAGE2B_DIR / "calibration" / "functional_specialization.npz",
                allow_pickle=False,
            ) as data:
                arrays = {key: np.array(data[key]) for key in data.files}
            arrays["specialization"] = arrays["specialization"].copy()
            arrays["specialization"][0, 0, 0] += 1e-6
            np.savez_compressed(
                fake / "calibration" / "functional_specialization.npz", **arrays
            )
            with self.assertRaises(RuntimeError):
                load_frozen_stage2b_scores(fake)


if __name__ == "__main__":
    unittest.main()
