from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.measured_damage import (
    DamageChunk,
    STAGE3_PROFILE_BITS,
    additivity_decision,
    additivity_gates_for_regime,
    build_damage_record,
    damage_chunk_path,
    damage_deterministic_content,
    load_damage_chunk,
    predicted_domain_delta_nll,
    save_damage_chunk,
    save_damage_matrix,
    load_frozen_damage,
    verify_damage_record,
)
from expert_analysis.specialist_preservation import (
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
)

SHAPE = (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS), len(STAGE3_PROFILE_BITS))


def _delta() -> np.ndarray:
    delta = np.zeros(SHAPE, dtype=np.float64)
    delta[0, 0, :, 0] = [0.30, 0.01, 0.02, 0.03]  # 3-bit
    delta[0, 0, :, 1] = [0.10, 0.005, 0.01, 0.015]  # 4-bit
    delta[0, 0, :, 2] = [0.001, 0.0001, 0.0002, 0.0003]  # 8-bit
    delta[2, 5, :, 1] = [0.02, 0.20, 0.01, 0.04]
    delta[2, 5, :, 2] = [0.002, 0.02, 0.001, 0.004]
    return delta


class PredictedDeltaTests(unittest.TestCase):
    def test_bf16_contributes_zero(self) -> None:
        bits = np.full((NUM_MOE_LAYERS, NUM_EXPERTS), 16, dtype=np.int64)
        predicted = predicted_domain_delta_nll(bits, _delta())
        np.testing.assert_array_equal(predicted, np.zeros(4))

    def test_additive_sum_over_assigned_bits(self) -> None:
        delta = _delta()
        bits = np.full((NUM_MOE_LAYERS, NUM_EXPERTS), 16, dtype=np.int64)
        bits[0, 0] = 4
        bits[2, 5] = 8
        predicted = predicted_domain_delta_nll(bits, delta)
        expected = delta[0, 0, :, 1] + delta[2, 5, :, 2]
        np.testing.assert_allclose(predicted, expected)

    def test_invalid_bits_rejected(self) -> None:
        bits = np.full((NUM_MOE_LAYERS, NUM_EXPERTS), 16, dtype=np.int64)
        bits[0, 0] = 5
        with self.assertRaises(ValueError):
            predicted_domain_delta_nll(bits, _delta())


class AdditivityGateTests(unittest.TestCase):
    def _probes(self, scramble: bool = False) -> list[dict]:
        rng = np.random.default_rng(11)
        rows = []
        for index in range(10):
            measured = np.abs(rng.normal(0.05 * (index + 1), 0.002, size=4))
            predicted = measured * 1.1  # constant multiplicative bias only
            if scramble:
                predicted = np.abs(rng.normal(0.1, 0.05, size=4))
            rows.append(
                {"predicted": predicted.tolist(), "measured": measured.tolist()}
            )
        return rows

    def test_rank_preserving_probes_pass(self) -> None:
        gates = additivity_gates_for_regime(self._probes())
        self.assertTrue(gates["gate_add_1"]["passed"])
        self.assertTrue(gates["gate_add_2"]["passed"])
        self.assertTrue(gates["all_passed"])

    def test_scrambled_probes_fail(self) -> None:
        gates = additivity_gates_for_regime(self._probes(scramble=True))
        self.assertFalse(gates["all_passed"])

    def test_decision_authorizes_only_passing_regimes(self) -> None:
        passing = additivity_gates_for_regime(self._probes())
        failing = additivity_gates_for_regime(self._probes(scramble=True))
        decision = additivity_decision({"4to8": passing, "3to8": failing})
        self.assertEqual(decision["decision"], "ADDITIVITY_GO")
        self.assertEqual(decision["authorized_regimes"], ["4to8"])
        decision = additivity_decision({"4to8": failing, "3to8": failing})
        self.assertEqual(decision["decision"], "MEASURED_DAMAGE_NO_GO")
        self.assertEqual(decision["authorized_regimes"], [])


class DamageChunkTests(unittest.TestCase):
    def test_round_trip_and_tamper_detection(self) -> None:
        rng = np.random.default_rng(3)
        loss = rng.random((NUM_EXPERTS, 4, 25)) + 0.5
        counts = np.full((NUM_EXPERTS, 4, 25), 64, dtype=np.uint32)
        chunk = DamageChunk(layer=1, bits=4, loss_sums=loss,
                            token_counts=counts, metadata={})
        with tempfile.TemporaryDirectory() as tmp:
            damage_dir = Path(tmp)
            path = damage_chunk_path(damage_dir, 4, 1)
            expected = {"run_fingerprint": "abc", "layer": 1, "bits": 4}
            save_damage_chunk(path, chunk, expected, 25)
            loaded = load_damage_chunk(path, expected, 25)
            np.testing.assert_array_equal(loaded.loss_sums, loss)
            with self.assertRaises(RuntimeError):
                load_damage_chunk(
                    path, {"run_fingerprint": "other", "layer": 1, "bits": 4}, 25
                )
            with np.load(path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            arrays["loss_sums"] = arrays["loss_sums"] + 1.0
            np.savez_compressed(path, **arrays)
            with self.assertRaises(RuntimeError):
                load_damage_chunk(path, expected, 25)


class DamageRecordTests(unittest.TestCase):
    def _record_and_arrays(self):
        delta = _delta()
        bf16 = np.asarray([2.9, 2.3, 1.8, 1.9])
        arrays = {
            "mean_nll": delta + bf16[None, None, :, None],
            "delta_nll": delta,
            "bf16_nll": bf16,
        }
        record = build_damage_record(
            arrays=arrays,
            uniform_nll={
                state: {d: 2.0 for d in STAGE2B_DOMAINS}
                for state in ("uniform8", "uniform4", "uniform3")
            },
            frozen_reference_drift={"tolerance": 5e-6},
            calibration_subset_hashes={d: {"input_ids_sha256": "x"} for d in STAGE2B_DOMAINS},
            model_info={"model": "test"},
            qdq_config={"group_size": 128},
            environment={},
            reproduction={"all_reproduced": True},
            chunk_hashes={},
            examples_per_domain=25,
        )
        return record, arrays

    def test_integrity_hash_round_trip(self) -> None:
        record, _ = self._record_and_arrays()
        verify_damage_record(record)
        tampered = dict(record)
        tampered["bf16_nll"] = {**record["bf16_nll"], "general": 99.0}
        with self.assertRaises(RuntimeError):
            verify_damage_record(tampered)

    def test_negative_cell_counts_recorded(self) -> None:
        record, _ = self._record_and_arrays()
        for bits_key in ("bits3", "bits4", "bits8"):
            self.assertEqual(record["summary"][bits_key]["negative_damage_cells"], 0)

    def test_save_and_reload_frozen(self) -> None:
        record, arrays = self._record_and_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            damage_dir = Path(tmp)
            save_damage_matrix(record, arrays, damage_dir)
            loaded_record, loaded_arrays = load_frozen_damage(damage_dir)
            self.assertEqual(
                damage_deterministic_content(loaded_record),
                damage_deterministic_content(record),
            )
            np.testing.assert_array_equal(
                loaded_arrays["delta_nll"], arrays["delta_nll"]
            )
            different = dict(record)
            different["model"] = {"model": "other"}
            different.pop("damage_sha256")
            rebuilt = build_damage_record(
                arrays=arrays,
                uniform_nll=record["uniform_nll"],
                frozen_reference_drift=record["frozen_stage2c_reference_drift"],
                calibration_subset_hashes=record["calibration_subset_hashes"],
                model_info={"model": "other"},
                qdq_config=record["qdq_config"],
                environment={},
                reproduction=record["repeated_evaluation_reproduction"],
                chunk_hashes={},
                examples_per_domain=25,
            )
            with self.assertRaises(RuntimeError):
                save_damage_matrix(rebuilt, arrays, damage_dir)


if __name__ == "__main__":
    unittest.main()
