from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from helpers import make_trace, workload_config
from residency_headroom.trace import RoutingTrace, metadata_path_for
from residency_headroom.workloads import (
    build_calibration_workload,
    build_workloads,
    calibration_frequency_scores,
    split_sequences,
)


class TraceWorkloadTests(unittest.TestCase):
    def test_trace_serialization_round_trip_and_schema_validation(self) -> None:
        trace = make_trace()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.npz"
            trace.save(path)
            loaded = RoutingTrace.load(path)
            self.assertEqual(loaded.trace_hash, trace.trace_hash)
            self.assertTrue(metadata_path_for(path).exists())
            np.testing.assert_array_equal(
                loaded.requested_expert_ids, trace.requested_expert_ids
            )
            self.assertEqual(loaded.validate()["requested_experts"], trace.num_events * 2)

    def test_trace_rejects_duplicate_atomic_expert(self) -> None:
        trace = make_trace()
        arrays = trace.arrays()
        arrays["requested_expert_ids"] = arrays["requested_expert_ids"].copy()
        arrays["requested_expert_ids"][0] = [1, 1]
        with self.assertRaises(ValueError):
            RoutingTrace.from_mapping(arrays, trace.metadata, validate=True)

    def test_workload_construction_and_calibration_leakage(self) -> None:
        trace = make_trace()
        config = workload_config()
        split = split_sequences(trace, config["calibration_fraction"])
        workloads = build_workloads(trace, split, config)
        self.assertEqual(len(workloads), 10)
        by_name = {workload.name: workload for workload in workloads}
        self.assertEqual(by_name["general_to_coding"].domains, ("general", "coding"))
        repeated = by_name["repeated_domain_cycle"]
        self.assertEqual(
            [item.domain for item in repeated.sequences],
            ["general", "coding", "math", "reasoning", "general"],
        )
        calibration = build_calibration_workload(trace, split, seed=99)
        scores = calibration_frequency_scores(trace, calibration)
        expected = sum(
            item["generation_length"] * trace.num_layers * trace.top_k
            for item in trace.metadata["sequences"]
            if item["sequence_id"] in calibration.sequence_ids
        )
        self.assertEqual(int(scores.sum()), expected)
        self.assertFalse(set(calibration.sequence_ids) & set(by_name["mixed_interleaved"].sequence_ids))


if __name__ == "__main__":
    unittest.main()
