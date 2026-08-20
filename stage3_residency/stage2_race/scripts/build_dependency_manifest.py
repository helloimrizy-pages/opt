"""Verify and record every frozen Stage 0/Stage 1 dependency of Stage 2."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import PREREGISTRATION, ROOT, STAGE2

from race_stage2.advisers import adviser_parameters
from race_stage2.frozen import (
    load_and_verify_stage2_inputs,
    perfect_score_matches_oracle,
    stage2_source_bundle_hash,
)
from residency_headroom.common import atomic_write_json, resolve_git_commit, utc_now


def main() -> None:
    inputs = load_and_verify_stage2_inputs(ROOT, PREREGISTRATION)
    prereg = inputs.preregistration
    manifest = {
        "schema_version": "race_stage2_dependency_manifest_v1",
        "created_at_utc": utc_now(),
        "stage0_trace_logical_hash": inputs.trace.trace_hash,
        "stage0_routing_trace_npz_sha256": prereg["stage0_reference"][
            "routing_trace_npz_sha256"
        ],
        "stage0_archive_manifest_sha256": prereg["stage0_reference"][
            "final_archive_manifest_sha256"
        ],
        "stage1_archive_manifest_sha256": prereg["stage1_reference"][
            "final_archive_manifest_sha256"
        ],
        "stage1_frozen_config_file_sha256": inputs.stage1_frozen["file_sha256"],
        "stage1_winner_method_id": inputs.stage1_frozen["selected_predictor_id"],
        "stage1_winner_spec": inputs.stage1_frozen["selected_predictor"],
        "stage2_preregistration_sha256": inputs.preregistration_hash,
        "stage2_source_commit": resolve_git_commit(ROOT),
        "stage2_source_bundle_hash": stage2_source_bundle_hash(ROOT),
        "adviser_pools": {
            pool: adviser_parameters(pool) for pool in ("primary", "extended")
        },
        "verified_files": inputs.verification["verified_files"],
        "frozen_stage1_perfect_score_equals_stage0_oracle": perfect_score_matches_oracle(
            inputs
        ),
        "calibration_workload_hash": inputs.calibration.hash,
        "evaluation_workload_hashes": {
            workload.name: workload.hash for workload in inputs.workloads
        },
        "calibration_evaluation_disjoint": not bool(
            set(inputs.calibration.sequence_ids)
            & {
                sequence
                for workload in inputs.workloads
                for sequence in workload.sequence_ids
            }
        ),
    }
    path = STAGE2 / "reports/stage2_dependency_manifest.json"
    atomic_write_json(path, manifest)
    print(f"wrote {path.relative_to(ROOT)}")
    for key in (
        "stage0_trace_logical_hash",
        "stage0_archive_manifest_sha256",
        "stage1_archive_manifest_sha256",
        "stage1_frozen_config_file_sha256",
        "stage1_winner_method_id",
        "stage2_source_commit",
    ):
        print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()
