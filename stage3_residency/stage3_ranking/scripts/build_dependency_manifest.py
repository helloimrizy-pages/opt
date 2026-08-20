"""Verify and record every frozen Stage 0/1/2 dependency of Stage 3."""
from __future__ import annotations
import _bootstrap  # noqa: F401
from _bootstrap import PREREGISTRATION, ROOT, STAGE3
from race_stage3.frozen import (
    load_and_verify_stage3_inputs,
    stage1_per_sequence_rows,
    stage3_source_bundle_hash,
)
from residency_headroom.common import atomic_write_json, resolve_git_commit, utc_now


def main() -> None:
    inputs = load_and_verify_stage3_inputs(ROOT, PREREGISTRATION)
    prereg = inputs.preregistration
    reconstructed = len(stage1_per_sequence_rows(
        ROOT, prereg, set(inputs.stage1_condition_ids.values())))
    manifest = {
        "schema_version": "race_stage3_dependency_manifest_v1",
        "created_at_utc": utc_now(),
        "stage0_trace_logical_hash": inputs.trace.trace_hash,
        "stage0_archive_manifest_sha256": prereg["stage0_reference"]["final_archive_manifest_sha256"],
        "stage1_archive_manifest_sha256": prereg["stage1_reference"]["final_archive_manifest_sha256"],
        "stage2_archive_manifest_sha256": prereg["stage2_reference"]["final_archive_manifest_sha256"],
        "stage1_frozen_config_file_sha256": inputs.stage1_frozen["file_sha256"],
        "stage1_winner_method_id": inputs.stage1_frozen["selected_predictor_id"],
        "stage3_preregistration_sha256": inputs.preregistration_hash,
        "stage3_source_commit": resolve_git_commit(ROOT),
        "stage3_source_bundle_hash": stage3_source_bundle_hash(ROOT),
        "verified_files": inputs.verification["verified_files"],
        "stage1_per_sequence_rows_available": reconstructed,
        "calibration_workload_hash": inputs.calibration.hash,
        "evaluation_workload_hashes": {w.name: w.hash for w in inputs.workloads},
        "calibration_evaluation_disjoint": not bool(
            set(inputs.calibration.sequence_ids)
            & {s for w in inputs.workloads for s in w.sequence_ids}),
    }
    path = STAGE3 / "reports/stage3_dependency_manifest.json"
    atomic_write_json(path, manifest)
    print(f"wrote {path.relative_to(ROOT)}")
    for key in ("stage0_trace_logical_hash", "stage1_archive_manifest_sha256",
                "stage2_archive_manifest_sha256", "stage1_per_sequence_rows_available",
                "stage3_source_commit"):
        print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()
