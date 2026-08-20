"""Seal every reviewable Stage 2 artifact behind one hashed manifest."""

from __future__ import annotations

import hashlib

import _bootstrap  # noqa: F401
from _bootstrap import EVALUATION_DIR, FROZEN_CONFIG, PREREGISTRATION, ROOT, STAGE2

from race_stage2.calibration import load_and_verify_stage2_frozen
from race_stage2.frozen import load_and_verify_stage2_inputs, stage2_source_bundle_hash
from residency_headroom.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    utc_now,
)


CRITICAL = (
    "configs/stage2_preregistered.json",
    "configs/stage2_preregistered.sha256",
    "results/calibration/transition_models.npz",
    "results/calibration/transition_models.metadata.json",
    "results/calibration/selection.json",
    "results/calibration/selection.sha256",
    "results/calibration/stage2_frozen_config.json",
    "results/calibration/stage2_frozen_config.sha256",
    "results/pilot/pilot_audit.json",
    "results/full/results.jsonl",
    "results/full/per_sequence_results.jsonl",
    "results/full/diagnostics.jsonl",
    "results/full/weight_trajectories.jsonl",
    "results/full/sanity_checks.json",
    "results/full/evaluation_manifest.json",
    "reports/stage2_dependency_manifest.json",
    "reports/analysis.json",
    "reports/analysis_audit.json",
    "reports/race_stage2_report.md",
    "reports/race_stage2_report.sha256",
    "reports/race_stage2_theory_notes.md",
)


def _tree_hash(directory) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        count += 1
    return {"files": count, "sha256": digest.hexdigest()}


def main() -> None:
    inputs = load_and_verify_stage2_inputs(ROOT, PREREGISTRATION)
    frozen = load_and_verify_stage2_frozen(FROZEN_CONFIG)
    analysis = read_json(STAGE2 / "reports/analysis.json")
    manifest = {
        "schema_version": "race_stage2_final_archive_v1",
        "sealed_at_utc": utc_now(),
        "verdict": analysis["verdict"],
        "stage0_verdict": "RACE_STAGE0_STRONG_GO",
        "stage1_verdict": "RACE_STAGE1_STRONG_GO",
        "stage0_trace_hash": inputs.trace.trace_hash,
        "stage0_archive_manifest_sha256": frozen["stage0_archive_manifest_sha256"],
        "stage1_archive_manifest_sha256": frozen["stage1_archive_manifest_sha256"],
        "stage1_frozen_config_file_sha256": frozen["stage1_frozen_config_file_sha256"],
        "stage1_winner_method_id": frozen["stage1_winner_method_id"],
        "stage2_preregistration_hash": inputs.preregistration_hash,
        "stage2_frozen_config_file_sha256": frozen["file_sha256"],
        "stage2_repository_head_at_freeze": frozen["stage2_repository_head"],
        "stage2_evaluation_source_bundle_hash": frozen["stage2_source_bundle_hash"],
        "stage2_final_source_bundle_hash": stage2_source_bundle_hash(ROOT),
        "selected_eta": frozen["selected_eta"],
        "selected_initialization": frozen["selected_initialization"],
        "primary_loss": frozen["primary_loss"],
        "primary_variant_id": frozen["primary_variant_id"],
        "weight_scope_primary": frozen["weight_scope_primary"],
        "H_max": frozen["H_max"],
        "adviser_pools": frozen["adviser_pools"],
        "transition_model_hash": frozen["transition_model_hash"],
        "evaluation_manifest_sha256": sha256_file(
            EVALUATION_DIR / "evaluation_manifest.json"
        ),
        "critical_files": {
            relative: sha256_file(STAGE2 / relative)
            for relative in CRITICAL
            if (STAGE2 / relative).exists()
        },
        "missing_critical_files": [
            relative for relative in CRITICAL if not (STAGE2 / relative).exists()
        ],
        "source_tree": _tree_hash(STAGE2 / "src"),
        "script_tree": _tree_hash(STAGE2 / "scripts"),
        "test_tree": _tree_hash(STAGE2 / "tests"),
        "table_tree": _tree_hash(STAGE2 / "tables"),
        "figure_tree": _tree_hash(STAGE2 / "figures"),
        "checkpoint_tree": _tree_hash(EVALUATION_DIR / "checkpoints"),
        "simulation_scope": (
            "Simulated expert residency, miss and transfer counts; no end-to-end latency "
            "or hardware-speedup claim."
        ),
        "bootstrap_scope": frozen["statistics"]["conditionality"],
    }
    path = STAGE2 / "reports/final_archive_manifest.json"
    atomic_write_json(path, manifest)
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    print(f"sealed {path.relative_to(ROOT)}")
    print(f"  verdict           : {manifest['verdict']}")
    print(f"  archive sha256    : {digest}")
    if manifest["missing_critical_files"]:
        raise SystemExit(
            f"Missing critical Stage 2 artifacts: {manifest['missing_critical_files']}"
        )


if __name__ == "__main__":
    main()
