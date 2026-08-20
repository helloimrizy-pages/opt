"""Seal every reviewable Stage 3 artifact behind one hashed manifest."""
from __future__ import annotations
import hashlib
import _bootstrap  # noqa: F401
from _bootstrap import EVALUATION_DIR, FROZEN_CONFIG, PREREGISTRATION, ROOT, STAGE3
from race_stage3.calibration import load_and_verify_stage3_frozen
from race_stage3.frozen import load_and_verify_stage3_inputs, stage3_source_bundle_hash
from residency_headroom.common import (
    atomic_write_json, atomic_write_text, read_json, sha256_file, utc_now,
)

CRITICAL = (
    "configs/stage3_preregistered.json", "configs/stage3_preregistered.sha256",
    "results/calibration/selection.json", "results/calibration/selection.sha256",
    "results/calibration/stage3_frozen_config.json",
    "results/calibration/stage3_frozen_config.sha256",
    "results/pilot/pilot_audit.json",
    "results/full/results.jsonl", "results/full/per_sequence_results.jsonl",
    "results/full/diagnostics.jsonl", "results/full/sanity_checks.json",
    "results/full/evaluation_manifest.json",
    "reports/stage3_dependency_manifest.json", "reports/analysis.json",
    "reports/analysis_audit.json", "reports/race_stage3_report.md",
    "reports/race_stage3_report.sha256",
)


def _tree(directory):
    digest = hashlib.sha256(); count = 0
    if not directory.exists():
        return {"files": 0, "sha256": digest.hexdigest()}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        digest.update(path.relative_to(directory).as_posix().encode()); digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        count += 1
    return {"files": count, "sha256": digest.hexdigest()}


def main() -> None:
    inputs = load_and_verify_stage3_inputs(ROOT, PREREGISTRATION)
    frozen = load_and_verify_stage3_frozen(FROZEN_CONFIG)
    analysis = read_json(STAGE3 / "reports/analysis.json")
    manifest = {
        "schema_version": "race_stage3_final_archive_v1",
        "sealed_at_utc": utc_now(),
        "verdict": analysis["verdict"],
        "stage0_verdict": "RACE_STAGE0_STRONG_GO",
        "stage1_verdict": "RACE_STAGE1_STRONG_GO",
        "stage2_verdict": "RACE_STAGE2_NO_GO",
        "stage0_trace_hash": inputs.trace.trace_hash,
        "stage0_archive_manifest_sha256": frozen["stage0_archive_manifest_sha256"],
        "stage1_archive_manifest_sha256": frozen["stage1_archive_manifest_sha256"],
        "stage2_archive_manifest_sha256": frozen["stage2_archive_manifest_sha256"],
        "stage3_preregistration_hash": inputs.preregistration_hash,
        "stage3_frozen_config_file_sha256": frozen["file_sha256"],
        "stage3_repository_head_at_freeze": frozen["stage3_repository_head"],
        "stage3_evaluation_source_bundle_hash": frozen["stage3_source_bundle_hash"],
        "stage3_final_source_bundle_hash": stage3_source_bundle_hash(ROOT),
        "primary_variant": frozen["primary_variant"],
        "evaluation_manifest_sha256": sha256_file(EVALUATION_DIR / "evaluation_manifest.json"),
        "critical_files": {rel: sha256_file(STAGE3 / rel) for rel in CRITICAL
                           if (STAGE3 / rel).exists()},
        "missing_critical_files": [rel for rel in CRITICAL if not (STAGE3 / rel).exists()],
        "source_tree": _tree(STAGE3 / "src"), "script_tree": _tree(STAGE3 / "scripts"),
        "test_tree": _tree(STAGE3 / "tests"), "table_tree": _tree(STAGE3 / "tables"),
        "figure_tree": _tree(STAGE3 / "figures"),
        "checkpoint_tree": _tree(EVALUATION_DIR / "checkpoints"),
        "simulation_scope": ("Simulated expert residency, miss and transfer counts; no end-to-end "
                             "latency or hardware-speedup claim."),
        "bootstrap_scope": frozen["statistics"]["conditionality"],
    }
    path = STAGE3 / "reports/final_archive_manifest.json"
    atomic_write_json(path, manifest)
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    print(f"sealed {path.relative_to(ROOT)}")
    print(f"  verdict        : {manifest['verdict']}")
    print(f"  archive sha256 : {digest}")
    if manifest["missing_critical_files"]:
        raise SystemExit(f"Missing critical Stage 3 artifacts: {manifest['missing_critical_files']}")


if __name__ == "__main__":
    main()
