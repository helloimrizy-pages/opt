from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from residency_headroom.common import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    sha256_file,
    utc_now,
)

from .calibration import load_and_verify_stage1_frozen
from .frozen import source_bundle_hash


def freeze_archive(repository_root: Path, stage1_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    stage1_root = stage1_root.resolve()
    frozen_path = stage1_root / "results/calibration/stage1_frozen_config.json"
    frozen = load_and_verify_stage1_frozen(frozen_path)
    current_source = source_bundle_hash(stage1_root)
    if current_source != frozen["stage1_source_bundle_hash"]:
        attestation_path = stage1_root / "reports/post_evaluation_reporting_fixes.json"
        attestation = read_json(attestation_path)
        if (
            attestation["frozen_evaluation_source_bundle_hash"]
            != frozen["stage1_source_bundle_hash"]
            or attestation["reconstructed_pre_fix_source_bundle_hash"]
            != frozen["stage1_source_bundle_hash"]
            or attestation["final_source_bundle_hash"] != current_source
            or attestation["scientific_results_changed"]
        ):
            raise ValueError("Post-evaluation reporting-fix attestation is invalid")
    evaluation_manifest = read_json(stage1_root / "results/full/evaluation_manifest.json")
    analysis = read_json(stage1_root / "reports/analysis.json")
    audit = read_json(stage1_root / "reports/analysis_audit.json")
    sanity = read_json(stage1_root / "results/full/sanity_checks.json")
    if not evaluation_manifest["passed"] or not audit["passed"] or not sanity["passed"]:
        raise ValueError("Cannot seal Stage 1 because a required audit failed")
    critical_relative = [
        "configs/stage1_preregistered.json",
        "configs/stage1_preregistered.sha256",
        "results/calibration/transition_models.npz",
        "results/calibration/transition_models.metadata.json",
        "results/calibration/calibration_results.jsonl",
        "results/calibration/calibration_per_sequence.jsonl",
        "results/calibration/lookahead_validation.json",
        "results/calibration/selection.json",
        "results/calibration/selection.sha256",
        "results/calibration/stage1_frozen_config.json",
        "results/calibration/stage1_frozen_config.sha256",
        "results/full/results.jsonl",
        "results/full/per_sequence_results.jsonl",
        "results/full/prediction_quality.jsonl",
        "results/full/sanity_checks.json",
        "results/full/evaluation_manifest.json",
        "reports/analysis.json",
        "reports/analysis_audit.json",
        "reports/post_evaluation_reporting_fixes.json",
        "reports/race_stage1_prediction_headroom_report.md",
        "reports/race_stage1_prediction_headroom_report.sha256",
        "tables/required_tables.md",
        "tables/causal_by_workload.csv",
        "tables/table1_causal_costs_by_regime.csv",
        "tables/table2_oracle_gap_closed.csv",
        "tables/table3_residual_headroom.csv",
        "tables/causal_sensitivity.csv",
        "tables/lookahead_curve.csv",
        "figures/figure1_normalized_transfer_cost.png",
        "figures/figure1_normalized_transfer_cost.pdf",
        "figures/figure2_gap_closed_vs_spare.png",
        "figures/figure2_gap_closed_vs_spare.pdf",
        "figures/figure3_lookahead_curve.png",
        "figures/figure3_lookahead_curve.pdf",
        "figures/figure4_prediction_quality_vs_residency.png",
        "figures/figure4_prediction_quality_vs_residency.pdf",
    ]
    missing = [name for name in critical_relative if not (stage1_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Stage 1 archive is incomplete: {missing}")
    critical_hashes = {
        name: sha256_file(stage1_root / name) for name in critical_relative
    }
    manifest = {
        "schema_version": "race_stage1_final_archive_v1",
        "sealed_at_utc": utc_now(),
        "verdict": analysis["verdict"],
        "stage0_verdict": "RACE_STAGE0_STRONG_GO",
        "stage0_source_base_commit": frozen["stage0_source_base_commit"],
        "stage0_actual_runtime_commit": frozen["stage0_actual_runtime_commit"],
        "stage1_repository_head": frozen["stage1_repository_head"],
        "stage1_evaluation_source_bundle_hash": frozen["stage1_source_bundle_hash"],
        "stage1_final_source_bundle_hash": current_source,
        "post_evaluation_reporting_fixes_sha256": sha256_file(
            stage1_root / "reports/post_evaluation_reporting_fixes.json"
        ),
        "trace_hash": frozen["trace_hash"],
        "preregistration_hash": frozen["preregistration_hash"],
        "frozen_config_file_sha256": frozen["file_sha256"],
        "transition_model_hash": frozen["transition_model_hash"],
        "evaluation_manifest_sha256": sha256_file(
            stage1_root / "results/full/evaluation_manifest.json"
        ),
        "report_sha256": sha256_file(
            stage1_root / "reports/race_stage1_prediction_headroom_report.md"
        ),
        "critical_files": critical_hashes,
        "checkpoint_tree": _tree_record(stage1_root / "results/full/checkpoints"),
        "source_tree": _tree_record(stage1_root / "src"),
        "test_tree": _tree_record(stage1_root / "tests"),
        "simulation_scope": (
            "Simulated expert residency/miss and transfer counts; no end-to-end "
            "latency or hardware-speedup claim."
        ),
        "bootstrap_scope": (
            "Conditional on frozen workload ordering; saved per-sequence contributions "
            "are reweighted without regenerating stateful cache trajectories."
        ),
    }
    path = stage1_root / "reports/final_archive_manifest.json"
    atomic_write_json(path, manifest)
    digest = sha256_file(path)
    atomic_write_text(
        path.with_suffix(".sha256"), f"{digest}  {path.name}\n"
    )
    return {**manifest, "manifest_sha256": digest}


def verify_archive(stage1_root: Path) -> dict[str, Any]:
    path = stage1_root / "reports/final_archive_manifest.json"
    manifest = read_json(path)
    sidecar = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    failures = []
    if sha256_file(path) != sidecar:
        failures.append("manifest")
    for name, expected in manifest["critical_files"].items():
        target = stage1_root / name
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(name)
    for key, directory in (("checkpoint_tree", "results/full/checkpoints"), ("source_tree", "src"), ("test_tree", "tests")):
        if _tree_record(stage1_root / directory) != manifest[key]:
            failures.append(key)
    return {"passed": not failures, "failures": failures, "manifest_sha256": sidecar}


def _tree_record(root: Path) -> dict[str, Any]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"})
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"files": len(paths), "sha256": digest.hexdigest()}
