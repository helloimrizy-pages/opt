"""Stage 2 dependency verification.

Nothing in Stage 2 recomputes a Stage 0 or Stage 1 number. Every reference cost is
read from the sealed archives after its SHA-256 has been re-verified here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from race_stage1.calibration import load_and_verify_stage1_frozen
from race_stage1.frozen import (
    FrozenInputs,
    iter_primary_stage0_rows,
    load_and_verify_frozen_inputs,
    stage0_references,
)
from residency_headroom.common import read_json, resolve_git_commit, sha256_file
from residency_headroom.workloads import Workload


STAGE2_ROOT = "stage3_residency/stage2_race"


@dataclass(frozen=True)
class Stage2Inputs:
    preregistration: dict[str, Any]
    preregistration_hash: str
    stage0: FrozenInputs
    stage1_frozen: dict[str, Any]
    stage0_references: dict[tuple[str, int], dict[str, Any]]
    stage1_costs: dict[tuple[str, int], int]
    stage1_perfect_costs: dict[tuple[str, int], int]
    stage1_sequence_condition_ids: dict[tuple[str, int], str]
    verification: dict[str, Any]

    @property
    def trace(self):
        return self.stage0.trace

    @property
    def calibration(self) -> Workload:
        return self.stage0.calibration

    @property
    def workloads(self) -> tuple[Workload, ...]:
        return self.stage0.workloads


def read_sidecar_hash(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip().split()
    if not value or len(value[0]) != 64:
        raise ValueError(f"Invalid SHA-256 sidecar {path}")
    return value[0]


def load_stage2_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    actual = sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        expected = read_sidecar_hash(sidecar)
        if actual != expected:
            raise ValueError(f"Stage 2 preregistration hash mismatch: {actual} != {expected}")
    return read_json(path), actual


def stage2_source_bundle_hash(root: Path) -> str:
    """Hash the reviewable Stage 2 source, scripts, configs and tests."""

    digest = hashlib.sha256()
    base = root / STAGE2_ROOT
    for directory in ("src", "scripts", "configs", "tests"):
        target = base / directory
        if not target.exists():
            continue
        for path in sorted(item for item in target.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            digest.update(path.relative_to(base).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _verify_files(root: Path, entries: Sequence[tuple[str, str, str]]) -> list[dict[str, str]]:
    records = []
    for label, relative, expected in entries:
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen {label} changed: {relative} ({actual} != {expected})")
        records.append({"label": label, "path": relative, "sha256": actual})
    return records


def load_and_verify_stage2_inputs(
    repository_root: Path, preregistration_path: Path
) -> Stage2Inputs:
    repository_root = repository_root.resolve()
    preregistration, preregistration_hash = load_stage2_preregistration(preregistration_path)
    stage0_ref = preregistration["stage0_reference"]
    stage1_ref = preregistration["stage1_reference"]

    verified = _verify_files(
        repository_root,
        [
            (
                "stage0 archive manifest",
                stage0_ref["final_archive_manifest_path"],
                stage0_ref["final_archive_manifest_sha256"],
            ),
            (
                "stage0 routing trace",
                stage0_ref["trace_path"],
                stage0_ref["routing_trace_npz_sha256"],
            ),
            (
                "stage0 frozen evaluation config",
                stage0_ref["frozen_evaluation_config_path"],
                stage0_ref["frozen_evaluation_config_file_sha256"],
            ),
            ("stage0 results", stage0_ref["result_path"], stage0_ref["results_jsonl_sha256"]),
            (
                "stage0 per-sequence results",
                stage0_ref["per_sequence_result_path"],
                stage0_ref["per_sequence_results_jsonl_sha256"],
            ),
            (
                "stage1 archive manifest",
                stage1_ref["final_archive_manifest_path"],
                stage1_ref["final_archive_manifest_sha256"],
            ),
            (
                "stage1 preregistration",
                stage1_ref["preregistration_path"],
                stage1_ref["preregistration_hash"],
            ),
            (
                "stage1 frozen config",
                stage1_ref["frozen_config_path"],
                stage1_ref["frozen_config_file_sha256"],
            ),
            ("stage1 results", stage1_ref["results_path"], stage1_ref["results_jsonl_sha256"]),
            (
                "stage1 per-sequence results",
                stage1_ref["per_sequence_results_path"],
                stage1_ref["per_sequence_results_jsonl_sha256"],
            ),
            (
                "stage1 transition models",
                stage1_ref["transition_model_path"],
                stage1_ref["transition_model_sha256"],
            ),
        ],
    )

    stage0 = load_and_verify_frozen_inputs(
        repository_root, repository_root / stage1_ref["preregistration_path"]
    )
    if stage0.trace.trace_hash != stage0_ref["trace_logical_hash"]:
        raise ValueError("Loaded trace does not match the Stage 2 preregistered logical hash")
    stage1_frozen = load_and_verify_stage1_frozen(
        repository_root / stage1_ref["frozen_config_path"]
    )
    if stage1_frozen["selected_predictor_id"] != stage1_ref["winner_method_id"]:
        raise ValueError("Frozen Stage 1 winner identifier changed")
    stage1_archive = read_json(repository_root / stage1_ref["final_archive_manifest_path"])
    if stage1_archive["verdict"] != stage1_ref["verdict"]:
        raise ValueError("Stage 1 archived verdict changed")
    if stage1_archive["frozen_config_file_sha256"] != stage1_frozen["file_sha256"]:
        raise ValueError("Stage 1 archive references another frozen config")

    references = stage0_references(repository_root / stage0_ref["result_path"])
    stage1_costs: dict[tuple[str, int], int] = {}
    stage1_perfect: dict[tuple[str, int], int] = {}
    stage1_conditions: dict[tuple[str, int], str] = {}
    winner = stage1_ref["winner_method_id"]
    for row in _iter_jsonl(repository_root / stage1_ref["results_path"]):
        key = (str(row["workload"]), int(row["capacity"]))
        if str(row["method_id"]) == winner:
            stage1_costs[key] = int(row["misses"])
            stage1_conditions[key] = str(row["condition_id"])
        elif str(row["method_id"]) == "perfect_score_simple_policy":
            stage1_perfect[key] = int(row["misses"])
    expected_keys = {
        (workload.name, int(capacity))
        for workload in stage0.workloads
        for capacity in preregistration["cache_capacities"]
    }
    for label, table in (("winner", stage1_costs), ("perfect score", stage1_perfect)):
        missing = sorted(expected_keys - set(table))
        if missing:
            raise ValueError(f"Frozen Stage 1 {label} rows are incomplete: {missing[:5]}")

    return Stage2Inputs(
        preregistration=preregistration,
        preregistration_hash=preregistration_hash,
        stage0=stage0,
        stage1_frozen=stage1_frozen,
        stage0_references=references,
        stage1_costs=stage1_costs,
        stage1_perfect_costs=stage1_perfect,
        stage1_sequence_condition_ids=stage1_conditions,
        verification={
            "verified_files": verified,
            "stage0_preregistration_hash": stage0.preregistration_hash,
            "stage1_frozen_config_file_sha256": stage1_frozen["file_sha256"],
            "stage1_winner_method_id": winner,
            "repository_head": resolve_git_commit(repository_root),
            "stage2_source_bundle_hash": stage2_source_bundle_hash(repository_root),
        },
    )


def perfect_score_matches_oracle(inputs: Stage2Inputs) -> dict[str, Any]:
    """Re-check the frozen Stage 1 perfect-score identity from sealed artifacts."""

    mismatches = []
    for key, cost in sorted(inputs.stage1_perfect_costs.items()):
        oracle = int(inputs.stage0_references[key]["oracle"]["misses"])
        if cost != oracle:
            mismatches.append({"workload": key[0], "capacity": key[1], "perfect": cost, "oracle": oracle})
    return {
        "passed": not mismatches,
        "conditions_checked": len(inputs.stage1_perfect_costs),
        "mismatches": mismatches,
    }


def truncated_workload(workload: Workload, sequences: int, name: str) -> Workload:
    """A frozen-prefix pilot workload; only used on calibration data."""

    from residency_headroom.workloads import WorkloadSequence

    if sequences < 1 or sequences > len(workload.sequences):
        raise ValueError("Pilot prefix length is out of range")
    prefix = tuple(
        WorkloadSequence(
            source_sequence_id=item.source_sequence_id,
            position=position,
            segment_index=item.segment_index,
            segment_label=item.segment_label,
            domain=item.domain,
        )
        for position, item in enumerate(workload.sequences[:sequences])
    )
    return Workload(name=name, regime=workload.regime, sequences=prefix)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stage0_primary_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    return iter_primary_stage0_rows(path)
