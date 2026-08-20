"""Stage 3 dependency verification.

Stage 3 recomputes no Stage 0, Stage 1 or Stage 2 number. Every reference cost is
read from a sealed archive whose SHA-256 has been re-verified here first.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from race_stage1.calibration import load_and_verify_stage1_frozen
from race_stage1.frozen import (
    FrozenInputs,
    load_and_verify_frozen_inputs,
    stage0_references,
)
from residency_headroom.common import read_json, resolve_git_commit, sha256_file
from residency_headroom.workloads import Workload, WorkloadSequence


STAGE3_ROOT = "stage3_residency/stage3_ranking"


@dataclass(frozen=True)
class Stage3Inputs:
    preregistration: dict[str, Any]
    preregistration_hash: str
    stage0: FrozenInputs
    stage1_frozen: dict[str, Any]
    stage0_references: dict[tuple[str, int], dict[str, Any]]
    stage1_costs: dict[tuple[str, int], int]
    stage1_condition_ids: dict[tuple[str, int], str]
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


def load_stage3_preregistration(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    actual = sha256_file(path)
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        expected = read_sidecar_hash(sidecar)
        if actual != expected:
            raise ValueError(f"Stage 3 preregistration hash mismatch: {actual} != {expected}")
    return read_json(path), actual


def stage3_source_bundle_hash(root: Path) -> str:
    digest = hashlib.sha256()
    base = root / STAGE3_ROOT
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


def stage1_per_sequence_rows(
    repository_root: Path, preregistration: dict[str, Any], wanted: set[str]
) -> list[dict[str, Any]]:
    """Stage 1 per-sequence rows, reconstructing the aggregate when it is absent.

    The 117.5 MiB aggregate is excluded from version control because it exceeds the
    GitHub per-file limit. It is the byte-exact concatenation, in frozen Stage 0
    workload order, of the committed per-workload checkpoints, so when the aggregate
    is missing this rebuilds the stream from those checkpoints and verifies the
    recorded SHA-256 before returning anything.
    """

    reference = preregistration["stage1_reference"]
    aggregate = repository_root / reference["per_sequence_results_path"]
    expected = reference["per_sequence_results_jsonl_sha256"]
    if aggregate.exists():
        if sha256_file(aggregate) != expected:
            raise ValueError("Frozen Stage 1 per-sequence rows changed")
        return list(_filtered(_iter_jsonl_path(aggregate), wanted))

    stage0_frozen = read_json(
        repository_root / preregistration["stage0_reference"]["frozen_evaluation_config_path"]
    )
    order = [item["name"] for item in stage0_frozen["workloads"]]
    checkpoints = aggregate.parent / "checkpoints"
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for name in order:
        path = checkpoints / name / "per_sequence_results.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Stage 1 per-sequence rows are unavailable: neither {aggregate} nor {path}"
            )
        data = path.read_bytes()
        digest.update(data)
        rows.extend(_filtered(_iter_jsonl_bytes(data), wanted))
    if digest.hexdigest() != expected:
        raise ValueError(
            "Reconstructed Stage 1 per-sequence stream does not match the sealed SHA-256"
        )
    return rows


def load_and_verify_stage3_inputs(
    repository_root: Path, preregistration_path: Path
) -> Stage3Inputs:
    repository_root = repository_root.resolve()
    preregistration, preregistration_hash = load_stage3_preregistration(preregistration_path)
    stage0_ref = preregistration["stage0_reference"]
    stage1_ref = preregistration["stage1_reference"]
    stage2_ref = preregistration["stage2_reference"]

    entries = [
        ("stage0 archive manifest", stage0_ref["final_archive_manifest_path"],
         stage0_ref["final_archive_manifest_sha256"]),
        ("stage0 routing trace", stage0_ref["trace_path"], stage0_ref["routing_trace_npz_sha256"]),
        ("stage0 frozen evaluation config", stage0_ref["frozen_evaluation_config_path"],
         stage0_ref["frozen_evaluation_config_file_sha256"]),
        ("stage0 results", stage0_ref["result_path"], stage0_ref["results_jsonl_sha256"]),
        ("stage0 per-sequence results", stage0_ref["per_sequence_result_path"],
         stage0_ref["per_sequence_results_jsonl_sha256"]),
        ("stage1 archive manifest", stage1_ref["final_archive_manifest_path"],
         stage1_ref["final_archive_manifest_sha256"]),
        ("stage1 preregistration", stage1_ref["preregistration_path"],
         stage1_ref["preregistration_hash"]),
        ("stage1 frozen config", stage1_ref["frozen_config_path"],
         stage1_ref["frozen_config_file_sha256"]),
        ("stage1 results", stage1_ref["results_path"], stage1_ref["results_jsonl_sha256"]),
        ("stage1 transition models", stage1_ref["transition_model_path"],
         stage1_ref["transition_model_sha256"]),
        ("stage2 archive manifest", stage2_ref["final_archive_manifest_path"],
         stage2_ref["final_archive_manifest_sha256"]),
        ("stage2 transition models", stage2_ref["transition_model_path"],
         stage2_ref["transition_model_sha256"]),
    ]
    verified = []
    for label, relative, expected in entries:
        path = repository_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Frozen {label} changed: {relative} ({actual} != {expected})")
        verified.append({"label": label, "path": relative, "sha256": actual})

    stage0 = load_and_verify_frozen_inputs(
        repository_root, repository_root / stage1_ref["preregistration_path"]
    )
    if stage0.trace.trace_hash != stage0_ref["trace_logical_hash"]:
        raise ValueError("Loaded trace does not match the Stage 3 preregistered logical hash")
    stage1_frozen = load_and_verify_stage1_frozen(
        repository_root / stage1_ref["frozen_config_path"]
    )
    if stage1_frozen["selected_predictor_id"] != stage1_ref["winner_method_id"]:
        raise ValueError("Frozen Stage 1 winner identifier changed")
    stage2_archive = read_json(repository_root / stage2_ref["final_archive_manifest_path"])
    if stage2_archive["verdict"] != stage2_ref["verdict"]:
        raise ValueError("Stage 2 archived verdict changed")

    references = stage0_references(repository_root / stage0_ref["result_path"])
    stage1_costs: dict[tuple[str, int], int] = {}
    stage1_conditions: dict[tuple[str, int], str] = {}
    winner = stage1_ref["winner_method_id"]
    for row in _iter_jsonl_path(repository_root / stage1_ref["results_path"]):
        if str(row["method_id"]) != winner:
            continue
        key = (str(row["workload"]), int(row["capacity"]))
        stage1_costs[key] = int(row["misses"])
        stage1_conditions[key] = str(row["condition_id"])
    expected_keys = {
        (workload.name, int(capacity))
        for workload in stage0.workloads
        for capacity in preregistration["cache_capacities"]
    }
    missing = sorted(expected_keys - set(stage1_costs))
    if missing:
        raise ValueError(f"Frozen Stage 1 winner rows are incomplete: {missing[:5]}")

    return Stage3Inputs(
        preregistration=preregistration,
        preregistration_hash=preregistration_hash,
        stage0=stage0,
        stage1_frozen=stage1_frozen,
        stage0_references=references,
        stage1_costs=stage1_costs,
        stage1_condition_ids=stage1_conditions,
        verification={
            "verified_files": verified,
            "stage1_winner_method_id": winner,
            "stage2_verdict": stage2_archive["verdict"],
            "repository_head": resolve_git_commit(repository_root),
            "stage3_source_bundle_hash": stage3_source_bundle_hash(repository_root),
        },
    )


def truncated_workload(workload: Workload, sequences: int, name: str) -> Workload:
    """A frozen-prefix pilot workload; only ever used on calibration data."""

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


def _iter_jsonl_path(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _iter_jsonl_bytes(data: bytes) -> Iterator[dict[str, Any]]:
    for line in data.splitlines():
        if line.strip():
            yield json.loads(line)


def _filtered(rows: Iterable[dict[str, Any]], wanted: set[str]) -> Iterator[dict[str, Any]]:
    for row in rows:
        if str(row.get("condition_id")) in wanted:
            yield row
