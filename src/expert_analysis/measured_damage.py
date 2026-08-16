"""Stage 3 measured per-expert quantization damage.

Stage 3 replaces every failed prediction of per-expert quantization damage
(Stage 2A surrogates, Stage 2B coverage, Stage 2C fragility-weighted coverage)
with direct measurement. For each MoE layer ``l``, expert ``e``, domain ``d``,
and bit width ``b`` in {3, 4, 8}, on the frozen 25-example/domain Stage 2B
calibration subset:

``m[l,e,d,b] = NLL_cal[d | only expert (l,e) QDQ-quantized to b bits]
              - NLL_cal[d | clean BF16]``

using the exact audited Stage-1 symmetric group-wise QDQ (group size 128) and
the frozen calibration NLL aggregation (mean over examples of per-example mean
NLL). Nothing is fitted, estimated, or predicted from weights, activations, or
gradients: every value is a measured ground-truth loss difference.

The additive working model for a full allocation ``x`` (protected experts at
8-bit, all others at the regime base precision) is

``PredictedDelta_d(x) = sum_{l,e} m[l, e, d, bits(l,e)]``

with BF16 cells contributing exactly zero. Whether this additive model is
usable is itself a preregistered gate: it must rank the frozen probe
allocations correctly on calibration data before any seed-46 evaluation is
authorized. If additivity fails, the stage stops with the negative result
preserved and the development split unevaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .balanced import array_sha256, canonical_sha256, file_sha256
from .io_utils import atomic_save_npz, atomic_write_json, read_json, write_csv
from .specialist_preservation import (
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
)
from .statistics import safe_spearman

STAGE3_STAGE = "stage3_measured_damage_preservation"
DAMAGE_SCHEMA_VERSION = "stage3_damage_matrix_v1"
DAMAGE_CHUNK_SCHEMA_VERSION = "stage3_damage_chunk_v1"
# Measured bit widths: both regime base precisions plus the 8-bit protected
# precision, so the additive model covers protected experts exactly instead of
# assuming their damage is zero.
STAGE3_PROFILE_BITS = (3, 4, 8)
STAGE3_REGIMES = {"4to8": 4, "3to8": 3}

# Preregistered additivity gates (fixed in code before any probe NLL exists).
# Gate ADD-1: for every domain, Spearman across the frozen 20%-budget probe
# allocations between additive predicted delta NLL and measured delta NLL must
# be at least this value.
ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN = 0.80
# Gate ADD-2: Spearman across probes between predicted and measured
# worst-domain delta NLL must be at least this value.
ADDITIVITY_MIN_WORST_DELTA_SPEARMAN = 0.80

ADDITIVITY_GO_DECISION = "ADDITIVITY_GO"
STAGE3_NO_GO_DECISION = "MEASURED_DAMAGE_NO_GO"


def profile_bits_index(bits: int) -> int:
    if bits not in STAGE3_PROFILE_BITS:
        raise ValueError(f"Bit width {bits} is not profiled in Stage 3")
    return STAGE3_PROFILE_BITS.index(bits)


@dataclass(frozen=True)
class DamageChunk:
    """Per-expert calibration losses for one (layer, bit width) pair."""

    layer: int
    bits: int
    loss_sums: np.ndarray
    token_counts: np.ndarray
    metadata: dict[str, Any]

    def validate(self, examples_per_domain: int) -> None:
        expected = (NUM_EXPERTS, len(STAGE2B_DOMAINS), examples_per_domain)
        if self.loss_sums.shape != expected or self.token_counts.shape != expected:
            raise ValueError(
                f"Damage chunk layer {self.layer} bits {self.bits} has the wrong "
                f"shape; expected {expected}"
            )
        if not np.all(np.isfinite(self.loss_sums)) or np.any(self.loss_sums < 0):
            raise ValueError("Damage chunk loss sums contain invalid values")
        if np.any(self.token_counts <= 0):
            raise ValueError("Damage chunk token counts must be positive")

    def mean_nll(self) -> np.ndarray:
        """``[expert, domain]`` mean over examples of per-example mean NLL."""

        per_example = self.loss_sums / self.token_counts.astype(np.float64)
        return per_example.mean(axis=2)


def damage_chunk_path(damage_dir: Path, bits: int, layer: int) -> Path:
    return damage_dir / "chunks" / f"bits{bits}" / f"layer_{layer:02d}.npz"


def save_damage_chunk(
    path: Path,
    chunk: DamageChunk,
    expected_metadata: Mapping[str, Any],
    examples_per_domain: int,
) -> dict[str, Any]:
    chunk.validate(examples_per_domain)
    arrays = {
        "loss_sums": chunk.loss_sums.astype(np.float64, copy=False),
        "token_counts": chunk.token_counts.astype(np.uint32, copy=False),
    }
    atomic_save_npz(path, **arrays)
    payload = {
        **dict(expected_metadata),
        "schema": DAMAGE_CHUNK_SCHEMA_VERSION,
        "layer": chunk.layer,
        "bits": chunk.bits,
        "num_experts": NUM_EXPERTS,
        "domains": list(STAGE2B_DOMAINS),
        "examples_per_domain": examples_per_domain,
        "array_sha256": {
            key: array_sha256(np.asarray(value)) for key, value in arrays.items()
        },
        "npz_sha256": file_sha256(path),
    }
    atomic_write_json(path.with_suffix(".metadata.json"), payload)
    return payload


def load_damage_chunk(
    path: Path,
    expected_metadata: Mapping[str, Any],
    examples_per_domain: int,
) -> DamageChunk:
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Damage chunk is incomplete: {path}")
    metadata = read_json(metadata_path)
    mismatches = [
        key for key, value in expected_metadata.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Damage chunk fingerprint mismatch for {path}: " + ", ".join(mismatches)
        )
    if metadata.get("npz_sha256") != file_sha256(path):
        raise RuntimeError(f"Damage chunk file hash mismatch for {path}")
    with np.load(path, allow_pickle=False) as data:
        loss_sums = np.asarray(data["loss_sums"], dtype=np.float64)
        token_counts = np.asarray(data["token_counts"], dtype=np.uint32)
    for key, values in (("loss_sums", loss_sums), ("token_counts", token_counts)):
        if metadata.get("array_sha256", {}).get(key) != array_sha256(values):
            raise RuntimeError(f"Damage chunk array hash mismatch for {path}:{key}")
    chunk = DamageChunk(
        layer=int(metadata["layer"]),
        bits=int(metadata["bits"]),
        loss_sums=loss_sums,
        token_counts=token_counts,
        metadata=metadata,
    )
    chunk.validate(examples_per_domain)
    return chunk


def assemble_damage_arrays(
    damage_dir: Path,
    expected_metadata: Mapping[str, Any],
    bf16_nll_by_domain: Mapping[str, float],
    examples_per_domain: int,
) -> dict[str, np.ndarray]:
    """Assemble ``mean_nll`` and ``delta_nll`` [layer, expert, domain, bits]."""

    shape = (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS), len(STAGE3_PROFILE_BITS))
    mean_nll = np.zeros(shape, dtype=np.float64)
    bf16 = np.asarray(
        [float(bf16_nll_by_domain[d]) for d in STAGE2B_DOMAINS], dtype=np.float64
    )
    if np.any(bf16 <= 0) or not np.all(np.isfinite(bf16)):
        raise ValueError("BF16 calibration NLL values must be finite and positive")
    for bit_index, bits in enumerate(STAGE3_PROFILE_BITS):
        for layer in range(NUM_MOE_LAYERS):
            chunk = load_damage_chunk(
                damage_chunk_path(damage_dir, bits, layer),
                {**dict(expected_metadata), "layer": layer, "bits": bits},
                examples_per_domain,
            )
            mean_nll[layer, :, :, bit_index] = chunk.mean_nll()
    delta_nll = mean_nll - bf16[None, None, :, None]
    if not np.all(np.isfinite(delta_nll)):
        raise RuntimeError("Assembled damage deltas contain non-finite values")
    return {"mean_nll": mean_nll, "delta_nll": delta_nll, "bf16_nll": bf16}


def damage_deterministic_content(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in ("damage_sha256", "created_at_utc")
    }


def build_damage_record(
    arrays: Mapping[str, np.ndarray],
    uniform_nll: Mapping[str, Mapping[str, float]],
    frozen_reference_drift: Mapping[str, Any],
    calibration_subset_hashes: Mapping[str, Mapping[str, Any]],
    model_info: Mapping[str, Any],
    qdq_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    reproduction: Mapping[str, Any],
    chunk_hashes: Mapping[str, str],
    examples_per_domain: int,
) -> dict[str, Any]:
    """Assemble the frozen ``damage_matrix.json`` payload."""

    delta = np.asarray(arrays["delta_nll"], dtype=np.float64)
    record: dict[str, Any] = {
        "schema": DAMAGE_SCHEMA_VERSION,
        "stage": STAGE3_STAGE,
        "damage_definition": (
            "m[l,e,d,b] = NLL_cal[d | only expert (l,e) at b-bit QDQ] - "
            "NLL_cal[d | clean BF16]; measured, never estimated"
        ),
        "nll_aggregation": "mean over calibration examples of per-example mean NLL",
        "no_clipping_note": (
            "negative measured damages are retained exactly; no clipping, "
            "normalization, or reweighting is applied to any measured value"
        ),
        "profile_bits": list(STAGE3_PROFILE_BITS),
        "domains": list(STAGE2B_DOMAINS),
        "num_moe_layers": NUM_MOE_LAYERS,
        "num_experts": NUM_EXPERTS,
        "calibration_examples_per_domain": examples_per_domain,
        "bf16_nll": {
            domain: float(arrays["bf16_nll"][index])
            for index, domain in enumerate(STAGE2B_DOMAINS)
        },
        "uniform_nll": {
            state: {domain: float(values[domain]) for domain in STAGE2B_DOMAINS}
            for state, values in uniform_nll.items()
        },
        "frozen_stage2c_reference_drift": dict(frozen_reference_drift),
        "summary": {
            f"bits{bits}": {
                "total_delta_nll_by_domain": {
                    domain: float(delta[:, :, index, bit_index].sum())
                    for index, domain in enumerate(STAGE2B_DOMAINS)
                },
                "negative_damage_cells": int(
                    (delta[:, :, :, bit_index] < 0).sum()
                ),
                "max_single_expert_delta": float(delta[:, :, :, bit_index].max()),
            }
            for bit_index, bits in enumerate(STAGE3_PROFILE_BITS)
        },
        "array_sha256": {
            "mean_nll": array_sha256(np.asarray(arrays["mean_nll"])),
            "delta_nll": array_sha256(delta),
            "bf16_nll": array_sha256(np.asarray(arrays["bf16_nll"])),
        },
        "chunk_sha256": dict(chunk_hashes),
        "calibration_subset_hashes": {
            domain: dict(calibration_subset_hashes[domain])
            for domain in STAGE2B_DOMAINS
        },
        "model": dict(model_info),
        "qdq_config": dict(qdq_config),
        "deterministic_environment": dict(environment),
        "repeated_evaluation_reproduction": dict(reproduction),
    }
    record["damage_sha256"] = canonical_sha256(damage_deterministic_content(record))
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def verify_damage_record(record: Mapping[str, Any]) -> None:
    expected = canonical_sha256(damage_deterministic_content(record))
    if record.get("damage_sha256") != expected:
        raise RuntimeError("damage_matrix.json failed its SHA-256 integrity check")


def save_damage_matrix(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    damage_dir: Path,
) -> tuple[Path, Path]:
    """Freeze the damage record, arrays, and CSV table."""

    verify_damage_record(record)
    json_path = damage_dir / "damage_matrix.json"
    npz_path = damage_dir / "damage_matrix.npz"
    if json_path.is_file():
        existing = read_json(json_path)
        if damage_deterministic_content(existing) != damage_deterministic_content(
            dict(record)
        ):
            raise RuntimeError(
                "damage_matrix.json is already frozen with different values; the "
                "damage matrix must not be recomputed after freezing"
            )
        return json_path, npz_path
    atomic_save_npz(
        npz_path,
        mean_nll=np.asarray(arrays["mean_nll"], dtype=np.float64),
        delta_nll=np.asarray(arrays["delta_nll"], dtype=np.float64),
        bf16_nll=np.asarray(arrays["bf16_nll"], dtype=np.float64),
        domains=np.asarray(STAGE2B_DOMAINS, dtype=np.str_),
        profile_bits=np.asarray(STAGE3_PROFILE_BITS, dtype=np.int8),
    )
    atomic_write_json(json_path, record)
    delta = np.asarray(arrays["delta_nll"], dtype=np.float64)
    rows = [
        {
            "layer": layer,
            "expert_id": expert,
            "domain": domain,
            "bits": bits,
            "delta_nll": delta[layer, expert, domain_index, bit_index],
        }
        for layer in range(NUM_MOE_LAYERS)
        for expert in range(NUM_EXPERTS)
        for domain_index, domain in enumerate(STAGE2B_DOMAINS)
        for bit_index, bits in enumerate(STAGE3_PROFILE_BITS)
    ]
    write_csv(
        damage_dir / "damage_matrix.csv",
        rows,
        ["layer", "expert_id", "domain", "bits", "delta_nll"],
    )
    return json_path, npz_path


def load_frozen_damage(damage_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load the frozen damage record and arrays, verifying every hash."""

    record = read_json(damage_dir / "damage_matrix.json")
    if record.get("schema") != DAMAGE_SCHEMA_VERSION:
        raise RuntimeError("Unexpected damage matrix schema")
    verify_damage_record(record)
    with np.load(damage_dir / "damage_matrix.npz", allow_pickle=False) as data:
        arrays = {
            "mean_nll": np.asarray(data["mean_nll"], dtype=np.float64),
            "delta_nll": np.asarray(data["delta_nll"], dtype=np.float64),
            "bf16_nll": np.asarray(data["bf16_nll"], dtype=np.float64),
        }
        stored_bits = tuple(int(v) for v in data["profile_bits"])
        stored_domains = tuple(str(v) for v in data["domains"])
    if stored_bits != STAGE3_PROFILE_BITS or stored_domains != STAGE2B_DOMAINS:
        raise RuntimeError("Damage matrix axes do not match the frozen definition")
    for name, values in arrays.items():
        if record["array_sha256"][name] != array_sha256(values):
            raise RuntimeError(f"Damage array {name} does not match its frozen hash")
    return record, arrays


def predicted_domain_delta_nll(
    expert_bits: np.ndarray, delta_nll: np.ndarray
) -> np.ndarray:
    """Additive predicted delta NLL per domain for one full bit assignment.

    ``expert_bits`` is the ``[layer, expert]`` matrix with values in
    {3, 4, 8, 16}; a 16-bit (BF16) cell contributes exactly zero.
    """

    bits = np.asarray(expert_bits)
    delta = np.asarray(delta_nll, dtype=np.float64)
    expected = (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS), len(STAGE3_PROFILE_BITS))
    if delta.shape != expected:
        raise ValueError(f"Damage deltas must have shape {expected}")
    if bits.shape != (NUM_MOE_LAYERS, NUM_EXPERTS):
        raise ValueError("Bit-assignment matrix does not match the model layout")
    allowed = set(STAGE3_PROFILE_BITS) | {16}
    if not set(np.unique(bits).tolist()) <= allowed:
        raise ValueError(f"Bit assignments must be within {sorted(allowed)}")
    predicted = np.zeros(len(STAGE2B_DOMAINS), dtype=np.float64)
    for bit_index, width in enumerate(STAGE3_PROFILE_BITS):
        mask = bits == width
        if mask.any():
            predicted += delta[:, :, :, bit_index][mask].sum(axis=0)
    return predicted


def additivity_gates_for_regime(
    probe_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the two preregistered additivity gates to one regime's probes.

    Every probe row supplies ``predicted`` and ``measured`` per-domain delta
    NLL lists in the fixed domain order.
    """

    if len(probe_rows) < 3:
        raise ValueError("Additivity gates require at least three probe allocations")
    predicted = np.asarray(
        [row["predicted"] for row in probe_rows], dtype=np.float64
    )
    measured = np.asarray([row["measured"] for row in probe_rows], dtype=np.float64)
    if predicted.shape != measured.shape or predicted.shape[1] != len(STAGE2B_DOMAINS):
        raise ValueError("Probe predicted/measured arrays are misaligned")
    if not (np.all(np.isfinite(predicted)) and np.all(np.isfinite(measured))):
        raise ValueError("Probe deltas contain non-finite values")

    per_domain = {
        domain: safe_spearman(predicted[:, index], measured[:, index])
        for index, domain in enumerate(STAGE2B_DOMAINS)
    }
    worst_spearman = safe_spearman(predicted.max(axis=1), measured.max(axis=1))
    predicted_worst_domain = [
        STAGE2B_DOMAINS[int(index)] for index in predicted.argmax(axis=1)
    ]
    measured_worst_domain = [
        STAGE2B_DOMAINS[int(index)] for index in measured.argmax(axis=1)
    ]
    agreement = float(
        np.mean(
            [p == m for p, m in zip(predicted_worst_domain, measured_worst_domain)]
        )
    )
    ratio = predicted.sum() / measured.sum() if measured.sum() != 0 else float("nan")
    gate_add_1 = {
        "name": "per_domain_probe_ranking",
        "spearman_by_domain": per_domain,
        "threshold": ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN,
        "passed": bool(
            all(
                np.isfinite(value) and value >= ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN
                for value in per_domain.values()
            )
        ),
    }
    gate_add_2 = {
        "name": "worst_domain_delta_ranking",
        "spearman": worst_spearman,
        "threshold": ADDITIVITY_MIN_WORST_DELTA_SPEARMAN,
        "passed": bool(
            np.isfinite(worst_spearman)
            and worst_spearman >= ADDITIVITY_MIN_WORST_DELTA_SPEARMAN
        ),
    }
    gates = {
        "gate_add_1": gate_add_1,
        "gate_add_2": gate_add_2,
        "all_passed": bool(gate_add_1["passed"] and gate_add_2["passed"]),
        "diagnostics_not_gated": {
            "worst_domain_identification_agreement": agreement,
            "total_predicted_over_measured_ratio": float(ratio),
            "median_absolute_relative_error": float(
                np.median(
                    np.abs(predicted - measured)
                    / np.maximum(np.abs(measured), 1e-12)
                )
            ),
            "probe_count": len(probe_rows),
        },
    }
    return gates


def additivity_decision(
    gates_by_regime: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    authorized = [
        regime for regime, gates in gates_by_regime.items() if gates["all_passed"]
    ]
    decision = ADDITIVITY_GO_DECISION if authorized else STAGE3_NO_GO_DECISION
    return {
        "decision": decision,
        "authorized_regimes": authorized,
        "regimes_evaluated": {
            regime: "PASS" if gates["all_passed"] else "FAIL"
            for regime, gates in gates_by_regime.items()
        },
        "rule": (
            "A regime is authorized for seed-46 development evaluation only if "
            "the additive damage model passes both preregistered ranking gates "
            "on the frozen calibration probes. If no regime passes, the stage "
            "decision is MEASURED_DAMAGE_NO_GO, the negative result is "
            "preserved, and neither seed 46 nor seed 44 is ever evaluated. "
            "Gate thresholds are never adjusted after any probe NLL exists."
        ),
    }
