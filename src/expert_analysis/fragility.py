"""Stage 2C calibration domain fragility.

Domain fragility is the only new scientific quantity introduced by Stage 2C.
For each domain ``d`` and base precision ``b`` in {3, 4}, evaluated on the same
frozen 25-example/domain Stage 2B calibration subset:

``q_raw[d,b] = (NLL_base_cal[d,b] - NLL_BF16_cal[d]) / NLL_BF16_cal[d]``
``q[d,b] = max(q_raw[d,b], 0)``
``q_norm[d,b] = q[d,b] / mean_d q[d,b]``  (regime invalid if the mean is zero)

The clipping rule is preregistered: absolute value is never used, so a domain
that improves under uniform base quantization has zero fragility rather than a
reversed priority. No alternative fragility formula, coefficient fit, exponent,
or domain weight may be introduced. Expert-level delta-NLL estimation remains
blocked by the frozen Stage 2A SURROGATE_NO_GO decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .balanced import ControlledSource, array_sha256, canonical_sha256
from .io_utils import atomic_write_json, read_json, write_csv
from .masking import LossStatistics
from .specialist_preservation import (
    CALIBRATION_EXAMPLES_PER_DOMAIN,
    CALIBRATION_SEED,
    NUM_EXPERTS,
    NUM_MOE_LAYERS,
    STAGE2B_DOMAINS,
    select_calibration_indices,
)

STAGE2C_STAGE = "stage2c_fragility_robust_preservation"
FRAGILITY_SCHEMA_VERSION = "stage2c_calibration_fragility_v1"
STAGE2C_REGIMES = {"4to8": 4, "3to8": 3}
FRAGILITY_CLIP_RULE = "q = max(q_raw, 0); absolute value is never used"


def raw_relative_fragility(bf16_nll: float, base_nll: float) -> float:
    """``q_raw = (NLL_base - NLL_BF16) / NLL_BF16`` on calibration data."""

    bf16 = float(bf16_nll)
    base = float(base_nll)
    if not (np.isfinite(bf16) and np.isfinite(base)):
        raise ValueError("Calibration NLL values must be finite")
    if bf16 <= 0:
        raise ValueError("BF16 calibration NLL must be positive")
    return (base - bf16) / bf16


def clipped_fragility(raw: float) -> float:
    """Preregistered nonnegative clipping: ``max(q_raw, 0)``, never abs()."""

    value = float(raw)
    if not np.isfinite(value):
        raise ValueError("Raw fragility must be finite")
    return max(value, 0.0)


def normalized_fragility(
    clipped_by_domain: Mapping[str, float],
) -> tuple[dict[str, float] | None, bool]:
    """``q_norm = q / mean_d q`` (mean one across domains), or regime invalid.

    Returns ``(q_norm, regime_valid)``. When all four domains have zero clipped
    fragility the regime is invalid and must not be evaluated; no alternative
    normalization is permitted.
    """

    if tuple(clipped_by_domain) != STAGE2B_DOMAINS:
        raise ValueError("Fragility must be supplied for exactly the four domains")
    values = np.asarray([clipped_by_domain[d] for d in STAGE2B_DOMAINS], dtype=np.float64)
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("Clipped fragility values must be finite and nonnegative")
    mean = float(values.mean())
    if mean <= 0.0:
        return None, False
    normalized = values / mean
    return {
        domain: float(normalized[index])
        for index, domain in enumerate(STAGE2B_DOMAINS)
    }, True


def mean_calibration_nll(statistics: LossStatistics) -> float:
    """Domain calibration NLL: mean over examples of per-example mean NLL.

    This matches how every Stage 2B/2C domain-level NLL is aggregated.
    """

    return float(statistics.per_token_nll.mean())


def compute_regime_fragility(
    bf16_nll_by_domain: Mapping[str, float],
    base_nll_by_domain: Mapping[str, float],
    base_bits: int,
) -> dict[str, Any]:
    """All preregistered fragility quantities for one base-precision regime."""

    if base_bits not in STAGE2C_REGIMES.values():
        raise ValueError(f"Base precision {base_bits} is not preregistered")
    domains: dict[str, Any] = {}
    clipped: dict[str, float] = {}
    for domain in STAGE2B_DOMAINS:
        bf16 = float(bf16_nll_by_domain[domain])
        base = float(base_nll_by_domain[domain])
        raw = raw_relative_fragility(bf16, base)
        clipped[domain] = clipped_fragility(raw)
        domains[domain] = {
            "bf16_nll": bf16,
            "base_nll": base,
            "raw_delta_nll": base - bf16,
            "relative_delta": raw,
            "clipped_fragility": clipped[domain],
        }
    q_norm, regime_valid = normalized_fragility(clipped)
    for domain in STAGE2B_DOMAINS:
        domains[domain]["normalized_fragility"] = (
            q_norm[domain] if regime_valid else None
        )
    return {
        "base_bits": base_bits,
        "regime_valid": regime_valid,
        "regime_invalid_reason": (
            None
            if regime_valid
            else "all four domains have zero clipped fragility at this base precision"
        ),
        "mean_clipped_fragility": float(np.mean([clipped[d] for d in STAGE2B_DOMAINS])),
        "domains": domains,
    }


@dataclass
class Stage2BScoreArtifacts:
    """Frozen Stage 2B score arrays reloaded with full hash verification."""

    functional: np.ndarray
    functional_specialization: np.ndarray
    routing_specialization: np.ndarray
    single_domain: np.ndarray
    global_importance: np.ndarray
    calibration_indices: dict[str, list[int]]
    metadata: dict[str, Any]

    @property
    def calibration_fingerprint(self) -> str:
        return str(self.metadata["calibration_fingerprint"])

    @property
    def score_hashes(self) -> dict[str, str]:
        return dict(self.metadata["score_hashes"])


def load_frozen_stage2b_scores(stage2b_results_dir: Path) -> Stage2BScoreArtifacts:
    """Reuse the frozen Stage 2B specialization artifacts, verifying provenance.

    The scores are never recomputed from seed-43/44/45 data: the frozen arrays
    are loaded and every recorded SHA-256 plus the deterministic calibration
    selection is re-verified before use.
    """

    calibration_dir = stage2b_results_dir / "calibration"
    metadata = read_json(calibration_dir / "calibration_metadata.json")
    if metadata["calibration_seed"] != CALIBRATION_SEED:
        raise RuntimeError("Stage 2B calibration seed does not match the frozen value")
    if metadata["calibration_examples_per_domain"] != CALIBRATION_EXAMPLES_PER_DOMAIN:
        raise RuntimeError("Stage 2B calibration subset size changed")
    expected_selection = select_calibration_indices()
    indices: dict[str, list[int]] = {}
    for domain in STAGE2B_DOMAINS:
        recorded = list(
            metadata["domains_detail"][domain]["calibration_indices_into_frozen_set"]
        )
        if recorded != expected_selection.indices[domain].tolist():
            raise RuntimeError(
                f"Stage 2B calibration indices for {domain} are not the "
                "deterministic frozen selection"
            )
        indices[domain] = recorded

    with np.load(
        calibration_dir / "functional_importance.npz", allow_pickle=False
    ) as data:
        functional = np.asarray(data["functional"], dtype=np.float64)
        single_domain = np.asarray(data["single_domain"], dtype=np.float64)
        global_importance = np.asarray(data["global_importance"], dtype=np.float64)
    with np.load(
        calibration_dir / "functional_specialization.npz", allow_pickle=False
    ) as data:
        functional_specialization = np.asarray(
            data["specialization"], dtype=np.float64
        )
    with np.load(
        calibration_dir / "routing_specialization.npz", allow_pickle=False
    ) as data:
        routing_specialization = np.asarray(data["specialization"], dtype=np.float64)

    observed_hashes = {
        "functional_importance_sha256": array_sha256(functional),
        "functional_specialization_sha256": array_sha256(functional_specialization),
        "routing_specialization_sha256": array_sha256(routing_specialization),
        "single_domain_importance_sha256": array_sha256(single_domain),
        "global_importance_sha256": array_sha256(global_importance),
    }
    for name, expected in metadata["score_hashes"].items():
        if observed_hashes.get(name) != expected:
            raise RuntimeError(
                f"Frozen Stage 2B score hash {name} does not match; refusing to "
                "reuse the specialization artifacts"
            )
    expected_shape = (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS))
    if functional_specialization.shape != expected_shape:
        raise RuntimeError("Frozen specialization array has the wrong shape")
    if not np.allclose(functional_specialization.sum(axis=(0, 1)), 1.0, atol=1e-9):
        raise RuntimeError("Frozen specialization mass does not sum to one per domain")
    return Stage2BScoreArtifacts(
        functional=functional,
        functional_specialization=functional_specialization,
        routing_specialization=routing_specialization,
        single_domain=single_domain,
        global_importance=global_importance,
        calibration_indices=indices,
        metadata=metadata,
    )


def calibration_subset_inputs(
    source: ControlledSource,
    scores: Stage2BScoreArtifacts,
) -> dict[str, Any]:
    """Slice the frozen controlled inputs down to the 25-example subsets.

    Every selected row hash is verified against the Stage 2B calibration
    metadata before use, so the fragility evaluation provably runs on the
    identical frozen calibration examples.
    """

    from .controlled import PreparedDomainExamples

    subsets: dict[str, PreparedDomainExamples] = {}
    for domain in STAGE2B_DOMAINS:
        prepared = source.prepared[domain]
        indices = scores.calibration_indices[domain]
        recorded_hashes = scores.metadata["domains_detail"][domain][
            "calibration_input_row_sha256"
        ]
        observed_hashes = [
            array_sha256(np.ascontiguousarray(prepared.input_ids[i])) for i in indices
        ]
        if observed_hashes != list(recorded_hashes):
            raise RuntimeError(
                f"Calibration subset rows for {domain} do not match the frozen "
                "Stage 2B row hashes"
            )
        subset = PreparedDomainExamples(
            domain=domain,
            input_ids=np.ascontiguousarray(prepared.input_ids[indices]),
            attention_mask=np.ascontiguousarray(prepared.attention_mask[indices]),
            measurement_mask=np.ascontiguousarray(prepared.measurement_mask[indices]),
            metadata={
                "stage": STAGE2C_STAGE,
                "role": "frozen_stage2b_calibration_subset",
                "calibration_seed": CALIBRATION_SEED,
                "calibration_indices_into_frozen_set": list(indices),
                "calibration_input_row_sha256": list(recorded_hashes),
            },
        )
        subset.validate()
        subsets[domain] = subset
    return subsets


def fragility_deterministic_content(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in ("fragility_sha256", "created_at_utc")
    }


def build_calibration_fragility_record(
    regime_results: Mapping[str, Mapping[str, Any]],
    calibration_subset_hashes: Mapping[str, Mapping[str, Any]],
    model_info: Mapping[str, Any],
    qdq_config: Mapping[str, Any],
    environment: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the frozen ``calibration_fragility.json`` payload."""

    if set(regime_results) != set(STAGE2C_REGIMES):
        raise ValueError("Fragility must be recorded for exactly the two regimes")
    record: dict[str, Any] = {
        "schema": FRAGILITY_SCHEMA_VERSION,
        "stage": STAGE2C_STAGE,
        "fragility_formula": (
            "q_raw[d,b] = (NLL_base_cal[d,b] - NLL_BF16_cal[d]) / NLL_BF16_cal[d]"
        ),
        "clipping_rule": FRAGILITY_CLIP_RULE,
        "normalization_rule": (
            "q_norm[d,b] = q[d,b] / mean_d q[d,b]; regime invalid if the mean is zero"
        ),
        "nll_aggregation": "mean over calibration examples of per-example mean NLL",
        "calibration_examples_per_domain": CALIBRATION_EXAMPLES_PER_DOMAIN,
        "calibration_seed": CALIBRATION_SEED,
        "regimes": {regime: dict(regime_results[regime]) for regime in STAGE2C_REGIMES},
        "calibration_subset_hashes": {
            domain: dict(calibration_subset_hashes[domain])
            for domain in STAGE2B_DOMAINS
        },
        "model": dict(model_info),
        "qdq_config": dict(qdq_config),
        "deterministic_environment": dict(environment),
        "repeated_baseline_reproduction": dict(reproduction),
    }
    record["fragility_sha256"] = canonical_sha256(
        fragility_deterministic_content(record)
    )
    record["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    return record


def verify_fragility_record(record: Mapping[str, Any]) -> None:
    expected = canonical_sha256(fragility_deterministic_content(record))
    if record.get("fragility_sha256") != expected:
        raise RuntimeError(
            "calibration_fragility.json failed its SHA-256 integrity check"
        )


def save_calibration_fragility(
    record: Mapping[str, Any], calibration_dir: Path
) -> tuple[Path, Path]:
    """Freeze the fragility record and its CSV table."""

    verify_fragility_record(record)
    json_path = calibration_dir / "calibration_fragility.json"
    if json_path.is_file():
        existing = read_json(json_path)
        if fragility_deterministic_content(existing) != fragility_deterministic_content(
            dict(record)
        ):
            raise RuntimeError(
                "calibration_fragility.json is already frozen with different "
                "values; fragility must not be recomputed after freezing"
            )
        return json_path, calibration_dir / "calibration_fragility.csv"
    atomic_write_json(json_path, record)
    rows = []
    for regime in STAGE2C_REGIMES:
        for domain in STAGE2B_DOMAINS:
            entry = record["regimes"][regime]["domains"][domain]
            rows.append(
                {
                    "regime": regime,
                    "base_bits": record["regimes"][regime]["base_bits"],
                    "domain": domain,
                    "bf16_nll": entry["bf16_nll"],
                    "base_nll": entry["base_nll"],
                    "raw_delta_nll": entry["raw_delta_nll"],
                    "relative_fragility": entry["relative_delta"],
                    "clipped_fragility": entry["clipped_fragility"],
                    "normalized_fragility": entry["normalized_fragility"],
                    "regime_valid": record["regimes"][regime]["regime_valid"],
                }
            )
    csv_path = calibration_dir / "calibration_fragility.csv"
    write_csv(
        csv_path,
        rows,
        [
            "regime", "base_bits", "domain", "bf16_nll", "base_nll",
            "raw_delta_nll", "relative_fragility", "clipped_fragility",
            "normalized_fragility", "regime_valid",
        ],
    )
    return json_path, csv_path


def load_frozen_fragility(calibration_dir: Path) -> dict[str, Any]:
    """Load the frozen fragility record and fail loudly if it was altered."""

    record = read_json(calibration_dir / "calibration_fragility.json")
    if record.get("schema") != FRAGILITY_SCHEMA_VERSION:
        raise RuntimeError("Unexpected calibration fragility schema")
    verify_fragility_record(record)
    return record


def fragility_vector(record: Mapping[str, Any], regime: str) -> np.ndarray:
    """Normalized fragility for one regime in the fixed domain order."""

    if regime not in STAGE2C_REGIMES:
        raise ValueError(f"Regime {regime!r} is not preregistered")
    regime_record = record["regimes"][regime]
    if not regime_record["regime_valid"]:
        raise RuntimeError(
            f"Regime {regime} is invalid (all-zero fragility) and must not be "
            "optimized or evaluated"
        )
    return np.asarray(
        [
            regime_record["domains"][domain]["normalized_fragility"]
            for domain in STAGE2B_DOMAINS
        ],
        dtype=np.float64,
    )
