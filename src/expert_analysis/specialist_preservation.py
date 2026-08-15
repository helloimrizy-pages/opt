"""Stage 2B calibration scores: functional importance and specialization.

This module builds the frozen optimization inputs for robust specialist
preservation. It reads only the audited controlled baseline artifacts
(per-example routing and functional-contribution arrays) and never any
quantization outcome, masking outcome, or held-out NLL. The formulas are fixed:

``F_layer[l,e,d] = F_raw[l,e,d] / sum_e F_raw[l,e,d]``
``F[l,e,d] = F_layer[l,e,d] / L``
``S_raw[l,e,d] = F[l,e,d] - max_{d' != d} F[l,e,d']``
``S[l,e,d] = max(0, S_raw[l,e,d]) / sum_{l,e} max(0, S_raw[l,e,d])``

The routing analogue replaces functional contribution with routing counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .balanced import ControlledSource, array_sha256, canonical_sha256
from .datasets import DOMAIN_SEED_OFFSETS
from .io_utils import atomic_save_npz, atomic_write_json

STAGE2B_DOMAINS = ("general", "math", "coding", "reasoning")
CALIBRATION_SEED = 20260815
CALIBRATION_EXAMPLES_PER_DOMAIN = 25
SINGLE_DOMAIN_CALIBRATION_EXAMPLES = 100
TOTAL_CALIBRATION_BUDGET = 100
NUM_MOE_LAYERS = 16
NUM_EXPERTS = 64


@dataclass(frozen=True)
class CalibrationSelection:
    """Deterministic per-domain calibration subset of the frozen controlled set."""

    seed: int
    per_domain: int
    indices: dict[str, np.ndarray]

    def validate(self, num_available: int) -> None:
        if tuple(self.indices) != STAGE2B_DOMAINS:
            raise ValueError("Calibration selection must cover exactly the four domains")
        total = 0
        for domain, values in self.indices.items():
            if values.ndim != 1 or len(values) != self.per_domain:
                raise ValueError(f"Calibration subset for {domain} has the wrong size")
            if len(np.unique(values)) != len(values):
                raise ValueError(f"Calibration subset for {domain} repeats an example")
            if np.any(values < 0) or np.any(values >= num_available):
                raise ValueError(f"Calibration subset for {domain} is out of range")
            if not np.array_equal(values, np.sort(values)):
                raise ValueError(f"Calibration subset for {domain} must be sorted")
            total += len(values)
        if total != TOTAL_CALIBRATION_BUDGET:
            raise ValueError(
                f"Multi-domain calibration must use exactly {TOTAL_CALIBRATION_BUDGET} "
                f"examples in total, found {total}"
            )


def select_calibration_indices(
    num_available: int = 100,
    per_domain: int = CALIBRATION_EXAMPLES_PER_DOMAIN,
    seed: int = CALIBRATION_SEED,
) -> CalibrationSelection:
    """Choose the fixed 25-example calibration subset per domain.

    Every domain uses an independent generator seeded from the calibration seed
    and the domain's fixed offset already used by the dataset loaders, so the
    subset does not depend on iteration order.
    """

    if per_domain < 1 or per_domain > num_available:
        raise ValueError("per_domain must be within the available example count")
    indices: dict[str, np.ndarray] = {}
    for domain in STAGE2B_DOMAINS:
        rng = np.random.default_rng([seed, DOMAIN_SEED_OFFSETS[domain]])
        chosen = rng.choice(num_available, size=per_domain, replace=False)
        indices[domain] = np.sort(chosen.astype(np.int64))
    selection = CalibrationSelection(seed=seed, per_domain=per_domain, indices=indices)
    selection.validate(num_available)
    return selection


def layer_normalized_importance(raw: np.ndarray, num_layers: int | None = None) -> np.ndarray:
    """Normalize within each layer, then give every layer equal total mass.

    ``raw`` has shape ``[layer, expert]``. The result sums to exactly one over
    all layer/expert cells.
    """

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Raw importance must have shape [layer, expert]")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Raw importance contains invalid values")
    layer_totals = values.sum(axis=1, keepdims=True)
    if np.any(layer_totals <= 0):
        empty = np.flatnonzero(layer_totals[:, 0] <= 0).tolist()
        raise ValueError(f"Layers {empty} have zero total importance mass")
    layers = num_layers if num_layers is not None else values.shape[0]
    if layers != values.shape[0]:
        raise ValueError("num_layers does not match the raw importance array")
    return values / layer_totals / float(layers)


def build_importance_tensor(
    raw_by_domain: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Stack per-domain layer-equalized importance into ``[layer, expert, domain]``."""

    if tuple(raw_by_domain) != STAGE2B_DOMAINS:
        raise ValueError("Importance must be supplied for exactly the four domains in order")
    stacked = np.stack(
        [layer_normalized_importance(raw_by_domain[domain]) for domain in STAGE2B_DOMAINS],
        axis=-1,
    )
    totals = stacked.sum(axis=(0, 1))
    if not np.allclose(totals, 1.0, atol=1e-9):
        raise RuntimeError("Importance normalization did not sum to one per domain")
    return stacked


def specialization_margins(importance: np.ndarray) -> np.ndarray:
    """``S_raw[l,e,d] = F[l,e,d] - max over the other domains of F[l,e,d']``."""

    values = np.asarray(importance, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(STAGE2B_DOMAINS):
        raise ValueError("Importance must have shape [layer, expert, domain]")
    margins = np.empty_like(values)
    for domain_index in range(values.shape[2]):
        others = np.delete(values, domain_index, axis=2)
        margins[:, :, domain_index] = values[:, :, domain_index] - others.max(axis=2)
    return margins


def normalized_specialization(margins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S_pos, S)`` where S is positive-part mass normalized per domain.

    Raises if any domain has zero positive specialist mass; the preregistration
    forbids redefining the score in that case.
    """

    positive = np.clip(np.asarray(margins, dtype=np.float64), 0.0, None)
    totals = positive.sum(axis=(0, 1))
    empty = [
        STAGE2B_DOMAINS[index]
        for index in range(len(STAGE2B_DOMAINS))
        if totals[index] <= 0.0
    ]
    if empty:
        raise RuntimeError(
            "Domains with zero positive specialist mass: "
            + ", ".join(empty)
            + "; aborting instead of redefining the specialization score"
        )
    return positive, positive / totals[None, None, :]


def specialist_coverage(specialization: np.ndarray, protected: np.ndarray) -> np.ndarray:
    """``Coverage_d(x) = sum_{l,e} S[l,e,d] * x[l,e]`` for binary ``x``."""

    scores = np.asarray(specialization, dtype=np.float64)
    x = np.asarray(protected)
    if scores.ndim != 3 or x.shape != scores.shape[:2]:
        raise ValueError("Protection matrix must have shape [layer, expert]")
    if not np.all(np.isin(x, (0, 1))):
        raise ValueError("Protection variables must be binary")
    return np.einsum("led,le->d", scores, x.astype(np.float64))


@dataclass
class SpecialistScores:
    """All frozen Stage 2B score arrays plus their hashes."""

    selection: CalibrationSelection
    functional_raw: np.ndarray
    functional: np.ndarray
    functional_specialization_raw: np.ndarray
    functional_specialization_pos: np.ndarray
    functional_specialization: np.ndarray
    routing_raw: np.ndarray
    routing: np.ndarray
    routing_specialization_raw: np.ndarray
    routing_specialization_pos: np.ndarray
    routing_specialization: np.ndarray
    single_domain_raw: np.ndarray
    single_domain: np.ndarray
    global_importance: np.ndarray
    metadata: dict[str, Any]

    def hashes(self) -> dict[str, str]:
        return {
            "functional_importance_sha256": array_sha256(self.functional),
            "functional_specialization_sha256": array_sha256(
                self.functional_specialization
            ),
            "routing_specialization_sha256": array_sha256(self.routing_specialization),
            "single_domain_importance_sha256": array_sha256(self.single_domain),
            "global_importance_sha256": array_sha256(self.global_importance),
        }


def build_specialist_scores(source: ControlledSource) -> SpecialistScores:
    """Construct every frozen Stage 2B score from the audited controlled source."""

    selection = select_calibration_indices()
    selection.validate(num_available=source.statistics[STAGE2B_DOMAINS[0]].num_examples)

    functional_raw = np.zeros(
        (NUM_MOE_LAYERS, NUM_EXPERTS, len(STAGE2B_DOMAINS)), dtype=np.float64
    )
    routing_raw = np.zeros_like(functional_raw)
    single_raw = np.zeros_like(functional_raw)
    calibration_rows: dict[str, Any] = {}
    for domain_index, domain in enumerate(STAGE2B_DOMAINS):
        statistics = source.statistics[domain]
        if statistics.routing_counts.shape != (100, NUM_MOE_LAYERS, NUM_EXPERTS):
            raise RuntimeError(f"Unexpected controlled statistic shape for {domain}")
        indices = selection.indices[domain]
        functional_raw[:, :, domain_index] = statistics.contribution_sums[indices].sum(
            axis=0, dtype=np.float64
        )
        routing_raw[:, :, domain_index] = statistics.routing_counts[indices].sum(
            axis=0, dtype=np.float64
        )
        single_raw[:, :, domain_index] = statistics.contribution_sums.sum(
            axis=0, dtype=np.float64
        )
        prepared = source.prepared[domain]
        selected_ids = list(prepared.metadata.get("selected_example_ids", []))
        if len(selected_ids) != statistics.num_examples:
            raise RuntimeError(f"Selected example IDs are incomplete for {domain}")
        calibration_rows[domain] = {
            "calibration_indices_into_frozen_set": indices.tolist(),
            "calibration_example_ids": [selected_ids[i] for i in indices.tolist()],
            "calibration_input_row_sha256": [
                array_sha256(np.ascontiguousarray(prepared.input_ids[i]))
                for i in indices.tolist()
            ],
            "single_domain_examples_used": statistics.num_examples,
            "single_domain_input_ids_sha256": array_sha256(prepared.input_ids),
        }

    raw_by_domain = {
        domain: functional_raw[:, :, index]
        for index, domain in enumerate(STAGE2B_DOMAINS)
    }
    functional = build_importance_tensor(raw_by_domain)
    functional_margins = specialization_margins(functional)
    functional_pos, functional_spec = normalized_specialization(functional_margins)

    routing_by_domain = {
        domain: routing_raw[:, :, index]
        for index, domain in enumerate(STAGE2B_DOMAINS)
    }
    routing = build_importance_tensor(routing_by_domain)
    routing_margins = specialization_margins(routing)
    routing_pos, routing_spec = normalized_specialization(routing_margins)

    single_by_domain = {
        domain: single_raw[:, :, index]
        for index, domain in enumerate(STAGE2B_DOMAINS)
    }
    single = build_importance_tensor(single_by_domain)
    global_importance = functional.mean(axis=2)

    metadata = {
        "stage": "stage2b_robust_specialist_preservation",
        "score_source": "frozen_controlled_baseline_arrays_only",
        "uses_quantization_outcomes": False,
        "uses_masking_outcomes": False,
        "uses_delta_nll_surrogates": False,
        "domains": list(STAGE2B_DOMAINS),
        "num_moe_layers": NUM_MOE_LAYERS,
        "num_experts": NUM_EXPERTS,
        "calibration_seed": selection.seed,
        "calibration_examples_per_domain": selection.per_domain,
        "multi_domain_total_calibration_examples": TOTAL_CALIBRATION_BUDGET,
        "single_domain_total_calibration_examples": SINGLE_DOMAIN_CALIBRATION_EXAMPLES,
        "equal_total_calibration_budget": True,
        "functional_statistic": "contribution_sums (L2 norm of selected gate weight times expert output)",
        "routing_statistic": "routing_counts (selected top-k route assignments)",
        "layer_equalization": f"per-layer normalization divided by L={NUM_MOE_LAYERS}",
        "specialization_formula": "S_raw[l,e,d] = F[l,e,d] - max_{d'!=d} F[l,e,d']; "
        "S = positive part normalized per domain over all layer/expert cells",
        "source_root": str(source.root),
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "source_selection_input_fingerprint": source.input_fingerprint,
        "source_resolved_model_revision": source.config["resolved_model_revision"],
        "dataset_revisions": source.config["dataset_revisions"],
        "domains_detail": calibration_rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    scores = SpecialistScores(
        selection=selection,
        functional_raw=functional_raw,
        functional=functional,
        functional_specialization_raw=functional_margins,
        functional_specialization_pos=functional_pos,
        functional_specialization=functional_spec,
        routing_raw=routing_raw,
        routing=routing,
        routing_specialization_raw=routing_margins,
        routing_specialization_pos=routing_pos,
        routing_specialization=routing_spec,
        single_domain_raw=single_raw,
        single_domain=single,
        global_importance=global_importance,
        metadata=metadata,
    )
    metadata["score_hashes"] = scores.hashes()
    metadata["calibration_fingerprint"] = canonical_sha256(
        {
            "calibration_seed": selection.seed,
            "calibration_examples_per_domain": selection.per_domain,
            "indices": {
                domain: values.tolist() for domain, values in selection.indices.items()
            },
            "source_collection_fingerprint": source.config["collection_fingerprint"],
            "score_hashes": metadata["score_hashes"],
        }
    )
    return scores


def save_specialist_scores(scores: SpecialistScores, calibration_dir: Path) -> dict[str, Path]:
    """Write the frozen score package used by the allocation solver."""

    calibration_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "functional_importance": calibration_dir / "functional_importance.npz",
        "functional_specialization": calibration_dir / "functional_specialization.npz",
        "routing_specialization": calibration_dir / "routing_specialization.npz",
        "calibration_metadata": calibration_dir / "calibration_metadata.json",
    }
    atomic_save_npz(
        paths["functional_importance"],
        functional_raw=scores.functional_raw,
        functional=scores.functional,
        single_domain_raw=scores.single_domain_raw,
        single_domain=scores.single_domain,
        global_importance=scores.global_importance,
        domains=np.asarray(STAGE2B_DOMAINS, dtype=np.str_),
    )
    atomic_save_npz(
        paths["functional_specialization"],
        specialization_raw=scores.functional_specialization_raw,
        specialization_pos=scores.functional_specialization_pos,
        specialization=scores.functional_specialization,
        domains=np.asarray(STAGE2B_DOMAINS, dtype=np.str_),
    )
    atomic_save_npz(
        paths["routing_specialization"],
        routing_raw=scores.routing_raw,
        routing=scores.routing,
        specialization_raw=scores.routing_specialization_raw,
        specialization_pos=scores.routing_specialization_pos,
        specialization=scores.routing_specialization,
        domains=np.asarray(STAGE2B_DOMAINS, dtype=np.str_),
    )
    atomic_write_json(paths["calibration_metadata"], scores.metadata)
    return paths
