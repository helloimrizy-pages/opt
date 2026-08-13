from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .controlled import PreparedDomainExamples
from .io_utils import atomic_write_json, read_json, write_csv
from .metrics import DomainStatistics
from .statistics import descending_ranks


BALANCED_DOMAINS = ("general", "math", "coding", "reasoning")
EXPECTED_MODEL = "allenai/OLMoE-1B-7B-0924"
EXPECTED_MODEL_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"
EXPECTED_DATASET_REVISIONS = {
    "general": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "math": "740312add88f781978c0658806c59bc2815b9866",
    "coding": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "reasoning": "210d026faf9955653af8916fad021475a3f00453",
}
EXPECTED_PREFIX = "Input:\n"
EXPECTED_PREFIX_IDS = (8982, 27, 187)
EXPECTED_EXAMPLES = 100
EXPECTED_MEASURED_POSITIONS = 64
EXPECTED_SEQUENCE_LENGTH = 68
EXPECTED_MOE_LAYERS = 16
EXPECTED_EXPERTS = 64
EXPECTED_TOP_K = 8
SELECTION_ALGORITHM_VERSION = "functional_margin_layer_diverse_v1"
CONTROL_ALGORITHM_VERSION = "same_layer_routing_assignment_v1"


@dataclass
class ControlledSource:
    root: Path
    config: dict[str, Any]
    architecture: dict[str, Any]
    corpus: dict[str, Any]
    manifest: dict[str, Any]
    statistics: dict[str, DomainStatistics]
    prepared: dict[str, PreparedDomainExamples]
    file_sha256: dict[str, str]
    input_fingerprint: str
    audit: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_controlled_source(source_dir: Path) -> ControlledSource:
    """Load and independently audit the baseline-only controlled artifacts.

    Deliberately does not read ``results.json``, masking CSVs, or any file below
    ``masking/``. This boundary keeps expert selection independent of interventions.
    """

    root = source_dir.resolve()
    required = [
        root / "collection_config.json",
        root / "architecture.json",
        root / "controlled_corpus.json",
        root / "collection_manifest.json",
    ]
    for domain in BALANCED_DOMAINS:
        required.extend(
            [
                root / "domains" / f"{domain}.npz",
                root / "domains" / f"{domain}.metadata.json",
                root / "controlled_inputs" / f"{domain}.npz",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Controlled source is incomplete; missing: " + ", ".join(missing)
        )

    config = read_json(root / "collection_config.json")
    architecture = read_json(root / "architecture.json")
    corpus = read_json(root / "controlled_corpus.json")
    manifest = read_json(root / "collection_manifest.json")
    _validate_source_metadata(config, architecture, corpus, manifest)

    statistics: dict[str, DomainStatistics] = {}
    prepared: dict[str, PreparedDomainExamples] = {}
    domain_checks: dict[str, Any] = {}
    reference_measurement_mask: np.ndarray | None = None
    for domain in BALANCED_DOMAINS:
        statistics_path = root / "domains" / f"{domain}.npz"
        metadata_path = root / "domains" / f"{domain}.metadata.json"
        input_path = root / "controlled_inputs" / f"{domain}.npz"
        metadata = read_json(metadata_path)
        item = DomainStatistics.load(statistics_path)
        examples = PreparedDomainExamples.load(input_path, domain, metadata)
        checks = _validate_domain_artifacts(domain, item, examples, metadata, config)
        if reference_measurement_mask is None:
            reference_measurement_mask = examples.measurement_mask
        elif not np.array_equal(reference_measurement_mask, examples.measurement_mask):
            raise RuntimeError("Controlled measurement masks differ across domains")
        statistics[domain] = item
        prepared[domain] = examples
        domain_checks[domain] = checks

    file_hashes = {
        str(path.relative_to(root)): file_sha256(path) for path in sorted(required)
    }
    input_fingerprint = canonical_sha256(file_hashes)
    audit = {
        "passed": True,
        "scope": "baseline_collection_and_controlled_inputs_only",
        "selection_outcome_files_read": False,
        "model": config["model"],
        "resolved_model_revision": config["resolved_model_revision"],
        "collection_fingerprint": config["collection_fingerprint"],
        "dataset_revisions": config["dataset_revisions"],
        "domains": list(BALANCED_DOMAINS),
        "examples_per_domain": EXPECTED_EXAMPLES,
        "measured_positions_per_example": EXPECTED_MEASURED_POSITIONS,
        "measured_positions_per_domain": (
            EXPECTED_EXAMPLES * EXPECTED_MEASURED_POSITIONS
        ),
        "top_k": EXPECTED_TOP_K,
        "neutral_prefix": EXPECTED_PREFIX,
        "neutral_prefix_token_ids": list(EXPECTED_PREFIX_IDS),
        "same_measurement_mask_across_domains": True,
        "domain_checks": domain_checks,
        "input_files_sha256": file_hashes,
        "selection_input_fingerprint": input_fingerprint,
    }
    return ControlledSource(
        root=root,
        config=config,
        architecture=architecture,
        corpus=corpus,
        manifest=manifest,
        statistics=statistics,
        prepared=prepared,
        file_sha256=file_hashes,
        input_fingerprint=input_fingerprint,
        audit=audit,
    )


def _validate_source_metadata(
    config: Mapping[str, Any],
    architecture: Mapping[str, Any],
    corpus: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    expected_config = {
        "model": EXPECTED_MODEL,
        "resolved_model_revision": EXPECTED_MODEL_REVISION,
        "domains": list(BALANCED_DOMAINS),
        "num_examples": EXPECTED_EXAMPLES,
        "tokens_per_example": EXPECTED_MEASURED_POSITIONS,
        "max_length": 512,
        "batch_size": 1,
        "seed": 42,
        "dtype": "bfloat16",
        "device": "cuda",
        "neutral_prefix": EXPECTED_PREFIX,
        "candidate_pool_size": 1000,
        "include_reference_answers": False,
        "allow_dataset_substitution": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"Controlled configuration mismatch for {key}: "
                f"expected {expected!r}, found {config.get(key)!r}"
            )
    if config.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("Requested model revision does not match the validated revision")
    if config.get("dataset_revisions") != EXPECTED_DATASET_REVISIONS:
        raise RuntimeError("Dataset revisions do not match the pinned controlled run")
    selected_inputs = config.get("selected_inputs", {})
    for domain, revision in EXPECTED_DATASET_REVISIONS.items():
        selected = selected_inputs.get(domain, {})
        if selected.get("resolved_revision") != revision:
            raise RuntimeError(f"Resolved dataset revision mismatch for {domain}")
        if len(selected.get("selected_example_ids", [])) != EXPECTED_EXAMPLES:
            raise RuntimeError(f"Selected example count mismatch for {domain}")
        if not selected.get("input_ids_sha256"):
            raise RuntimeError(f"Missing controlled-input hash for {domain}")

    if architecture.get("num_moe_layers") != EXPECTED_MOE_LAYERS:
        raise RuntimeError("Unexpected number of MoE layers")
    if architecture.get("num_experts") != EXPECTED_EXPERTS:
        raise RuntimeError("Unexpected expert count")
    if architecture.get("top_k") != [EXPECTED_TOP_K]:
        raise RuntimeError("Unexpected router top-k")
    layers = architecture.get("layers", [])
    if len(layers) != EXPECTED_MOE_LAYERS:
        raise RuntimeError("Architecture layer metadata is incomplete")
    if [item.get("model_layer_index") for item in layers] != list(
        range(EXPECTED_MOE_LAYERS)
    ):
        raise RuntimeError("MoE model-layer indices are not the expected 0..15")
    for layer in layers:
        if (
            layer.get("num_experts") != EXPECTED_EXPERTS
            or layer.get("top_k") != EXPECTED_TOP_K
        ):
            raise RuntimeError("Layer-level architecture metadata is inconsistent")

    expected_corpus = {
        "prompt_style": "neutral_fixed_token_control",
        "neutral_prefix": EXPECTED_PREFIX,
        "neutral_prefix_token_ids": list(EXPECTED_PREFIX_IDS),
        "measured_tokens_per_example": EXPECTED_MEASURED_POSITIONS,
        "lookahead_tokens_per_example": 1,
        "examples_per_domain": EXPECTED_EXAMPLES,
        "measured_tokens_per_domain": EXPECTED_EXAMPLES
        * EXPECTED_MEASURED_POSITIONS,
        "model_sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "same_prefix_token_ids": True,
        "same_model_sequence_length": True,
        "same_measurement_length_distribution": True,
        "same_total_measurement_budget": True,
    }
    for key, expected in expected_corpus.items():
        if corpus.get(key) != expected:
            raise RuntimeError(
                f"Controlled-corpus mismatch for {key}: expected {expected!r}, "
                f"found {corpus.get(key)!r}"
            )
    fingerprint = config.get("collection_fingerprint")
    if not fingerprint or manifest.get("collection_fingerprint") != fingerprint:
        raise RuntimeError("Collection fingerprints are missing or inconsistent")
    if manifest.get("completed_domains") != list(BALANCED_DOMAINS):
        raise RuntimeError("Controlled collection does not contain all completed domains")


def _validate_domain_artifacts(
    domain: str,
    statistics: DomainStatistics,
    examples: PreparedDomainExamples,
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_shape = (EXPECTED_EXAMPLES, EXPECTED_MOE_LAYERS, EXPECTED_EXPERTS)
    if statistics.routing_counts.shape != expected_shape:
        raise RuntimeError(f"Unexpected statistic shape for {domain}")
    if statistics.token_counts.shape != (EXPECTED_EXAMPLES,) or not np.all(
        statistics.token_counts == EXPECTED_MEASURED_POSITIONS
    ):
        raise RuntimeError(f"Measured token counts are not exactly 64 for {domain}")
    expected_assignments = np.broadcast_to(
        statistics.token_counts[:, None] * EXPECTED_TOP_K,
        (EXPECTED_EXAMPLES, EXPECTED_MOE_LAYERS),
    )
    observed_assignments = statistics.routing_counts.sum(axis=2, dtype=np.uint64)
    if not np.array_equal(observed_assignments, expected_assignments):
        raise RuntimeError(f"Top-k routing assignment counts are invalid for {domain}")
    if np.any(statistics.routing_counts > statistics.token_counts[:, None, None]):
        raise RuntimeError(f"An expert is routed more than once per measured token in {domain}")
    for name, values in (
        ("routing_counts", statistics.routing_counts),
        ("gate_sums", statistics.gate_sums),
        ("contribution_sums", statistics.contribution_sums),
        ("token_counts", statistics.token_counts),
    ):
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise RuntimeError(f"{name} contains invalid values for {domain}")
    if np.any(statistics.contribution_sums.sum(axis=(0, 2)) <= 0):
        raise RuntimeError(f"A layer has no functional contribution for {domain}")

    if examples.input_ids.shape != (EXPECTED_EXAMPLES, EXPECTED_SEQUENCE_LENGTH):
        raise RuntimeError(f"Controlled input shape mismatch for {domain}")
    if np.any(examples.measurement_mask.sum(axis=1) != EXPECTED_MEASURED_POSITIONS):
        raise RuntimeError(f"Measurement geometry mismatch for {domain}")
    control = metadata.get("control", {})
    if control.get("prefix_token_ids") != list(EXPECTED_PREFIX_IDS):
        raise RuntimeError(f"Neutral prefix metadata mismatch for {domain}")
    prefix_positions = control.get("prefix_model_positions")
    if prefix_positions != list(range(len(EXPECTED_PREFIX_IDS))):
        raise RuntimeError(f"Neutral prefix positions mismatch for {domain}")
    observed_prefix = examples.input_ids[:, prefix_positions]
    expected_prefix = np.broadcast_to(
        np.asarray(EXPECTED_PREFIX_IDS, dtype=observed_prefix.dtype),
        observed_prefix.shape,
    )
    if not np.array_equal(observed_prefix, expected_prefix):
        raise RuntimeError(f"Neutral prefix token IDs mismatch for {domain}")
    input_hash = array_sha256(examples.input_ids)
    measurement_hash = array_sha256(examples.measurement_mask)
    selected = config["selected_inputs"][domain]
    if input_hash != selected["input_ids_sha256"] or input_hash != control.get(
        "input_ids_sha256"
    ):
        raise RuntimeError(f"Controlled input hash mismatch for {domain}")
    if measurement_hash != control.get("measurement_mask_sha256"):
        raise RuntimeError(f"Measurement-mask hash mismatch for {domain}")
    if metadata.get("resolved_revision") != EXPECTED_DATASET_REVISIONS[domain]:
        raise RuntimeError(f"Dataset revision mismatch in metadata for {domain}")
    if metadata.get("substituted") is not False:
        raise RuntimeError(f"Dataset substitution was observed for {domain}")
    if metadata.get("num_examples") != EXPECTED_EXAMPLES:
        raise RuntimeError(f"Metadata example count mismatch for {domain}")
    return {
        "statistics_shape": list(expected_shape),
        "all_arrays_finite_and_nonnegative": True,
        "exact_top_k_assignments_per_example_layer": True,
        "top_k": EXPECTED_TOP_K,
        "total_measured_positions": int(statistics.token_counts.sum()),
        "input_ids_sha256": input_hash,
        "measurement_mask_sha256": measurement_hash,
        "neutral_prefix_exact": True,
    }


def build_preregistration(source: ControlledSource) -> dict[str, Any]:
    candidates = rank_candidate_experts(source.statistics)
    selected, domain_tiers = select_specialized_experts(candidates)
    controls, control_tiers = match_routing_controls(candidates, selected)
    deterministic = {
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "selection_input_fingerprint": source.input_fingerprint,
        "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
        "control_algorithm_version": CONTROL_ALGORITHM_VERSION,
        "selection_algorithm": selection_algorithm_metadata(),
        "domain_selection_tiers": domain_tiers,
        "control_matching_tiers": control_tiers,
        "ranked_candidate_pool": candidates,
        "selected_experts": selected,
        "matched_controls": controls,
        "analysis_preregistration": analysis_preregistration_metadata(),
    }
    fingerprint = canonical_sha256(deterministic)
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_MASKING",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "masking_outcomes_used_for_selection": False,
        "allowed_selection_inputs": [
            "baseline routing counts",
            "baseline selected gate mass",
            "baseline functional-contribution sums",
            "controlled-run metadata and fingerprints",
        ],
        "excluded_selection_inputs": [
            "masked losses",
            "loss changes",
            "intervention contrasts",
            "all post-masking quantities",
        ],
        "source": {
            "artifact_directory": str(source.root),
            "model": source.config["model"],
            "model_revision": source.config["resolved_model_revision"],
            "collection_fingerprint": source.config["collection_fingerprint"],
            "selection_input_fingerprint": source.input_fingerprint,
            "input_files_sha256": source.file_sha256,
            "integrity_audit": source.audit,
        },
        **{key: value for key, value in deterministic.items() if not key.startswith("source_")},
        "preregistration_fingerprint": fingerprint,
    }


def selection_algorithm_metadata() -> dict[str, Any]:
    return {
        "primary_statistic": "within-layer normalized functional contribution",
        "specialization_score": (
            "I(target) - max(I(non-target)); larger is more target-specialized"
        ),
        "strict_thresholds": {
            "target_rank_max": math.ceil(EXPECTED_EXPERTS * 0.10),
            "at_least_one_non_target_rank_min": EXPECTED_EXPERTS // 2 + 1,
            "specialization_margin_min_exclusive": 0.0,
            "target_routing_coverage_min": 0.01,
        },
        "ranking_order": [
            "specialization margin descending",
            "target functional rank ascending",
            "worst non-target rank descending",
            "layer ascending",
            "expert ID ascending",
        ],
        "selection_count_per_domain": 3,
        "layer_diversity": (
            "take the highest-ranked eligible expert from unused layers first; "
            "allow a repeated layer only if fewer than three eligible layers exist"
        ),
        "predefined_relaxation_tiers": [
            {
                "name": "strict",
                "target_rank_max": 7,
                "bottom_half_non_target_required": True,
                "positive_margin_required": True,
                "target_routing_coverage_min": 0.01,
            },
            {
                "name": "top20",
                "target_rank_max": 13,
                "bottom_half_non_target_required": True,
                "positive_margin_required": True,
                "target_routing_coverage_min": 0.01,
            },
            {
                "name": "coverage_relaxed",
                "target_rank_max": 13,
                "bottom_half_non_target_required": True,
                "positive_margin_required": True,
                "target_routing_coverage_min": 0.005,
            },
            {
                "name": "positive_routed_fallback",
                "target_rank_max": 16,
                "bottom_half_non_target_required": False,
                "positive_margin_required": True,
                "target_routing_coverage_min_exclusive": 0.0,
            },
        ],
        "control_matching": {
            "one_control_per_specialist": True,
            "controls_unique": True,
            "specialists_excluded_from_control_pool": True,
            "same_layer_required_in_primary_tiers": True,
            "primary_specialization_cap": (
                "control target-margin <= 0.25 * specialist target-margin"
            ),
            "assignment": "minimum-cost one-to-one Hungarian assignment within layer",
            "cost": (
                "absolute target routing-coverage difference + "
                "0.02 * normalized cross-domain rank range + "
                "0.01 * positive control/specialist margin ratio"
            ),
            "predefined_relaxation_caps": [0.25, 0.50, 1.00],
            "cross_layer_fallback": (
                "only if no complete same-layer assignment exists after predefined caps"
            ),
        },
    }


def analysis_preregistration_metadata() -> dict[str, Any]:
    return {
        "bootstrap_replicates": 1000,
        "bootstrap_seed": 42,
        "primary_contrast": (
            "target delta NLL minus arithmetic mean of the three non-target delta NLLs"
        ),
        "pairwise_contrasts": "target delta NLL minus each named non-target domain",
        "resampling": (
            "examples resampled within domain; paired specialist/control and aggregate "
            "analyses reuse indices within each domain"
        ),
        "decision_rule": {
            "strong_go": (
                "at least three domains have positive mean specialized contrast and "
                "positive mean specialized-minus-control difference; overall mean "
                "specialized-minus-control is positive; at least 75% of paired "
                "differences are positive"
            ),
            "go_with_qualifications": (
                "overall mean specialized contrast and specialized-minus-control "
                "difference are positive, with a positive specialized contrast in at "
                "least one non-Coding domain, but STRONG GO is not met"
            ),
            "weak_no_go": "all other outcomes",
        },
    }


def rank_candidate_experts(
    statistics: Mapping[str, DomainStatistics],
) -> list[dict[str, Any]]:
    if tuple(statistics) != BALANCED_DOMAINS:
        if set(statistics) != set(BALANCED_DOMAINS):
            raise ValueError("Candidate ranking requires exactly the four balanced domains")
    aggregates = {domain: statistics[domain].aggregate() for domain in BALANCED_DOMAINS}
    ranks = {
        domain: np.stack(
            [
                descending_ranks(layer)
                for layer in aggregates[domain]["normalized_contribution"]
            ]
        )
        for domain in BALANCED_DOMAINS
    }
    rows: list[dict[str, Any]] = []
    for target in BALANCED_DOMAINS:
        target_statistics = statistics[target]
        total_tokens = float(target_statistics.token_counts.sum())
        for layer in range(target_statistics.num_layers):
            for expert_id in range(target_statistics.num_experts):
                contributions = {
                    domain: float(
                        aggregates[domain]["normalized_contribution"][layer, expert_id]
                    )
                    for domain in BALANCED_DOMAINS
                }
                domain_ranks = {
                    domain: float(ranks[domain][layer, expert_id])
                    for domain in BALANCED_DOMAINS
                }
                route_frequencies = {
                    domain: float(
                        aggregates[domain]["routing_frequency"][layer, expert_id]
                    )
                    for domain in BALANCED_DOMAINS
                }
                route_coverages = {
                    domain: float(
                        statistics[domain].routing_counts[:, layer, expert_id].sum()
                        / statistics[domain].token_counts.sum()
                    )
                    for domain in BALANCED_DOMAINS
                }
                example_coverages = {
                    domain: float(
                        np.mean(
                            statistics[domain].routing_counts[:, layer, expert_id] > 0
                        )
                    )
                    for domain in BALANCED_DOMAINS
                }
                others = [domain for domain in BALANCED_DOMAINS if domain != target]
                max_non_target_domain = max(
                    others,
                    key=lambda domain: (
                        contributions[domain],
                        -BALANCED_DOMAINS.index(domain),
                    ),
                )
                max_non_target = contributions[max_non_target_domain]
                margin = contributions[target] - max_non_target
                routing_margin = route_frequencies[target] - max(
                    route_frequencies[domain] for domain in others
                )
                coverage_margin = route_coverages[target] - max(
                    route_coverages[domain] for domain in others
                )
                worst_non_target_rank = max(domain_ranks[domain] for domain in others)
                strict_checks = {
                    "target_top_10_percent": domain_ranks[target] <= 7,
                    "non_target_bottom_half": worst_non_target_rank > 32,
                    "positive_specialization_margin": margin > 0,
                    "target_routing_coverage_at_least_1_percent": (
                        route_coverages[target] >= 0.01
                    ),
                }
                row: dict[str, Any] = {
                    "target_domain": target,
                    "layer": layer,
                    "expert_id": expert_id,
                    "specialization_margin": margin,
                    "target_normalized_contribution": contributions[target],
                    "max_non_target_normalized_contribution": max_non_target,
                    "max_non_target_domain": max_non_target_domain,
                    "target_functional_contribution": float(
                        aggregates[target]["functional_contribution"][layer, expert_id]
                    ),
                    "target_gate_mass": float(
                        aggregates[target]["gate_mass"][layer, expert_id]
                    ),
                    "target_rank": domain_ranks[target],
                    "worst_non_target_rank": worst_non_target_rank,
                    "cross_domain_rank_range": max(domain_ranks.values())
                    - min(domain_ranks.values()),
                    "target_routing_frequency": route_frequencies[target],
                    "target_routing_coverage": route_coverages[target],
                    "target_example_coverage": example_coverages[target],
                    "routing_specialization_margin": routing_margin,
                    "routing_coverage_specialization_margin": coverage_margin,
                    "eligible_strict": all(strict_checks.values()),
                    "strict_checks": strict_checks,
                    "normalized_contribution_by_domain": contributions,
                    "functional_rank_by_domain": domain_ranks,
                    "routing_frequency_by_domain": route_frequencies,
                    "routing_coverage_by_domain": route_coverages,
                    "routing_example_coverage_by_domain": example_coverages,
                    "target_measured_positions": int(total_tokens),
                }
                rows.append(row)

    for target in BALANCED_DOMAINS:
        domain_rows = [row for row in rows if row["target_domain"] == target]
        domain_rows.sort(key=_candidate_sort_key)
        eligible_rank = 0
        for candidate_rank, row in enumerate(domain_rows, 1):
            row["candidate_rank"] = candidate_rank
            if row["eligible_strict"]:
                eligible_rank += 1
                row["strict_eligible_rank"] = eligible_rank
            else:
                row["strict_eligible_rank"] = None
    rows.sort(
        key=lambda row: (BALANCED_DOMAINS.index(row["target_domain"]), row["candidate_rank"])
    )
    return rows


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["specialization_margin"]),
        float(row["target_rank"]),
        -float(row["worst_non_target_rank"]),
        int(row["layer"]),
        int(row["expert_id"]),
    )


def select_specialized_experts(
    candidates: Sequence[dict[str, Any]], count_per_domain: int = 3
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected: list[dict[str, Any]] = []
    tiers_used: dict[str, str] = {}
    tier_definitions = [
        (
            "strict",
            lambda row: bool(row["eligible_strict"]),
        ),
        (
            "top20",
            lambda row: (
                row["target_rank"] <= 13
                and row["worst_non_target_rank"] > 32
                and row["specialization_margin"] > 0
                and row["target_routing_coverage"] >= 0.01
            ),
        ),
        (
            "coverage_relaxed",
            lambda row: (
                row["target_rank"] <= 13
                and row["worst_non_target_rank"] > 32
                and row["specialization_margin"] > 0
                and row["target_routing_coverage"] >= 0.005
            ),
        ),
        (
            "positive_routed_fallback",
            lambda row: (
                row["target_rank"] <= 16
                and row["specialization_margin"] > 0
                and row["target_routing_coverage"] > 0
            ),
        ),
    ]
    for target in BALANCED_DOMAINS:
        domain_rows = sorted(
            [row for row in candidates if row["target_domain"] == target],
            key=_candidate_sort_key,
        )
        chosen_pool: list[dict[str, Any]] | None = None
        chosen_tier = ""
        for tier, predicate in tier_definitions:
            pool = [row for row in domain_rows if predicate(row)]
            if len(pool) >= count_per_domain:
                chosen_pool = pool
                chosen_tier = tier
                break
        if chosen_pool is None:
            raise RuntimeError(f"Unable to select {count_per_domain} specialists for {target}")
        chosen: list[dict[str, Any]] = []
        used_layers: set[int] = set()
        for row in chosen_pool:
            if row["layer"] not in used_layers:
                chosen.append(row)
                used_layers.add(int(row["layer"]))
            if len(chosen) == count_per_domain:
                break
        if len(chosen) < count_per_domain:
            selected_ids = {(row["layer"], row["expert_id"]) for row in chosen}
            for row in chosen_pool:
                identity = (row["layer"], row["expert_id"])
                if identity not in selected_ids:
                    chosen.append(row)
                    selected_ids.add(identity)
                if len(chosen) == count_per_domain:
                    break
        for selection_order, row in enumerate(chosen, 1):
            selected.append(
                {
                    **row,
                    "role": "specialized",
                    "selection_order_within_domain": selection_order,
                    "selection_tier": chosen_tier,
                    "selection_rationale": (
                        f"{chosen_tier} eligibility; highest remaining functional "
                        "specialization margin subject to layer diversity"
                    ),
                }
            )
        tiers_used[target] = chosen_tier
    return selected, tiers_used


def match_routing_controls(
    candidates: Sequence[dict[str, Any]],
    selected: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = {
        (row["target_domain"], row["layer"], row["expert_id"]): row
        for row in candidates
    }
    specialist_ids = {(row["layer"], row["expert_id"]) for row in selected}
    matched: list[dict[str, Any]] = []
    tiers: dict[str, Any] = {}
    for layer in sorted({int(row["layer"]) for row in selected}):
        layer_specialists = [row for row in selected if row["layer"] == layer]
        controls = [
            expert_id
            for expert_id in range(EXPECTED_EXPERTS)
            if (layer, expert_id) not in specialist_ids
        ]
        assignment: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
        for cap in (0.25, 0.50, 1.00):
            costs = _control_cost_matrix(
                layer_specialists, controls, lookup, specialization_cap=cap
            )
            row_indices, column_indices = linear_sum_assignment(costs)
            if len(row_indices) == len(layer_specialists) and np.all(
                costs[row_indices, column_indices] < 1e5
            ):
                assignment = (row_indices, column_indices, costs, cap)
                break
        if assignment is None:
            raise RuntimeError(
                f"No complete unique same-layer control assignment exists for layer {layer}"
            )
        row_indices, column_indices, costs, cap = assignment
        tiers[str(layer)] = {
            "same_layer": True,
            "specialization_cap_ratio": cap,
            "fallback_used": cap != 0.25,
        }
        for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
            specialist = layer_specialists[row_index]
            control_expert = controls[column_index]
            target = specialist["target_domain"]
            control = lookup[(target, layer, control_expert)]
            route_distance = abs(
                specialist["target_routing_coverage"]
                - control["target_routing_coverage"]
            )
            positive_ratio = max(float(control["specialization_margin"]), 0.0) / float(
                specialist["specialization_margin"]
            )
            pair_id = (
                f"{target}_L{specialist['layer']}_E{specialist['expert_id']}"
            )
            matched.append(
                {
                    "pair_id": pair_id,
                    "target_domain": target,
                    "specialized": specialist,
                    "control": {
                        **control,
                        "role": "control",
                        "paired_specialist_layer": specialist["layer"],
                        "paired_specialist_expert_id": specialist["expert_id"],
                    },
                    "same_layer": True,
                    "control_reused": False,
                    "matching_tier_specialization_cap_ratio": cap,
                    "fallback_used": cap != 0.25,
                    "target_routing_coverage_absolute_difference": route_distance,
                    "target_routing_frequency_absolute_difference": abs(
                        specialist["target_routing_frequency"]
                        - control["target_routing_frequency"]
                    ),
                    "control_to_specialist_positive_margin_ratio": positive_ratio,
                    "matching_cost": float(costs[row_index, column_index]),
                    "matching_rationale": (
                        "unique same-layer minimum-cost assignment using target routing "
                        "coverage, rank balance, and residual specialization"
                    ),
                }
            )
    selected_order = {
        (
            row["target_domain"],
            row["layer"],
            row["expert_id"],
        ): index
        for index, row in enumerate(selected)
    }
    matched.sort(
        key=lambda pair: selected_order[
            (
                pair["target_domain"],
                pair["specialized"]["layer"],
                pair["specialized"]["expert_id"],
            )
        ]
    )
    control_ids = {
        (pair["control"]["layer"], pair["control"]["expert_id"])
        for pair in matched
    }
    if len(control_ids) != len(matched):
        raise RuntimeError("Control assignment unexpectedly reused an expert")
    return matched, tiers


def _control_cost_matrix(
    specialists: Sequence[dict[str, Any]],
    controls: Sequence[int],
    lookup: Mapping[tuple[str, int, int], dict[str, Any]],
    specialization_cap: float,
) -> np.ndarray:
    costs = np.full((len(specialists), len(controls)), 1e6, dtype=np.float64)
    for row_index, specialist in enumerate(specialists):
        target = specialist["target_domain"]
        layer = int(specialist["layer"])
        specialist_margin = float(specialist["specialization_margin"])
        for column_index, expert_id in enumerate(controls):
            candidate = lookup[(target, layer, expert_id)]
            control_margin = float(candidate["specialization_margin"])
            if candidate["target_routing_coverage"] <= 0:
                continue
            if control_margin > specialization_cap * specialist_margin:
                continue
            route_distance = abs(
                specialist["target_routing_coverage"]
                - candidate["target_routing_coverage"]
            )
            rank_penalty = 0.02 * float(candidate["cross_domain_rank_range"]) / max(
                EXPECTED_EXPERTS - 1, 1
            )
            specialization_penalty = 0.01 * max(control_margin, 0.0) / max(
                specialist_margin, np.finfo(float).eps
            )
            costs[row_index, column_index] = (
                route_distance
                + rank_penalty
                + specialization_penalty
                + expert_id * 1e-12
            )
    return costs


def write_preregistration_artifacts(payload: dict[str, Any], output_dir: Path) -> None:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "selected_experts_preregistered.json"
    if frozen_path.exists():
        existing = read_json(frozen_path)
        if existing.get("preregistration_fingerprint") != payload.get(
            "preregistration_fingerprint"
        ):
            raise RuntimeError(
                "A different frozen preregistration already exists; use a new output directory"
            )
        payload = existing
    else:
        forbidden = [
            output / "masking_results.csv",
            output / "per_example_loss_changes.npz",
            output / "masking",
        ]
        if any(path.exists() for path in forbidden):
            raise RuntimeError("Masking artifacts exist before preregistration was frozen")
        atomic_write_json(frozen_path, payload)

    candidate_rows = [flatten_candidate(row) for row in payload["ranked_candidate_pool"]]
    write_csv(output / "candidate_experts.csv", candidate_rows, candidate_csv_fields())
    control_rows = [flatten_control_pair(pair) for pair in payload["matched_controls"]]
    write_csv(output / "matched_controls.csv", control_rows, matched_control_csv_fields())


def flatten_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = {
        key: value
        for key, value in row.items()
        if not isinstance(value, (dict, list, tuple))
    }
    for domain in BALANCED_DOMAINS:
        flattened[f"{domain}_normalized_contribution"] = row[
            "normalized_contribution_by_domain"
        ][domain]
        flattened[f"{domain}_functional_rank"] = row["functional_rank_by_domain"][
            domain
        ]
        flattened[f"{domain}_routing_frequency"] = row[
            "routing_frequency_by_domain"
        ][domain]
        flattened[f"{domain}_routing_coverage"] = row[
            "routing_coverage_by_domain"
        ][domain]
        flattened[f"{domain}_example_coverage"] = row[
            "routing_example_coverage_by_domain"
        ][domain]
    flattened["strict_checks"] = json.dumps(row["strict_checks"], sort_keys=True)
    return flattened


def candidate_csv_fields() -> list[str]:
    fields = [
        "target_domain",
        "candidate_rank",
        "strict_eligible_rank",
        "eligible_strict",
        "layer",
        "expert_id",
        "target_rank",
        "worst_non_target_rank",
        "specialization_margin",
        "target_normalized_contribution",
        "max_non_target_normalized_contribution",
        "max_non_target_domain",
        "target_functional_contribution",
        "target_gate_mass",
        "target_routing_frequency",
        "target_routing_coverage",
        "target_example_coverage",
        "routing_specialization_margin",
        "routing_coverage_specialization_margin",
        "cross_domain_rank_range",
        "target_measured_positions",
        "strict_checks",
    ]
    for domain in BALANCED_DOMAINS:
        fields.extend(
            [
                f"{domain}_normalized_contribution",
                f"{domain}_functional_rank",
                f"{domain}_routing_frequency",
                f"{domain}_routing_coverage",
                f"{domain}_example_coverage",
            ]
        )
    return fields


def flatten_control_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
    specialist = pair["specialized"]
    control = pair["control"]
    return {
        "pair_id": pair["pair_id"],
        "target_domain": pair["target_domain"],
        "specialized_layer": specialist["layer"],
        "specialized_expert_id": specialist["expert_id"],
        "specialized_target_rank": specialist["target_rank"],
        "specialized_specialization_margin": specialist["specialization_margin"],
        "specialized_target_normalized_contribution": specialist[
            "target_normalized_contribution"
        ],
        "specialized_target_routing_frequency": specialist[
            "target_routing_frequency"
        ],
        "specialized_target_routing_coverage": specialist[
            "target_routing_coverage"
        ],
        "specialized_target_example_coverage": specialist[
            "target_example_coverage"
        ],
        "control_layer": control["layer"],
        "control_expert_id": control["expert_id"],
        "control_target_rank": control["target_rank"],
        "control_specialization_margin": control["specialization_margin"],
        "control_target_normalized_contribution": control[
            "target_normalized_contribution"
        ],
        "control_target_routing_frequency": control["target_routing_frequency"],
        "control_target_routing_coverage": control["target_routing_coverage"],
        "control_target_example_coverage": control["target_example_coverage"],
        "same_layer": pair["same_layer"],
        "control_reused": pair["control_reused"],
        "target_routing_frequency_absolute_difference": pair[
            "target_routing_frequency_absolute_difference"
        ],
        "target_routing_coverage_absolute_difference": pair[
            "target_routing_coverage_absolute_difference"
        ],
        "control_to_specialist_positive_margin_ratio": pair[
            "control_to_specialist_positive_margin_ratio"
        ],
        "matching_tier_specialization_cap_ratio": pair[
            "matching_tier_specialization_cap_ratio"
        ],
        "matching_cost": pair["matching_cost"],
        "fallback_used": pair["fallback_used"],
        "matching_rationale": pair["matching_rationale"],
    }


def matched_control_csv_fields() -> list[str]:
    return [
        "pair_id",
        "target_domain",
        "specialized_layer",
        "specialized_expert_id",
        "specialized_target_rank",
        "specialized_specialization_margin",
        "specialized_target_normalized_contribution",
        "specialized_target_routing_frequency",
        "specialized_target_routing_coverage",
        "specialized_target_example_coverage",
        "control_layer",
        "control_expert_id",
        "control_target_rank",
        "control_specialization_margin",
        "control_target_normalized_contribution",
        "control_target_routing_frequency",
        "control_target_routing_coverage",
        "control_target_example_coverage",
        "same_layer",
        "control_reused",
        "target_routing_frequency_absolute_difference",
        "target_routing_coverage_absolute_difference",
        "control_to_specialist_positive_margin_ratio",
        "matching_tier_specialization_cap_ratio",
        "matching_cost",
        "fallback_used",
        "matching_rationale",
    ]
