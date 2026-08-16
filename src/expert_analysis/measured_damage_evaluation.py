"""Stage 3 evaluation support: seed-46 split, preregistration, run records.

The seed-46 development split is entirely new: it is disjoint from the frozen
calibration data, the original controlled 100/domain set, the contaminated
seed-43 Stage 2B development split, the Stage 2C seed-45 development split
(contaminated for Stage 3 because Stage 3 was conceived after its results were
observed), and the untouched seed-44 final split. Model evaluation itself
reuses the audited Stage 2B mixed-precision manager and loss checkpointing
unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .balanced import (
    EXPECTED_PREFIX,
    EXPECTED_PREFIX_IDS,
    ControlledSource,
    array_sha256,
    canonical_sha256,
    file_sha256,
)
from .controlled import PreparedDomainExamples
from .fragility_evaluation import STAGE2C_SPLIT_SCHEMA
from .fragility_statistics import (
    GATE_D_REQUIRED_POSITIVE_DOMAINS,
    GATE_E_DENOMINATOR_EPSILON,
    GATE_E_RELATIVE_TOLERANCE,
    STAGE2C_BOOTSTRAP_REPLICATES,
    STAGE2C_BOOTSTRAP_SEED,
    STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
)
from .heldout_splits import (
    DEVELOPMENT_EXAMPLES_PER_DOMAIN,
    FULL_SPLIT_REQUEST,
    MEASURED_TOKENS_PER_EXAMPLE,
    REQUIRED_CONTENT_TOKENS,
    content_row_key,
    prepare_heldout_split,
    reconstruct_prior_usage,
    select_heldout_examples,
    split_row_keys,
)
from .io_utils import atomic_write_json, read_json
from .measured_damage import (
    ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN,
    ADDITIVITY_MIN_WORST_DELTA_SPEARMAN,
    STAGE3_PROFILE_BITS,
    STAGE3_REGIMES,
    STAGE3_STAGE,
)
from .measured_damage_optimization import (
    MEASURED_DAMAGE_METHOD,
    STAGE3_DEVELOPMENT_SEED,
    STAGE3_FINAL_SEED,
    load_stage3_allocation,
    stage3_registry_entries,
)
from .modeling import ModelBundle
from .protection_optimization import PROTECTION_FRACTIONS
from .specialist_preservation import STAGE2B_DOMAINS

STAGE3_SPLIT_SCHEMA = "stage3_development_split_v1"
STAGE3_SPLIT_NAME = "development"
STAGE3_EXAMPLES_PER_DOMAIN = DEVELOPMENT_EXAMPLES_PER_DOMAIN
PREREGISTRATION_FILE = "stage3_preregistration.json"
PREREGISTRATION_SHA_FILE = "stage3_preregistration_sha256.txt"

STAGE3_PHASE_DIRS = {
    "additivity": "additivity",
    "development": "development_seed46",
    "final": "final_seed44",
}


def stage3_phase_dir_name(phase: str) -> str:
    if phase not in STAGE3_PHASE_DIRS:
        raise ValueError(f"Unknown Stage 3 phase {phase!r}")
    return STAGE3_PHASE_DIRS[phase]


def _frozen_split_rows_and_texts(
    splits_dir: Path,
    domain: str,
    split_name: str,
    manifest_entry: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    """Content-row keys and text hashes of one frozen prior split domain."""

    path = splits_dir / split_name / f"{domain}.npz"
    with np.load(path, allow_pickle=False) as data:
        input_ids = np.asarray(data["input_ids"])
    if array_sha256(input_ids) != manifest_entry["input_ids_sha256"]:
        raise RuntimeError(
            f"Frozen {split_name} split hash changed for {domain} in {splits_dir}"
        )
    rows = {
        content_row_key(input_ids[i, len(EXPECTED_PREFIX_IDS):])
        for i in range(input_ids.shape[0])
    }
    return rows, set(manifest_entry["example_text_sha256"])


def build_stage3_development_split(
    source: ControlledSource,
    tokenizer: Any,
    stage2b_results_dir: Path,
    stage2c_results_dir: Path,
    output_dir: Path,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Construct the new seed-46 development split with strict disjointness.

    Exclusions per domain: the seed-42 prior usage rule frozen by Stage 2B
    (entire inspected candidate pool where practical, otherwise the 100
    previously evaluated examples), every seed-43 development text, every
    seed-44 final text, and every seed-45 Stage 2C development text. Any
    overlap aborts the build.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    stage2b_manifest = read_json(
        stage2b_results_dir / "splits" / "split_manifest.json"
    )
    if (
        stage2b_manifest["development_seed"] != 43
        or stage2b_manifest["final_seed"] != STAGE3_FINAL_SEED
    ):
        raise RuntimeError("Stage 2B split manifest does not match the frozen seeds")
    stage2c_manifest = read_json(
        stage2c_results_dir / "splits" / "split_manifest.json"
    )
    if (
        stage2c_manifest.get("schema") != STAGE2C_SPLIT_SCHEMA
        or stage2c_manifest["development_seed"] != 45
    ):
        raise RuntimeError("Stage 2C split manifest does not match the frozen seed 45")

    reference_row = source.prepared[STAGE2B_DOMAINS[0]].measurement_mask[0]
    manifest_domains: dict[str, Any] = {}
    for domain in STAGE2B_DOMAINS:
        prior = reconstruct_prior_usage(domain, source, tokenizer, cache_dir)
        prior_rows: dict[str, set[str]] = {}
        prior_texts: dict[str, set[str]] = {}
        for split_name, splits_dir, manifest_entry in (
            (
                "development",
                stage2b_results_dir / "splits",
                stage2b_manifest["domains"][domain]["development"],
            ),
            (
                "final",
                stage2b_results_dir / "splits",
                stage2b_manifest["domains"][domain]["final"],
            ),
            (
                "development",
                stage2c_results_dir / "splits",
                stage2c_manifest["domains"][domain],
            ),
        ):
            key = f"{splits_dir}:{split_name}"
            rows, texts = _frozen_split_rows_and_texts(
                splits_dir, domain, split_name, manifest_entry
            )
            prior_rows[key] = rows
            prior_texts[key] = texts
        excluded_texts_beyond_prior: set[str] = set()
        for texts in prior_texts.values():
            excluded_texts_beyond_prior |= texts

        # Full-pool exclusion decision from counts alone, before selection.
        from .datasets import load_domain_examples
        from .heldout_splits import (
            EXPECTED_DATASET_REVISIONS,
            _encode_texts,
            text_sha256,
        )

        probe_pool = load_domain_examples(
            domain=domain,
            num_examples=FULL_SPLIT_REQUEST,
            seed=STAGE3_DEVELOPMENT_SEED,
            cache_dir=cache_dir,
            revision=EXPECTED_DATASET_REVISIONS[domain],
            include_answers=False,
            allow_substitution=False,
            format_style="neutral_content",
        )
        probe_encoded = _encode_texts(tokenizer, probe_pool.texts)
        eligible_hashes = {
            text_sha256(probe_pool.texts[index])
            for index, token_ids in enumerate(probe_encoded)
            if len(token_ids) >= REQUIRED_CONTENT_TOKENS
        }
        available_excluding_pool = len(
            eligible_hashes - prior.pool_text_hashes - excluded_texts_beyond_prior
        )
        exclude_full_pool = available_excluding_pool >= STAGE3_EXAMPLES_PER_DOMAIN
        exclusion = (
            set(prior.pool_text_hashes)
            if exclude_full_pool
            else set(prior.selected_text_hashes)
        ) | excluded_texts_beyond_prior

        selection = select_heldout_examples(
            domain,
            STAGE3_SPLIT_NAME,
            STAGE3_DEVELOPMENT_SEED,
            STAGE3_EXAMPLES_PER_DOMAIN,
            exclusion,
            tokenizer,
            cache_dir,
        )
        prepared = prepare_heldout_split(selection, tokenizer, reference_row)
        row_keys = split_row_keys(prepared)
        if len(set(row_keys)) != len(row_keys):
            raise RuntimeError(
                f"The seed-46 split for {domain} contains duplicate content rows"
            )
        all_prior_rows: set[str] = set(prior.selected_row_keys)
        for rows in prior_rows.values():
            all_prior_rows |= rows
        all_prior_texts = set(prior.selected_text_hashes) | excluded_texts_beyond_prior
        overlaps = {
            "seed46_rows_vs_all_prior_rows": sorted(set(row_keys) & all_prior_rows),
            "seed46_texts_vs_all_prior_texts": sorted(
                set(selection.text_hashes) & all_prior_texts
            ),
        }
        if any(overlaps.values()):
            raise RuntimeError(
                f"Seed-46 split overlap detected for {domain}: "
                + ", ".join(name for name, values in overlaps.items() if values)
            )

        split_dir = output_dir / STAGE3_SPLIT_NAME
        prepared.save(split_dir / f"{domain}.npz")
        atomic_write_json(split_dir / f"{domain}.metadata.json", prepared.metadata)
        manifest_domains[domain] = {
            "dataset_revision": EXPECTED_DATASET_REVISIONS[domain],
            "prior_pool_size": prior.pool_size,
            "prior_selection_reconstruction_verified": prior.reconstruction_verified,
            "prior_pool_fully_excluded": exclude_full_pool,
            "prior_pool_exclusion_note": (
                "entire inspected seed-42 candidate pool excluded"
                if exclude_full_pool
                else "full-pool exclusion impractical (dataset too small); the 100 "
                "previously evaluated examples plus all seed-43/seed-44/seed-45 "
                "texts were excluded"
            ),
            "eligible_after_full_exclusion": available_excluding_pool,
            "num_examples": prepared.num_examples,
            "sequence_length": prepared.sequence_length,
            "input_ids_sha256": array_sha256(prepared.input_ids),
            "measurement_mask_sha256": array_sha256(prepared.measurement_mask),
            "dataset_example_ids": list(selection.example_ids),
            "example_text_sha256": list(selection.text_hashes),
            "exclusion_set_size": len(exclusion),
            "seed45_texts_excluded": len(
                stage2c_manifest["domains"][domain]["example_text_sha256"]
            ),
            "overlap_checks": {name: len(values) for name, values in overlaps.items()},
            "disjointness_verified": True,
        }

    manifest = {
        "schema": STAGE3_SPLIT_SCHEMA,
        "stage": STAGE3_STAGE,
        "development_seed": STAGE3_DEVELOPMENT_SEED,
        "final_seed": STAGE3_FINAL_SEED,
        "final_split_source": str(stage2b_results_dir / "splits" / "final"),
        "final_split_note": (
            "the untouched Stage 2B seed-44 final split is reused for final "
            "confirmation and must not be evaluated before FINAL_CONFIRMATION_GO"
        ),
        "examples_per_domain": STAGE3_EXAMPLES_PER_DOMAIN,
        "measured_tokens_per_example": MEASURED_TOKENS_PER_EXAMPLE,
        "lookahead_tokens_per_example": 1,
        "neutral_prefix": EXPECTED_PREFIX,
        "neutral_prefix_token_ids": list(EXPECTED_PREFIX_IDS),
        "model_sequence_length": len(EXPECTED_PREFIX_IDS) + REQUIRED_CONTENT_TOKENS,
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "domains": manifest_domains,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def load_stage3_development_split(
    splits_dir: Path, domain: str
) -> PreparedDomainExamples:
    """Load one seed-46 split domain and verify it against the manifest."""

    manifest = read_json(splits_dir / "split_manifest.json")
    if manifest.get("schema") != STAGE3_SPLIT_SCHEMA:
        raise RuntimeError("Unexpected Stage 3 split schema")
    if manifest["development_seed"] != STAGE3_DEVELOPMENT_SEED:
        raise RuntimeError("The Stage 3 development split seed is not 46")
    entry = manifest["domains"][domain]
    metadata = read_json(splits_dir / STAGE3_SPLIT_NAME / f"{domain}.metadata.json")
    prepared = PreparedDomainExamples.load(
        splits_dir / STAGE3_SPLIT_NAME / f"{domain}.npz", domain, metadata
    )
    if array_sha256(prepared.input_ids) != entry["input_ids_sha256"]:
        raise RuntimeError(f"Seed-46 {domain} input hash mismatch against the manifest")
    if array_sha256(prepared.measurement_mask) != entry["measurement_mask_sha256"]:
        raise RuntimeError(f"Seed-46 {domain} measurement-mask hash mismatch")
    if prepared.num_examples != entry["num_examples"]:
        raise RuntimeError(f"Seed-46 {domain} example count mismatch")
    if not entry.get("disjointness_verified"):
        raise RuntimeError(f"Seed-46 {domain} disjointness was never verified")
    return prepared


def build_stage3_preregistration(
    registry: Mapping[str, Any],
    damage_record: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """The frozen Stage 3 preregistration payload.

    Built after the damage matrix and allocations are frozen but before any
    probe (additivity) NLL or seed-46 NLL is computed.
    """

    return {
        "schema": "stage3_preregistration_v1",
        "stage": STAGE3_STAGE,
        "research_hypothesis": (
            "Per-expert quantization damage that is directly measured on frozen "
            "calibration data composes additively well enough to rank "
            "allocations, and minimizing the worst additively predicted domain "
            "delta NLL under a fixed memory budget reduces worst-domain "
            "degradation better than score-based specialist preservation, "
            "fragility-weighted preservation, and simpler importance-based "
            "protection."
        ),
        "primary_method": MEASURED_DAMAGE_METHOD,
        "objective_formula": (
            "minimize z subject to PredictedDelta_d(x) <= z for every domain "
            "and sum_{l,e} DeltaM[l,e,b] * x[l,e] <= Budget_p(b), x binary, "
            "with PredictedDelta_d(x) = sum_{l,e} m[l,e,d,bits(l,e)]; no "
            "clipping, weighting, fitted coefficient, or added term"
        ),
        "damage_definition": damage_record["damage_definition"],
        "measured_not_estimated": (
            "every m[l,e,d,b] is a measured single-expert QDQ calibration loss "
            "difference; the frozen Stage 2A SURROGATE_NO_GO decision blocks "
            "only predictive surrogates and is preserved unchanged, and the "
            "Stage 2C rule against searching alternative fragility weightings "
            "is respected because no score-based weighting is used at all"
        ),
        "no_new_metric_search": (
            "no clipping, normalization, reweighting, fitted coefficient, "
            "surrogate, seed-43/45 outcome, or post-hoc objective change is "
            "permitted; if any Stage 3 gate fails, the negative result is "
            "preserved and the stage stops"
        ),
        "damage_sha256": damage_record["damage_sha256"],
        "profile_bits": list(STAGE3_PROFILE_BITS),
        "calibration": {
            "examples_per_domain": damage_record["calibration_examples_per_domain"],
            "calibration_subset_hashes": damage_record["calibration_subset_hashes"],
        },
        "precision_regimes": dict(STAGE3_REGIMES),
        "protected_bits": 8,
        "protection_budgets": list(PROTECTION_FRACTIONS),
        "quantization": (
            "Stage-1 symmetric group-wise expert-only QDQ, group size 128, "
            "FP16 scales; routers, attention, embeddings, normalization, "
            "lm_head, and all non-expert parameters stay BF16"
        ),
        "additivity_gates": {
            "probes": (
                "all frozen 20%-budget allocations of both regimes: the eight "
                "Stage 2B deterministic methods, the five Stage 2B random "
                "allocations, the Stage 2C Fragility-Robust allocation, and "
                "the new Measured-Damage-Robust allocation, evaluated on the "
                "frozen calibration subsets"
            ),
            "gate_add_1": (
                "for every domain, Spearman across probes between additively "
                "predicted and measured delta NLL >= "
                f"{ADDITIVITY_MIN_PER_DOMAIN_SPEARMAN}"
            ),
            "gate_add_2": (
                "Spearman across probes between predicted and measured "
                "worst-domain delta NLL >= "
                f"{ADDITIVITY_MIN_WORST_DELTA_SPEARMAN}"
            ),
            "rule": (
                "only regimes passing both gates are authorized for seed-46 "
                "development evaluation; if no regime passes, the stage "
                "decision is MEASURED_DAMAGE_NO_GO with seed 46 and seed 44 "
                "unevaluated; thresholds were fixed in code before any probe "
                "NLL existed and are frozen here"
            ),
            "uniform_diagnostic": (
                "predicted vs measured uniform 3/4/8-bit deltas are reported "
                "as descriptive additivity evidence and never gate anything"
            ),
        },
        "comparators": [
            "robust_functional (frozen Stage 2B)",
            "robust_routing (frozen Stage 2B)",
            "average_specialization (frozen Stage 2B)",
            "global_importance (frozen Stage 2B)",
            "general_only (frozen Stage 2B)",
            "math_only (frozen Stage 2B)",
            "coding_only (frozen Stage 2B)",
            "reasoning_only (frozen Stage 2B)",
            "random_seed1001..random_seed1005 (frozen Stage 2B)",
            "fragility_robust (frozen Stage 2C)",
        ],
        "development_seed": STAGE3_DEVELOPMENT_SEED,
        "development_examples_per_domain": split_manifest["examples_per_domain"],
        "development_budget_fraction": STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
        "final_seed": STAGE3_FINAL_SEED,
        "development_split_input_hashes": {
            domain: split_manifest["domains"][domain]["input_ids_sha256"]
            for domain in STAGE2B_DOMAINS
        },
        "development_gates": {
            "gate_a": (
                "lower point-estimate WorstRelativeDelta than BOTH frozen "
                "Robust-Functional and frozen Fragility-Robust"
            ),
            "gate_b": (
                "lower point-estimate WorstRelativeDelta than the mean of five "
                "frozen random allocations"
            ),
            "gate_c": (
                "lower point-estimate WorstRelativeDelta than BOTH "
                "Global-Importance and Average-Specialization"
            ),
            "gate_d": (
                f"positive Recovery vs the uniform base model in at least "
                f"{GATE_D_REQUIRED_POSITIVE_DOMAINS} of 4 domains"
            ),
            "gate_e": (
                f"MeanRelativeDelta at most {GATE_E_RELATIVE_TOLERANCE:.0%} "
                "relatively worse than the lower of Global-Importance and "
                "Average-Specialization; epsilon "
                f"{GATE_E_DENOMINATOR_EPSILON} for denominator safety only"
            ),
            "ci_significance_required_at_development": False,
        },
        "development_decision_rule": (
            "FINAL_CONFIRMATION_GO if at least one additivity-authorized "
            "regime passes all five gates on the new seed-46 split, else "
            "MEASURED_DAMAGE_NO_GO; only passing regimes are authorized for "
            "seed-44 final confirmation and a failing regime is never "
            "final-tested"
        ),
        "final_success_criteria": {
            "requirement_1": (
                "lower point WorstRelativeDelta than Robust-Functional, "
                "Fragility-Robust, Global-Importance, Average-Specialization, "
                "and the random mean at >= 3 of 4 budgets"
            ),
            "requirement_2": (
                "positive average worst-domain improvement over "
                "Average-Specialization across all four budgets"
            ),
            "requirement_3": (
                ">= 2 of 4 worst-domain comparisons vs Average-Specialization "
                "with 95% bootstrap CIs entirely favoring Measured-Damage-Robust"
            ),
            "requirement_4": (
                ">= 2 of 4 worst-domain comparisons vs Global-Importance with "
                "point estimates favoring Measured-Damage-Robust"
            ),
            "requirement_5": (
                "no domain with negative recovery vs the uniform base at >= 3 "
                "of 4 budgets"
            ),
            "strong_success": "requirements 1-5 in at least one authorized regime",
            "qualified_success": (
                "not strong; positive average improvement over "
                "Average-Specialization; no systematically negative domain; and "
                "either point wins over both Global-Importance and "
                "Average-Specialization at >= 2 of 4 budgets, or point wins "
                "over both Robust-Functional and Fragility-Robust plus the "
                "random mean at >= 3 of 4 budgets"
            ),
            "otherwise": "NEGATIVE RESULT; goalposts are never moved",
        },
        "bootstrap": {
            "replicates": STAGE2C_BOOTSTRAP_REPLICATES,
            "seed": STAGE2C_BOOTSTRAP_SEED,
            "paired": True,
            "worst_domain_recomputed_per_replicate": True,
        },
        "allocation_registry_sha256": registry["registry_sha256"],
        "measured_damage_allocation_hashes": {
            f"{entry['regime']}_budget{int(round(entry['budget_fraction'] * 100))}": (
                entry["allocation_sha256"]
            )
            for entry in registry["new_entries"]
        },
        "stage2b_registry_sha256": registry["stage2b_registry_sha256"],
        "stage2c_registry_sha256": registry["stage2c_registry_sha256"],
        "prior_stage_motivation_note": (
            "Stage 3 was conceived after observing the Stage 2C development "
            "NO_GO; seeds 43 and 45 are contaminated and are never used, and "
            "no Stage 2B/2C development number enters any Stage 3 measurement, "
            "objective, or allocation"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_stage3_preregistration(
    results_dir: Path, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    """Freeze the preregistration file and record its SHA-256 separately."""

    path = results_dir / PREREGISTRATION_FILE
    sha_path = results_dir / PREREGISTRATION_SHA_FILE
    if path.is_file():
        existing = read_json(path)
        volatile = ("created_at_utc",)
        if {k: v for k, v in existing.items() if k not in volatile} != {
            k: v for k, v in dict(payload).items() if k not in volatile
        }:
            raise RuntimeError(
                "stage3_preregistration.json is already frozen with different "
                "content; refusing to overwrite the preregistration"
            )
        digest = file_sha256(path)
        if sha_path.is_file() and sha_path.read_text().strip() != digest:
            raise RuntimeError("stage3_preregistration_sha256.txt does not match")
        return path, digest
    atomic_write_json(path, payload)
    digest = file_sha256(path)
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return path, digest


def verify_stage3_preregistration_unchanged(results_dir: Path) -> str:
    """Refuse to proceed if the frozen preregistration file changed."""

    path = results_dir / PREREGISTRATION_FILE
    sha_path = results_dir / PREREGISTRATION_SHA_FILE
    if not path.is_file() or not sha_path.is_file():
        raise RuntimeError(
            "The Stage 3 preregistration is missing; freeze it before evaluation"
        )
    recorded = sha_path.read_text().strip()
    observed = file_sha256(path)
    if observed != recorded:
        raise RuntimeError(
            "stage3_preregistration.json changed after freezing "
            f"(recorded {recorded[:16]}..., observed {observed[:16]}...); "
            "refusing to proceed"
        )
    return observed


def stage3_run_fingerprint(
    bundle: ModelBundle,
    registry: Mapping[str, Any],
    preregistration_sha256: str,
    split_hashes: Mapping[str, str],
    phase: str,
    batch_size: int,
    determinism: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "stage": "stage3_measured_damage_evaluation",
            "phase": phase,
            "model": bundle.checkpoint,
            "resolved_model_revision": bundle.resolved_revision,
            "dtype": str(bundle.runtime.dtype).replace("torch.", ""),
            "batch_size": batch_size,
            "registry_sha256": registry["registry_sha256"],
            "preregistration_sha256": preregistration_sha256,
            "split_input_hashes": dict(split_hashes),
            "deterministic_settings": {
                key: value
                for key, value in determinism.items()
                if key != "torch_version"
            },
        }
    )


UNIFORM_REFERENCE_ORDER = {
    "bf16_reference": 0,
    "uniform_8bit_reference": 1,
    "uniform_4bit_reference": 2,
    "uniform_3bit_reference": 3,
}


def stage3_phase_records(
    registry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    stage2c_allocations_dir: Path,
    phase: str,
    authorized_regimes: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (uniform reference records, matched-budget records) for a phase.

    The additivity phase uses every 20%-budget competitor of both regimes as
    an additive-model probe. Development uses the additivity-authorized
    regimes at the 20% budget; final uses the development-authorized regimes
    at all budgets.
    """

    if phase not in STAGE3_PHASE_DIRS:
        raise ValueError(f"Unknown Stage 3 phase {phase!r}")
    if phase in ("development", "final") and not authorized_regimes:
        raise ValueError(f"The {phase} phase requires explicitly authorized regimes")
    unauthorized = set(authorized_regimes or []) - set(STAGE3_REGIMES)
    if unauthorized:
        raise RuntimeError(f"Regimes {sorted(unauthorized)} are not preregistered")
    references: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for entry in stage3_registry_entries(registry):
        record = load_stage3_allocation(
            entry, allocations_dir, stage2b_allocations_dir, stage2c_allocations_dir
        )
        if record["method_kind"] == "uniform_reference":
            references.append(record)
            continue
        if phase == "additivity":
            if record["budget_fraction"] == STAGE2C_DEVELOPMENT_BUDGET_FRACTION:
                competitors.append(record)
        elif phase == "development":
            if (
                record["regime"] in (authorized_regimes or [])
                and record["budget_fraction"] == STAGE2C_DEVELOPMENT_BUDGET_FRACTION
            ):
                competitors.append(record)
        else:
            if record["regime"] in (authorized_regimes or []):
                competitors.append(record)
    references.sort(key=lambda record: UNIFORM_REFERENCE_ORDER.get(record["method"], 9))
    competitors.sort(
        key=lambda record: (
            record["regime"],
            record["budget_fraction"],
            record["method_kind"],
            record["method"],
        )
    )
    return references, competitors
