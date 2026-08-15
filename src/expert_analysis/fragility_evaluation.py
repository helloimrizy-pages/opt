"""Stage 2C evaluation support: seed-45 split, preregistration, run records.

The seed-45 development split is entirely new: it is disjoint from the frozen
calibration data, the original controlled 100/domain set (used by masking,
Stage 1, and Stage 2A), the contaminated seed-43 Stage 2B development split,
and the untouched seed-44 final split. Model evaluation itself reuses the
audited Stage 2B mixed-precision manager and loss checkpointing unchanged.
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
from .fragility import STAGE2C_STAGE, STAGE2C_REGIMES
from .fragility_optimization import (
    FRAGILITY_ROBUST_METHOD,
    STAGE2C_DEVELOPMENT_SEED,
    STAGE2C_FINAL_SEED,
    load_stage2c_allocation,
)
from .fragility_statistics import (
    FINAL_REQUIRED_CI_WINS_VS_AVERAGE,
    FINAL_REQUIRED_POINT_BUDGETS,
    FINAL_REQUIRED_POINT_WINS_VS_GLOBAL,
    GATE_D_REQUIRED_POSITIVE_DOMAINS,
    GATE_E_DENOMINATOR_EPSILON,
    GATE_E_RELATIVE_TOLERANCE,
    QUALIFIED_REQUIRED_POINT_BUDGETS,
    STAGE2C_BOOTSTRAP_REPLICATES,
    STAGE2C_BOOTSTRAP_SEED,
    STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
    SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS,
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
from .modeling import ModelBundle
from .protection_optimization import PROTECTION_FRACTIONS
from .specialist_preservation import STAGE2B_DOMAINS

STAGE2C_SPLIT_SCHEMA = "stage2c_development_split_v1"
STAGE2C_SPLIT_NAME = "development"
STAGE2C_EXAMPLES_PER_DOMAIN = DEVELOPMENT_EXAMPLES_PER_DOMAIN
PREREGISTRATION_FILE = "stage2c_preregistration.json"
PREREGISTRATION_SHA_FILE = "preregistration_sha256.txt"


def build_stage2c_development_split(
    source: ControlledSource,
    tokenizer: Any,
    stage2b_results_dir: Path,
    output_dir: Path,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Construct the new seed-45 development split with strict disjointness.

    Exclusions per domain: the seed-42 prior usage rule frozen by Stage 2B
    (entire inspected candidate pool where practical, otherwise the 100
    previously evaluated examples), plus every seed-43 development text and
    every seed-44 final text. Any overlap aborts the build.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    stage2b_manifest = read_json(
        stage2b_results_dir / "splits" / "split_manifest.json"
    )
    if (
        stage2b_manifest["development_seed"] != 43
        or stage2b_manifest["final_seed"] != STAGE2C_FINAL_SEED
    ):
        raise RuntimeError("Stage 2B split manifest does not match the frozen seeds")
    reference_row = source.prepared[STAGE2B_DOMAINS[0]].measurement_mask[0]
    manifest_domains: dict[str, Any] = {}
    for domain in STAGE2B_DOMAINS:
        prior = reconstruct_prior_usage(domain, source, tokenizer, cache_dir)
        stage2b_domain = stage2b_manifest["domains"][domain]
        stage2b_rows: dict[str, set[str]] = {}
        stage2b_texts: dict[str, set[str]] = {}
        for split_name in ("development", "final"):
            entry = stage2b_domain[split_name]
            path = stage2b_results_dir / "splits" / split_name / f"{domain}.npz"
            with np.load(path, allow_pickle=False) as data:
                input_ids = np.asarray(data["input_ids"])
            if array_sha256(input_ids) != entry["input_ids_sha256"]:
                raise RuntimeError(
                    f"Frozen Stage 2B {split_name} split hash changed for {domain}"
                )
            stage2b_rows[split_name] = {
                content_row_key(input_ids[i, len(EXPECTED_PREFIX_IDS):])
                for i in range(input_ids.shape[0])
            }
            stage2b_texts[split_name] = set(entry["example_text_sha256"])

        excluded_texts_beyond_prior = (
            stage2b_texts["development"] | stage2b_texts["final"]
        )
        # Full-pool exclusion decision from counts alone, before selection.
        from .datasets import load_domain_examples
        from .heldout_splits import EXPECTED_DATASET_REVISIONS, _encode_texts, text_sha256

        probe_pool = load_domain_examples(
            domain=domain,
            num_examples=FULL_SPLIT_REQUEST,
            seed=STAGE2C_DEVELOPMENT_SEED,
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
        exclude_full_pool = available_excluding_pool >= STAGE2C_EXAMPLES_PER_DOMAIN
        exclusion = (
            set(prior.pool_text_hashes)
            if exclude_full_pool
            else set(prior.selected_text_hashes)
        ) | excluded_texts_beyond_prior

        selection = select_heldout_examples(
            domain,
            STAGE2C_SPLIT_NAME,
            STAGE2C_DEVELOPMENT_SEED,
            STAGE2C_EXAMPLES_PER_DOMAIN,
            exclusion,
            tokenizer,
            cache_dir,
        )
        prepared = prepare_heldout_split(selection, tokenizer, reference_row)
        row_keys = split_row_keys(prepared)
        if len(set(row_keys)) != len(row_keys):
            raise RuntimeError(
                f"The seed-45 split for {domain} contains duplicate content rows"
            )
        overlaps = {
            "seed45_vs_prior_evaluated": sorted(
                set(row_keys) & prior.selected_row_keys
            ),
            "seed45_vs_seed43_development": sorted(
                set(row_keys) & stage2b_rows["development"]
            ),
            "seed45_vs_seed44_final": sorted(set(row_keys) & stage2b_rows["final"]),
            "seed45_text_vs_prior": sorted(
                set(selection.text_hashes) & prior.selected_text_hashes
            ),
            "seed45_text_vs_seed43": sorted(
                set(selection.text_hashes) & stage2b_texts["development"]
            ),
            "seed45_text_vs_seed44": sorted(
                set(selection.text_hashes) & stage2b_texts["final"]
            ),
        }
        if any(overlaps.values()):
            raise RuntimeError(
                f"Seed-45 split overlap detected for {domain}: "
                + ", ".join(name for name, values in overlaps.items() if values)
            )

        split_dir = output_dir / STAGE2C_SPLIT_NAME
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
                "previously evaluated examples plus all seed-43/seed-44 texts "
                "were excluded"
            ),
            "eligible_after_full_exclusion": available_excluding_pool,
            "num_examples": prepared.num_examples,
            "sequence_length": prepared.sequence_length,
            "input_ids_sha256": array_sha256(prepared.input_ids),
            "measurement_mask_sha256": array_sha256(prepared.measurement_mask),
            "dataset_example_ids": list(selection.example_ids),
            "example_text_sha256": list(selection.text_hashes),
            "exclusion_set_size": len(exclusion),
            "seed43_texts_excluded": len(stage2b_texts["development"]),
            "seed44_texts_excluded": len(stage2b_texts["final"]),
            "overlap_checks": {name: len(values) for name, values in overlaps.items()},
            "disjointness_verified": True,
        }

    manifest = {
        "schema": STAGE2C_SPLIT_SCHEMA,
        "stage": STAGE2C_STAGE,
        "development_seed": STAGE2C_DEVELOPMENT_SEED,
        "final_seed": STAGE2C_FINAL_SEED,
        "final_split_source": str(stage2b_results_dir / "splits" / "final"),
        "final_split_note": (
            "the untouched Stage 2B seed-44 final split is reused for final "
            "confirmation and must not be evaluated before FINAL_CONFIRMATION_GO"
        ),
        "examples_per_domain": STAGE2C_EXAMPLES_PER_DOMAIN,
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


def load_stage2c_development_split(
    splits_dir: Path, domain: str
) -> PreparedDomainExamples:
    """Load one seed-45 split domain and verify it against the manifest."""

    manifest = read_json(splits_dir / "split_manifest.json")
    if manifest.get("schema") != STAGE2C_SPLIT_SCHEMA:
        raise RuntimeError("Unexpected Stage 2C split schema")
    if manifest["development_seed"] != STAGE2C_DEVELOPMENT_SEED:
        raise RuntimeError("The Stage 2C development split seed is not 45")
    entry = manifest["domains"][domain]
    metadata = read_json(
        splits_dir / STAGE2C_SPLIT_NAME / f"{domain}.metadata.json"
    )
    prepared = PreparedDomainExamples.load(
        splits_dir / STAGE2C_SPLIT_NAME / f"{domain}.npz", domain, metadata
    )
    if array_sha256(prepared.input_ids) != entry["input_ids_sha256"]:
        raise RuntimeError(f"Seed-45 {domain} input hash mismatch against the manifest")
    if array_sha256(prepared.measurement_mask) != entry["measurement_mask_sha256"]:
        raise RuntimeError(f"Seed-45 {domain} measurement-mask hash mismatch")
    if prepared.num_examples != entry["num_examples"]:
        raise RuntimeError(f"Seed-45 {domain} example count mismatch")
    if not entry.get("disjointness_verified"):
        raise RuntimeError(f"Seed-45 {domain} disjointness was never verified")
    return prepared


def build_stage2c_preregistration(
    registry: Mapping[str, Any],
    fragility_record: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """The frozen Stage 2C preregistration payload (built before seed-45 NLL)."""

    return {
        "schema": "stage2c_preregistration_v1",
        "stage": STAGE2C_STAGE,
        "research_hypothesis": (
            "At a fixed expert-weight memory budget, calibration-measured "
            "domain quantization fragility combined with domain-specialist "
            "coverage reduces worst-domain quality degradation better than "
            "unweighted specialist preservation and simpler importance-based "
            "protection."
        ),
        "primary_method": FRAGILITY_ROBUST_METHOD,
        "objective_formula": (
            "minimize z subject to q_norm[d] * (1 - Coverage_d(x)) <= z for "
            "every domain, sum_{l,e} DeltaM[l,e,b] * x[l,e] <= Budget_p(b), "
            "x binary; no mean-risk penalty, no tuned lambda, no added term"
        ),
        "fragility_formula": (
            "q_raw[d,b] = (NLL_base_cal[d,b] - NLL_BF16_cal[d]) / NLL_BF16_cal[d]"
        ),
        "clipping_rule": "q[d,b] = max(q_raw[d,b], 0); absolute value is never used",
        "normalization_rule": (
            "q_norm[d,b] = q[d,b] / mean_d q[d,b]; a regime with all-zero "
            "fragility is invalid and is not evaluated"
        ),
        "no_new_metric_search": (
            "no alternative fragility formulas, fitted coefficients, tuned "
            "exponents, domain weights, Stage 2A metric combinations, "
            "expert-level delta-NLL, seed-43 outcomes, or post-hoc objective "
            "changes are permitted; if Stage 2C fails the negative result is "
            "preserved and optimization development on this branch stops"
        ),
        "calibration": {
            "examples_per_domain": fragility_record["calibration_examples_per_domain"],
            "calibration_seed": fragility_record["calibration_seed"],
            "calibration_subset_hashes": fragility_record["calibration_subset_hashes"],
            "fragility_sha256": fragility_record["fragility_sha256"],
            "fragility_values": {
                regime: {
                    domain: fragility_record["regimes"][regime]["domains"][domain][
                        "normalized_fragility"
                    ]
                    for domain in STAGE2B_DOMAINS
                }
                for regime in STAGE2C_REGIMES
            },
        },
        "precision_regimes": dict(STAGE2C_REGIMES),
        "protected_bits": 8,
        "protection_budgets": list(PROTECTION_FRACTIONS),
        "quantization": (
            "Stage-1 symmetric group-wise expert-only QDQ, group size 128, "
            "FP16 scales; routers, attention, embeddings, normalization, "
            "lm_head, and all non-expert parameters stay BF16"
        ),
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
        ],
        "development_seed": STAGE2C_DEVELOPMENT_SEED,
        "development_examples_per_domain": STAGE2C_EXAMPLES_PER_DOMAIN,
        "development_budget_fraction": STAGE2C_DEVELOPMENT_BUDGET_FRACTION,
        "final_seed": STAGE2C_FINAL_SEED,
        "development_split_input_hashes": {
            domain: split_manifest["domains"][domain]["input_ids_sha256"]
            for domain in STAGE2B_DOMAINS
        },
        "development_gates": {
            "gate_a": "lower point-estimate WorstRelativeDelta than frozen Robust-Functional",
            "gate_b": "lower point-estimate WorstRelativeDelta than the mean of five frozen random allocations",
            "gate_c": "lower point-estimate WorstRelativeDelta than BOTH Global-Importance and Average-Specialization",
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
            "FINAL_CONFIRMATION_GO if at least one regime passes all five "
            "gates, else FRAGILITY_ROBUST_NO_GO; only passing regimes are "
            "authorized for seed-44 final confirmation and a failing regime is "
            "never final-tested"
        ),
        "final_success_criteria": {
            "requirement_1": (
                "lower point WorstRelativeDelta than Robust-Functional, "
                "Global-Importance, Average-Specialization, and the random "
                f"mean at >= {FINAL_REQUIRED_POINT_BUDGETS} of 4 budgets"
            ),
            "requirement_2": (
                "positive average worst-domain improvement over "
                "Average-Specialization across all four budgets"
            ),
            "requirement_3": (
                f">= {FINAL_REQUIRED_CI_WINS_VS_AVERAGE} of 4 worst-domain "
                "comparisons vs Average-Specialization with 95% bootstrap CIs "
                "entirely favoring Fragility-Robust"
            ),
            "requirement_4": (
                f">= {FINAL_REQUIRED_POINT_WINS_VS_GLOBAL} of 4 worst-domain "
                "comparisons vs Global-Importance with point estimates "
                "favoring Fragility-Robust"
            ),
            "requirement_5": (
                "no domain with negative recovery vs the uniform base at >= "
                f"{SYSTEMATIC_NEGATIVE_RECOVERY_BUDGETS} of 4 budgets"
            ),
            "strong_success": "requirements 1-5 in at least one authorized regime",
            "qualified_success": (
                "not strong; positive average improvement over "
                "Average-Specialization; no systematically negative domain; and "
                "either point wins over both Global-Importance and "
                f"Average-Specialization at >= {QUALIFIED_REQUIRED_POINT_BUDGETS} "
                "of 4 budgets, or a clear Stage 2B fix (beats Robust-Functional "
                f"and the random mean at >= {FINAL_REQUIRED_POINT_BUDGETS} of 4 "
                "budgets)"
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
        "fragility_robust_allocation_hashes": {
            f"{entry['regime']}_budget{int(round(entry['budget_fraction'] * 100))}": (
                entry["allocation_sha256"]
            )
            for entry in registry["new_entries"]
        },
        "stage2b_registry_sha256": registry["stage2b_registry_sha256"],
        "stage2b_motivation_note": (
            "Stage 2C was conceived after observing the Stage 2B development "
            "NO_GO; seed 43 is contaminated and is never used, and Stage 2B "
            "development numbers never enter the objective or allocations"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_stage2c_preregistration(
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
                "stage2c_preregistration.json is already frozen with different "
                "content; refusing to overwrite the preregistration"
            )
        digest = file_sha256(path)
        if sha_path.is_file() and sha_path.read_text().strip() != digest:
            raise RuntimeError("preregistration_sha256.txt does not match the file")
        return path, digest
    atomic_write_json(path, payload)
    digest = file_sha256(path)
    sha_path.write_text(digest + "\n", encoding="utf-8")
    return path, digest


def verify_preregistration_unchanged(results_dir: Path) -> str:
    """Refuse to proceed if the frozen preregistration file changed."""

    path = results_dir / PREREGISTRATION_FILE
    sha_path = results_dir / PREREGISTRATION_SHA_FILE
    if not path.is_file() or not sha_path.is_file():
        raise RuntimeError(
            "The Stage 2C preregistration is missing; freeze it before evaluation"
        )
    recorded = sha_path.read_text().strip()
    observed = file_sha256(path)
    if observed != recorded:
        raise RuntimeError(
            "stage2c_preregistration.json changed after freezing "
            f"(recorded {recorded[:16]}..., observed {observed[:16]}...); "
            "refusing to proceed"
        )
    return observed


def stage2c_run_fingerprint(
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
            "stage": "stage2c_fragility_evaluation",
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


def stage2c_phase_records(
    registry: Mapping[str, Any],
    allocations_dir: Path,
    stage2b_allocations_dir: Path,
    phase: str,
    authorized_regimes: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (uniform reference records, matched-budget records) for a phase."""

    if phase not in ("development", "final"):
        raise ValueError(f"Unknown phase {phase!r}")
    if phase == "final" and not authorized_regimes:
        raise ValueError("Final evaluation requires explicitly authorized regimes")
    valid_regimes = list(registry.get("valid_regimes", list(STAGE2C_REGIMES)))
    if phase == "final":
        unauthorized = set(authorized_regimes or []) - set(valid_regimes)
        if unauthorized:
            raise RuntimeError(
                f"Regimes {sorted(unauthorized)} are not valid in the frozen registry"
            )
    references: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for entry in list(registry["new_entries"]) + list(registry["reused_entries"]):
        record = load_stage2c_allocation(
            entry, allocations_dir, stage2b_allocations_dir
        )
        if record["method_kind"] == "uniform_reference":
            references.append(record)
            continue
        if record["regime"] not in valid_regimes:
            continue
        if phase == "development":
            if record["budget_fraction"] == STAGE2C_DEVELOPMENT_BUDGET_FRACTION:
                competitors.append(record)
        else:
            if record["regime"] in authorized_regimes:
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
