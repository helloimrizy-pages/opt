"""Stage 2B development/final split construction with strict disjointness.

The optimization uses only the frozen calibration data, so evaluation requires
two new controlled splits that never overlap the 100 previously evaluated
examples per domain (calibration, masking, Stage-1 pilot, and Stage-2A all used
exactly that frozen set), nor one another. Where the underlying split is large
enough, the entire seed-42 candidate pool inspected during the original
controlled selection is excluded as well; where that would exhaust the dataset
the limitation is recorded instead of silently relaxed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .balanced import (
    EXPECTED_DATASET_REVISIONS,
    EXPECTED_PREFIX,
    EXPECTED_PREFIX_IDS,
    ControlledSource,
    array_sha256,
)
from .controlled import PreparedDomainExamples, prepare_controlled_domain
from .datasets import DomainExamples, load_domain_examples
from .io_utils import atomic_write_json, read_json
from .specialist_preservation import STAGE2B_DOMAINS

DEVELOPMENT_SPLIT_SEED = 43
FINAL_SPLIT_SEED = 44
DEVELOPMENT_EXAMPLES_PER_DOMAIN = 50
FINAL_EXAMPLES_PER_DOMAIN = 100
MEASURED_TOKENS_PER_EXAMPLE = 64
REQUIRED_CONTENT_TOKENS = MEASURED_TOKENS_PER_EXAMPLE + 1
MAX_LENGTH = 512
PRIOR_POOL_SEED = 42
PRIOR_POOL_SIZE = 1000
FULL_SPLIT_REQUEST = 10_000


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_row_key(token_ids: np.ndarray) -> str:
    """Hash of the exact measured content tokens (prefix excluded)."""

    contiguous = np.ascontiguousarray(np.asarray(token_ids, dtype=np.int32))
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _encode_texts(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False, padding=False)
    return [[int(v) for v in row] for row in encoded["input_ids"]]


@dataclass
class PriorControlledUsage:
    """Reconstruction of the seed-42 candidate pool and its selected 100 texts."""

    domain: str
    pool_text_hashes: set[str]
    selected_text_hashes: set[str]
    selected_row_keys: set[str]
    pool_size: int
    eligible_pool_size: int
    reconstruction_verified: bool


def reconstruct_prior_usage(
    domain: str,
    source: ControlledSource,
    tokenizer: Any,
    cache_dir: str | None = None,
) -> PriorControlledUsage:
    """Rebuild the original candidate pool and verify the frozen 100 selection.

    The reconstruction re-runs the deterministic seed-42 shuffle and the exact
    first-100-eligible rule, then proves correctness by matching the frozen
    controlled ``input_ids`` hash bit for bit.
    """

    pool = load_domain_examples(
        domain=domain,
        num_examples=PRIOR_POOL_SIZE,
        seed=PRIOR_POOL_SEED,
        cache_dir=cache_dir,
        revision=EXPECTED_DATASET_REVISIONS[domain],
        include_answers=False,
        allow_substitution=False,
        format_style="neutral_content",
    )
    encoded = _encode_texts(tokenizer, pool.texts)
    eligible = [
        index
        for index, token_ids in enumerate(encoded)
        if len(token_ids) >= REQUIRED_CONTENT_TOKENS
    ]
    frozen = source.prepared[domain]
    selected = eligible[: frozen.num_examples]
    if len(selected) != frozen.num_examples:
        raise RuntimeError(
            f"Reconstructed prior pool for {domain} has only {len(selected)} eligible "
            f"candidates; expected {frozen.num_examples}"
        )
    prefix = list(EXPECTED_PREFIX_IDS)
    rows = np.asarray(
        [prefix + encoded[index][:REQUIRED_CONTENT_TOKENS] for index in selected],
        dtype=np.int32,
    )
    if array_sha256(rows) != array_sha256(frozen.input_ids):
        raise RuntimeError(
            f"Reconstructed prior selection for {domain} does not reproduce the frozen "
            "controlled input hash; refusing to trust the exclusion set"
        )
    selected_row_keys = {
        content_row_key(frozen.input_ids[i, len(prefix):]) for i in range(frozen.num_examples)
    }
    return PriorControlledUsage(
        domain=domain,
        pool_text_hashes={text_sha256(text) for text in pool.texts},
        selected_text_hashes={text_sha256(pool.texts[index]) for index in selected},
        selected_row_keys=selected_row_keys,
        pool_size=len(pool.texts),
        eligible_pool_size=len(eligible),
        reconstruction_verified=True,
    )


@dataclass
class SplitSelection:
    """One domain's held-out selection before geometry preparation."""

    domain: str
    split: str
    seed: int
    texts: list[str]
    example_ids: list[str]
    text_hashes: list[str]
    metadata: dict[str, Any]


def select_heldout_examples(
    domain: str,
    split: str,
    seed: int,
    num_examples: int,
    excluded_text_hashes: set[str],
    tokenizer: Any,
    cache_dir: str | None = None,
) -> SplitSelection:
    """Deterministically choose held-out examples disjoint from the exclusions."""

    pool = load_domain_examples(
        domain=domain,
        num_examples=FULL_SPLIT_REQUEST,
        seed=seed,
        cache_dir=cache_dir,
        revision=EXPECTED_DATASET_REVISIONS[domain],
        include_answers=False,
        allow_substitution=False,
        format_style="neutral_content",
    )
    encoded = _encode_texts(tokenizer, pool.texts)
    pool_ids = list(pool.metadata.get("selected_example_ids", []))
    chosen: list[int] = []
    skipped_excluded = 0
    skipped_short = 0
    seen_hashes: set[str] = set()
    for index, token_ids in enumerate(encoded):
        if len(token_ids) < REQUIRED_CONTENT_TOKENS:
            skipped_short += 1
            continue
        digest = text_sha256(pool.texts[index])
        if digest in excluded_text_hashes or digest in seen_hashes:
            skipped_excluded += 1
            continue
        seen_hashes.add(digest)
        chosen.append(index)
        if len(chosen) >= num_examples:
            break
    if len(chosen) < num_examples:
        raise RuntimeError(
            f"Domain {domain!r} cannot supply {num_examples} disjoint {split} examples "
            f"(found {len(chosen)}; pool {len(pool.texts)}, short {skipped_short}, "
            f"excluded {skipped_excluded})"
        )
    metadata = dict(pool.metadata)
    metadata.update(
        {
            "split": split,
            "split_seed": seed,
            "candidate_pool_loaded": len(pool.texts),
            "skipped_below_length": skipped_short,
            "skipped_excluded_or_duplicate": skipped_excluded,
            "exclusion_set_size": len(excluded_text_hashes),
        }
    )
    return SplitSelection(
        domain=domain,
        split=split,
        seed=seed,
        texts=[pool.texts[index] for index in chosen],
        example_ids=[str(pool_ids[index]) for index in chosen],
        text_hashes=[text_sha256(pool.texts[index]) for index in chosen],
        metadata=metadata,
    )


def prepare_heldout_split(
    selection: SplitSelection,
    tokenizer: Any,
    reference_measurement_row: np.ndarray,
) -> PreparedDomainExamples:
    """Build controlled-geometry inputs identical to the frozen evaluation setup."""

    candidates = DomainExamples(
        domain=selection.domain,
        texts=list(selection.texts),
        metadata={
            **selection.metadata,
            "selected_example_ids": list(selection.example_ids),
        },
    )
    prepared = prepare_controlled_domain(
        candidates,
        tokenizer=tokenizer,
        num_examples=len(selection.texts),
        measured_tokens_per_example=MEASURED_TOKENS_PER_EXAMPLE,
        prefix_ids=list(EXPECTED_PREFIX_IDS),
        neutral_prefix=EXPECTED_PREFIX,
        max_length=MAX_LENGTH,
    )
    if prepared.sequence_length != len(EXPECTED_PREFIX_IDS) + REQUIRED_CONTENT_TOKENS:
        raise RuntimeError(
            f"Held-out split geometry for {selection.domain} does not match the "
            "controlled 68-token layout"
        )
    for row in prepared.measurement_mask:
        if not np.array_equal(row, reference_measurement_row):
            raise RuntimeError(
                "Held-out measurement mask differs from the frozen controlled pattern"
            )
    prepared.metadata.update(
        {
            "split": selection.split,
            "split_seed": selection.seed,
            "text_sha256": list(selection.text_hashes),
        }
    )
    return prepared


def split_row_keys(prepared: PreparedDomainExamples) -> list[str]:
    prefix_length = len(EXPECTED_PREFIX_IDS)
    return [
        content_row_key(prepared.input_ids[i, prefix_length:])
        for i in range(prepared.num_examples)
    ]


def build_heldout_splits(
    source: ControlledSource,
    tokenizer: Any,
    output_dir: Path,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Construct, verify, and persist both held-out splits for all domains."""

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_row = source.prepared[STAGE2B_DOMAINS[0]].measurement_mask[0]
    manifest_domains: dict[str, Any] = {}
    for domain in STAGE2B_DOMAINS:
        prior = reconstruct_prior_usage(domain, source, tokenizer, cache_dir)
        frozen = source.prepared[domain]
        frozen_row_keys = {
            content_row_key(frozen.input_ids[i, len(EXPECTED_PREFIX_IDS):])
            for i in range(frozen.num_examples)
        }
        if frozen_row_keys != prior.selected_row_keys:
            raise RuntimeError(f"Frozen row keys are inconsistent for {domain}")

        # Decide whether excluding the entire inspected candidate pool is
        # practical: the remaining shuffled split must still supply both new
        # splits. This is decided from counts alone, before any selection.
        needed = DEVELOPMENT_EXAMPLES_PER_DOMAIN + FINAL_EXAMPLES_PER_DOMAIN
        probe_pool = load_domain_examples(
            domain=domain,
            num_examples=FULL_SPLIT_REQUEST,
            seed=DEVELOPMENT_SPLIT_SEED,
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
        available_excluding_pool = len(eligible_hashes - prior.pool_text_hashes)
        exclude_full_pool = available_excluding_pool >= needed
        exclusion = set(
            prior.pool_text_hashes if exclude_full_pool else prior.selected_text_hashes
        )

        development_selection = select_heldout_examples(
            domain, "development", DEVELOPMENT_SPLIT_SEED,
            DEVELOPMENT_EXAMPLES_PER_DOMAIN, exclusion, tokenizer, cache_dir,
        )
        final_exclusion = exclusion | set(development_selection.text_hashes)
        final_selection = select_heldout_examples(
            domain, "final", FINAL_SPLIT_SEED,
            FINAL_EXAMPLES_PER_DOMAIN, final_exclusion, tokenizer, cache_dir,
        )
        development = prepare_heldout_split(development_selection, tokenizer, reference_row)
        final = prepare_heldout_split(final_selection, tokenizer, reference_row)

        development_keys = split_row_keys(development)
        final_keys = split_row_keys(final)
        for split_name, keys in (("development", development_keys), ("final", final_keys)):
            if len(set(keys)) != len(keys):
                raise RuntimeError(
                    f"The {split_name} split for {domain} contains two examples with "
                    "identical measured content tokens"
                )
        overlaps = {
            "development_vs_prior_evaluated": sorted(
                set(development_keys) & prior.selected_row_keys
            ),
            "final_vs_prior_evaluated": sorted(set(final_keys) & prior.selected_row_keys),
            "development_vs_final": sorted(set(development_keys) & set(final_keys)),
            "development_text_vs_prior": sorted(
                set(development_selection.text_hashes) & prior.selected_text_hashes
            ),
            "final_text_vs_prior": sorted(
                set(final_selection.text_hashes) & prior.selected_text_hashes
            ),
            "development_text_vs_final_text": sorted(
                set(development_selection.text_hashes)
                & set(final_selection.text_hashes)
            ),
        }
        if any(overlaps.values()):
            raise RuntimeError(
                f"Held-out split overlap detected for {domain}: "
                + ", ".join(name for name, values in overlaps.items() if values)
            )

        for split_name, prepared in (("development", development), ("final", final)):
            split_dir = output_dir / split_name
            prepared.save(split_dir / f"{domain}.npz")
            atomic_write_json(split_dir / f"{domain}.metadata.json", prepared.metadata)
        manifest_domains[domain] = {
            "dataset_revision": EXPECTED_DATASET_REVISIONS[domain],
            "prior_pool_size": prior.pool_size,
            "prior_pool_eligible": prior.eligible_pool_size,
            "prior_selection_reconstruction_verified": prior.reconstruction_verified,
            "prior_pool_fully_excluded": exclude_full_pool,
            "prior_pool_exclusion_note": (
                "entire inspected seed-42 candidate pool excluded"
                if exclude_full_pool
                else "full-pool exclusion impractical (dataset too small); only the "
                "100 previously evaluated examples were excluded, so some new "
                "examples may share the previously length-inspected pool"
            ),
            "eligible_after_full_pool_exclusion": available_excluding_pool,
            "development": _split_manifest(development, development_selection),
            "final": _split_manifest(final, final_selection),
            "overlap_checks": {name: len(values) for name, values in overlaps.items()},
            "disjointness_verified": True,
        }

    manifest = {
        "schema": "stage2b_heldout_splits_v1",
        "stage": "stage2b_robust_specialist_preservation",
        "development_seed": DEVELOPMENT_SPLIT_SEED,
        "final_seed": FINAL_SPLIT_SEED,
        "development_examples_per_domain": DEVELOPMENT_EXAMPLES_PER_DOMAIN,
        "final_examples_per_domain": FINAL_EXAMPLES_PER_DOMAIN,
        "measured_tokens_per_example": MEASURED_TOKENS_PER_EXAMPLE,
        "lookahead_tokens_per_example": 1,
        "neutral_prefix": EXPECTED_PREFIX,
        "neutral_prefix_token_ids": list(EXPECTED_PREFIX_IDS),
        "model_sequence_length": len(EXPECTED_PREFIX_IDS) + REQUIRED_CONTENT_TOKENS,
        "source_collection_fingerprint": source.config["collection_fingerprint"],
        "domains": manifest_domains,
        "final_split_evaluation_note": (
            "final inputs are constructed and frozen here but must not be "
            "evaluated before the development GO gate passes"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def _split_manifest(
    prepared: PreparedDomainExamples, selection: SplitSelection
) -> dict[str, Any]:
    return {
        "num_examples": prepared.num_examples,
        "sequence_length": prepared.sequence_length,
        "input_ids_sha256": array_sha256(prepared.input_ids),
        "measurement_mask_sha256": array_sha256(prepared.measurement_mask),
        "dataset_example_ids": list(selection.example_ids),
        "example_text_sha256": list(selection.text_hashes),
        "candidate_pool_loaded": selection.metadata["candidate_pool_loaded"],
        "skipped_below_length": selection.metadata["skipped_below_length"],
        "skipped_excluded_or_duplicate": selection.metadata[
            "skipped_excluded_or_duplicate"
        ],
        "exclusion_set_size": selection.metadata["exclusion_set_size"],
        "dataset_id_note": (
            "IDs without a stable dataset key fall back to the shuffled row index; "
            "example_text_sha256 is the durable identity"
        ),
    }


def load_heldout_split(
    output_dir: Path, split: str, domain: str
) -> PreparedDomainExamples:
    """Load one prepared split domain and verify it against the manifest."""

    manifest = read_json(output_dir / "split_manifest.json")
    entry = manifest["domains"][domain][split]
    metadata = read_json(output_dir / split / f"{domain}.metadata.json")
    prepared = PreparedDomainExamples.load(
        output_dir / split / f"{domain}.npz", domain, metadata
    )
    if array_sha256(prepared.input_ids) != entry["input_ids_sha256"]:
        raise RuntimeError(f"{split}/{domain} input hash mismatch against the manifest")
    if array_sha256(prepared.measurement_mask) != entry["measurement_mask_sha256"]:
        raise RuntimeError(f"{split}/{domain} measurement-mask hash mismatch")
    if prepared.num_examples != entry["num_examples"]:
        raise RuntimeError(f"{split}/{domain} example count mismatch")
    return prepared
