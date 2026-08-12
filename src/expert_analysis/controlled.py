from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .datasets import DomainExamples
from .io_utils import atomic_save_npz


@dataclass
class PreparedDomainExamples:
    """Fixed-length, pretokenized inputs and the positions used for measurement."""

    domain: str
    input_ids: np.ndarray
    attention_mask: np.ndarray
    measurement_mask: np.ndarray
    metadata: dict[str, Any]

    @property
    def num_examples(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])

    def validate(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [example, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask shape does not match input_ids")
        if self.measurement_mask.shape != self.input_ids.shape:
            raise ValueError("measurement_mask shape does not match input_ids")
        if self.num_examples < 1 or self.sequence_length < 2:
            raise ValueError("Prepared inputs must contain examples with at least two tokens")
        if np.any(self.input_ids < 0):
            raise ValueError("input_ids contains negative token IDs")
        if not np.all(np.isin(self.attention_mask, (0, 1))):
            raise ValueError("attention_mask must be binary")
        if not np.all(np.isin(self.measurement_mask, (0, 1))):
            raise ValueError("measurement_mask must be binary")
        if np.any(self.measurement_mask > self.attention_mask):
            raise ValueError("measurement positions must also be attended positions")
        measured = self.measurement_mask.sum(axis=1)
        if np.any(measured <= 0) or np.any(measured != measured[0]):
            raise ValueError("Every example must have the same positive measurement budget")
        # Each measured source position must have an attended next-token label.
        if np.any(self.measurement_mask[:, -1] != 0):
            raise ValueError("The final token cannot be a next-token prediction source")
        if np.any(
            (self.measurement_mask[:, :-1] == 1)
            & (self.attention_mask[:, 1:] == 0)
        ):
            raise ValueError("A measured source position has no attended next-token target")

    def save(self, path: Path) -> None:
        self.validate()
        atomic_save_npz(
            path,
            input_ids=self.input_ids.astype(np.int32, copy=False),
            attention_mask=self.attention_mask.astype(np.uint8, copy=False),
            measurement_mask=self.measurement_mask.astype(np.uint8, copy=False),
        )

    @classmethod
    def load(
        cls, path: Path, domain: str, metadata: dict[str, Any]
    ) -> "PreparedDomainExamples":
        with np.load(path, allow_pickle=False) as data:
            instance = cls(
                domain=domain,
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                measurement_mask=data["measurement_mask"],
                metadata=metadata,
            )
        instance.validate()
        return instance


def prepare_controlled_domains(
    candidates: Mapping[str, DomainExamples],
    tokenizer: Any,
    num_examples: int,
    measured_tokens_per_example: int,
    neutral_prefix: str,
    max_length: int,
) -> tuple[dict[str, PreparedDomainExamples], dict[str, Any]]:
    """Create domains with an identical prefix and exact shared token budget.

    Each example contains ``measured_tokens_per_example`` source positions plus one
    look-ahead content token. The source positions are used for both expert metrics
    and next-token loss, so the activation proxy and masking intervention are aligned.
    """
    if num_examples < 2:
        raise ValueError("Controlled validation requires at least two examples per domain")
    if measured_tokens_per_example < 2:
        raise ValueError("measured_tokens_per_example must be at least two")
    if not neutral_prefix:
        raise ValueError("neutral_prefix cannot be empty")

    prefix_ids = _encode_one(tokenizer, neutral_prefix)
    if not prefix_ids:
        raise RuntimeError("The neutral prefix tokenized to an empty sequence")
    prepared: dict[str, PreparedDomainExamples] = {}
    domain_manifest: dict[str, Any] = {}
    for domain, pool in candidates.items():
        item = prepare_controlled_domain(
            pool,
            tokenizer=tokenizer,
            num_examples=num_examples,
            measured_tokens_per_example=measured_tokens_per_example,
            prefix_ids=prefix_ids,
            neutral_prefix=neutral_prefix,
            max_length=max_length,
        )
        prepared[domain] = item
        domain_manifest[domain] = item.metadata

    sequence_lengths = {item.sequence_length for item in prepared.values()}
    measured_budgets = {
        int(item.measurement_mask.sum()) for item in prepared.values()
    }
    prefix_positions = {
        tuple(item.metadata["control"]["prefix_model_positions"])
        for item in prepared.values()
    }
    if len(sequence_lengths) != 1 or len(measured_budgets) != 1 or len(prefix_positions) != 1:
        raise RuntimeError("Controlled domains do not have identical sequence/budget geometry")
    return prepared, {
        "prompt_style": "neutral_fixed_token_control",
        "neutral_prefix": neutral_prefix,
        "neutral_prefix_token_ids": prefix_ids,
        "neutral_prefix_token_sha256": _array_sha256(
            np.asarray(prefix_ids, dtype=np.int32)
        ),
        "measured_tokens_per_example": measured_tokens_per_example,
        "lookahead_tokens_per_example": 1,
        "examples_per_domain": num_examples,
        "measured_tokens_per_domain": num_examples * measured_tokens_per_example,
        "model_sequence_length": next(iter(sequence_lengths)),
        "same_prefix_token_ids": True,
        "same_model_sequence_length": True,
        "same_measurement_length_distribution": True,
        "same_total_measurement_budget": True,
        "domains": domain_manifest,
    }


def prepare_controlled_domain(
    candidates: DomainExamples,
    tokenizer: Any,
    num_examples: int,
    measured_tokens_per_example: int,
    prefix_ids: Sequence[int],
    neutral_prefix: str,
    max_length: int,
) -> PreparedDomainExamples:
    ids = list(candidates.metadata.get("selected_example_ids", []))
    if len(ids) != len(candidates.texts):
        raise RuntimeError(
            f"Candidate IDs and texts are misaligned for domain {candidates.domain!r}"
        )
    encoded_contents = _encode_many(tokenizer, candidates.texts)
    required_content_tokens = measured_tokens_per_example + 1
    eligible = [
        index
        for index, token_ids in enumerate(encoded_contents)
        if len(token_ids) >= required_content_tokens
    ]
    if len(eligible) < num_examples:
        lengths = np.asarray([len(item) for item in encoded_contents], dtype=np.int64)
        maximum = int(lengths.max()) if len(lengths) else 0
        raise RuntimeError(
            f"Domain {candidates.domain!r} has only {len(eligible)} candidates with at least "
            f"{required_content_tokens} content tokens (need {num_examples}; pool has "
            f"{len(encoded_contents)}, maximum length {maximum}). Increase "
            "--candidate-pool-size or lower --tokens-per-example."
        )

    selected = eligible[:num_examples]
    sequences: list[list[int]] = []
    attention_masks: list[list[int]] = []
    measurement_masks: list[list[int]] = []
    prefix_model_positions: list[int] | None = None
    selected_original_lengths: list[int] = []
    for index in selected:
        content_ids = encoded_contents[index]
        selected_original_lengths.append(len(content_ids))
        raw_ids = list(prefix_ids) + content_ids[:required_content_tokens]
        sequence, special_mask = _add_model_special_tokens(tokenizer, raw_ids)
        if len(sequence) > max_length:
            raise RuntimeError(
                f"Controlled sequence length {len(sequence)} exceeds --max-length {max_length}"
            )
        measurement: list[int] = []
        current_prefix_positions: list[int] = []
        raw_index = 0
        for position, special in enumerate(special_mask):
            if special:
                measurement.append(0)
                continue
            if raw_index < len(prefix_ids):
                current_prefix_positions.append(position)
            measurement.append(
                int(
                    len(prefix_ids)
                    <= raw_index
                    < len(prefix_ids) + measured_tokens_per_example
                )
            )
            raw_index += 1
        if raw_index != len(raw_ids):
            raise RuntimeError("Special-token mapping did not preserve every raw token")
        if sum(measurement) != measured_tokens_per_example or measurement[-1] != 0:
            raise RuntimeError("Failed to construct the exact next-token measurement mask")
        if prefix_model_positions is None:
            prefix_model_positions = current_prefix_positions
        elif prefix_model_positions != current_prefix_positions:
            raise RuntimeError("Neutral prefix positions differ between examples")
        sequences.append(sequence)
        attention_masks.append([1] * len(sequence))
        measurement_masks.append(measurement)

    lengths = {len(item) for item in sequences}
    if len(lengths) != 1:
        raise RuntimeError("Fixed-token controlled inputs produced different sequence lengths")
    input_array = np.asarray(sequences, dtype=np.int32)
    attention_array = np.asarray(attention_masks, dtype=np.uint8)
    measurement_array = np.asarray(measurement_masks, dtype=np.uint8)
    metadata = dict(candidates.metadata)
    metadata.update(
        {
            "candidate_pool_requested": candidates.metadata.get("requested_examples"),
            "candidate_pool_actual": len(candidates.texts),
            "candidate_pool_eligible": len(eligible),
            "requested_examples": num_examples,
            "actual_examples": num_examples,
            "selected_example_ids": [ids[index] for index in selected],
            "include_reference_answers": False,
            "format_style": "neutral_content",
            "control": {
                "neutral_prefix": neutral_prefix,
                "prefix_token_ids": list(prefix_ids),
                "prefix_model_positions": prefix_model_positions or [],
                "measured_tokens_per_example": measured_tokens_per_example,
                "lookahead_tokens_per_example": 1,
                "model_sequence_length": int(input_array.shape[1]),
                "selected_original_content_token_min": min(selected_original_lengths),
                "selected_original_content_token_max": max(selected_original_lengths),
                "selected_original_content_token_mean": float(
                    np.mean(selected_original_lengths)
                ),
                "input_ids_sha256": _array_sha256(input_array),
                "measurement_mask_sha256": _array_sha256(measurement_array),
            },
        }
    )
    result = PreparedDomainExamples(
        domain=candidates.domain,
        input_ids=input_array,
        attention_mask=attention_array,
        measurement_mask=measurement_array,
        metadata=metadata,
    )
    result.validate()
    return result


def _encode_one(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    values = encoded["input_ids"]
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [int(item) for item in values]


def _encode_many(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    values = encoded["input_ids"]
    if isinstance(values, np.ndarray):
        values = values.tolist()
    return [[int(item) for item in row] for row in values]


def _add_model_special_tokens(
    tokenizer: Any, raw_ids: Sequence[int]
) -> tuple[list[int], list[int]]:
    sequence = [int(item) for item in tokenizer.build_inputs_with_special_tokens(list(raw_ids))]
    special_mask = [
        int(item)
        for item in tokenizer.get_special_tokens_mask(
            list(raw_ids), already_has_special_tokens=False
        )
    ]
    if len(sequence) != len(special_mask):
        raise RuntimeError("Tokenizer returned inconsistent special-token metadata")
    if sum(1 for item in special_mask if item == 0) != len(raw_ids):
        raise RuntimeError("Tokenizer special-token mask does not map the raw token sequence")
    return sequence, special_mask


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
