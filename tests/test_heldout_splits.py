from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import expert_analysis.heldout_splits as heldout_splits
from expert_analysis.datasets import DomainExamples
from expert_analysis.heldout_splits import (
    prepare_heldout_split,
    select_heldout_examples,
    split_row_keys,
    text_sha256,
)


class FakeTokenizer:
    """Deterministic whitespace tokenizer with no special tokens."""

    def __call__(self, texts, add_special_tokens=False, truncation=False, padding=False):
        del add_special_tokens, truncation, padding
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        return {"input_ids": [self._encode(text) for text in texts]}

    @staticmethod
    def _encode(text: str) -> list[int]:
        return [1000 + (hash(word) % 40_000) for word in text.split()]

    @staticmethod
    def num_special_tokens_to_add(pair: bool = False) -> int:
        del pair
        return 0


def make_text(index: int, words: int = 70) -> str:
    return " ".join(f"d{index}w{i}" for i in range(words))


def fake_pool(texts: list[str]) -> DomainExamples:
    return DomainExamples(
        domain="general",
        texts=list(texts),
        metadata={"selected_example_ids": [str(i) for i in range(len(texts))]},
    )


class SelectionTests(unittest.TestCase):
    def test_exclusion_and_length_filters_are_applied(self) -> None:
        long_texts = [make_text(i) for i in range(10)]
        short_text = "too short"
        pool = fake_pool([short_text, *long_texts])
        excluded = {text_sha256(long_texts[0]), text_sha256(long_texts[3])}
        with mock.patch.object(
            heldout_splits, "load_domain_examples", return_value=pool
        ):
            selection = select_heldout_examples(
                "general", "development", 43, 4, excluded, FakeTokenizer()
            )
        self.assertEqual(len(selection.texts), 4)
        self.assertNotIn(long_texts[0], selection.texts)
        self.assertNotIn(long_texts[3], selection.texts)
        self.assertNotIn(short_text, selection.texts)
        self.assertEqual(selection.metadata["skipped_below_length"], 1)
        self.assertEqual(selection.metadata["skipped_excluded_or_duplicate"], 2)

    def test_duplicate_texts_are_used_once(self) -> None:
        repeated = make_text(1)
        pool = fake_pool([repeated, repeated, make_text(2), make_text(3)])
        with mock.patch.object(
            heldout_splits, "load_domain_examples", return_value=pool
        ):
            selection = select_heldout_examples(
                "general", "development", 43, 3, set(), FakeTokenizer()
            )
        self.assertEqual(len(set(selection.text_hashes)), 3)

    def test_insufficient_disjoint_examples_aborts(self) -> None:
        texts = [make_text(i) for i in range(3)]
        pool = fake_pool(texts)
        excluded = {text_sha256(texts[0])}
        with mock.patch.object(
            heldout_splits, "load_domain_examples", return_value=pool
        ):
            with self.assertRaises(RuntimeError):
                select_heldout_examples(
                    "general", "final", 44, 3, excluded, FakeTokenizer()
                )


class GeometryTests(unittest.TestCase):
    def test_prepared_split_matches_controlled_geometry(self) -> None:
        texts = [make_text(i) for i in range(4)]
        pool = fake_pool(texts)
        with mock.patch.object(
            heldout_splits, "load_domain_examples", return_value=pool
        ):
            selection = select_heldout_examples(
                "general", "development", 43, 4, set(), FakeTokenizer()
            )
        reference_row = np.asarray([0, 0, 0] + [1] * 64 + [0], dtype=np.uint8)
        prepared = prepare_heldout_split(selection, FakeTokenizer(), reference_row)
        self.assertEqual(prepared.sequence_length, 68)
        self.assertTrue(np.all(prepared.input_ids[:, :3] == (8982, 27, 187)))
        self.assertTrue(np.all(prepared.measurement_mask.sum(axis=1) == 64))
        keys = split_row_keys(prepared)
        self.assertEqual(len(set(keys)), 4)

    def test_development_and_final_disjointness_via_exclusion(self) -> None:
        texts = [make_text(i) for i in range(12)]
        pool = fake_pool(texts)
        tokenizer = FakeTokenizer()
        with mock.patch.object(
            heldout_splits, "load_domain_examples", return_value=pool
        ):
            development = select_heldout_examples(
                "general", "development", 43, 5, set(), tokenizer
            )
            final = select_heldout_examples(
                "general", "final", 44, 5, set(development.text_hashes), tokenizer
            )
        self.assertFalse(set(development.text_hashes) & set(final.text_hashes))


if __name__ == "__main__":
    unittest.main()
