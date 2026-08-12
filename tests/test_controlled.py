from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from expert_analysis.controlled import PreparedDomainExamples, prepare_controlled_domains
from expert_analysis.datasets import DomainExamples


class CharacterTokenizer:
    def __call__(
        self,
        texts,
        add_special_tokens: bool = False,
        truncation: bool = False,
        padding: bool = False,
    ):
        del truncation, padding
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        encoded = [[3 + ord(character) % 97 for character in text] for text in items]
        if add_special_tokens:
            encoded = [self.build_inputs_with_special_tokens(item) for item in encoded]
        return {"input_ids": encoded[0] if single else encoded}

    @staticmethod
    def build_inputs_with_special_tokens(token_ids):
        return [1, *token_ids, 2]

    @staticmethod
    def get_special_tokens_mask(token_ids, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [int(item in (1, 2)) for item in token_ids]
        return [1, *([0] * len(token_ids)), 1]


class NoSpecialTokenBuilder(CharacterTokenizer):
    build_inputs_with_special_tokens = None

    @staticmethod
    def num_special_tokens_to_add(pair=False):
        del pair
        return 0


class ControlledInputTests(unittest.TestCase):
    def test_exact_shared_prefix_length_and_budget(self) -> None:
        candidates = {}
        for domain in ("general", "math", "coding", "reasoning"):
            texts = ["short", "abcdefghijk", "lmnopqrstuv", "wxyzABCDEFG"]
            candidates[domain] = DomainExamples(
                domain=domain,
                texts=texts,
                metadata={
                    "repository": f"test/{domain}",
                    "config": "main",
                    "split": "test",
                    "requested_examples": 4,
                    "selected_example_ids": [f"{domain}-{index}" for index in range(4)],
                },
            )
        prepared, manifest = prepare_controlled_domains(
            candidates,
            CharacterTokenizer(),
            num_examples=2,
            measured_tokens_per_example=6,
            neutral_prefix="Input:\n",
            max_length=32,
        )
        self.assertTrue(manifest["same_prefix_token_ids"])
        self.assertTrue(manifest["same_model_sequence_length"])
        self.assertTrue(manifest["same_measurement_length_distribution"])
        self.assertTrue(manifest["same_total_measurement_budget"])
        self.assertEqual(manifest["measured_tokens_per_domain"], 12)
        expected_prefix = manifest["neutral_prefix_token_ids"]
        sequence_lengths = set()
        for domain, item in prepared.items():
            item.validate()
            sequence_lengths.add(item.sequence_length)
            np.testing.assert_array_equal(item.measurement_mask.sum(axis=1), [6, 6])
            positions = item.metadata["control"]["prefix_model_positions"]
            self.assertEqual(item.input_ids[0, positions].tolist(), expected_prefix)
            self.assertNotIn(f"{domain.title()} problem", "Input:\n")
            self.assertEqual(len(item.metadata["selected_example_ids"]), 2)
        self.assertEqual(len(sequence_lengths), 1)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "general.npz"
            prepared["general"].save(path)
            loaded = PreparedDomainExamples.load(
                path, "general", prepared["general"].metadata
            )
            np.testing.assert_array_equal(
                loaded.measurement_mask, prepared["general"].measurement_mask
            )

    def test_tokenizer_without_sequence_builder_when_no_specials_are_added(self) -> None:
        domain = DomainExamples(
            domain="general",
            texts=["abcdefghijk", "lmnopqrstuv"],
            metadata={
                "repository": "test/general",
                "split": "test",
                "requested_examples": 2,
                "selected_example_ids": ["0", "1"],
            },
        )
        prepared, manifest = prepare_controlled_domains(
            {"general": domain},
            NoSpecialTokenBuilder(),
            num_examples=2,
            measured_tokens_per_example=6,
            neutral_prefix="Input:\n",
            max_length=32,
        )
        self.assertEqual(prepared["general"].sequence_length, 14)
        self.assertEqual(manifest["measured_tokens_per_domain"], 12)


if __name__ == "__main__":
    unittest.main()
