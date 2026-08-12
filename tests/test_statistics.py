from __future__ import annotations

import unittest

import numpy as np

from expert_analysis.metrics import DomainStatistics
from expert_analysis.statistics import (
    bootstrap_spearman_pair,
    descending_ranks,
    topk_similarity,
    topk_size,
)


class StatisticsTests(unittest.TestCase):
    def test_descending_ranks_with_ties(self) -> None:
        np.testing.assert_allclose(
            descending_ranks(np.asarray([1.0, 3.0, 3.0, 0.0])),
            [3.0, 1.5, 1.5, 4.0],
        )

    def test_topk_similarity(self) -> None:
        first = np.asarray([4.0, 3.0, 2.0, 1.0])
        second = np.asarray([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(topk_size(4, 0.25), 1)
        k, intersection, overlap, jaccard = topk_similarity(first, second, 0.5)
        self.assertEqual((k, intersection), (2, 1))
        self.assertAlmostEqual(overlap, 0.5)
        self.assertAlmostEqual(jaccard, 1 / 3)

    def test_bootstrap_shape_and_reproducibility(self) -> None:
        first = DomainStatistics.zeros(6, 2, 4, ["a", "b"])
        second = DomainStatistics.zeros(6, 2, 4, ["a", "b"])
        base = np.arange(6 * 2 * 4).reshape(6, 2, 4) % 7 + 1
        first.routing_counts[:] = base
        second.routing_counts[:] = base[:, :, ::-1]
        first.token_counts[:] = 10
        second.token_counts[:] = 10
        one = bootstrap_spearman_pair(
            first, second, "routing_frequency", 8, np.random.default_rng(5)
        )
        two = bootstrap_spearman_pair(
            first, second, "routing_frequency", 8, np.random.default_rng(5)
        )
        self.assertEqual(one.shape, (8, 2))
        np.testing.assert_allclose(one, two, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
