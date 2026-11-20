"""Kiểm thử tối thiểu cho các phép tính cốt lõi bằng dữ liệu tổng hợp."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from two_sample_utils import effect_sizes, welch_test


class WelchTestTests(unittest.TestCase):
    def test_matches_scipy(self) -> None:
        x = np.array([1.0, 2.0, 2.0, 4.0, 8.0])
        y = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0])
        actual = welch_test(x, y)
        expected = stats.ttest_ind(x, y, equal_var=False)
        self.assertAlmostEqual(actual["t_statistic"], expected.statistic, places=12)
        self.assertAlmostEqual(actual["p_value"], expected.pvalue, places=12)
        self.assertAlmostEqual(actual["df"], expected.df, places=12)

    def test_difference_direction_and_ci(self) -> None:
        x = np.arange(20, dtype=float) + 10
        y = np.arange(20, dtype=float)
        result = welch_test(x, y)
        self.assertGreater(result["mean_difference"], 0)
        self.assertGreater(result["ci95_low"], 0)

    def test_effect_direction(self) -> None:
        x = np.array([3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = effect_sizes(x, y)
        self.assertGreater(result["cohen_d"], 0)
        self.assertLess(result["hedges_g"], result["cohen_d"])


if __name__ == "__main__":
    unittest.main()

