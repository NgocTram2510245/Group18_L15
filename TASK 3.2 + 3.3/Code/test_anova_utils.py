"""Smoke tests bang du lieu tong hop, khong thay the kiem thtuêxáu tren NSL-KDD."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from anova_utils import GROUP_ORDER, duration_two_part, games_howell, run_omnibus
from report_assets import REPORT_FIGURE_STEMS


class TestAnovaUtilities(unittest.TestCase):
    def test_games_howell_returns_ten_pairs(self) -> None:
        rng = np.random.default_rng(7)
        samples = {group: rng.normal(i, 1 + i / 4, 80) for i, group in enumerate(GROUP_ORDER)}
        result = games_howell(samples)
        self.assertEqual(len(result), 10)
        self.assertTrue(result["adjusted_p_value"].between(0, 1).all())
        self.assertTrue((result["ci95_lower"] <= result["ci95_upper"]).all())

    def test_known_group_shift_is_detected(self) -> None:
        rng = np.random.default_rng(11)
        rows = []
        for i, group in enumerate(GROUP_ORDER):
            n = 100 if group != "U2R" else 60
            for value in rng.normal(i * 1.5, 1, n):
                rows.append(
                    {
                        "attack_category": group,
                        "count": max(0, value + 10),
                        "serror_rate": np.clip(value / 10 + 0.2, 0, 1),
                        "src_bytes": np.expm1(max(0, value + 5)),
                        "duration": np.expm1(max(0, value + 3)),
                    }
                )
        data = pd.DataFrame(rows)
        data["attack_category"] = pd.Categorical(data["attack_category"], GROUP_ORDER, ordered=True)
        omnibus, posthoc, sensitivity = run_omnibus(data)
        self.assertEqual(len(omnibus), 4)
        self.assertTrue(omnibus["reject_h0_holm"].all())
        self.assertEqual(len(posthoc), 40)
        self.assertEqual(len(sensitivity), 8)

    def test_duration_zero_output_is_descriptive_only(self) -> None:
        rows = []
        for group_index, group in enumerate(GROUP_ORDER):
            for index in range(20):
                rows.append(
                    {
                        "attack_category": group,
                        "duration": 0 if index < group_index + 2 else index + 1,
                    }
                )
        data = pd.DataFrame(rows)
        zero, positive = duration_two_part(data)
        self.assertEqual(len(zero), 5)
        self.assertNotIn("chi2_overall", zero.columns)
        self.assertTrue(zero["zero_proportion"].between(0, 1).all())
        self.assertEqual(len(positive), 1)

    def test_report_figure_contract(self) -> None:
        self.assertEqual(len(REPORT_FIGURE_STEMS), 5)
        self.assertEqual(len(set(REPORT_FIGURE_STEMS)), 5)


if __name__ == "__main__":
    unittest.main()

