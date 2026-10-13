"""
Unit and integration tests for prediction target building and labeling.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.target_builder import TargetBuilder


class TestTargetBuilder(unittest.TestCase):
    """Test suite for target computation, horizon labeling, and censoring handling."""

    @classmethod
    def setUpClass(cls):
        cls.df_model = pd.read_csv("data/processed/modeling_dataset.csv")

    def test_eligible_targets_are_finite_and_consistent(self):
        """Verify that eligible cells have valid targets and RUL = EOL - cutoff."""
        eligible = self.df_model[self.df_model["is_benchmark_eligible"] == True]
        self.assertEqual(len(eligible), 372, "Expected 372 eligible rows (124 * 3).")

        # RUL consistency
        rul_diff = eligible["target_rul"] - (eligible["target_eol_cycle"] - eligible["cutoff"])
        self.assertTrue(
            (rul_diff.abs() < 1e-6).all(),
            "target_rul does not equal target_eol_cycle - cutoff!",
        )

        # Check binary labels match horizon limits
        for cutoff in [40, 70, 100]:
            sub = eligible[eligible["cutoff"] == cutoff]
            # Horizon 300
            exp_300 = (sub["target_eol_cycle"] <= cutoff + 300).astype(int)
            self.assertTrue(
                (sub["target_fail_within_300"] == exp_300).all(),
                f"Mismatch in target_fail_within_300 for cutoff {cutoff}",
            )
            # Horizon 500
            exp_500 = (sub["target_eol_cycle"] <= cutoff + 500).astype(int)
            self.assertTrue(
                (sub["target_fail_within_500"] == exp_500).all(),
                f"Mismatch in target_fail_within_500 for cutoff {cutoff}",
            )
            # Early quartile 700
            exp_700 = (sub["target_eol_cycle"] <= 700).astype(int)
            self.assertTrue(
                (sub["target_fail_by_700"] == exp_700).all(),
                f"Mismatch in target_fail_by_700 for cutoff {cutoff}",
            )

    def test_censored_targets_for_excluded_cells(self):
        """Verify that excluded cells are strictly labeled as censored with NaN values."""
        excluded = self.df_model[self.df_model["is_benchmark_eligible"] == False]
        self.assertEqual(len(excluded), 48, "Expected 48 excluded rows (16 * 3).")

        self.assertTrue(
            (excluded["status_fail_within_500"] == "censored").all(),
            "Non-eligible cell marked uncensored!",
        )
        self.assertTrue(
            excluded["target_fail_within_500"].isna().all(),
            "Non-eligible cell has non-NaN target value!",
        )

    def test_target_class_counts(self):
        """Verify exact positive/negative counts for verified benchmark cohort."""
        eligible = self.df_model[self.df_model["is_benchmark_eligible"] == True]

        for cutoff, expected_pos_500 in [(40, 42), (70, 45), (100, 46)]:
            sub = eligible[eligible["cutoff"] == cutoff]
            pos_count = int((sub["target_fail_within_500"] == 1).sum())
            neg_count = int((sub["target_fail_within_500"] == 0).sum())
            self.assertEqual(
                pos_count,
                expected_pos_500,
                f"Unexpected pos_count for cutoff {cutoff}: {pos_count} != {expected_pos_500}",
            )
            self.assertEqual(pos_count + neg_count, 124)


if __name__ == "__main__":
    unittest.main()
