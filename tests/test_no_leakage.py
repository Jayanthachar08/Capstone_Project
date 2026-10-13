"""
Unit and integration tests for data leakage prevention and split integrity.
"""

import json
import unittest
import numpy as np
import pandas as pd

from src.features.modeling_dataset import get_feature_columns, get_target_columns


class TestLeakagePrevention(unittest.TestCase):
    """Test suite verifying strict prevention of data leakage across partitions, cutoffs, and targets."""

    @classmethod
    def setUpClass(cls):
        cls.df_model = pd.read_csv("data/processed/modeling_dataset.csv")
        with open("data/splits/splits_benchmark.json") as f:
            cls.splits = json.load(f)
        cls.feature_cols = get_feature_columns(cls.df_model)
        cls.target_cols = get_target_columns(cls.df_model)

    def test_battery_split_invariance_across_cutoffs(self):
        """Rule 1: All rows from the same battery across all cutoffs must belong to the exact same split."""
        for cid, group in self.df_model.groupby("cell_id"):
            partitions = group["benchmark_partition"].unique()
            self.assertEqual(
                len(partitions),
                1,
                f"Cell {cid} is assigned to multiple splits across cutoffs: {partitions}",
            )

    def test_no_split_overlap(self):
        """Rule 2: No battery identity overlaps between any pair of splits."""
        train = set(self.splits["train"])
        primary_test = set(self.splits["primary_test"])
        secondary_test = set(self.splits["secondary_test"])
        excluded = set(self.splits["excluded"])

        self.assertTrue(train.isdisjoint(primary_test), "Leakage: Train and Primary Test overlap!")
        self.assertTrue(train.isdisjoint(secondary_test), "Leakage: Train and Secondary Test overlap!")
        self.assertTrue(primary_test.isdisjoint(secondary_test), "Leakage: Primary and Secondary Test overlap!")
        self.assertTrue(train.isdisjoint(excluded), "Leakage: Train and Excluded overlap!")

    def test_features_do_not_contain_target_leakage(self):
        """Rule 4: Input feature columns must never contain target or future cycle life information."""
        forbidden_keywords = ["eol", "life", "rul", "target", "status", "fail"]
        for fcol in self.feature_cols:
            fcol_lower = fcol.lower()
            for kw in forbidden_keywords:
                self.assertNotIn(
                    kw,
                    fcol_lower,
                    f"Potential leakage in feature column: '{fcol}' contains forbidden keyword '{kw}'!",
                )

    def test_disjoint_feature_and_target_sets(self):
        """Verify that feature columns and target columns have zero intersection."""
        feat_set = set(self.feature_cols)
        targ_set = set(self.target_cols)
        self.assertTrue(
            feat_set.isdisjoint(targ_set),
            f"Leakage: Features and Targets overlap on: {feat_set & targ_set}",
        )

    def test_phase2_splits_preserved_faithfully(self):
        """Rule 8: Benchmark split assignments match Phase 2 split definitions exactly."""
        for split_name in ["train", "primary_test", "secondary_test", "excluded"]:
            expected_cells = set(self.splits[split_name])
            actual_cells = set(
                self.df_model[self.df_model["benchmark_partition"] == split_name]["cell_id"]
            )
            self.assertEqual(
                actual_cells,
                expected_cells,
                f"Split mismatch for '{split_name}': {actual_cells ^ expected_cells}",
            )


if __name__ == "__main__":
    unittest.main()
