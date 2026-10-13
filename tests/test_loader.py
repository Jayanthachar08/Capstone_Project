"""
Unit and integration tests for data loading, quality filtering, and split management.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd

from src.data.loader import (
    BatteryDataLoader,
    BATCH_FILES,
    EXCLUSION_REGISTRY,
    CARRYOVER_MAPPING,
    load_cells_metadata,
    load_benchmark_splits,
)


class TestBatteryDataLoader(unittest.TestCase):
    """Test suite verifying data extraction, validation, and split integrity."""

    @classmethod
    def setUpClass(cls):
        """Ensures the extractor has executed and output artifacts exist."""
        cls.loader = BatteryDataLoader()
        meta_path = cls.loader.output_dir / "cells_metadata.csv"
        splits_path = cls.loader.splits_dir / "splits_benchmark.json"

        # If artifacts do not exist, run extraction
        if not meta_path.exists() or not splits_path.exists():
            cls.loader.extract_and_export_all()

        cls.df_meta = load_cells_metadata(str(meta_path))
        cls.splits = load_benchmark_splits(str(splits_path))

    def test_raw_dataset_files_exist(self):
        """Verify that all three canonical MATR batch files exist and are non-empty."""
        for bname, fname in BATCH_FILES.items():
            fpath = os.path.join("Datasets", fname)
            self.assertTrue(os.path.exists(fpath), f"Missing raw file: {fpath}")
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            self.assertGreater(size_mb, 1500, f"File {fpath} is suspiciously small: {size_mb} MB")

    def test_battery_identity_uniqueness_and_counts(self):
        """Verify total raw count is 140 and each internal cell_id is globally unique."""
        self.assertEqual(len(self.df_meta), 140, "Expected exactly 140 raw battery records.")
        unique_ids = self.df_meta["cell_id"].unique()
        self.assertEqual(len(unique_ids), 140, "Duplicate cell_id found in metadata!")

        # Batch counts
        b1_count = len(self.df_meta[self.df_meta["batch"] == "batch1"])
        b2_count = len(self.df_meta[self.df_meta["batch"] == "batch2"])
        b3_count = len(self.df_meta[self.df_meta["batch"] == "batch3"])
        self.assertEqual(b1_count, 46, "Batch 1 must contain 46 cells.")
        self.assertEqual(b2_count, 48, "Batch 2 must contain 48 cells.")
        self.assertEqual(b3_count, 46, "Batch 3 must contain 46 cells.")

    def test_quality_filter_and_exclusions(self):
        """Verify the 16 authoritative benchmark exclusions and 124 retained cells."""
        excluded_df = self.df_meta[self.df_meta["is_benchmark_eligible"] == False]
        eligible_df = self.df_meta[self.df_meta["is_benchmark_eligible"] == True]

        self.assertEqual(len(excluded_df), 16, "Expected exactly 16 excluded cells.")
        self.assertEqual(len(eligible_df), 124, "Expected exactly 124 eligible benchmark cells.")

        # Check every excluded cell matches the authoritative registry
        for cell_id in EXCLUSION_REGISTRY.keys():
            self.assertIn(
                cell_id,
                excluded_df["cell_id"].values,
                f"Expected excluded cell {cell_id} not flagged as excluded!",
            )

    def test_benchmark_splits_leakage_safety(self):
        """Verify that train, primary_test, and secondary_test are pairwise disjoint and sum to 124."""
        train = set(self.splits["train"])
        primary_test = set(self.splits["primary_test"])
        secondary_test = set(self.splits["secondary_test"])
        excluded = set(self.splits["excluded"])

        self.assertEqual(len(train), 41, "Train reference pool must have 41 cells.")
        self.assertEqual(len(primary_test), 43, "Primary test set must have 43 cells.")
        self.assertEqual(len(secondary_test), 40, "Secondary test set must have 40 cells.")
        self.assertEqual(len(excluded), 16, "Excluded set must have 16 cells.")

        # Disjoint checks (leakage prevention)
        self.assertTrue(train.isdisjoint(primary_test), "Train and Primary Test overlap!")
        self.assertTrue(train.isdisjoint(secondary_test), "Train and Secondary Test overlap!")
        self.assertTrue(primary_test.isdisjoint(secondary_test), "Primary and Secondary Test overlap!")
        self.assertTrue(train.isdisjoint(excluded), "Train and Excluded overlap!")

        # Union check
        total_eligible = train | primary_test | secondary_test
        self.assertEqual(len(total_eligible), 124, "Union of splits does not equal 124 cells.")

    def test_cycle_life_validity_for_eligible_cells(self):
        """Verify all 124 eligible cells have finite, positive cycle life values."""
        eligible_df = self.df_meta[self.df_meta["is_benchmark_eligible"] == True]
        self.assertFalse(
            eligible_df["benchmark_cycle_life"].isna().any(),
            "Found NaN benchmark_cycle_life in eligible cells!",
        )
        self.assertTrue(
            (eligible_df["benchmark_cycle_life"] > 0).all(),
            "Found non-positive cycle life in eligible cells!",
        )

    def test_carryover_adjustments(self):
        """Verify the 5 resumed cells have their continuation cycles reflected."""
        for b1_id, (b2_id, add_l) in CARRYOVER_MAPPING.items():
            b1_row = self.df_meta[self.df_meta["cell_id"] == b1_id].iloc[0]
            raw_life = b1_row["raw_cycle_life"]
            adj_life = b1_row["benchmark_cycle_life"]
            self.assertEqual(
                adj_life,
                raw_life + add_l,
                f"Carryover addition mismatch for {b1_id}: {adj_life} != {raw_life} + {add_l}",
            )


if __name__ == "__main__":
    unittest.main()
