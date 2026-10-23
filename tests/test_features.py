"""
Unit and integration tests for feature engineering integrity and consistency.
"""

import unittest
import numpy as np
import pandas as pd

from src.features.modeling_dataset import get_feature_columns


class TestFeatureEngineering(unittest.TestCase):
    """Test suite verifying feature computation, separation, and physical bounds."""

    @classmethod
    def setUpClass(cls):
        cls.df_model = pd.read_csv("data/processed/modeling_dataset.csv")
        cls.feature_cols = get_feature_columns(cls.df_model)

    def test_feature_counts_and_naming_groups(self):
        """Verify that all feature columns belong to one of the four documented groups."""
        self.assertEqual(len(self.feature_cols), 34, "Expected exactly 34 engineered features.")
        init_cols = [c for c in self.feature_cols if c.startswith("init_")]
        snap_cols = [c for c in self.feature_cols if c.startswith("snap_")]
        traj_cols = [c for c in self.feature_cols if c.startswith("traj_")]
        curve_cols = [c for c in self.feature_cols if c.startswith("curve_")]

        self.assertEqual(len(init_cols), 11, "Expected 11 initial-condition features.")
        self.assertEqual(len(snap_cols), 7, "Expected 7 snapshot features.")
        self.assertEqual(len(traj_cols), 11, "Expected 11 trajectory features.")
        self.assertEqual(len(curve_cols), 5, "Expected 5 differential curve features.")

    def test_initial_features_are_invariant_to_cutoff(self):
        """Initial features (measured at cycle 2) must be invariant to observation cutoff."""
        eligible = self.df_model[self.df_model["is_benchmark_eligible"] == True]
        init_cols = [c for c in self.feature_cols if c.startswith("init_")]

        pivoted = eligible.pivot(index="cell_id", columns="cutoff", values="init_q_discharge")
        self.assertTrue(
            (pivoted[40] == pivoted[70]).all() and (pivoted[70] == pivoted[100]).all(),
            "init_q_discharge varies across observation cutoffs for the same cell!",
        )

        pivoted_ir = eligible.pivot(
            index="cell_id", columns="cutoff", values="init_internal_resistance"
        )
        self.assertTrue(
            (pivoted_ir[40] == pivoted_ir[70]).all() and (pivoted_ir[70] == pivoted_ir[100]).all(),
            "init_internal_resistance varies across observation cutoffs for the same cell!",
        )

    def test_no_null_features_in_eligible_cells(self):
        """All 34 features must be completely non-null and finite for all eligible cells."""
        eligible = self.df_model[self.df_model["is_benchmark_eligible"] == True]
        null_counts = eligible[self.feature_cols].isna().sum()
        self.assertEqual(
            null_counts.sum(),
            0,
            f"Found null values in eligible features:\n{null_counts[null_counts > 0]}",
        )

    def test_physical_plausibility_and_bounds(self):
        """Verify engineered features conform to physical battery operating bounds."""
        eligible = self.df_model[self.df_model["is_benchmark_eligible"] == True]

        # Capacity near nominal 1.1 Ah (A123 18650 LFP cell)
        self.assertTrue(
            (eligible["init_q_discharge"] >= 0.95).all()
            and (eligible["init_q_discharge"] <= 1.15).all(),
            "Initial discharge capacity outside plausible 0.95-1.15 Ah range!",
        )

        # Internal resistance positive (0.01 to 0.05 Ohms)
        self.assertTrue(
            (eligible["init_internal_resistance"] > 0.01).all()
            and (eligible["init_internal_resistance"] < 0.05).all(),
            "Initial internal resistance outside plausible range!",
        )

        # Delta Q(V) log-variance negative and finite
        self.assertTrue(
            (eligible["curve_delta_q_var_log10"] < -1.0).all()
            and (eligible["curve_delta_q_var_log10"] > -8.0).all(),
            "Differential curve log10 variance outside plausible range!",
        )

    def test_post_cutoff_measurements_do_not_leak_into_features(self):
        """Rule: Modifying measurements after cutoff leaves extracted features strictly invariant."""
        from src.features.feature_engineering import FeatureEngineer

        # Create synthetic cycle series up to 200 cycles
        cycles = np.arange(1, 201)
        base_df = pd.DataFrame({
            "cycle": cycles,
            "QDischarge": 1.05 - 0.0001 * cycles,
            "QCharge": 1.06 - 0.0001 * cycles,
            "IR": 0.015 + 0.00001 * cycles,
            "Tmax": 35.0 + 0.01 * cycles,
            "Tavg": 32.0 + 0.005 * cycles,
            "Tmin": 30.0 + 0.002 * cycles,
            "chargetime": 45.0 - 0.01 * cycles,
        })
        meta = pd.Series({"policy": "3.6C(80%)-3.6C"})
        curve_dummy = {
            "curve_delta_q_var_log10": -4.0,
            "curve_delta_q_min": -0.05,
            "curve_delta_q_mean": -0.02,
            "curve_delta_q_skew": 0.1,
            "curve_delta_q_kurtosis": 2.5,
        }

        # Extract features at cutoff 40 on clean data
        feats_original = FeatureEngineer.extract_cell_features_at_cutoff(
            base_df, meta, cutoff=40, curve_features=curve_dummy
        )

        # Corrupt data strictly after cutoff (cycle > 40)
        corrupted_df = base_df.copy()
        corrupted_df.loc[corrupted_df["cycle"] > 40, "QDischarge"] = 0.001
        corrupted_df.loc[corrupted_df["cycle"] > 40, "IR"] = 999.0
        corrupted_df.loc[corrupted_df["cycle"] > 40, "Tmax"] = 1000.0

        feats_after_corruption = FeatureEngineer.extract_cell_features_at_cutoff(
            corrupted_df, meta, cutoff=40, curve_features=curve_dummy
        )

        for k in feats_original:
            val_orig = feats_original[k]
            val_corrupt = feats_after_corruption[k]
            if np.isnan(val_orig):
                self.assertTrue(np.isnan(val_corrupt))
            else:
                self.assertEqual(
                    val_orig,
                    val_corrupt,
                    f"Feature '{k}' changed when post-cutoff cycle data was modified!",
                )



if __name__ == "__main__":
    unittest.main()
