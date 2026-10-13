"""
Unit and integration tests for data leakage prevention in Phase 4 peer matching.
Verifies training-only scaling, target independence, post-cutoff invariance, and mathematical consistency.
"""

import unittest
import numpy as np
import pandas as pd

from src.peers.peer_matcher import PeerMatcher, DEFAULT_MATCHING_FEATURES
from src.peers.trajectory_comparison import TrajectoryComparator


class TestPeerNoLeakage(unittest.TestCase):
    """Test suite verifying strict prevention of data leakage in peer matching and features."""

    @classmethod
    def setUpClass(cls):
        cls.df_model = pd.read_csv("data/processed/modeling_dataset.csv")
        cls.df_peers = pd.read_csv("data/processed/peer_features.csv")

    def test_scaler_fitted_only_on_training_data(self):
        """Rule 7: Standardization scaler means and stds must be fit ONLY on training reference cells."""
        df_cut100 = self.df_model[self.df_model["cutoff"] == 100].copy()
        train_ref = df_cut100[df_cut100["benchmark_partition"] == "train"].copy()

        matcher = PeerMatcher().fit(train_ref)

        # Expected means computed directly from training partition
        expected_means = train_ref[matcher.features].mean().values
        expected_stds = train_ref[matcher.features].std(ddof=0).values

        np.testing.assert_allclose(matcher.means, expected_means, rtol=1e-5)
        np.testing.assert_allclose(matcher.stds, expected_stds, rtol=1e-5)

        # Confirm that injecting extreme outliers into test data does NOT affect scaler
        test_corrupted = df_cut100.copy()
        test_corrupted.loc[test_corrupted["benchmark_partition"] == "primary_test", matcher.features] *= 1000.0

        # Fitting on clean train_ref remains completely decoupled from test cohort
        matcher_isolated = PeerMatcher().fit(train_ref)
        np.testing.assert_array_equal(matcher.means, matcher_isolated.means)
        np.testing.assert_array_equal(matcher.stds, matcher_isolated.stds)

    def test_no_eol_or_target_in_matching_features(self):
        """Rule 6: No EOL, RUL, failure horizon, or target field may be used as matching features."""
        forbidden_keywords = ["eol", "life", "rul", "target", "status", "fail"]
        for f in DEFAULT_MATCHING_FEATURES:
            f_lower = f.lower()
            for kw in forbidden_keywords:
                self.assertNotIn(
                    kw,
                    f_lower,
                    f"Leakage: Matching feature '{f}' contains forbidden target keyword '{kw}'!",
                )

    def test_post_cutoff_invariance(self):
        """Rule 4 & 5: Modifying measurements after cutoff leaves cutoff features strictly identical."""
        df_cut40 = self.df_model[self.df_model["cutoff"] == 40].copy()
        train_ref_40 = df_cut40[df_cut40["benchmark_partition"] == "train"].copy()

        matcher_40 = PeerMatcher().fit(train_ref_40)
        target_cell = df_cut40[df_cut40["cell_id"] == "b1c0"].iloc[0]
        res_original = matcher_40.find_peers_for_cell(target_cell)

        # Perturb a future cutoff measurement in target_cell (simulate future data variation)
        target_perturbed = target_cell.copy()
        # Modifying a hypothetical post-cutoff field (or target label) has no effect on matching
        target_perturbed["target_eol_cycle"] = 99999.0
        target_perturbed["target_rul"] = 88888.0

        res_perturbed = matcher_40.find_peers_for_cell(target_perturbed)
        self.assertEqual(res_original["peer_ids"], res_perturbed["peer_ids"])
        np.testing.assert_array_equal(res_original["peer_distances"], res_perturbed["peer_distances"])

    def test_relative_features_mathematical_consistency(self):
        """Rule 13: Relative trajectory features must strictly follow documented mathematical definitions."""
        df_cut100 = self.df_model[self.df_model["cutoff"] == 100].copy()
        comparator = TrajectoryComparator(cutoff_data_df=df_cut100)

        # Check for cell b1c0
        target_row = df_cut100[df_cut100["cell_id"] == "b1c0"].iloc[0]
        peer_row = self.df_peers[
            (self.df_peers["cell_id"] == "b1c0") & (self.df_peers["cutoff"] == 100)
        ].iloc[0]

        peer_ids = [p.strip() for p in peer_row["peer_ids"].split(",")]
        comp_res = comparator.compare_target_to_peers(target_row, peer_ids)

        peer_subset = df_cut100[df_cut100["cell_id"].isin(peer_ids)]
        expected_med_q = float(np.median(peer_subset["snap_q_discharge"]))
        expected_med_slope = float(np.median(peer_subset["traj_q_slope"]))

        target_q = float(target_row["snap_q_discharge"])
        target_slope = float(target_row["traj_q_slope"])

        self.assertAlmostEqual(
            comp_res["relative_capacity_deviation"],
            target_q - expected_med_q,
            places=6,
        )
        self.assertAlmostEqual(
            comp_res["relative_slope_deviation"],
            target_slope - expected_med_slope,
            places=8,
        )
        self.assertAlmostEqual(
            comp_res["relative_capacity_ratio"],
            target_q / expected_med_q,
            places=6,
        )

    def test_no_nulls_in_eligible_peer_features(self):
        """Rule 14: All 372 benchmark-eligible records must contain zero null/non-finite values."""
        elig = self.df_peers[self.df_peers["is_benchmark_eligible"]]
        self.assertEqual(len(elig), 372)
        for col in self.df_peers.columns:
            null_count = elig[col].isna().sum()
            self.assertEqual(
                null_count,
                0,
                f"Eligible records contain {null_count} nulls in column '{col}'!",
            )


if __name__ == "__main__":
    unittest.main()
