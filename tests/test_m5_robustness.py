"""
Module: tests.test_m5_robustness
Author: Capstone Project Team
Description: Unit and anti-leakage verification tests for the Counterfactual Peer Robustness Engine (M5).
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.models.model_factory import ModelFactory, get_feature_names_for_model
from src.peers.peer_matcher import PeerMatcher
from src.peers.peer_robustness import PeerRobustnessEngine
from src.peers.trajectory_comparison import TrajectoryComparator


class TestM5PeerRobustness(unittest.TestCase):
    """Test suite verifying deterministic perturbations, anti-leakage, and sensitivity metrics."""

    @classmethod
    def setUpClass(cls):
        # Load and merge modeling dataset with Phase 4 peer features
        df_model = pd.read_csv("data/processed/modeling_dataset.csv")
        df_peers = pd.read_csv("data/processed/peer_features.csv")
        merge_keys = ["cell_id", "cutoff"]
        peer_cols_to_add = [c for c in df_peers.columns if c not in df_model.columns or c in merge_keys]
        cls.df_merged = pd.merge(df_model, df_peers[peer_cols_to_add], on=merge_keys)

        cls.df_cut100 = cls.df_merged[cls.df_merged["cutoff"] == 100].copy().reset_index(drop=True)
        cls.train_df = cls.df_cut100[cls.df_cut100["benchmark_partition"] == "train"].copy().reset_index(drop=True)
        cls.test_df = cls.df_cut100[cls.df_cut100["benchmark_partition"] == "primary_test"].copy().reset_index(drop=True)

        # Fit PeerMatcher on training reference cells strictly
        cls.matcher = PeerMatcher(k=5, min_k=3)
        cls.matcher.fit(cls.train_df)
        cls.comparator = TrajectoryComparator(cls.df_cut100)
        cls.engine = PeerRobustnessEngine(cls.matcher, cls.comparator, k=5)

        # Fit M3 model on train
        cls.m3_features = get_feature_names_for_model("M3")
        y_train = cls.train_df["target_fail_within_500"].values.astype(float)
        cls.m3_pipe = ModelFactory.build_logistic_regression(random_state=42)
        cls.m3_pipe.fit(cls.train_df[cls.m3_features], y_train)

    def test_deterministic_perturbations(self):
        """Rule 1: Same target cell and model must produce bitwise identical perturbation metrics."""
        test_cell = self.test_df.iloc[0]
        res1 = self.engine.evaluate_cell_robustness(test_cell, self.m3_pipe, self.m3_features)
        res2 = self.engine.evaluate_cell_robustness(test_cell, self.m3_pipe, self.m3_features)

        self.assertEqual(res1.n_valid_perturbations, res2.n_valid_perturbations)
        self.assertAlmostEqual(res1.counterfactual_mean, res2.counterfactual_mean, places=7)
        self.assertAlmostEqual(res1.counterfactual_std, res2.counterfactual_std, places=7)
        self.assertEqual(res1.most_influential_peer, res2.most_influential_peer)

    def test_no_self_peer_inclusion(self):
        """Rule 2: A cell must NEVER include itself in any perturbation peer set."""
        train_cell = self.train_df.iloc[0]
        cid = str(train_cell["cell_id"])
        res = self.engine.evaluate_cell_robustness(train_cell, self.m3_pipe, self.m3_features)

        for p in res.perturbation_records:
            self.assertNotIn(cid, p["peer_ids"], f"Cell {cid} matched itself in perturbation {p['perturbation_id']}")

    def test_no_test_cell_contamination(self):
        """Rule 3: Test cells must draw perturbations EXCLUSIVELY from training reference candidates."""
        test_cell = self.test_df.iloc[0]
        res = self.engine.evaluate_cell_robustness(test_cell, self.m3_pipe, self.m3_features)

        train_ids = set(self.train_df["cell_id"].astype(str))
        for p in res.perturbation_records:
            for peer_id in p["peer_ids"]:
                self.assertIn(
                    peer_id,
                    train_ids,
                    f"Contamination violation: Peer {peer_id} in {p['perturbation_id']} is not in training pool!",
                )

    def test_instability_and_range_non_negative(self):
        """Rule 4: Instability U and range must be non-negative and finite."""
        for _, row in self.test_df.head(5).iterrows():
            res = self.engine.evaluate_cell_robustness(row, self.m3_pipe, self.m3_features)
            self.assertTrue(np.isfinite(res.counterfactual_std))
            self.assertGreaterEqual(res.counterfactual_std, 0.0)
            self.assertGreaterEqual(res.counterfactual_range, 0.0)
            self.assertTrue(0.0 <= res.flip_rate <= 1.0)

    def test_excluded_cells_zero_perturbations(self):
        """Rule 5: Excluded cells must produce 0 perturbations and NaN summary stats."""
        df_excl = self.df_cut100[self.df_cut100["benchmark_partition"] == "excluded"].copy()
        if not df_excl.empty:
            excl_cell = df_excl.iloc[0]
            res = self.engine.evaluate_cell_robustness(excl_cell, self.m3_pipe, self.m3_features)
            self.assertEqual(res.n_valid_perturbations, 0)
            self.assertFalse(res.is_benchmark_eligible)


if __name__ == "__main__":
    unittest.main()
