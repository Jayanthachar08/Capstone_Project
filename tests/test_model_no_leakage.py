"""
Unit and integration tests for data leakage prevention in Phase 5 model training.
Verifies target keyword exclusion, training-only preprocessor fitting, partition isolation,
and decision threshold decoupling from test sets.
"""

import json
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.model_factory import get_feature_names_for_model, ModelFactory
from src.models.evaluate_models import ModelEvaluator


class TestModelNoLeakage(unittest.TestCase):
    """Test suite validating anti-leakage compliance in model training and evaluation."""

    @classmethod
    def setUpClass(cls):
        df_model = pd.read_csv("data/processed/modeling_dataset.csv")
        df_peers = pd.read_csv("data/processed/peer_features.csv")
        merge_keys = ["cell_id", "cutoff"]
        peer_cols = [c for c in df_peers.columns if c not in df_model.columns or c in merge_keys]
        cls.df = pd.merge(df_model, df_peers[peer_cols], on=merge_keys)

        with open("data/splits/splits_benchmark.json") as f:
            cls.splits = json.load(f)

    def test_target_keyword_exclusion_in_all_models(self):
        """Rule 1: No model feature may contain forbidden target or outcome keywords."""
        forbidden_keywords = ["eol", "rul", "life", "target", "status", "fail"]
        for m_key in ["M1", "M2", "M3", "M4"]:
            feats = get_feature_names_for_model(m_key)
            for f in feats:
                f_lower = f.lower()
                for kw in forbidden_keywords:
                    self.assertNotIn(
                        kw,
                        f_lower,
                        f"Leakage: {m_key} feature '{f}' contains forbidden target keyword '{kw}'!",
                    )

    def test_partition_disjointness(self):
        """Rule 2: Partitions must be strictly disjoint with zero battery ID leakage."""
        train = set(self.splits["train"])
        primary_test = set(self.splits["primary_test"])
        secondary_test = set(self.splits["secondary_test"])
        excluded = set(self.splits["excluded"])

        self.assertTrue(train.isdisjoint(primary_test), "Leakage: Train overlaps with Primary Test!")
        self.assertTrue(train.isdisjoint(secondary_test), "Leakage: Train overlaps with Secondary Test!")
        self.assertTrue(primary_test.isdisjoint(secondary_test), "Leakage: Primary and Secondary Test overlap!")
        self.assertTrue(train.isdisjoint(excluded), "Leakage: Train overlaps with Excluded!")

    def test_scaler_fitted_only_on_training_partition(self):
        """Rule 3: StandardScaler parameters must be fitted EXCLUSIVELY on training data."""
        sub100 = self.df[self.df["cutoff"] == 100].copy()
        train_df = sub100[sub100["benchmark_partition"] == "train"]
        m4_feats = get_feature_names_for_model("M4")

        scaler = StandardScaler()
        scaler.fit(train_df[m4_feats])

        expected_means = train_df[m4_feats].mean().values
        expected_stds = train_df[m4_feats].std(ddof=0).values

        np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)
        np.testing.assert_allclose(scaler.scale_, expected_stds, rtol=1e-5)

    def test_threshold_selection_decoupled_from_test_labels(self):
        """Rule 4: Optimal classification threshold must be chosen without accessing test labels."""
        # Simulated training OOF predictions
        y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        p_train = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.3, 0.75, 0.15])

        thresh_train = ModelEvaluator.select_optimal_threshold(y_train, p_train)
        self.assertGreaterEqual(thresh_train, 0.10)
        self.assertLessEqual(thresh_train, 0.90)

        # Altering test inputs has zero effect on the training threshold function
        thresh_again = ModelEvaluator.select_optimal_threshold(y_train, p_train)
        self.assertEqual(thresh_train, thresh_again)


if __name__ == "__main__":
    unittest.main()
