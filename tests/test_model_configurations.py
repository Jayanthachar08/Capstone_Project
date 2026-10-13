"""
Unit tests for Phase 5 model configurations and feature hierarchy.
Verifies exact feature counts, group memberships, orthogonality, and pipeline creation.
"""

import unittest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.models.model_factory import (
    FEATURE_GROUPS,
    MODEL_CONFIGS,
    ModelFactory,
    get_feature_names_for_model,
)


class TestModelConfigurations(unittest.TestCase):
    """Test suite validating the four model feature configurations M1, M2, M3, M4."""

    def test_feature_counts(self):
        """Verify exact expected feature counts: M1=18, M2=34, M3=42, M4=46."""
        m1 = get_feature_names_for_model("M1")
        m2 = get_feature_names_for_model("M2")
        m3 = get_feature_names_for_model("M3")
        m4 = get_feature_names_for_model("M4")

        self.assertEqual(len(m1), 18, f"M1 expected 18 features, got {len(m1)}")
        self.assertEqual(len(m2), 34, f"M2 expected 34 features, got {len(m2)}")
        self.assertEqual(len(m3), 42, f"M3 expected 42 features, got {len(m3)}")
        self.assertEqual(len(m4), 46, f"M4 expected 46 features, got {len(m4)}")

    def test_strict_feature_hierarchy(self):
        """Verify strict nested hierarchy: M1 subset of M2 subset of M3 subset of M4."""
        s1 = set(get_feature_names_for_model("M1"))
        s2 = set(get_feature_names_for_model("M2"))
        s3 = set(get_feature_names_for_model("M3"))
        s4 = set(get_feature_names_for_model("M4"))

        self.assertTrue(s1.issubset(s2), "M1 features must be a strict subset of M2")
        self.assertTrue(s2.issubset(s3), "M2 features must be a strict subset of M3")
        self.assertTrue(s3.issubset(s4), "M3 features must be a strict subset of M4")

    def test_m1_excludes_trajectory_and_peers(self):
        """M1 must contain zero trajectory, differential curve, peer, or PMRI features."""
        m1 = set(get_feature_names_for_model("M1"))
        traj_feats = set(FEATURE_GROUPS["traj"]) | set(FEATURE_GROUPS["curve"])
        peer_feats = set(FEATURE_GROUPS["peer_traj"])
        rel_feats = set(FEATURE_GROUPS["peer_reliability"])

        self.assertTrue(m1.isdisjoint(traj_feats), "M1 contains forbidden trajectory features!")
        self.assertTrue(m1.isdisjoint(peer_feats), "M1 contains forbidden peer features!")
        self.assertTrue(m1.isdisjoint(rel_feats), "M1 contains forbidden PMRI features!")

    def test_m2_excludes_peer_and_reliability(self):
        """M2 must contain zero peer-relative or PMRI features."""
        m2 = set(get_feature_names_for_model("M2"))
        peer_feats = set(FEATURE_GROUPS["peer_traj"])
        rel_feats = set(FEATURE_GROUPS["peer_reliability"])

        self.assertTrue(m2.isdisjoint(peer_feats), "M2 contains forbidden peer features!")
        self.assertTrue(m2.isdisjoint(rel_feats), "M2 contains forbidden PMRI features!")

    def test_m3_excludes_pmri_and_reliability(self):
        """M3 must strictly isolate peer-relative features and exclude PMRI/reliability features."""
        m3 = set(get_feature_names_for_model("M3"))
        rel_feats = set(FEATURE_GROUPS["peer_reliability"])

        self.assertTrue(m3.isdisjoint(rel_feats), "M3 contains forbidden reliability features!")
        # Confirm M3 has all 8 peer trajectory features
        peer_feats = set(FEATURE_GROUPS["peer_traj"])
        self.assertTrue(peer_feats.issubset(m3), "M3 missing peer-relative trajectory features!")

    def test_m4_includes_pmri_and_reliability(self):
        """M4 must include the PMRI and peer reliability features."""
        m4 = set(get_feature_names_for_model("M4"))
        rel_feats = set(FEATURE_GROUPS["peer_reliability"])
        self.assertTrue(rel_feats.issubset(m4), "M4 missing PMRI/reliability features!")
        self.assertIn("pmri", m4)

    def test_pipeline_construction(self):
        """Verify pipeline structure contains StandardScaler and LogisticRegression."""
        pipe = ModelFactory.build_logistic_regression()
        self.assertIsInstance(pipe, Pipeline)
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIsInstance(pipe.named_steps["classifier"], LogisticRegression)


if __name__ == "__main__":
    unittest.main()
