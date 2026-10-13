"""
Unit tests for the PeerReliabilityEvaluator subsystem.
Verifies PMRI bounds, component dynamics, agreement penalties, and warning triggers.
"""

import unittest
import numpy as np
import pandas as pd

from src.peers.peer_reliability import PeerReliabilityEvaluator


class TestPeerReliability(unittest.TestCase):
    """Test suite validating properties and safeguards of the Peer-Match Reliability Index (PMRI)."""

    @classmethod
    def setUpClass(cls):
        cls.df_peers = pd.read_csv("data/processed/peer_features.csv")
        cls.evaluator = PeerReliabilityEvaluator()

    def test_pmri_within_unit_interval(self):
        """Rule 9: PMRI must strictly lie within [0.0, 1.0] for all eligible benchmark cells."""
        elig = self.df_peers[self.df_peers["is_benchmark_eligible"]]
        for _, row in elig.iterrows():
            pmri = float(row["pmri"])
            self.assertTrue(
                np.isfinite(pmri),
                f"Non-finite PMRI detected for cell {row['cell_id']} at cutoff {row['cutoff']}",
            )
            self.assertGreaterEqual(pmri, 0.0)
            self.assertLessEqual(pmri, 1.0)

    def test_pmri_components_bounded(self):
        """All four PMRI component scores must strictly lie within [0.0, 1.0]."""
        elig = self.df_peers[self.df_peers["is_benchmark_eligible"]]
        for col in [
            "peer_similarity_score",
            "peer_agreement_score",
            "peer_availability_score",
            "peer_feature_completeness",
        ]:
            vals = elig[col].values
            self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite values in {col}")
            self.assertTrue(np.all((vals >= 0.0) & (vals <= 1.0)), f"Values outside [0, 1] in {col}")

    def test_large_peer_disagreement_lowers_agreement_and_pmri(self):
        """Rule 12: High trajectory dispersion among peers must strictly lower agreement and composite PMRI."""
        match_base = {
            "peer_match_status": "MATCH_SUCCESS",
            "peer_count": 5,
            "mean_peer_distance": 0.5,
            "peer_feature_completeness": 1.0,
        }

        # Case A: Low dispersion (high consensus)
        comp_low_disp = {"peer_slope_dispersion": 1e-6}
        # Case B: High dispersion (disagreement)
        comp_high_disp = {"peer_slope_dispersion": 2e-4}

        res_low = self.evaluator.evaluate_reliability(match_base, comp_low_disp)
        res_high = self.evaluator.evaluate_reliability(match_base, comp_high_disp)

        self.assertGreater(
            res_low["peer_agreement_score"],
            res_high["peer_agreement_score"],
            "Peer agreement score should be strictly higher for low dispersion",
        )
        self.assertGreater(
            res_low["pmri"],
            res_high["pmri"],
            "Composite PMRI should be strictly higher for high-agreement peers",
        )
        self.assertIn("HIGH_PEER_DISPERSION", res_high["reliability_warning"])

    def test_large_distance_lowers_similarity_and_pmri(self):
        """Larger standardized distance must strictly lower peer similarity and composite PMRI."""
        comp_base = {"peer_slope_dispersion": 2e-5}

        match_close = {
            "peer_match_status": "MATCH_SUCCESS",
            "peer_count": 5,
            "mean_peer_distance": 0.2,
            "peer_feature_completeness": 1.0,
        }
        match_far = {
            "peer_match_status": "MATCH_SUCCESS",
            "peer_count": 5,
            "mean_peer_distance": 3.0,
            "peer_feature_completeness": 1.0,
        }

        res_close = self.evaluator.evaluate_reliability(match_close, comp_base)
        res_far = self.evaluator.evaluate_reliability(match_far, comp_base)

        self.assertGreater(res_close["peer_similarity_score"], res_far["peer_similarity_score"])
        self.assertGreater(res_close["pmri"], res_far["pmri"])
        self.assertIn("HIGH_PEER_DISTANCE", res_far["reliability_warning"])

    def test_insufficient_peers_yields_warning_and_zero_pmri(self):
        """Rule 10: Insufficient peers must trigger warning and zero availability/PMRI score."""
        match_insufficient = {
            "peer_match_status": "INSUFFICIENT_PEERS",
            "peer_count": 2,
            "mean_peer_distance": 0.5,
            "peer_feature_completeness": 1.0,
        }
        comp_dummy = {"peer_slope_dispersion": 2e-5}
        res = self.evaluator.evaluate_reliability(match_insufficient, comp_dummy)

        self.assertEqual(res["pmri"], 0.0)
        self.assertEqual(res["peer_availability_score"], 0.0)
        self.assertEqual(res["reliability_warning"], "INSUFFICIENT_PEERS")

    def test_non_finite_inputs_handled_safely(self):
        """Rule 11: Non-finite inputs (NaN, inf) must be handled gracefully without crashing."""
        match_corrupt = {
            "peer_match_status": "MATCH_SUCCESS",
            "peer_count": 5,
            "mean_peer_distance": np.nan,
            "peer_feature_completeness": np.nan,
        }
        comp_corrupt = {"peer_slope_dispersion": np.inf}
        res = self.evaluator.evaluate_reliability(match_corrupt, comp_corrupt)

        self.assertTrue(np.isfinite(res["pmri"]))
        self.assertGreaterEqual(res["pmri"], 0.0)
        self.assertLessEqual(res["pmri"], 1.0)


if __name__ == "__main__":
    unittest.main()
