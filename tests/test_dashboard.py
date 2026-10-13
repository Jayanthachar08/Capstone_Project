"""
Unit tests for the dashboard server data manager and API data serialization.
"""

import unittest
from dashboard.server import DATA_MANAGER


class TestDashboardDataManager(unittest.TestCase):
    """Test suite verifying data accuracy and consistency for the dashboard."""

    def test_overview_counts(self):
        """Verify overview statistics match verified benchmark totals."""
        ov = DATA_MANAGER.get_overview()
        self.assertEqual(ov["total_cells"], 140)
        self.assertEqual(ov["eligible_cells"], 124)
        self.assertEqual(ov["excluded_cells"], 16)
        self.assertEqual(ov["partitions"]["train"], 41)
        self.assertEqual(ov["partitions"]["primary_test"], 43)
        self.assertEqual(ov["partitions"]["secondary_test"], 40)
        self.assertEqual(ov["partitions"]["excluded"], 16)
        self.assertEqual(ov["total_cycle_records"], 118131)

    def test_battery_detail_integrity(self):
        """Verify detailed battery view retrieves valid measurements and features."""
        detail = DATA_MANAGER.get_battery_detail("b1c0", cutoff=100)
        self.assertIsNotNone(detail)
        meta = detail["metadata"]
        self.assertEqual(meta["cell_id"], "b1c0")
        self.assertEqual(meta["batch"], "batch1")
        self.assertTrue(meta["is_benchmark_eligible"])
        self.assertEqual(meta["benchmark_cycle_life"], 1852.0)

        # Check summary cards
        cards = detail["summary_cards"]
        self.assertEqual(cards["observed_cycles_in_view"], 100)
        self.assertEqual(cards["total_cycles_recorded"], 1851)
        self.assertGreater(cards["q_start_ah"], 1.0)
        self.assertGreater(cards["q_latest_ah"], 1.0)

        # Check feature groups
        feats = detail["features_grouped"]
        self.assertEqual(len(feats["Initial Condition"]), 11)
        self.assertEqual(len(feats["Instantaneous Snapshot"]), 7)
        self.assertEqual(len(feats["Emerging Trajectory"]), 11)
        self.assertEqual(len(feats["Differential Curves"]), 5)

    def test_comparison_series_output(self):
        """Verify multi-cell comparison query structure."""
        comp = DATA_MANAGER.get_comparison(["b1c0", "b1c1", "b2c0"], metric="QDischarge", cutoff=100)
        self.assertEqual(len(comp["series"]), 3)
        for s in comp["series"]:
            self.assertIn("cell_id", s)
            self.assertIn("points", s)
    def test_peer_info_api(self):
        """Verify peer features and PMRI retrieval for a benchmark battery."""
        peer_data = DATA_MANAGER.get_peer_info("b1c0", cutoff=100)
        self.assertIsNotNone(peer_data)
        self.assertEqual(peer_data["cell_id"], "b1c0")
        self.assertEqual(peer_data["peer_count"], 5)
        self.assertEqual(peer_data["peer_match_status"], "MATCH_SUCCESS")
        self.assertIn("b1c1", peer_data["peer_ids"])
        self.assertGreaterEqual(peer_data["pmri"], 0.5)
        self.assertEqual(peer_data["reliability_warning"], "NONE")

    def test_model_metrics_and_predictions_api(self):
        """Verify model metrics and per-battery prediction probability retrieval."""
        metrics = DATA_MANAGER.get_model_metrics()
        self.assertGreater(len(metrics), 0)
        self.assertTrue(any(m["model"] == "M4" for m in metrics))

    def test_overview_model_statistics(self):
        """Verify overview statistics include model counts and elevated risk tallies."""
        ov = DATA_MANAGER.get_overview()
        self.assertEqual(ov["total_models"], 4)
        self.assertEqual(ov["total_predictions"], 372)
        self.assertIn("elevated_risk_counts", ov)
        self.assertEqual(set(ov["elevated_risk_counts"].keys()), {40, 70, 100})
        self.assertIn("M4", ov["elevated_risk_counts"][100])
        self.assertGreater(ov["elevated_risk_counts"][100]["M4"], 0)

    def test_enriched_model_predictions(self):
        """Verify model predictions contain thresholds, classifications, and finite probabilities."""
        pred = DATA_MANAGER.get_model_predictions_for_cell("b1c0", cutoff=100)
        self.assertIsNotNone(pred)
        self.assertIn("model_details", pred)
        for m in ["M1", "M2", "M3", "M4"]:
            det = pred["model_details"][m]
            self.assertIn("probability", det)
            self.assertIn("threshold", det)
            self.assertIn("risk_category", det)
            self.assertTrue(0.0 <= det["probability"] <= 1.0)
            self.assertTrue(0.0 <= det["threshold"] <= 1.0)
            self.assertIn(det["risk_category"], ["ELEVATED_RISK", "NOMINAL_RISK"])

    def test_peer_trajectories_generation(self):
        """Verify aligned peer trajectories extraction and mean/deviation computation."""
        traj = DATA_MANAGER.get_peer_trajectories("b1c0", cutoff=100)
        self.assertIsNotNone(traj)
        self.assertEqual(traj["cell_id"], "b1c0")
        self.assertEqual(len(traj["peers"]), 5)
        self.assertGreater(len(traj["target_trajectory"]), 0)
        self.assertGreater(len(traj["mean_peer_trajectory"]), 0)
        self.assertGreater(len(traj["deviation_trajectory"]), 0)
        self.assertEqual(traj["reliability_warning"], "NONE")

    def test_research_artifacts_loaded(self):
        """Verify Phase 6 hypothesis evaluations and paired differences are indexed."""
        self.assertGreater(len(DATA_MANAGER.research_summary), 0)
        self.assertIn("H1_incremental_peer_relative_value", DATA_MANAGER.research_summary)
        self.assertIn("H2_peer_match_reliability_value", DATA_MANAGER.research_summary)
        self.assertEqual(len(DATA_MANAGER.research_differences), 24)
        self.assertEqual(len(DATA_MANAGER.research_robustness), 48)


if __name__ == "__main__":
    unittest.main()

