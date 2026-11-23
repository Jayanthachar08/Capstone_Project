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

        # Check summary cards & HeroUI aliases
        cards = detail["summary_cards"]
        self.assertEqual(cards["observed_cycles_in_view"], 100)
        self.assertEqual(cards["total_cycles_recorded"], 1851)
        self.assertGreater(cards["q_start_ah"], 1.0)
        self.assertGreater(cards["q_latest_ah"], 1.0)
        self.assertEqual(cards["qdischarge_init_ah"], cards["q_start_ah"])
        self.assertEqual(cards["qdischarge_latest_ah"], cards["q_latest_ah"])
        self.assertEqual(cards["fade_delta_ah"], cards["q_change_ah"])
        self.assertEqual(cards["ir_growth_pct"], cards["ir_change_pct"])
        self.assertEqual(cards["tmax_peak_c"], cards["tmax_latest_c"])

        # Check observed landmarks table
        self.assertIn("observed_landmarks", detail)
        self.assertGreaterEqual(len(detail["observed_landmarks"]), 5)
        for lm in detail["observed_landmarks"]:
            self.assertIn("cycle", lm)
            self.assertIn("q_discharge", lm)

        # Check ground truth targets
        self.assertIn("ground_truth_targets", detail)
        self.assertEqual(detail["ground_truth_targets"]["target_eol_cycle"], 1852.0)

        # Check all 46 features (flat and grouped across 6 groups)
        self.assertEqual(len(detail["features"]), 46)
        feats = detail["features_grouped"]
        self.assertEqual(len(feats["Initial Condition"]), 11)
        self.assertEqual(len(feats["Instantaneous Snapshot"]), 7)
        self.assertEqual(len(feats["Emerging Trajectory"]), 11)
        self.assertEqual(len(feats["Differential Curves"]), 5)
        self.assertEqual(len(feats["Peer-Relative"]), 8)
        self.assertEqual(len(feats["Peer Reliability (PMRI)"]), 4)

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
        """Verify overview statistics include model counts and elevated risk tallies for M1-M4."""
        ov = DATA_MANAGER.get_overview()
        self.assertEqual(ov["total_models"], 4)
        self.assertEqual(ov["models"], ["M1", "M2", "M3", "M4"])
        self.assertEqual(ov["total_predictions"], 372)
        self.assertIn("active_algorithm", ov)
        self.assertIn("supported_algorithms", ov)
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

        # Verify dual aliases for frontend compatibility
        self.assertIn("target_series", traj)
        self.assertIn("peer_series", traj)
        self.assertIn("centroid_series", traj)

        # Verify peer structure and normalized weights
        weights_sum = 0.0
        for peer in traj["peers"]:
            self.assertIn("peer_id", peer)
            self.assertIn("cell_id", peer)
            self.assertIn("distance", peer)
            self.assertIn("weight", peer)
            self.assertIn("points", peer)
            self.assertGreater(len(peer["points"]), 0)
            self.assertGreater(peer["weight"], 0.0)
            weights_sum += peer["weight"]
        self.assertAlmostEqual(weights_sum, 1.0, places=3)

    def test_research_artifacts_loaded(self):
        """Verify Phase 6 hypothesis evaluations and paired differences are indexed."""
        self.assertGreater(len(DATA_MANAGER.research_summary), 0)
        self.assertIn("H1_incremental_peer_relative_value", DATA_MANAGER.research_summary)
        self.assertIn("H2_peer_match_reliability_value", DATA_MANAGER.research_summary)
        self.assertEqual(len(DATA_MANAGER.research_differences), 24)
        self.assertEqual(len(DATA_MANAGER.research_robustness), 48)

    def test_algorithm_selector_api(self):
        """Verify algorithm selector queries and dynamic switching across M1-M4."""
        initial = DATA_MANAGER.get_active_algorithm()
        self.assertIn(initial, ["logistic_regression", "rbf_svm", "tabpfn_2_5"])

        # Switch to rbf_svm
        switched = DATA_MANAGER.set_active_algorithm("rbf_svm")
        self.assertEqual(switched, "rbf_svm")
        self.assertEqual(DATA_MANAGER.get_active_algorithm(), "rbf_svm")

        pred_rbf = DATA_MANAGER.get_model_predictions_for_cell("b1c0", cutoff=100)
        self.assertEqual(pred_rbf["active_algorithm"], "rbf_svm")

        # Switch to tabpfn_2_5
        switched2 = DATA_MANAGER.set_active_algorithm("tabpfn_2_5")
        self.assertEqual(switched2, "tabpfn_2_5")
        pred_tab = DATA_MANAGER.get_model_predictions_for_cell("b1c0", cutoff=100)
        self.assertEqual(pred_tab["active_algorithm"], "tabpfn_2_5")

        # Reset to logistic_regression
        DATA_MANAGER.set_active_algorithm("logistic_regression")
        self.assertEqual(DATA_MANAGER.get_active_algorithm(), "logistic_regression")

    def test_algorithm_benchmarks_api(self):
        """Verify 36-configuration benchmarks and ranking data retrieval."""
        bench = DATA_MANAGER.get_algorithm_benchmarks()
        self.assertIn("summary", bench)
        self.assertIn("rankings", bench)
        self.assertIn("distribution_shift", bench)
        self.assertEqual(len(bench["rankings"]), 3)
        algo_names = {r["algorithm"] for r in bench["rankings"]}
        self.assertEqual(algo_names, {"logistic_regression", "rbf_svm", "tabpfn_2_5"})

    def test_benchmark_export(self):
        """Verify benchmark data exports in CSV and JSON formats."""
        csv_content, csv_mime = DATA_MANAGER.get_benchmark_export("csv")
        self.assertIn("cell_id", csv_content)
        self.assertIn("algorithm", csv_content)
        self.assertIn("prob_M1", csv_content)
        self.assertIn("prob_M4", csv_content)
        self.assertEqual(csv_mime, "text/csv")

        json_content, json_mime = DATA_MANAGER.get_benchmark_export("json")
        self.assertIn("active_algorithm", json_content)
        self.assertIn("benchmarks", json_content)
        self.assertEqual(json_mime, "application/json")

    def test_hardware_csv_validation(self):
        """Verify laboratory cycler hardware CSV validation rules."""
        # Valid CSV
        valid_csv = (
            "Cycle_Index,Discharge_Capacity_Ah,Internal_Resistance_Ohm,Temperature_Max_C,Charge_Time_Min\n"
            + "\n".join(f"{i},1.05,0.015,31.2,42.5" for i in range(1, 45))
        )
        res_valid = DATA_MANAGER.validate_hardware_csv(valid_csv)
        self.assertTrue(res_valid["is_valid"])
        self.assertEqual(res_valid["cycle_count"], 44)

        # Invalid CSV (missing columns, non-monotonic)
        invalid_csv = "Cycle_Index,Discharge_Capacity_Ah\n1,1.05\n3,1.04\n2,1.03"
        res_invalid = DATA_MANAGER.validate_hardware_csv(invalid_csv)
        self.assertFalse(res_invalid["is_valid"])
        self.assertGreater(len(res_invalid["errors"]), 0)

    def test_multi_algorithm_metrics_api(self):
        """Verify multi-algorithm benchmark metrics are loaded and indexed."""
        algos = DATA_MANAGER.get_multi_algorithm_metrics()
        self.assertGreater(len(algos), 0)
        found_algos = {a["algorithm"] for a in algos}
        self.assertTrue({"logistic_regression", "rbf_svm", "tabpfn_2_5"}.issubset(found_algos))

    def test_peer_control_results_api(self):
        """Verify multi-seed peer control permutations and PMRI validation analysis."""
        data = DATA_MANAGER.get_peer_control_results()
        self.assertIn("permutations", data)
        self.assertIn("pmri_analysis", data)
        self.assertGreater(len(data["permutations"]), 20)
        self.assertIn("by_cutoff", data["pmri_analysis"])

    def test_selective_prediction_api(self):
        """Verify selective prediction risk-coverage curve points."""
        curve = DATA_MANAGER.get_selective_prediction_results()
        self.assertGreater(len(curve), 0)
        mechanisms = {pt["selection_mechanism"] for pt in curve}
        self.assertIn("pmri_threshold", mechanisms)


if __name__ == "__main__":
    unittest.main()

