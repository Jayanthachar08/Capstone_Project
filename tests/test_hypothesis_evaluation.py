"""
Unit tests for Phase 6 research hypothesis evaluations, paired bootstrap comparisons,
robustness analyses, and single-class partition handling.
"""

import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from src.evaluation.hypothesis_evaluator import HypothesisEvaluator


class TestHypothesisEvaluation(unittest.TestCase):
    """Test suite validating Phase 6 hypothesis evaluations and paired bootstrap comparisons."""

    @classmethod
    def setUpClass(cls):
        cls.evaluator = HypothesisEvaluator()
        cls.preds_df, cls.metrics_df = cls.evaluator.load_data()

    def test_paired_bootstrap_math_and_ordering(self):
        """Verifies that paired bootstrap deltas compute correct directions and ordered CI bounds."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 3)  # N=30
        prob_a = np.array([0.4, 0.6, 0.7, 0.8, 0.3, 0.5, 0.8, 0.2, 0.6, 0.3] * 3)  # Imperfect ranking
        prob_b = np.array([0.9, 0.1, 0.95, 0.05, 0.85, 0.2, 0.9, 0.1, 0.95, 0.15] * 3)  # Superior discrimination & calibration

        res = HypothesisEvaluator.paired_bootstrap_delta(
            y_true=y_true,
            prob_a=prob_a,
            prob_b=prob_b,
            n_bootstraps=200,
            random_seed=42,
        )

        # Model B is strictly better: AUROC delta > 0, Brier delta < 0
        self.assertGreater(res["auroc"]["point"], 0.0)
        self.assertLess(res["brier"]["point"], 0.0)

        # CI bounds ordering
        self.assertLessEqual(res["auroc"]["ci_lower"], res["auroc"]["ci_upper"])
        self.assertLessEqual(res["brier"]["ci_lower"], res["brier"]["ci_upper"])
        self.assertLessEqual(res["auprc"]["ci_lower"], res["auprc"]["ci_upper"])

        # Finite standard errors
        self.assertTrue(np.isfinite(res["auroc"]["std_err"]))
        self.assertTrue(np.isfinite(res["brier"]["std_err"]))

    def test_single_class_bootstrap_graceful_handling(self):
        """Verifies that single-class partitions return NaN for AUROC delta but finite Brier delta."""
        y_all_zero = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        prob_a = np.array([0.1, 0.2, 0.1, 0.15, 0.3, 0.05, 0.1, 0.2])
        prob_b = np.array([0.05, 0.02, 0.01, 0.03, 0.02, 0.01, 0.05, 0.02])

        res = HypothesisEvaluator.paired_bootstrap_delta(
            y_true=y_all_zero,
            prob_a=prob_a,
            prob_b=prob_b,
            n_bootstraps=100,
            random_seed=42,
        )

        self.assertTrue(np.isnan(res["auroc"]["point"]))
        self.assertTrue(np.isnan(res["auroc"]["ci_lower"]))
        self.assertTrue(np.isnan(res["auprc"]["point"]))

        # Brier is well-defined
        self.assertTrue(np.isfinite(res["brier"]["point"]))
        self.assertLessEqual(res["brier"]["ci_lower"], res["brier"]["ci_upper"])

    def test_model_differences_schema_and_values(self):
        """Verifies that phase6_model_differences.csv exists with correct schema and valid entries."""
        diff_path = Path("results/phase6_model_differences.csv")
        if not diff_path.exists():
            # If not yet written by evaluator, run it
            self.evaluator.evaluate_model_differences(n_bootstraps=100)

        df_diff = pd.read_csv(diff_path)
        required_cols = [
            "cutoff",
            "partition",
            "comparison",
            "model_a",
            "model_b",
            "delta_auroc",
            "delta_auroc_ci_lower",
            "delta_auroc_ci_upper",
            "delta_brier",
            "delta_brier_ci_lower",
            "delta_brier_ci_upper",
            "status",
        ]
        for col in required_cols:
            self.assertIn(col, df_diff.columns)

        # Primary test cutoffs must be [40, 70, 100]
        prim_diff = df_diff[df_diff["partition"] == "primary_test"]
        self.assertEqual(set(prim_diff["cutoff"].unique()), {40, 70, 100})

        # All primary test CI bounds must be ordered: ci_lower <= ci_upper
        for _, row in prim_diff.iterrows():
            if not np.isnan(row["delta_auroc_ci_lower"]):
                self.assertLessEqual(row["delta_auroc_ci_lower"], row["delta_auroc_ci_upper"])
            self.assertLessEqual(row["delta_brier_ci_lower"], row["delta_brier_ci_upper"])

    def test_hypothesis_summary_json_keys(self):
        """Verifies that phase6_hypothesis_summary.json exists and contains verdicts for H1-H4."""
        summary_path = Path("results/phase6_hypothesis_summary.json")
        if not summary_path.exists():
            self.evaluator.evaluate_hypotheses()

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_hypotheses = [
            "H1_incremental_peer_relative_value",
            "H2_peer_match_reliability_value",
            "H3_early_observation_usefulness",
            "H4_reliability_under_weak_matching",
        ]
        for h in expected_hypotheses:
            self.assertIn(h, data)
            self.assertIn("verdict", str(data[h]))

    def test_figures_created(self):
        """Verifies that all four publication figures exist in results/figures/."""
        fig_dir = Path("results/figures")
        expected_figs = [
            "cutoff_comparison.png",
            "model_differences.png",
            "pmri_stratification.png",
            "calibration_diagrams.png",
        ]
        for fig in expected_figs:
            fig_path = fig_dir / fig
            if not fig_path.exists():
                self.evaluator.generate_figures()
            self.assertTrue((fig_dir / fig).exists(), f"Missing figure {fig}")


if __name__ == "__main__":
    unittest.main()
