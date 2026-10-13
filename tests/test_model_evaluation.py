"""
Unit tests for model evaluation metrics, calibration calculations, and single-class edge cases.
"""

import unittest
import numpy as np
import pandas as pd

from src.models.evaluate_models import ModelEvaluator
from src.models.calibration import ProbabilityCalibrator


class TestModelEvaluation(unittest.TestCase):
    """Test suite validating metric correctness, probability bounds, and calibration diagnostics."""

    @classmethod
    def setUpClass(cls):
        cls.preds_df = pd.read_csv("results/phase5_predictions.csv")

    def test_multi_class_metrics_computation(self):
        """Verifies standard evaluation metrics calculation on balanced inputs."""
        y_true = np.array([1, 1, 0, 0])
        y_probs = np.array([0.9, 0.8, 0.2, 0.1])
        res = ModelEvaluator.evaluate(y_true, y_probs, threshold=0.5, partition_name="test")

        self.assertEqual(res["auroc"], 1.0)
        self.assertEqual(res["auprc"], 1.0)
        self.assertLess(res["brier_score"], 0.05)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["tn"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["f1_score"], 1.0)
        self.assertEqual(res["fpr"], 0.0)

    def test_single_class_partition_graceful_handling(self):
        """Single-class partitions must report AUROC as NaN with documented status, not crash."""
        y_all_zero = np.array([0, 0, 0, 0, 0])
        y_probs = np.array([0.1, 0.2, 0.15, 0.05, 0.1])

        res = ModelEvaluator.evaluate(y_all_zero, y_probs, threshold=0.5, partition_name="secondary_test")
        self.assertTrue(np.isnan(res["auroc"]))
        self.assertTrue(np.isnan(res["auprc"]))
        self.assertTrue(np.isfinite(res["brier_score"]))
        self.assertIn("SINGLE_CLASS_EVALUATION", res["status"])
        self.assertEqual(res["specificity"], 1.0)
        self.assertEqual(res["fpr"], 0.0)

    def test_predicted_probabilities_bounded_and_finite(self):
        """All model prediction probabilities in results/phase5_predictions.csv must be finite in [0, 1]."""
        for m in ["prob_M1", "prob_M2", "prob_M3", "prob_M4"]:
            vals = self.preds_df[m].values
            self.assertTrue(np.all(np.isfinite(vals)), f"Non-finite probability in {m}")
            self.assertTrue(np.all((vals >= 0.0) & (vals <= 1.0)), f"Probability outside [0, 1] in {m}")

    def test_platt_scaling_calibration_bounds(self):
        """Platt scaling calibrator must output finite probabilities bounded in [0, 1]."""
        y_tr = np.array([0, 0, 1, 1, 0, 1])
        p_tr = np.array([0.2, 0.3, 0.7, 0.8, 0.1, 0.9])

        calibrator = ProbabilityCalibrator()
        calibrator.fit_on_oof_predictions(y_tr, p_tr)

        test_p = np.array([0.1, 0.5, 0.95])
        cal_p = calibrator.calibrate(test_p)

        self.assertEqual(len(cal_p), 3)
        self.assertTrue(np.all((cal_p >= 0.0) & (cal_p <= 1.0)))
        # Verify monotonicity: higher input prob should yield higher calibrated prob
        self.assertLess(cal_p[0], cal_p[1])
        self.assertLess(cal_p[1], cal_p[2])


if __name__ == "__main__":
    unittest.main()
