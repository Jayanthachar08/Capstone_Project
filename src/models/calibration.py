"""
Module: src.models.calibration
Author: Capstone Project Team
Description: Probability calibration and uncertainty assessment module.
             Evaluates Brier scores, Expected Calibration Error (ECE), reliability diagrams,
             and fits training-only Platt scaling (sigmoid) calibration without test leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Manages empirical probability calibration and reliability diagnostics.

    NOTE: Calibration models are fit EXCLUSIVELY on training-partition
    out-of-fold predictions. Test partitions are never accessed during calibration.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.calibrator: Optional[LogisticRegression] = None
        self.is_fitted: bool = False

    def fit_on_oof_predictions(self, y_train_true: np.ndarray, y_train_oof_probs: np.ndarray) -> ProbabilityCalibrator:
        """Fits a univariate Platt scaling model (sigmoid calibration) on out-of-fold training predictions.

        Formula: P_calibrated(y=1 | p) = 1 / (1 + exp(-(a * logit(p) + b)))

        Args:
            y_train_true: Ground truth binary labels in training cohort.
            y_train_oof_probs: Out-of-fold predicted probabilities on training cohort.
        """
        # Clip to avoid infinite logits
        eps = 1e-6
        p_clipped = np.clip(y_train_oof_probs, eps, 1.0 - eps)
        # Compute log-odds (logits)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        # Fit univariate logistic regression on logits
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(logits, y_train_true)
        self.calibrator = clf
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Applies the fitted Platt scaler to new predicted probabilities."""
        if not self.is_fitted or self.calibrator is None:
            return probs.copy()

        eps = 1e-6
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        return self.calibrator.predict_proba(logits)[:, 1]

    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 5) -> float:
        """Calculates Expected Calibration Error (ECE).

        Formula: ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
        """
        if len(y_true) == 0:
            return 0.0

        # Uniform bin edges from 0 to 1
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            b_min, b_max = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                idx = np.where((y_probs >= b_min) & (y_probs <= b_max))[0]
            else:
                idx = np.where((y_probs >= b_min) & (y_probs < b_max))[0]

            if len(idx) > 0:
                bin_acc = np.mean(y_true[idx])
                bin_conf = np.mean(y_probs[idx])
                bin_weight = len(idx) / n_total
                ece += bin_weight * abs(bin_acc - bin_conf)

        return float(ece)

    @staticmethod
    def compute_calibration_diagnostics(
        y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 5
    ) -> Dict[str, Any]:
        """Calculates comprehensive calibration diagnostics (Brier score, ECE, reliability curve)."""
        brier = float(brier_score_loss(y_true, y_probs))
        ece = ProbabilityCalibrator.calculate_ece(y_true, y_probs, n_bins=n_bins)

        # Reliability curve if both classes are present
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2:
            try:
                prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy="uniform")
                curve_data = {
                    "prob_true": prob_true.tolist(),
                    "prob_pred": prob_pred.tolist(),
                }
            except Exception as e:
                logger.warning("Error generating calibration curve: %s", e)
                curve_data = {"prob_true": [], "prob_pred": []}
        else:
            curve_data = {"prob_true": [], "prob_pred": []}

        return {
            "brier_score": brier,
            "expected_calibration_error": ece,
            "reliability_curve": curve_data,
        }
