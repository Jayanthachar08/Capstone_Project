"""
Module: src.models.evaluate_models
Author: Capstone Project Team
Description: Comprehensive evaluation module computing discrimination metrics (AUROC, AUPRC),
             calibration metrics (Brier, ECE), and threshold-dependent classification metrics
             (Confusion Matrix, Sensitivity/Recall, Specificity, FPR, Precision, F1-Score).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.models.calibration import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes comprehensive classification and calibration metrics for model predictions."""

    @staticmethod
    def select_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Finds the optimal probability decision threshold maximizing Youden's J statistic

        (Sensitivity + Specificity - 1 = TPR - FPR) on validation/training predictions.
        """
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.50

        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        # Bound cleanly between 0.1 and 0.9 to avoid extreme edge thresholds
        return float(np.clip(best_thresh, 0.10, 0.90))

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float = 0.50,
        partition_name: str = "test",
    ) -> Dict[str, Any]:
        """Calculates discrimination, calibration, and classification metrics.

        Handles single-class partitions gracefully (e.g. secondary_test at cutoff 40 where all labels are 0).

        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_probs: Predicted probabilities of positive class in [0, 1].
            threshold: Probability decision threshold for binary classification.
            partition_name: Name of evaluation partition.

        Returns:
            Dict containing computed metrics and diagnostic status.
        """
        n_samples = len(y_true)
        pos_count = int(np.sum(y_true == 1))
        neg_count = int(np.sum(y_true == 0))
        unique_classes = np.unique(y_true)

        # Basic calibration: Brier score is always well-defined
        brier = float(brier_score_loss(y_true, y_probs))
        ece = ProbabilityCalibrator.calculate_ece(y_true, y_probs, n_bins=5)

        # Binary classifications at threshold
        y_pred = (y_probs >= threshold).astype(int)

        # Single-class evaluation edge case (e.g. secondary_test with no positive labels)
        if len(unique_classes) < 2:
            # AUROC and AUPRC are mathematically undefined
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            accuracy = float(np.mean(y_true == y_pred))
            spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
            fpr_val = float(fp / (tn + fp)) if (tn + fp) > 0 else np.nan

            return {
                "partition": partition_name,
                "n_samples": n_samples,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "prevalence": float(pos_count / n_samples) if n_samples > 0 else 0.0,
                "auroc": np.nan,
                "auprc": np.nan,
                "brier_score": brier,
                "ece": ece,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": np.nan,
                "recall": np.nan,
                "specificity": spec,
                "fpr": fpr_val,
                "precision": np.nan,
                "f1_score": np.nan,
                "accuracy": accuracy,
                "status": "SINGLE_CLASS_EVALUATION (AUROC/AUPRC mathematically undefined)",
            }

        # Multi-class evaluation (both 0 and 1 present)
        try:
            auroc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auroc = np.nan

        try:
            auprc = float(average_precision_score(y_true, y_probs))
        except Exception:
            auprc = np.nan

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fpr_val = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
        acc = float((tp + tn) / n_samples) if n_samples > 0 else 0.0

        return {
            "partition": partition_name,
            "n_samples": n_samples,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "prevalence": float(pos_count / n_samples) if n_samples > 0 else 0.0,
            "auroc": auroc,
            "auprc": auprc,
            "brier_score": brier,
            "ece": ece,
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "sensitivity": sens,
            "recall": sens,
            "specificity": spec,
            "fpr": fpr_val,
            "precision": prec,
            "f1_score": f1,
            "accuracy": acc,
            "status": "COMPLETE",
        }
