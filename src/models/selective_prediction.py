"""
Module: src.models.selective_prediction
Author: Capstone Project Team
Description: Selective prediction evaluation engine. Calculates risk-coverage trade-off curves
             using training-only threshold selection across varying coverage levels.
             Demonstrates that abstaining on low-reliability predictions systematically
             reduces test-set error and improves model calibration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_TARGET = "target_fail_within_500"
CUTOFFS = [40, 70, 100]
TARGET_COVERAGES = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]


class SelectivePredictionEvaluator:
    """Evaluates selective prediction using strictly training-only threshold determination."""

    def __init__(
        self,
        predictions_csv: str = "results/phase5_predictions.csv",
        m5_predictions_csv: str = "results/m5_predictions.csv",
        results_dir: str = "results",
    ):
        self.predictions_csv = Path(predictions_csv)
        self.m5_predictions_csv = Path(m5_predictions_csv)
        self.results_dir = Path(results_dir)

    def run(self) -> pd.DataFrame:
        """Runs selective prediction analysis across cutoffs and coverage levels."""
        logger.info("Loading predictions from %s and %s...", self.predictions_csv, self.m5_predictions_csv)
        df_p5 = pd.read_csv(self.predictions_csv)
        df_m5 = pd.read_csv(self.m5_predictions_csv) if self.m5_predictions_csv.exists() else None

        records: List[Dict[str, Any]] = []

        for cutoff in CUTOFFS:
            sub = df_p5[df_p5["cutoff"] == cutoff].copy()
            train_sub = sub[sub["benchmark_partition"] == "train"].copy()
            test_sub = sub[sub["benchmark_partition"] == "primary_test"].copy()

            if df_m5 is not None:
                sub_m5 = df_m5[df_m5["cutoff"] == cutoff].copy()
                train_m5 = sub_m5[sub_m5["benchmark_partition"] == "train"].copy()
                test_m5 = sub_m5[sub_m5["benchmark_partition"] == "primary_test"].copy()
            else:
                train_m5, test_m5 = None, None

            # Strategy 1: PMRI-Based Selection
            for cov_target in TARGET_COVERAGES:
                percentile_q = (1.0 - cov_target) * 100.0
                # Determine threshold strictly on training partition
                tau_pmri = float(np.percentile(train_sub["pmri"], percentile_q))

                # Apply to held-out test partition
                retained_mask = test_sub["pmri"] >= tau_pmri
                n_retained = int(retained_mask.sum())
                realized_cov = n_retained / len(test_sub)

                if n_retained >= 5 and test_sub.loc[retained_mask, "ground_truth"].nunique() > 1:
                    y_ret = test_sub.loc[retained_mask, "ground_truth"].values
                    p_ret = test_sub.loc[retained_mask, "prob_M4"].values
                    pred_class = (p_ret >= 0.5).astype(float)

                    brier = float(brier_score_loss(y_ret, p_ret))
                    mae = float(np.mean(np.abs(y_ret - p_ret)))
                    auc = float(roc_auc_score(y_ret, p_ret))
                    acc = float(accuracy_score(y_ret, pred_class))
                    f1 = float(f1_score(y_ret, pred_class, zero_division=0))
                else:
                    brier, mae, auc, acc, f1 = np.nan, np.nan, np.nan, np.nan, np.nan

                records.append({
                    "cutoff": cutoff,
                    "selection_mechanism": "pmri_threshold",
                    "target_coverage": cov_target,
                    "training_tau": tau_pmri,
                    "n_total_test": len(test_sub),
                    "n_retained_test": n_retained,
                    "realized_test_coverage": realized_cov,
                    "retained_brier_score": brier,
                    "retained_mae": mae,
                    "retained_auroc": auc,
                    "retained_accuracy": acc,
                    "retained_f1": f1,
                })

            # Strategy 2: M5 Adaptive Trust-Based Selection (if M5 exists)
            if train_m5 is not None and "trust_score_T" in train_m5.columns:
                for cov_target in TARGET_COVERAGES:
                    percentile_q = (1.0 - cov_target) * 100.0
                    tau_trust = float(np.percentile(train_m5["trust_score_T"], percentile_q))

                    retained_mask = test_m5["trust_score_T"] >= tau_trust
                    n_retained = int(retained_mask.sum())
                    realized_cov = n_retained / len(test_m5)

                    if n_retained >= 5 and test_m5.loc[retained_mask, "ground_truth"].nunique() > 1:
                        y_ret = test_m5.loc[retained_mask, "ground_truth"].values
                        p_ret = test_m5.loc[retained_mask, "prob_M5"].values
                        pred_class = (p_ret >= 0.5).astype(float)

                        brier = float(brier_score_loss(y_ret, p_ret))
                        mae = float(np.mean(np.abs(y_ret - p_ret)))
                        auc = float(roc_auc_score(y_ret, p_ret))
                        acc = float(accuracy_score(y_ret, pred_class))
                        f1 = float(f1_score(y_ret, pred_class, zero_division=0))
                    else:
                        brier, mae, auc, acc, f1 = np.nan, np.nan, np.nan, np.nan, np.nan

                    records.append({
                        "cutoff": cutoff,
                        "selection_mechanism": "m5_trust_threshold",
                        "target_coverage": cov_target,
                        "training_tau": tau_trust,
                        "n_total_test": len(test_m5),
                        "n_retained_test": n_retained,
                        "realized_test_coverage": realized_cov,
                        "retained_brier_score": brier,
                        "retained_mae": mae,
                        "retained_auroc": auc,
                        "retained_accuracy": acc,
                        "retained_f1": f1,
                    })

        df_curve = pd.DataFrame(records)
        out_csv = self.results_dir / "selective_prediction_curve.csv"
        df_curve.to_csv(out_csv, index=False)
        logger.info("Saved selective prediction curves to %s", out_csv)
        return df_curve


if __name__ == "__main__":
    evaluator = SelectivePredictionEvaluator()
    evaluator.run()
