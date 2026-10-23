"""
Module: src.models.multi_algorithm_benchmark
Author: Capstone Project Team
Description: Modular multi-algorithm benchmarking engine comparing Logistic Regression,
             Random Forest, RBF-SVM, and Histogram Gradient Boosting across M1–M4
             model configurations and observation cutoffs (40, 70, 100 cycles).
             Maintains strict anti-leakage guarantees and saves versioned artifacts.
"""

from __future__ import annotations

import json
import logging
import platform
from pathlib import Path
import sys

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.calibration import ProbabilityCalibrator
from src.models.evaluate_models import ModelEvaluator
from src.models.model_factory import (
    FEATURE_GROUPS,
    MODEL_CONFIGS,
    SUPPORTED_ALGORITHMS,
    ModelFactory,
    get_feature_names_for_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_TARGET = "target_fail_within_500"
DEFAULT_CUTOFFS = [40, 70, 100]
RANDOM_SEED = 42


class MultiAlgorithmBenchmark:
    """Executes fair, leak-free comparison of multiple ML algorithms across M1–M4."""

    def __init__(
        self,
        modeling_csv: str = "data/processed/modeling_dataset.csv",
        peer_csv: str = "data/processed/peer_features.csv",
        results_dir: str = "results",
        target_name: str = PRIMARY_TARGET,
        cutoffs: Optional[List[int]] = None,
        random_state: int = RANDOM_SEED,
        algorithms: Optional[List[str]] = None,
    ):
        self.modeling_csv = Path(modeling_csv)
        self.peer_csv = Path(peer_csv)
        self.results_dir = Path(results_dir)
        self.target_name = target_name
        self.cutoffs = cutoffs or DEFAULT_CUTOFFS
        self.random_state = random_state
        self.algorithms = algorithms or [
            "logistic_regression",
            "random_forest",
            "rbf_svm",
            "hist_gradient_boosting",
        ]

    def load_and_merge_data(self) -> pd.DataFrame:
        """Loads and merges modeling dataset with peer features."""
        df_model = pd.read_csv(self.modeling_csv)
        df_peers = pd.read_csv(self.peer_csv)

        merge_keys = ["cell_id", "cutoff"]
        peer_cols_to_add = [
            c for c in df_peers.columns if c not in df_model.columns or c in merge_keys
        ]
        df_merged = pd.merge(df_model, df_peers[peer_cols_to_add], on=merge_keys)
        return df_merged

    def run(self) -> Dict[str, Any]:
        """Runs the complete multi-algorithm benchmark across all configurations."""
        df_all = self.load_and_merge_data()
        metrics_records: List[Dict[str, Any]] = []
        predictions_records: List[Dict[str, Any]] = []

        logger.info(
            "Starting Multi-Algorithm Benchmark for %d algorithms across %d cutoffs...",
            len(self.algorithms),
            len(self.cutoffs),
        )

        for algo_key in self.algorithms:
            algo_meta = SUPPORTED_ALGORITHMS.get(algo_key, {"name": algo_key, "family": "Other"})
            algo_name = algo_meta.get("name", algo_key)
            logger.info("==================================================")
            logger.info("Benchmarking Algorithm: %s (%s)", algo_name, algo_key)
            logger.info("==================================================")

            for cutoff in self.cutoffs:
                df_cut = df_all[df_all["cutoff"] == cutoff].copy().reset_index(drop=True)
                df_cut_elig = df_cut[df_cut["is_benchmark_eligible"]].copy().reset_index(drop=True)

                train_df = df_cut_elig[df_cut_elig["benchmark_partition"] == "train"].copy().reset_index(drop=True)
                test_prim_df = df_cut_elig[df_cut_elig["benchmark_partition"] == "primary_test"].copy().reset_index(drop=True)
                test_sec_df = df_cut_elig[df_cut_elig["benchmark_partition"] == "secondary_test"].copy().reset_index(drop=True)

                y_train = train_df[self.target_name].values.astype(float)
                y_prim = test_prim_df[self.target_name].values.astype(float)
                y_sec = test_sec_df[self.target_name].values.astype(float)

                prim_pred_map: Dict[str, Dict[str, float]] = {cid: {} for cid in test_prim_df["cell_id"]}
                sec_pred_map: Dict[str, Dict[str, float]] = {cid: {} for cid in test_sec_df["cell_id"]}
                train_pred_map: Dict[str, Dict[str, float]] = {cid: {} for cid in train_df["cell_id"]}

                for m_key in ["M1", "M2", "M3", "M4"]:
                    m_info = MODEL_CONFIGS[m_key]
                    features = get_feature_names_for_model(m_key)

                    # 1. 5-Fold Stratified Cross-Validation on Training Set ONLY
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
                    oof_probs = np.zeros(len(train_df), dtype=float)

                    for train_idx, val_idx in skf.split(train_df[features], y_train):
                        cv_pipe = ModelFactory.build_pipeline(
                            algorithm=algo_key, random_state=self.random_state
                        )
                        cv_pipe.fit(train_df.iloc[train_idx][features], y_train[train_idx])
                        probs = cv_pipe.predict_proba(train_df.iloc[val_idx][features])[:, 1]
                        oof_probs[val_idx] = probs

                    # 2. Optimal Threshold and Calibration fit strictly on OOF predictions
                    opt_threshold = ModelEvaluator.select_optimal_threshold(y_train, oof_probs)
                    calibrator = ProbabilityCalibrator()
                    calibrator.fit_on_oof_predictions(y_train, oof_probs)

                    train_metrics = ModelEvaluator.evaluate(
                        y_train, oof_probs, threshold=opt_threshold, partition_name="train_cv"
                    )

                    # 3. Fit Final Pipeline on Full Training Partition
                    final_pipe = ModelFactory.build_pipeline(
                        algorithm=algo_key, random_state=self.random_state
                    )
                    final_pipe.fit(train_df[features], y_train)

                    # 4. Predict on Held-Out Primary Test
                    prim_raw_probs = final_pipe.predict_proba(test_prim_df[features])[:, 1]
                    prim_cal_probs = calibrator.calibrate(prim_raw_probs)
                    prim_metrics = ModelEvaluator.evaluate(
                        y_prim, prim_raw_probs, threshold=opt_threshold, partition_name="primary_test"
                    )

                    # 5. Predict on Held-Out Secondary Test (Batch 3 domain transfer)
                    sec_raw_probs = final_pipe.predict_proba(test_sec_df[features])[:, 1]
                    sec_cal_probs = calibrator.calibrate(sec_raw_probs)
                    sec_metrics = ModelEvaluator.evaluate(
                        y_sec, sec_raw_probs, threshold=opt_threshold, partition_name="secondary_test"
                    )

                    logger.info(
                        "[%s | %s | Cutoff %d] Primary Test: AUROC=%.4f, AUPRC=%.4f, Brier=%.4f",
                        algo_key,
                        m_key,
                        cutoff,
                        prim_metrics["auroc"],
                        prim_metrics["auprc"],
                        prim_metrics["brier_score"],
                    )

                    # Store per-battery predictions
                    for cid, p_raw in zip(test_prim_df["cell_id"], prim_raw_probs):
                        prim_pred_map[cid][m_key] = float(p_raw)
                    for cid, p_raw in zip(test_sec_df["cell_id"], sec_raw_probs):
                        sec_pred_map[cid][m_key] = float(p_raw)
                    for cid, p_raw in zip(train_df["cell_id"], oof_probs):
                        train_pred_map[cid][m_key] = float(p_raw)

                    # Append metric records
                    for met, part_label in [
                        (train_metrics, "train_cv"),
                        (prim_metrics, "primary_test"),
                        (sec_metrics, "secondary_test"),
                    ]:
                        rec = {
                            "algorithm": algo_key,
                            "algorithm_name": algo_name,
                            "algorithm_family": algo_meta.get("family", "Unknown"),
                            "cutoff": cutoff,
                            "model": m_key,
                            "model_name": m_info["name"],
                            "feature_count": len(features),
                            "partition": part_label,
                            "n_samples": met["n_samples"],
                            "positive_count": met["positive_count"],
                            "negative_count": met["negative_count"],
                            "prevalence": met["prevalence"],
                            "auroc": met["auroc"],
                            "auprc": met["auprc"],
                            "brier_score": met["brier_score"],
                            "ece": met["ece"],
                            "threshold": met["threshold"],
                            "tp": met["tp"],
                            "fp": met["fp"],
                            "tn": met["tn"],
                            "fn": met["fn"],
                            "sensitivity": met["sensitivity"],
                            "specificity": met["specificity"],
                            "fpr": met["fpr"],
                            "precision": met["precision"],
                            "f1_score": met["f1_score"],
                            "accuracy": met["accuracy"],
                            "status": met["status"],
                        }
                        metrics_records.append(rec)

                # Store per-battery rows for this algo and cutoff
                for df_sub, pred_map, part_name in [
                    (train_df, train_pred_map, "train"),
                    (test_prim_df, prim_pred_map, "primary_test"),
                    (test_sec_df, sec_pred_map, "secondary_test"),
                ]:
                    for _, row in df_sub.iterrows():
                        cid = str(row["cell_id"])
                        predictions_records.append({
                            "algorithm": algo_key,
                            "cell_id": cid,
                            "batch": str(row.get("batch", "")),
                            "cutoff": int(cutoff),
                            "benchmark_partition": part_name,
                            "target_name": self.target_name,
                            "ground_truth": float(row.get(self.target_name, np.nan)),
                            "prob_M1": pred_map[cid].get("M1", np.nan),
                            "prob_M2": pred_map[cid].get("M2", np.nan),
                            "prob_M3": pred_map[cid].get("M3", np.nan),
                            "prob_M4": pred_map[cid].get("M4", np.nan),
                            "pmri": float(row.get("pmri", np.nan)),
                        })

        df_metrics = pd.DataFrame(metrics_records)
        df_preds = pd.DataFrame(predictions_records)

        # Save Artifacts
        out_metrics = self.results_dir / "multi_algorithm_metrics.csv"
        out_preds = self.results_dir / "multi_algorithm_predictions.csv"
        out_config = self.results_dir / "multi_algorithm_config.json"
        out_report = self.results_dir / "multi_algorithm_report.md"

        df_metrics.to_csv(out_metrics, index=False)
        df_preds.to_csv(out_preds, index=False)

        config_data = {
            "algorithms": self.algorithms,
            "algorithm_metadata": SUPPORTED_ALGORITHMS,
            "cutoffs": self.cutoffs,
            "primary_target": self.target_name,
            "random_seed": self.random_state,
            "validation_protocol": "5_fold_stratified_cv_train_only",
            "threshold_selection": "Youden_J_on_train_cv",
            "calibration": "Platt_scaling_on_train_cv",
            "system_info": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        with open(out_config, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        self._write_report(out_report, df_metrics)
        logger.info("Multi-Algorithm Benchmark completed successfully.")
        return {"metrics_df": df_metrics, "predictions_df": df_preds}

    def _write_report(self, report_path: Path, df_metrics: pd.DataFrame):
        """Generates a structured comparative report across algorithms."""
        prim = df_metrics[df_metrics["partition"] == "primary_test"].copy()

        lines = [
            "# Multi-Algorithm Machine Learning Benchmark Report",
            "",
            "**Project Title:** Confidence-Aware Peer-Relative Degradation Trajectory Analysis for Early End-of-Life Risk Prediction in Lithium-Ion Batteries  ",
            "**Validation Protocol:** 5-Fold Stratified Training CV (Threshold & Calibration Tuning strictly Training-Only)  ",
            "**Primary Test Partition:** N = 43 independent cells (shared batch domain)  ",
            f"**Target Horizon:** `{self.target_name}` (Failure within 500 cycles post-cutoff)  ",
            "",
            "---",
            "",
            "## 1. Primary Test Partition Comparison (AUROC, AUPRC, Brier Score)",
            "",
            "| Cutoff | Algorithm | Family | Model | AUROC | AUPRC | Brier Score | Sensitivity | Specificity | F1-Score |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for cutoff in self.cutoffs:
            sub = prim[prim["cutoff"] == cutoff].sort_values(["algorithm", "model"])
            for _, r in sub.iterrows():
                lines.append(
                    f"| **{cutoff}** | `{r['algorithm']}` | {r['algorithm_family']} | **{r['model']}** | "
                    f"{r['auroc']:.4f} | {r['auprc']:.4f} | {r['brier_score']:.4f} | "
                    f"{r['sensitivity']:.4f} | {r['specificity']:.4f} | {r['f1_score']:.4f} |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## 2. Algorithm Summary Across Cutoffs (M4 Configuration)",
            "",
            "| Algorithm | Cutoff 40 AUROC | Cutoff 70 AUROC | Cutoff 100 AUROC | Cutoff 40 Brier | Cutoff 70 Brier | Cutoff 100 Brier |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for algo in self.algorithms:
            sub = prim[(prim["algorithm"] == algo) & (prim["model"] == "M4")]
            a40 = sub[sub["cutoff"] == 40]["auroc"].values[0] if len(sub[sub["cutoff"] == 40]) else np.nan
            a70 = sub[sub["cutoff"] == 70]["auroc"].values[0] if len(sub[sub["cutoff"] == 70]) else np.nan
            a100 = sub[sub["cutoff"] == 100]["auroc"].values[0] if len(sub[sub["cutoff"] == 100]) else np.nan

            b40 = sub[sub["cutoff"] == 40]["brier_score"].values[0] if len(sub[sub["cutoff"] == 40]) else np.nan
            b70 = sub[sub["cutoff"] == 70]["brier_score"].values[0] if len(sub[sub["cutoff"] == 70]) else np.nan
            b100 = sub[sub["cutoff"] == 100]["brier_score"].values[0] if len(sub[sub["cutoff"] == 100]) else np.nan

            lines.append(
                f"| `{algo}` | {a40:.4f} | {a70:.4f} | {a100:.4f} | {b40:.4f} | {b70:.4f} | {b100:.4f} |"
            )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    benchmark = MultiAlgorithmBenchmark()
    benchmark.run()
