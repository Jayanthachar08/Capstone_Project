"""
Module: src.models.benchmark_models
Author: Capstone Project Team
Description: Comprehensive benchmarking executor and analysis engine.
             Independently trains and evaluates 36 experimental configurations:
             3 algorithms (Logistic Regression, RBF-SVM, TabPFN 2.5) x
             4 model configurations (M1, M2, M3, M4) x
             3 early observation cutoffs (40, 70, 100 cycles).
             Saves versioned artifacts and statistical comparison summaries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.models.multi_algorithm_benchmark import MultiAlgorithmBenchmark
from src.models.model_factory import SUPPORTED_ALGORITHMS, MODEL_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark() -> Dict[str, Any]:
    """Executes the full 36-configuration benchmark and generates structured summaries."""
    benchmark_dir = BASE_DIR / "results" / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    benchmark = MultiAlgorithmBenchmark(
        results_dir=str(BASE_DIR / "results"),
        algorithms=["logistic_regression", "rbf_svm", "tabpfn_2_5"],
        cutoffs=[40, 70, 100],
    )
    result = benchmark.run()
    df_metrics: pd.DataFrame = result["metrics_df"]
    df_preds: pd.DataFrame = result["predictions_df"]

    # Save to results/benchmarks directory as well
    metrics_36_csv = benchmark_dir / "benchmark_metrics_36.csv"
    preds_36_csv = benchmark_dir / "benchmark_predictions_36.csv"
    df_metrics.to_csv(metrics_36_csv, index=False)
    df_preds.to_csv(preds_36_csv, index=False)
    logger.info("Saved 36-configuration metrics to %s", metrics_36_csv)
    logger.info("Saved 36-configuration predictions to %s", preds_36_csv)

    # 1. Primary Test Analysis
    prim_df = df_metrics[df_metrics["partition"] == "primary_test"].copy()
    sec_df = df_metrics[df_metrics["partition"] == "secondary_test"].copy()

    # Build comparison summary structure
    comparison_summary: Dict[str, Any] = {
        "title": "36-Configuration Comparative Benchmark (LR vs RBF-SVM vs TabPFN 2.5 across M1-M4)",
        "total_configurations": 36,
        "algorithms": ["logistic_regression", "rbf_svm", "tabpfn_2_5"],
        "models": ["M1", "M2", "M3", "M4"],
        "cutoffs": [40, 70, 100],
        "by_cutoff": {},
        "m4_summary": {},
    }

    for cutoff in [40, 70, 100]:
        c_sub = prim_df[prim_df["cutoff"] == cutoff]
        matrix: Dict[str, Dict[str, Dict[str, float]]] = {}
        for m in ["M1", "M2", "M3", "M4"]:
            matrix[m] = {}
            for a in ["logistic_regression", "rbf_svm", "tabpfn_2_5"]:
                row = c_sub[(c_sub["model"] == m) & (c_sub["algorithm"] == a)]
                if len(row) > 0:
                    r = row.iloc[0]
                    matrix[m][a] = {
                        "auroc": float(r["auroc"]),
                        "auprc": float(r["auprc"]),
                        "brier_score": float(r["brier_score"]),
                        "sensitivity": float(r["sensitivity"]),
                        "specificity": float(r["specificity"]),
                        "f1_score": float(r["f1_score"]),
                        "accuracy": float(r["accuracy"]),
                        "threshold": float(r["threshold"]),
                    }
        comparison_summary["by_cutoff"][str(cutoff)] = matrix

    for a in ["logistic_regression", "rbf_svm", "tabpfn_2_5"]:
        a_sub = prim_df[(prim_df["algorithm"] == a) & (prim_df["model"] == "M4")]
        comparison_summary["m4_summary"][a] = {
            "cutoff_40_auroc": float(a_sub[a_sub["cutoff"] == 40]["auroc"].values[0]) if len(a_sub[a_sub["cutoff"] == 40]) else None,
            "cutoff_70_auroc": float(a_sub[a_sub["cutoff"] == 70]["auroc"].values[0]) if len(a_sub[a_sub["cutoff"] == 70]) else None,
            "cutoff_100_auroc": float(a_sub[a_sub["cutoff"] == 100]["auroc"].values[0]) if len(a_sub[a_sub["cutoff"] == 100]) else None,
            "cutoff_40_brier": float(a_sub[a_sub["cutoff"] == 40]["brier_score"].values[0]) if len(a_sub[a_sub["cutoff"] == 40]) else None,
            "cutoff_70_brier": float(a_sub[a_sub["cutoff"] == 70]["brier_score"].values[0]) if len(a_sub[a_sub["cutoff"] == 70]) else None,
            "cutoff_100_brier": float(a_sub[a_sub["cutoff"] == 100]["brier_score"].values[0]) if len(a_sub[a_sub["cutoff"] == 100]) else None,
        }

    summary_file = benchmark_dir / "algorithm_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)
    logger.info("Saved comparison summary to %s", summary_file)

    # 2. Distribution Shift Analysis (Batch 3 Policy Shift)
    shift_analysis: Dict[str, Any] = {
        "title": "Distribution Shift Analysis: Primary Test (In-Domain) vs Secondary Test (Batch 3 Policy Shift)",
        "comparisons": [],
    }
    for cutoff in [40, 70, 100]:
        for a in ["logistic_regression", "rbf_svm", "tabpfn_2_5"]:
            for m in ["M1", "M2", "M3", "M4"]:
                p_row = prim_df[(prim_df["cutoff"] == cutoff) & (prim_df["algorithm"] == a) & (prim_df["model"] == m)]
                s_row = sec_df[(sec_df["cutoff"] == cutoff) & (sec_df["algorithm"] == a) & (sec_df["model"] == m)]
                if len(p_row) > 0 and len(s_row) > 0:
                    p = p_row.iloc[0]
                    s = s_row.iloc[0]
                    shift_analysis["comparisons"].append({
                        "cutoff": cutoff,
                        "algorithm": a,
                        "model": m,
                        "primary_auroc": float(p["auroc"]),
                        "secondary_auroc": float(s["auroc"]),
                        "delta_auroc": float(s["auroc"] - p["auroc"]),
                        "primary_brier": float(p["brier_score"]),
                        "secondary_brier": float(s["brier_score"]),
                        "delta_brier": float(s["brier_score"] - p["brier_score"]),
                    })

    shift_file = benchmark_dir / "distribution_shift_analysis.json"
    with open(shift_file, "w", encoding="utf-8") as f:
        json.dump(shift_analysis, f, indent=2)
    logger.info("Saved distribution shift analysis to %s", shift_file)

    # 3. Overall Algorithm Ranking
    # Ranking criterion based on average primary AUROC, Brier score, and Secondary generalization
    rankings: List[Dict[str, Any]] = []
    for a in ["logistic_regression", "rbf_svm", "tabpfn_2_5"]:
        a_prim = prim_df[prim_df["algorithm"] == a]
        a_sec = sec_df[sec_df["algorithm"] == a]
        rankings.append({
            "algorithm": a,
            "name": SUPPORTED_ALGORITHMS[a]["name"],
            "family": SUPPORTED_ALGORITHMS[a]["family"],
            "mean_primary_auroc": float(a_prim["auroc"].mean()),
            "mean_primary_brier": float(a_prim["brier_score"].mean()),
            "mean_primary_f1": float(a_prim["f1_score"].mean()),
            "mean_secondary_auroc": float(a_sec["auroc"].mean()),
            "mean_secondary_brier": float(a_sec["brier_score"].mean()),
            "m4_cutoff70_auroc": float(a_prim[(a_prim["model"] == "M4") & (a_prim["cutoff"] == 70)]["auroc"].values[0]),
            "m4_cutoff70_brier": float(a_prim[(a_prim["model"] == "M4") & (a_prim["cutoff"] == 70)]["brier_score"].values[0]),
        })

    rankings.sort(key=lambda x: -x["mean_primary_auroc"])
    for idx, r in enumerate(rankings):
        r["rank"] = idx + 1

    ranking_file = benchmark_dir / "algorithm_ranking.json"
    with open(ranking_file, "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2)
    logger.info("Saved algorithm ranking to %s", ranking_file)

    return {
        "metrics_df": df_metrics,
        "predictions_df": df_preds,
        "comparison_summary": comparison_summary,
        "shift_analysis": shift_analysis,
        "rankings": rankings,
    }


if __name__ == "__main__":
    run_benchmark()
