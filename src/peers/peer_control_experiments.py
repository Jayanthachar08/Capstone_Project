"""
Module: src.peers.peer_control_experiments
Author: Capstone Project Team
Description: Rigorous multi-seed randomized peer control experiments and PMRI empirical validation.
             Evaluates empirical null distributions over 20 random permutation seeds to prove
             that genuine peer-relative predictive signal is non-random, and validates PMRI association
             with peer similarity and prediction error.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.models.model_factory import FEATURE_GROUPS, ModelFactory, get_feature_names_for_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRIMARY_TARGET = "target_fail_within_500"
CUTOFFS = [40, 70, 100]
NUM_SEEDS = 20


class MultiSeedPeerControl:
    """Executes multi-seed negative peer control permutations and PMRI statistical analysis."""

    def __init__(
        self,
        modeling_csv: str = "data/processed/modeling_dataset.csv",
        peer_csv: str = "data/processed/peer_features.csv",
        results_dir: str = "results",
        target_name: str = PRIMARY_TARGET,
        n_seeds: int = NUM_SEEDS,
        seed_start: int = 42,
    ):
        self.modeling_csv = Path(modeling_csv)
        self.peer_csv = Path(peer_csv)
        self.results_dir = Path(results_dir)
        self.target_name = target_name
        self.n_seeds = n_seeds
        self.seeds = [seed_start + i for i in range(n_seeds)]

    def load_merged_data(self) -> pd.DataFrame:
        """Loads and merges modeling dataset with peer features."""
        df_model = pd.read_csv(self.modeling_csv)
        df_peers = pd.read_csv(self.peer_csv)
        merge_keys = ["cell_id", "cutoff"]
        peer_cols_to_add = [
            c for c in df_peers.columns if c not in df_model.columns or c in merge_keys
        ]
        return pd.merge(df_model, df_peers[peer_cols_to_add], on=merge_keys)

    def run_permutation_control(self) -> pd.DataFrame:
        """Runs multi-seed randomized peer feature permutations across cutoffs."""
        df_all = self.load_merged_data()
        records: List[Dict[str, Any]] = []

        m3_feats = get_feature_names_for_model("M3")
        peer_traj_feats = FEATURE_GROUPS["peer_traj"]

        logger.info(
            "Executing Multi-Seed Peer Control with %d seeds across cutoffs %s...",
            self.n_seeds,
            CUTOFFS,
        )

        for cutoff in CUTOFFS:
            sub = df_all[df_all["cutoff"] == cutoff].copy()
            train_sub = sub[sub["benchmark_partition"] == "train"].copy().reset_index(drop=True)
            test_sub = sub[sub["benchmark_partition"] == "primary_test"].copy().reset_index(drop=True)

            y_tr = train_sub[self.target_name].values.astype(float)
            y_te = test_sub[self.target_name].values.astype(float)

            # Fit Genuine M3 Model
            pipe_genuine = ModelFactory.build_logistic_regression(random_state=42)
            pipe_genuine.fit(train_sub[m3_feats], y_tr)
            p_gen = pipe_genuine.predict_proba(test_sub[m3_feats])[:, 1]
            genuine_auroc = float(roc_auc_score(y_te, p_gen))
            genuine_brier = float(brier_score_loss(y_te, p_gen))

            # Record Genuine Baseline
            records.append({
                "cutoff": cutoff,
                "trial_type": "genuine",
                "seed": 42,
                "auroc": genuine_auroc,
                "brier_score": genuine_brier,
                "delta_auroc_vs_genuine": 0.0,
                "delta_brier_vs_genuine": 0.0,
            })

            # Run Multi-Seed Permutations
            null_aurocs = []
            null_briers = []

            for s in self.seeds:
                rng = np.random.RandomState(s)
                train_scrambled = train_sub.copy()
                test_scrambled = test_sub.copy()

                # Independently shuffle each peer-relative feature to break peer pairing
                for feat in peer_traj_feats:
                    train_scrambled[feat] = rng.permutation(train_sub[feat].values)
                    test_scrambled[feat] = rng.permutation(test_sub[feat].values)

                pipe_scrambled = ModelFactory.build_logistic_regression(random_state=s)
                pipe_scrambled.fit(train_scrambled[m3_feats], y_tr)
                p_scr = pipe_scrambled.predict_proba(test_scrambled[m3_feats])[:, 1]

                scr_auroc = float(roc_auc_score(y_te, p_scr))
                scr_brier = float(brier_score_loss(y_te, p_scr))

                null_aurocs.append(scr_auroc)
                null_briers.append(scr_brier)

                records.append({
                    "cutoff": cutoff,
                    "trial_type": "randomized_null",
                    "seed": s,
                    "auroc": scr_auroc,
                    "brier_score": scr_brier,
                    "delta_auroc_vs_genuine": scr_auroc - genuine_auroc,
                    "delta_brier_vs_genuine": scr_brier - genuine_brier,
                })

            mean_null_auc = float(np.mean(null_aurocs))
            std_null_auc = float(np.std(null_aurocs))
            mean_null_brier = float(np.mean(null_briers))
            p_val_auc = float(np.mean([1.0 if a >= genuine_auroc else 0.0 for a in null_aurocs]))

            logger.info(
                "[Cutoff %d] Genuine AUROC=%.4f vs Null Mean=%.4f +/- %.4f (p=%.4f) | Genuine Brier=%.4f vs Null Mean=%.4f",
                cutoff,
                genuine_auroc,
                mean_null_auc,
                std_null_auc,
                p_val_auc,
                genuine_brier,
                mean_null_brier,
            )

        df_res = pd.DataFrame(records)
        out_csv = self.results_dir / "peer_randomization_control.csv"
        df_res.to_csv(out_csv, index=False)
        logger.info("Saved multi-seed peer control results to %s", out_csv)
        return df_res

    def run_pmri_validation(self) -> Dict[str, Any]:
        """Validates statistical association between PMRI, peer similarity, and prediction error."""
        df_all = self.load_merged_data()
        preds_p5 = pd.read_csv(self.results_dir / "phase5_predictions.csv")

        # Merge with M4 predictions
        merged = pd.merge(
            preds_p5,
            df_all[["cell_id", "cutoff", "mean_peer_distance", "peer_similarity_score", "peer_agreement_score"]],
            on=["cell_id", "cutoff"],
            how="left",
        )

        prim = merged[merged["benchmark_partition"] == "primary_test"].copy()
        prim["abs_error_m4"] = (prim["ground_truth"] - prim["prob_M4"]).abs()

        summary: Dict[str, Any] = {"by_cutoff": {}}

        for cutoff in CUTOFFS:
            sub = prim[prim["cutoff"] == cutoff].copy()
            # Pearson and Spearman correlation between PMRI and peer distance
            r_dist, p_dist = pearsonr(sub["pmri"], sub["mean_peer_distance"])
            # Correlation between PMRI and prediction error
            r_err, p_err = spearmanr(sub["pmri"], sub["abs_error_m4"])

            # Error in High PMRI vs Low PMRI (median split)
            med_pmri = float(sub["pmri"].median())
            high_sub = sub[sub["pmri"] >= med_pmri]
            low_sub = sub[sub["pmri"] < med_pmri]

            summary["by_cutoff"][str(cutoff)] = {
                "n_samples": len(sub),
                "pmri_median": med_pmri,
                "pmri_mean": float(sub["pmri"].mean()),
                "pmri_std": float(sub["pmri"].std()),
                "corr_pmri_vs_distance": {"r": float(r_dist), "p_value": float(p_dist)},
                "corr_pmri_vs_error": {"rho": float(r_err), "p_value": float(p_err)},
                "high_pmri_mae": float(high_sub["abs_error_m4"].mean()),
                "low_pmri_mae": float(low_sub["abs_error_m4"].mean()),
                "high_pmri_brier": float(brier_score_loss(high_sub["ground_truth"], high_sub["prob_M4"])),
                "low_pmri_brier": float(brier_score_loss(low_sub["ground_truth"], low_sub["prob_M4"])),
            }

        out_json = self.results_dir / "pmri_reliability_analysis.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved PMRI validation analysis to %s", out_json)
        return summary


if __name__ == "__main__":
    controller = MultiSeedPeerControl()
    controller.run_permutation_control()
    controller.run_pmri_validation()
