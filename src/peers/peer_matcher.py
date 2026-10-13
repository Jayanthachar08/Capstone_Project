"""
Module: src.peers.peer_matcher
Author: Capstone Project Team
Description: Similarity-based peer matching subsystem for lithium-ion batteries.
             Identifies nearest reference batteries from a permitted training pool
             using standardized weighted Euclidean distance over early-life features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Core matching features capturing operational protocol, initial state, and early impedance
DEFAULT_MATCHING_FEATURES: List[str] = [
    "init_policy_c1",
    "init_policy_soc",
    "init_policy_c2",
    "init_q_discharge",
    "init_internal_resistance",
    "init_temp_avg",
    "snap_internal_resistance",
]

# Physically motivated weights: 50% charging stress regime, 35% initial health state, 15% cutoff impedance
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "init_policy_c1": 0.20,
    "init_policy_soc": 0.15,
    "init_policy_c2": 0.15,
    "init_q_discharge": 0.15,
    "init_internal_resistance": 0.15,
    "init_temp_avg": 0.10,
    "snap_internal_resistance": 0.10,
}


class PeerMatcher:
    """Selects top-k nearest reference batteries from the permitted training pool

    strictly using measurements available up to the given observation cutoff.
    """

    def __init__(
        self,
        k: int = 5,
        min_k: int = 3,
        features: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initializes the peer matcher with configurable k and distance weights.

        Args:
            k: Target number of nearest peers to select (default: 5).
            min_k: Minimum number of valid peers required to declare match success (default: 3).
            features: List of feature names to use for matching.
            weights: Dictionary of feature weights (will be normalized to sum to 1.0).
        """
        self.k = int(k)
        self.min_k = int(min_k)
        self.features = list(features) if features is not None else list(DEFAULT_MATCHING_FEATURES)

        raw_weights = dict(weights) if weights is not None else dict(DEFAULT_FEATURE_WEIGHTS)
        # Normalize weights to sum to 1.0 across selected features
        total_w = sum(raw_weights.get(f, 1.0) for f in self.features)
        if total_w <= 0:
            raise ValueError("Sum of feature weights must be positive.")
        self.weights = {f: raw_weights.get(f, 1.0) / total_w for f in self.features}

        self.weights_arr: np.ndarray = np.array([self.weights[f] for f in self.features], dtype=np.float64)

        # Fitted training reference bank statistics
        self.is_fitted: bool = False
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.reference_df: Optional[pd.DataFrame] = None
        self.reference_z: Optional[np.ndarray] = None
        self.reference_cell_ids: Optional[List[str]] = None

    def fit(self, df_train_reference: pd.DataFrame) -> PeerMatcher:
        """Fits standardization scalers and stores reference feature vectors

        strictly from permitted training partition cells.

        Args:
            df_train_reference: DataFrame containing training reference cells at a single cutoff.
        """
        if df_train_reference.empty:
            raise ValueError("Cannot fit PeerMatcher on empty reference DataFrame.")

        # Ensure reference contains only training partition cells
        invalid_partitions = set(df_train_reference["benchmark_partition"].dropna()) - {"train"}
        if invalid_partitions:
            raise ValueError(
                f"Data leakage violation! Reference DataFrame contains non-training cells: {invalid_partitions}"
            )

        # Verify all matching features exist
        missing = [f for f in self.features if f not in df_train_reference.columns]
        if missing:
            raise KeyError(f"Missing required matching features in reference data: {missing}")

        X_ref = df_train_reference[self.features].values.astype(np.float64)

        # Calculate mean and std strictly on training reference pool
        self.means = np.nanmean(X_ref, axis=0)
        stds = np.nanstd(X_ref, axis=0)
        # Prevent division by zero for constant features
        self.stds = np.where(stds < 1e-8, 1.0, stds)

        # Store standardized reference vectors and cell identifiers
        self.reference_z = (X_ref - self.means) / self.stds
        self.reference_cell_ids = list(df_train_reference["cell_id"].astype(str))
        self.reference_df = df_train_reference.copy().reset_index(drop=True)
        self.is_fitted = True

        logger.debug(
            "Fitted PeerMatcher on %d training reference cells with %d features.",
            len(self.reference_cell_ids),
            len(self.features),
        )
        return self

    def standardize(self, x_vec: np.ndarray) -> np.ndarray:
        """Transforms a raw feature vector into standardized coordinates using fitted training stats."""
        if not self.is_fitted:
            raise RuntimeError("PeerMatcher must be fitted before standardizing features.")
        return (x_vec - self.means) / self.stds

    def find_peers_for_cell(
        self,
        target_series: pd.Series,
        out_of_fold: bool = False,
    ) -> Dict[str, Any]:
        """Finds nearest peer batteries for a single target cell.

        Args:
            target_series: pd.Series with metadata and feature columns for the target cell.
            out_of_fold: If True, explicitly excludes target_series['cell_id'] from candidate pool
                        (mandatory when matching training cells against the reference pool).

        Returns:
            Dict containing match status, selected peer IDs, distances, and matching metrics.
        """
        if not self.is_fitted:
            raise RuntimeError("PeerMatcher must be fitted before finding peers.")

        target_id = str(target_series.get("cell_id", ""))
        is_eligible = bool(target_series.get("is_benchmark_eligible", True))
        partition = str(target_series.get("benchmark_partition", ""))

        # Excluded cells cannot have valid benchmark peers
        if not is_eligible or partition == "excluded":
            return {
                "cell_id": target_id,
                "peer_match_status": "EXCLUDED_CELL",
                "peer_count": 0,
                "peer_ids": [],
                "peer_distances": [],
                "mean_peer_distance": np.nan,
                "peer_feature_completeness": 0.0,
            }

        # Extract target features and assess completeness
        target_vals = []
        n_finite = 0
        for f in self.features:
            val = target_series.get(f, np.nan)
            try:
                f_val = float(val)
                target_vals.append(f_val)
                if np.isfinite(f_val):
                    n_finite += 1
            except (ValueError, TypeError):
                target_vals.append(np.nan)

        completeness = float(n_finite / len(self.features))
        target_x = np.array(target_vals, dtype=np.float64)

        # If critical features are completely missing, return insufficient info
        if n_finite < len(self.features) - 1:
            return {
                "cell_id": target_id,
                "peer_match_status": "INSUFFICIENT_FEATURES",
                "peer_count": 0,
                "peer_ids": [],
                "peer_distances": [],
                "mean_peer_distance": np.nan,
                "peer_feature_completeness": completeness,
            }

        target_z = self.standardize(target_x)

        # Compute standardized weighted Euclidean distance to all reference candidates
        # Vectorized over all reference cells
        diff = self.reference_z - target_z  # Shape: (N_ref, n_feats)
        sq_diff = np.square(diff)
        weighted_sq_diff = sq_diff * self.weights_arr  # broadcasts over features
        distances = np.sqrt(np.nansum(weighted_sq_diff, axis=1))

        # Build candidate list, enforcing leave-one-out out-of-fold restriction
        eligible_candidates: List[Tuple[str, float]] = []
        for ref_id, dist in zip(self.reference_cell_ids, distances):
            # A battery can NEVER match itself
            if ref_id == target_id:
                continue
            if np.isfinite(dist):
                eligible_candidates.append((ref_id, float(dist)))

        # Check candidate availability
        if len(eligible_candidates) < self.min_k:
            return {
                "cell_id": target_id,
                "peer_match_status": "INSUFFICIENT_PEERS",
                "peer_count": len(eligible_candidates),
                "peer_ids": [c[0] for c in eligible_candidates],
                "peer_distances": [c[1] for c in eligible_candidates],
                "mean_peer_distance": (
                    float(np.mean([c[1] for c in eligible_candidates]))
                    if eligible_candidates
                    else np.nan
                ),
                "peer_feature_completeness": completeness,
            }

        # Sort by distance ascending and pick top k
        eligible_candidates.sort(key=lambda item: item[1])
        top_k = eligible_candidates[: self.k]
        top_ids = [c[0] for c in top_k]
        top_dists = [c[1] for c in top_k]
        mean_dist = float(np.mean(top_dists))

        return {
            "cell_id": target_id,
            "peer_match_status": "MATCH_SUCCESS",
            "peer_count": len(top_k),
            "peer_ids": top_ids,
            "peer_distances": top_dists,
            "mean_peer_distance": mean_dist,
            "peer_feature_completeness": completeness,
        }
