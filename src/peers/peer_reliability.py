"""
Module: src.peers.peer_reliability
Author: Capstone Project Team
Description: Peer-Match Reliability Index (PMRI) and reliability warning subsystem.
             Measures the suitability and quality of peer comparisons using four
             interpretable dimensions: similarity, agreement, availability, and completeness.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default transparent PMRI component weights summing to 1.0
DEFAULT_PMRI_WEIGHTS: Dict[str, float] = {
    "similarity": 0.35,
    "agreement": 0.30,
    "availability": 0.20,
    "completeness": 0.15,
}

# Calibration scale constants for exponential decay scoring
DEFAULT_DISTANCE_SCALE: float = 2.0  # d0: characteristic standardized distance scale
DEFAULT_SLOPE_DISPERSION_SCALE: float = 5e-5  # sigma0: characteristic slope dispersion (Ah/cycle)
LOW_PMRI_THRESHOLD: float = 0.50
HIGH_DISTANCE_THRESHOLD: float = 2.0
HIGH_DISPERSION_THRESHOLD: float = 1e-4


class PeerReliabilityEvaluator:
    """Calculates the Peer-Match Reliability Index (PMRI) and flags potential warnings.

    NOTE: PMRI is an interpretable heuristic metric measuring peer match quality,
    NOT a calibrated probability of battery failure or end-of-life risk.
    """

    def __init__(
        self,
        target_k: int = 5,
        min_k: int = 3,
        d0: float = DEFAULT_DISTANCE_SCALE,
        sigma0: float = DEFAULT_SLOPE_DISPERSION_SCALE,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initializes the reliability evaluator.

        Args:
            target_k: Target number of peers expected (default: 5).
            min_k: Minimum acceptable peers to consider match valid (default: 3).
            d0: Characteristic scale parameter for distance decay (default: 2.0).
            sigma0: Characteristic scale parameter for slope agreement decay (default: 5e-5 Ah/cycle).
            weights: Component weights for PMRI composition (default: 0.35, 0.30, 0.20, 0.15).
        """
        self.target_k = int(target_k)
        self.min_k = int(min_k)
        self.d0 = float(d0)
        self.sigma0 = float(sigma0)

        raw_weights = dict(weights) if weights is not None else dict(DEFAULT_PMRI_WEIGHTS)
        total_w = sum(raw_weights.values())
        if total_w <= 0:
            raise ValueError("PMRI weights must sum to a positive number.")
        self.weights = {k: v / total_w for k, v in raw_weights.items()}

    def calculate_similarity_score(self, mean_distance: float) -> float:
        """Computes peer similarity score S_sim in [0.0, 1.0] via exponential decay.

        Formula: S_sim = exp(-mean_distance / d0)
        """
        if not np.isfinite(mean_distance) or mean_distance < 0:
            return 0.0
        return float(np.clip(np.exp(-mean_distance / self.d0), 0.0, 1.0))

    def calculate_agreement_score(self, peer_slope_dispersion: float) -> float:
        """Computes peer trajectory agreement score S_agree in [0.0, 1.0].

        High agreement corresponds to low variance among peer degradation slopes.
        Formula: S_agree = exp(-peer_slope_dispersion / sigma0)
        """
        if not np.isfinite(peer_slope_dispersion) or peer_slope_dispersion < 0:
            return 0.0
        return float(np.clip(np.exp(-peer_slope_dispersion / self.sigma0), 0.0, 1.0))

    def calculate_availability_score(self, k_matched: int) -> float:
        """Computes peer availability score S_avail in [0.0, 1.0].

        Full score if k_matched == target_k; penalized if fewer; 0.0 if below min_k.
        """
        if k_matched < self.min_k:
            return 0.0
        return float(np.clip(k_matched / self.target_k, 0.0, 1.0))

    def calculate_completeness_score(self, feature_completeness: float) -> float:
        """Validates and bounds feature completeness in [0.0, 1.0]."""
        if not np.isfinite(feature_completeness):
            return 0.0
        return float(np.clip(feature_completeness, 0.0, 1.0))

    def evaluate_reliability(
        self,
        match_result: Dict[str, Any],
        comparison_result: Dict[str, float],
    ) -> Dict[str, Any]:
        """Evaluates PMRI composite index and determines reliability warnings.

        Args:
            match_result: Output from PeerMatcher.find_peers_for_cell().
            comparison_result: Output from TrajectoryComparator.compare_target_to_peers().

        Returns:
            Dict containing:
            - peer_similarity_score: float in [0, 1]
            - peer_agreement_score: float in [0, 1]
            - peer_availability_score: float in [0, 1]
            - peer_feature_completeness: float in [0, 1]
            - pmri: float in [0, 1] (or NaN for excluded cells)
            - reliability_warning: str description of warning flags
        """
        status = str(match_result.get("peer_match_status", "UNKNOWN"))

        # Excluded cell handling
        if status == "EXCLUDED_CELL":
            return {
                "peer_similarity_score": np.nan,
                "peer_agreement_score": np.nan,
                "peer_availability_score": 0.0,
                "peer_feature_completeness": 0.0,
                "pmri": np.nan,
                "reliability_warning": "EXCLUDED_CELL",
            }

        # Insufficient features or peers handling
        if status in ("INSUFFICIENT_PEERS", "INSUFFICIENT_FEATURES"):
            completeness = float(match_result.get("peer_feature_completeness", 0.0))
            return {
                "peer_similarity_score": 0.0,
                "peer_agreement_score": 0.0,
                "peer_availability_score": 0.0,
                "peer_feature_completeness": completeness,
                "pmri": 0.0,
                "reliability_warning": status,
            }

        k_matched = int(match_result.get("peer_count", 0))
        mean_dist = float(match_result.get("mean_peer_distance", np.nan))
        slope_disp = float(comparison_result.get("peer_slope_dispersion", np.nan))
        completeness = float(match_result.get("peer_feature_completeness", 1.0))

        # Compute component scores
        s_sim = self.calculate_similarity_score(mean_dist)
        s_agree = self.calculate_agreement_score(slope_disp)
        s_avail = self.calculate_availability_score(k_matched)
        s_comp = self.calculate_completeness_score(completeness)

        # Composite PMRI
        pmri = (
            self.weights["similarity"] * s_sim
            + self.weights["agreement"] * s_agree
            + self.weights["availability"] * s_avail
            + self.weights["completeness"] * s_comp
        )
        pmri = float(np.clip(pmri, 0.0, 1.0))

        # Compile diagnostic warning flags
        warnings: List[str] = []
        if k_matched < self.target_k:
            warnings.append("LOW_PEER_COUNT")
        if mean_dist > HIGH_DISTANCE_THRESHOLD:
            warnings.append("HIGH_PEER_DISTANCE")
        if slope_disp > HIGH_DISPERSION_THRESHOLD:
            warnings.append("HIGH_PEER_DISPERSION")
        if pmri < LOW_PMRI_THRESHOLD:
            warnings.append("LOW_RELIABILITY")

        warning_str = "; ".join(warnings) if warnings else "NONE"

        return {
            "peer_similarity_score": s_sim,
            "peer_agreement_score": s_agree,
            "peer_availability_score": s_avail,
            "peer_feature_completeness": s_comp,
            "pmri": pmri,
            "reliability_warning": warning_str,
        }
