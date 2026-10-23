"""
Module: src.models.adaptive_peer_trust
Author: Capstone Project Team
Description: Adaptive Peer-Trust and Counterfactual Risk Engine (M5).
             Combines static covariate peer reliability (PMRI) with dynamic counterfactual
             peer-set instability (U) to adaptively balance peer-relative and historical
             degradation signals, providing calibrated risk estimates and selective abstention.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA_CANDIDATES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]


@dataclass
class M5PredictionRecord:
    """Detailed record for a single M5 adaptive trust prediction."""

    cell_id: str
    cutoff: int
    benchmark_partition: str
    p_history: float  # M2 individual trajectory probability
    p_peer: float  # M3 peer-relative probability
    pmri: float  # R: Covariate match reliability
    instability: float  # U: Counterfactual prediction standard deviation
    lambda_param: float  # Instability sensitivity penalty
    instability_penalty: float  # exp(-lambda * U)
    trust_score: float  # T = R * exp(-lambda * U)
    p_m5: float  # Blended probability: T * p_peer + (1-T) * p_history
    decision_threshold: float
    is_elevated_risk: bool
    risk_category: str  # "ELEVATED_RISK" vs "NOMINAL_RISK"
    abstained: bool
    abstention_reason: str
    ablation_mode: str = "full"


class AdaptivePeerTrustEngine:
    """Computes adaptive peer-trust scores and dynamically blends peer-relative

    and history-only degradation risk models while enforcing rigorous anti-leakage rules.
    """

    def __init__(
        self,
        lambda_param: float = 3.0,
        pmri_abstention_threshold: float = 0.50,
        instability_abstention_threshold: Optional[float] = 0.15,
        trust_abstention_threshold: Optional[float] = 0.30,
        random_seed: int = 42,
    ):
        """Initializes the adaptive trust engine.

        Args:
            lambda_param: Instability penalty coefficient in T = R * exp(-lambda * U).
            pmri_abstention_threshold: Advisory PMRI cutoff (default: 0.50).
            instability_abstention_threshold: Threshold above which U triggers instability abstention.
            trust_abstention_threshold: Threshold below which composite trust T triggers abstention.
            random_seed: Deterministic random seed.
        """
        self.lambda_param = float(lambda_param)
        self.pmri_abstention_threshold = float(pmri_abstention_threshold)
        self.instability_abstention_threshold = (
            float(instability_abstention_threshold)
            if instability_abstention_threshold is not None
            else None
        )
        self.trust_abstention_threshold = (
            float(trust_abstention_threshold)
            if trust_abstention_threshold is not None
            else None
        )
        self.random_seed = int(random_seed)
        self.is_tuned = False

    @staticmethod
    def calculate_trust(
        pmri: float,
        instability: float,
        lambda_param: float,
        ablation_mode: str = "full",
    ) -> Tuple[float, float]:
        """Calculates adaptive peer-trust score T in [0, 1].

        Ablation Modes:
        - "full": T = R * exp(-lambda * U)
        - "no_pmri": T = exp(-lambda * U) (ignoring R, assuming R=1)
        - "no_instability": T = R (lambda = 0, ignoring counterfactual variance U)
        - "fixed_blend": T = 0.5 (constant 50/50 blending)

        Returns:
            Tuple of (trust_score T, instability_penalty exp(-lambda * U)).
        """
        # Validate inputs
        r = float(pmri) if np.isfinite(pmri) else 0.0
        r = max(0.0, min(1.0, r))

        u = float(instability) if np.isfinite(instability) else 0.0
        u = max(0.0, u)

        lam = max(0.0, float(lambda_param))

        penalty = float(np.exp(-lam * u))

        if ablation_mode == "full":
            t = r * penalty
        elif ablation_mode == "no_pmri":
            t = penalty
        elif ablation_mode == "no_instability":
            t = r
        elif ablation_mode == "fixed_blend":
            t = 0.5
        else:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")

        t = max(0.0, min(1.0, t))
        return t, penalty

    @classmethod
    def blend_probability(
        cls,
        p_history: float,
        p_peer: float,
        trust_score: float,
    ) -> float:
        """Blends history-only and peer-relative probabilities using trust score T.

        p_M5 = T * p_peer + (1 - T) * p_history
        """
        p_h = float(p_history) if np.isfinite(p_history) else 0.5
        p_p = float(p_peer) if np.isfinite(p_peer) else 0.5

        p_h = max(0.0, min(1.0, p_h))
        p_p = max(0.0, min(1.0, p_p))
        t = max(0.0, min(1.0, float(trust_score)))

        p_m5 = t * p_p + (1.0 - t) * p_h
        return max(0.0, min(1.0, float(p_m5)))

    def evaluate_abstention(
        self,
        pmri: float,
        instability: float,
        trust_score: float,
        peer_match_status: str = "MATCH_SUCCESS",
    ) -> Tuple[bool, str]:
        """Evaluates whether to abstain from automated decision making and returns the documented reason."""
        if peer_match_status != "MATCH_SUCCESS":
            return True, f"UNSUPPORTED_OPERATING_CONDITIONS ({peer_match_status})"

        if np.isnan(pmri) or pmri < self.pmri_abstention_threshold:
            return True, f"INSUFFICIENT_REFERENCE_COVERAGE (PMRI={pmri:.3f} < {self.pmri_abstention_threshold:.2f})"

        if (
            self.instability_abstention_threshold is not None
            and instability > self.instability_abstention_threshold
        ):
            return True, f"EXCESSIVE_COUNTERFACTUAL_INSTABILITY (U={instability:.3f} > {self.instability_abstention_threshold:.2f})"

        if (
            self.trust_abstention_threshold is not None
            and trust_score < self.trust_abstention_threshold
        ):
            return True, f"LOW_ADAPTIVE_TRUST (T={trust_score:.3f} < {self.trust_abstention_threshold:.2f})"

        return False, "NOMINAL_TRUST"

    def predict_single(
        self,
        cell_id: str,
        cutoff: int,
        benchmark_partition: str,
        p_history: float,
        p_peer: float,
        pmri: float,
        instability: float,
        decision_threshold: float = 0.50,
        ablation_mode: str = "full",
        peer_match_status: str = "MATCH_SUCCESS",
    ) -> M5PredictionRecord:
        """Generates an M5 prediction record for a single battery cell."""
        t, penalty = self.calculate_trust(
            pmri=pmri,
            instability=instability,
            lambda_param=self.lambda_param,
            ablation_mode=ablation_mode,
        )

        p_m5 = self.blend_probability(
            p_history=p_history,
            p_peer=p_peer,
            trust_score=t,
        )

        is_elevated = bool(p_m5 >= decision_threshold)
        risk_cat = "ELEVATED_RISK" if is_elevated else "NOMINAL_RISK"

        abstained, reason = self.evaluate_abstention(
            pmri=pmri,
            instability=instability,
            trust_score=t,
            peer_match_status=peer_match_status,
        )

        return M5PredictionRecord(
            cell_id=cell_id,
            cutoff=cutoff,
            benchmark_partition=benchmark_partition,
            p_history=p_history,
            p_peer=p_peer,
            pmri=pmri,
            instability=instability,
            lambda_param=self.lambda_param,
            instability_penalty=penalty,
            trust_score=t,
            p_m5=p_m5,
            decision_threshold=decision_threshold,
            is_elevated_risk=is_elevated,
            risk_category=risk_cat,
            abstained=abstained,
            abstention_reason=reason,
            ablation_mode=ablation_mode,
        )

    def tune_lambda_on_training_cv(
        self,
        train_y: np.ndarray,
        train_p_history: np.ndarray,
        train_p_peer: np.ndarray,
        train_pmri: np.ndarray,
        train_instability: np.ndarray,
        lambda_candidates: Optional[List[float]] = None,
        objective: str = "brier",
    ) -> float:
        """Selects optimal lambda strictly on training partition out-of-fold predictions.

        NEVER accesses test labels. Optimizes Brier score loss.
        """
        candidates = lambda_candidates or DEFAULT_LAMBDA_CANDIDATES
        best_lambda = candidates[0]
        best_score = float("inf")

        for lam in candidates:
            # Vectorized blend calculation for this lambda
            p_blended = []
            for y_t, ph, pp, r, u in zip(
                train_y, train_p_history, train_p_peer, train_pmri, train_instability
            ):
                t, _ = self.calculate_trust(r, u, lam, ablation_mode="full")
                pm5 = self.blend_probability(ph, pp, t)
                p_blended.append(pm5)

            p_arr = np.array(p_blended)
            score = float(brier_score_loss(train_y, p_arr))

            logger.debug("Tuning lambda=%.2f -> Training CV Brier=%.5f", lam, score)
            if score < best_score:
                best_score = score
                best_lambda = lam

        logger.info(
            "Selected optimal lambda=%.2f on training partition (Training Brier=%.5f)",
            best_lambda,
            best_score,
        )
        self.lambda_param = best_lambda
        self.is_tuned = True
        return best_lambda
