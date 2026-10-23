"""
Module: src.peers.peer_robustness
Author: Capstone Project Team
Description: Counterfactual Peer Robustness Engine for Lithium-Ion Battery Degradation Analysis.
             Evaluates prediction stability under systematic perturbations of reference peer sets
             (Leave-One-Peer-Out, Single-Peer Replacement, and Sub-Cohort Re-weighting)
             to quantify peer-set dependence (U) and peer-level sensitivity.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.peers.peer_matcher import PeerMatcher
from src.peers.trajectory_comparison import TrajectoryComparator

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualPerturbation:
    """Represents a single counterfactual peer set perturbation and its prediction."""

    perturbation_id: str
    perturbation_type: str  # "ORIGINAL", "LEAVE_ONE_OUT", "REPLACEMENT", "LEAVE_TWO_OUT"
    excluded_peer: Optional[str]
    added_peer: Optional[str]
    peer_ids: List[str]
    predicted_prob: float
    prob_delta: float  # predicted_prob - original_prob
    prediction_flip: bool
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class CellRobustnessSummary:
    """Summary of counterfactual stability metrics for a single cell at a cutoff."""

    cell_id: str
    cutoff: int
    benchmark_partition: str
    is_benchmark_eligible: bool
    original_prob: float
    decision_threshold: float
    original_classification: int
    n_valid_perturbations: int
    counterfactual_mean: float
    counterfactual_std: float  # U: Instability metric
    counterfactual_min: float
    counterfactual_max: float
    counterfactual_range: float
    flip_rate: float
    most_influential_peer: Optional[str]
    max_peer_delta: float
    peer_sensitivities: Dict[str, float] = field(default_factory=dict)
    perturbation_records: List[Dict[str, Any]] = field(default_factory=list)


class PeerRobustnessEngine:
    """Engine for systematically testing prediction sensitivity to reference peer selections.

    Constructs valid counterfactual peer sets using:
    1. Leave-One-Peer-Out (LOO) perturbations across all k peers.
    2. Replacement of each peer with the next nearest available reference neighbour.
    3. Leave-Two-Out perturbations to test cohort co-dependence.
    """

    def __init__(
        self,
        matcher: PeerMatcher,
        comparator: TrajectoryComparator,
        k: int = 5,
        n_replacement_pool: int = 3,
        random_seed: int = 42,
    ):
        """Initializes the robustness engine.

        Args:
            matcher: Fitted PeerMatcher instance on the training reference pool.
            comparator: TrajectoryComparator initialized with cutoff records.
            k: Target peer set size (default: 5).
            n_replacement_pool: Number of extra neighbours to consider for replacement (default: 3).
            random_seed: Deterministic random seed.
        """
        self.matcher = matcher
        self.comparator = comparator
        self.k = int(k)
        self.n_replacement_pool = int(n_replacement_pool)
        self.random_seed = int(random_seed)

    def _get_candidate_ranking(
        self, target_series: pd.Series, out_of_fold: bool = False
    ) -> List[Tuple[str, float]]:
        """Retrieves all eligible reference candidates sorted by distance ascending."""
        target_id = str(target_series.get("cell_id", ""))
        target_vals = []
        for f in self.matcher.features:
            val = target_series.get(f, np.nan)
            try:
                target_vals.append(float(val))
            except (ValueError, TypeError):
                target_vals.append(np.nan)

        target_x = np.array(target_vals, dtype=np.float64)
        target_z = self.matcher.standardize(target_x)

        diff = self.matcher.reference_z - target_z
        sq_diff = np.square(diff)
        weighted_sq_diff = sq_diff * self.matcher.weights_arr
        distances = np.sqrt(np.nansum(weighted_sq_diff, axis=1))

        candidates: List[Tuple[str, float]] = []
        for ref_id, dist in zip(self.matcher.reference_cell_ids, distances):
            if ref_id == target_id:
                continue  # Never match self
            if np.isfinite(dist):
                candidates.append((ref_id, float(dist)))

        candidates.sort(key=lambda item: item[1])
        return candidates

    def _build_feature_vector(
        self,
        target_series: pd.Series,
        peer_relative_dict: Dict[str, float],
        feature_names: List[str],
    ) -> pd.DataFrame:
        """Assembles a single-row feature vector combining target individual features

        and recomputed peer-relative features in the exact column ordering expected by the model.
        """
        row_dict = {}
        for f in feature_names:
            if f in peer_relative_dict:
                row_dict[f] = peer_relative_dict[f]
            else:
                row_dict[f] = float(target_series.get(f, np.nan))
        return pd.DataFrame([row_dict], columns=feature_names)

    def evaluate_cell_robustness(
        self,
        target_series: pd.Series,
        model_pipeline: Pipeline,
        feature_names: List[str],
        decision_threshold: float = 0.50,
        original_prob: Optional[float] = None,
    ) -> CellRobustnessSummary:
        """Evaluates counterfactual stability for a single battery cell at the active cutoff.

        Args:
            target_series: pd.Series containing target battery features.
            model_pipeline: Fitted scikit-learn Pipeline (e.g. M3 peer-relative model).
            feature_names: Exact list of feature names matching model_pipeline inputs.
            decision_threshold: Classification decision threshold for flip detection.
            original_prob: Pre-computed baseline probability (if available, else recomputed).

        Returns:
            CellRobustnessSummary containing instability U, flip rate, and peer sensitivities.
        """
        cid = str(target_series.get("cell_id", ""))
        cutoff = int(target_series.get("cutoff", 100))
        partition = str(target_series.get("benchmark_partition", ""))
        is_eligible = bool(target_series.get("is_benchmark_eligible", True))

        if not is_eligible or partition == "excluded":
            return CellRobustnessSummary(
                cell_id=cid,
                cutoff=cutoff,
                benchmark_partition=partition,
                is_benchmark_eligible=False,
                original_prob=np.nan,
                decision_threshold=decision_threshold,
                original_classification=-1,
                n_valid_perturbations=0,
                counterfactual_mean=np.nan,
                counterfactual_std=0.0,
                counterfactual_min=np.nan,
                counterfactual_max=np.nan,
                counterfactual_range=0.0,
                flip_rate=0.0,
                most_influential_peer=None,
                max_peer_delta=0.0,
                peer_sensitivities={},
                perturbation_records=[],
            )

        # 1. Obtain ranked reference candidate list strictly from training pool
        is_train = partition == "train"
        candidates = self._get_candidate_ranking(target_series, out_of_fold=is_train)

        if len(candidates) < self.matcher.min_k:
            logger.warning("Cell %s has fewer than min_k candidates (%d)", cid, len(candidates))
            return CellRobustnessSummary(
                cell_id=cid,
                cutoff=cutoff,
                benchmark_partition=partition,
                is_benchmark_eligible=True,
                original_prob=original_prob if original_prob is not None else np.nan,
                decision_threshold=decision_threshold,
                original_classification=-1,
                n_valid_perturbations=0,
                counterfactual_mean=np.nan,
                counterfactual_std=0.0,
                counterfactual_min=np.nan,
                counterfactual_max=np.nan,
                counterfactual_range=0.0,
                flip_rate=0.0,
                most_influential_peer=None,
                max_peer_delta=0.0,
                peer_sensitivities={},
                perturbation_records=[],
            )

        # Baseline top-k peer set
        original_peer_ids = [c[0] for c in candidates[: self.k]]
        next_candidates = [c[0] for c in candidates[self.k : self.k + self.n_replacement_pool]]

        # Baseline prediction
        if original_prob is None or np.isnan(original_prob):
            base_comp = self.comparator.compare_target_to_peers(target_series, original_peer_ids)
            X_base = self._build_feature_vector(target_series, base_comp, feature_names)
            original_prob = float(model_pipeline.predict_proba(X_base)[0, 1])

        original_class = int(original_prob >= decision_threshold)

        perturbations: List[CounterfactualPerturbation] = []

        # Record original
        perturbations.append(
            CounterfactualPerturbation(
                perturbation_id="P_ORIGINAL",
                perturbation_type="ORIGINAL",
                excluded_peer=None,
                added_peer=None,
                peer_ids=list(original_peer_ids),
                predicted_prob=float(original_prob),
                prob_delta=0.0,
                prediction_flip=False,
                is_valid=True,
            )
        )

        peer_sensitivities: Dict[str, float] = {}

        # -------------------------------------------------------------
        # Perturbation Type 1: Leave-One-Peer-Out (LOO) across all k peers
        # -------------------------------------------------------------
        for idx, peer_to_drop in enumerate(original_peer_ids):
            loo_peers = [p for p in original_peer_ids if p != peer_to_drop]
            if len(loo_peers) < self.matcher.min_k:
                perturbations.append(
                    CounterfactualPerturbation(
                        perturbation_id=f"LOO_peer_{idx+1}_{peer_to_drop}",
                        perturbation_type="LEAVE_ONE_OUT",
                        excluded_peer=peer_to_drop,
                        added_peer=None,
                        peer_ids=loo_peers,
                        predicted_prob=np.nan,
                        prob_delta=0.0,
                        prediction_flip=False,
                        is_valid=False,
                        error_message="Sub-cohort size below min_k",
                    )
                )
                continue

            comp_loo = self.comparator.compare_target_to_peers(target_series, loo_peers)
            X_loo = self._build_feature_vector(target_series, comp_loo, feature_names)
            p_loo = float(model_pipeline.predict_proba(X_loo)[0, 1])
            delta_loo = p_loo - original_prob
            flip_loo = bool((p_loo >= decision_threshold) != (original_class == 1))

            peer_sensitivities[peer_to_drop] = abs(delta_loo)

            perturbations.append(
                CounterfactualPerturbation(
                    perturbation_id=f"LOO_peer_{idx+1}_{peer_to_drop}",
                    perturbation_type="LEAVE_ONE_OUT",
                    excluded_peer=peer_to_drop,
                    added_peer=None,
                    peer_ids=loo_peers,
                    predicted_prob=p_loo,
                    prob_delta=delta_loo,
                    prediction_flip=flip_loo,
                    is_valid=True,
                )
            )

        # -------------------------------------------------------------
        # Perturbation Type 2: Single-Peer Replacement with Next Candidate
        # -------------------------------------------------------------
        if next_candidates:
            replacement_peer = next_candidates[0]  # Peer k+1
            for idx, peer_to_replace in enumerate(original_peer_ids):
                rep_peers = [replacement_peer if p == peer_to_replace else p for p in original_peer_ids]
                comp_rep = self.comparator.compare_target_to_peers(target_series, rep_peers)
                X_rep = self._build_feature_vector(target_series, comp_rep, feature_names)
                p_rep = float(model_pipeline.predict_proba(X_rep)[0, 1])
                delta_rep = p_rep - original_prob
                flip_rep = bool((p_rep >= decision_threshold) != (original_class == 1))

                perturbations.append(
                    CounterfactualPerturbation(
                        perturbation_id=f"REP_peer_{idx+1}_{peer_to_replace}_with_{replacement_peer}",
                        perturbation_type="REPLACEMENT",
                        excluded_peer=peer_to_replace,
                        added_peer=replacement_peer,
                        peer_ids=rep_peers,
                        predicted_prob=p_rep,
                        prob_delta=delta_rep,
                        prediction_flip=flip_rep,
                        is_valid=True,
                    )
                )

        # -------------------------------------------------------------
        # Perturbation Type 3: Leave-Two-Out (Furthest pair, closest pair)
        # -------------------------------------------------------------
        if self.k >= 4 and len(original_peer_ids) >= 4:
            # Drop the two least similar peers (peers 4 & 5)
            l2o_peers = original_peer_ids[:3]
            comp_l2o = self.comparator.compare_target_to_peers(target_series, l2o_peers)
            X_l2o = self._build_feature_vector(target_series, comp_l2o, feature_names)
            p_l2o = float(model_pipeline.predict_proba(X_l2o)[0, 1])
            delta_l2o = p_l2o - original_prob
            flip_l2o = bool((p_l2o >= decision_threshold) != (original_class == 1))

            perturbations.append(
                CounterfactualPerturbation(
                    perturbation_id="L2O_drop_furthest_pair",
                    perturbation_type="LEAVE_TWO_OUT",
                    excluded_peer=f"{original_peer_ids[-2]},{original_peer_ids[-1]}",
                    added_peer=None,
                    peer_ids=l2o_peers,
                    predicted_prob=p_l2o,
                    prob_delta=delta_l2o,
                    prediction_flip=flip_l2o,
                    is_valid=True,
                )
            )

        # -------------------------------------------------------------
        # Aggregate Counterfactual Stability Statistics
        # -------------------------------------------------------------
        # Exclude the original baseline from perturbation dispersion to strictly measure counterfactual variance
        cf_probs = [
            p.predicted_prob
            for p in perturbations
            if p.is_valid and p.perturbation_type != "ORIGINAL" and np.isfinite(p.predicted_prob)
        ]

        if len(cf_probs) > 0:
            cf_mean = float(np.mean(cf_probs))
            # Population or sample standard deviation across counterfactual peer sets
            cf_std = float(np.std(cf_probs, ddof=1)) if len(cf_probs) > 1 else 0.0
            cf_min = float(np.min(cf_probs))
            cf_max = float(np.max(cf_probs))
            cf_range = float(cf_max - cf_min)
            n_flips = sum(
                1
                for p in perturbations
                if p.is_valid and p.perturbation_type != "ORIGINAL" and p.prediction_flip
            )
            flip_rate = float(n_flips / len(cf_probs))
        else:
            cf_mean = original_prob
            cf_std = 0.0
            cf_min = original_prob
            cf_max = original_prob
            cf_range = 0.0
            flip_rate = 0.0

        # Identify most influential peer
        if peer_sensitivities:
            most_influential = max(peer_sensitivities, key=peer_sensitivities.get)
            max_delta = peer_sensitivities[most_influential]
        else:
            most_influential = None
            max_delta = 0.0

        pert_records = [asdict(p) for p in perturbations]

        return CellRobustnessSummary(
            cell_id=cid,
            cutoff=cutoff,
            benchmark_partition=partition,
            is_benchmark_eligible=is_eligible,
            original_prob=original_prob,
            decision_threshold=decision_threshold,
            original_classification=original_class,
            n_valid_perturbations=len(cf_probs),
            counterfactual_mean=cf_mean,
            counterfactual_std=cf_std,
            counterfactual_min=cf_min,
            counterfactual_max=cf_max,
            counterfactual_range=cf_range,
            flip_rate=flip_rate,
            most_influential_peer=most_influential,
            max_peer_delta=max_delta,
            peer_sensitivities=peer_sensitivities,
            perturbation_records=pert_records,
        )

    def evaluate_cohort_robustness(
        self,
        df_cutoff: pd.DataFrame,
        model_pipeline: Pipeline,
        feature_names: List[str],
        decision_threshold: float = 0.50,
        original_probs: Optional[Dict[str, float]] = None,
    ) -> List[CellRobustnessSummary]:
        """Evaluates counterfactual robustness across all eligible cells in a cutoff dataframe."""
        results: List[CellRobustnessSummary] = []
        for _, row in df_cutoff.iterrows():
            cid = str(row["cell_id"])
            p_orig = original_probs.get(cid) if original_probs else None
            summary = self.evaluate_cell_robustness(
                target_series=row,
                model_pipeline=model_pipeline,
                feature_names=feature_names,
                decision_threshold=decision_threshold,
                original_prob=p_orig,
            )
            results.append(summary)
        return results
