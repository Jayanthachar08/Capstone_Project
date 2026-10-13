"""
Module: src.peers.trajectory_comparison
Author: Capstone Project Team
Description: Calculates interpretable peer-relative degradation trajectory features
             by benchmarking a target battery against its matched reference peer cohort
             over a strictly compatible early-life observation window [2, N_obs].
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrajectoryComparator:
    """Computes relative degradation deviations and peer-cohort dispersion statistics

    using cycle measurements strictly restricted to the observation cutoff.
    """

    def __init__(self, cutoff_data_df: pd.DataFrame):
        """Initializes the trajectory comparator with cutoff-indexed records.

        Args:
            cutoff_data_df: Modeling dataset DataFrame filtered to a single observation cutoff.
        """
        # Index records by cell_id for instant O(1) lookup
        self.records_by_cell: Dict[str, Dict[str, Any]] = {}
        for _, row in cutoff_data_df.iterrows():
            cid = str(row["cell_id"])
            self.records_by_cell[cid] = row.to_dict()

    def compare_target_to_peers(
        self,
        target_series: pd.Series,
        peer_ids: List[str],
    ) -> Dict[str, float]:
        """Calculates trajectory divergence metrics between target cell and its matched peers.

        Features calculated:
        - relative_capacity_deviation: Q_target(N_obs) - median(Q_peers(N_obs)) [Ah]
        - relative_slope_deviation: (dQ/dn)_target - median((dQ/dn)_peers) [Ah/cycle]
        - relative_capacity_ratio: Q_target(N_obs) / median(Q_peers(N_obs)) [fraction]
        - relative_fade_deviation: Delta Q_fade,target - median(Delta Q_fade,peers) [Ah]
        - relative_ir_deviation: IR_target(N_obs) - median(IR_peers(N_obs)) [Ohm]
        - peer_trajectory_dispersion: std(Q_peers(N_obs)) [Ah]
        - peer_slope_dispersion: std((dQ/dn)_peers) [Ah/cycle]
        - relative_z_score_capacity: (Q_target - mean(Q_peers)) / (std(Q_peers) + 1e-6) [z-score]

        Args:
            target_series: Series containing features of the target cell.
            peer_ids: List of string identifiers for matched peers.

        Returns:
            Dict containing relative trajectory metrics.
        """
        nan_output: Dict[str, float] = {
            "relative_capacity_deviation": np.nan,
            "relative_slope_deviation": np.nan,
            "relative_capacity_ratio": np.nan,
            "relative_fade_deviation": np.nan,
            "relative_ir_deviation": np.nan,
            "peer_trajectory_dispersion": np.nan,
            "peer_slope_dispersion": np.nan,
            "relative_z_score_capacity": np.nan,
        }

        if not peer_ids or len(peer_ids) == 0:
            return nan_output

        target_id = str(target_series.get("cell_id", ""))
        is_eligible = bool(target_series.get("is_benchmark_eligible", True))
        if not is_eligible or str(target_series.get("benchmark_partition", "")) == "excluded":
            return nan_output

        # Extract target trajectory metrics
        q_target = float(target_series.get("snap_q_discharge", np.nan))
        slope_target = float(target_series.get("traj_q_slope", np.nan))
        fade_target = float(target_series.get("traj_q_fade", np.nan))
        ir_target = float(target_series.get("snap_internal_resistance", np.nan))

        if not np.isfinite(q_target) or not np.isfinite(slope_target):
            return nan_output

        # Collect peer values
        peer_q_vals: List[float] = []
        peer_slope_vals: List[float] = []
        peer_fade_vals: List[float] = []
        peer_ir_vals: List[float] = []

        for pid in peer_ids:
            if pid not in self.records_by_cell:
                logger.warning("Peer cell_id %s not found in cutoff dataset.", pid)
                continue
            prec = self.records_by_cell[pid]
            q_p = float(prec.get("snap_q_discharge", np.nan))
            sl_p = float(prec.get("traj_q_slope", np.nan))
            fd_p = float(prec.get("traj_q_fade", np.nan))
            ir_p = float(prec.get("snap_internal_resistance", np.nan))

            if np.isfinite(q_p):
                peer_q_vals.append(q_p)
            if np.isfinite(sl_p):
                peer_slope_vals.append(sl_p)
            if np.isfinite(fd_p):
                peer_fade_vals.append(fd_p)
            if np.isfinite(ir_p):
                peer_ir_vals.append(ir_p)

        if len(peer_q_vals) == 0 or len(peer_slope_vals) == 0:
            return nan_output

        # Peer medians and dispersions
        med_q_peer = float(np.median(peer_q_vals))
        med_slope_peer = float(np.median(peer_slope_vals))
        med_fade_peer = float(np.median(peer_fade_vals)) if peer_fade_vals else np.nan
        med_ir_peer = float(np.median(peer_ir_vals)) if peer_ir_vals else np.nan

        std_q_peer = float(np.std(peer_q_vals))
        std_slope_peer = float(np.std(peer_slope_vals))
        mean_q_peer = float(np.mean(peer_q_vals))

        # Relative deviations
        rel_q_dev = q_target - med_q_peer
        rel_slope_dev = slope_target - med_slope_peer
        rel_q_ratio = (q_target / med_q_peer) if med_q_peer > 0 else np.nan
        rel_fade_dev = (fade_target - med_fade_peer) if np.isfinite(med_fade_peer) else np.nan
        rel_ir_dev = (ir_target - med_ir_peer) if np.isfinite(med_ir_peer) else np.nan
        z_score_q = (q_target - mean_q_peer) / (std_q_peer + 1e-6)

        return {
            "relative_capacity_deviation": rel_q_dev,
            "relative_slope_deviation": rel_slope_dev,
            "relative_capacity_ratio": rel_q_ratio,
            "relative_fade_deviation": rel_fade_dev,
            "relative_ir_deviation": rel_ir_dev,
            "peer_trajectory_dispersion": std_q_peer,
            "peer_slope_dispersion": std_slope_peer,
            "relative_z_score_capacity": z_score_q,
        }
