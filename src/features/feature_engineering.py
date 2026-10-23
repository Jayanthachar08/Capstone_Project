"""
Module: src.features.feature_engineering
Author: Capstone Project Team
Description: Feature engineering module extracting initial-condition features,
             instantaneous snapshot features, emerging individual trajectory features,
             and differential discharge curve statistics across observation cutoffs.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

logger = logging.getLogger(__name__)

BATCH_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}


def parse_charging_policy(policy_str: str) -> Tuple[float, float, float]:
    """Extracts fast-charging protocol parameters (C1, SOC switch point %, C2) from string.

    Example: '3.6C(80%)-3.6C' -> (3.6, 80.0, 3.6)
             '5C(67%)-4C-newstructure' -> (5.0, 67.0, 4.0)
             '4C(31%)-5' -> (4.0, 31.0, 5.0)
    """
    if not isinstance(policy_str, str) or not policy_str:
        return np.nan, np.nan, np.nan

    match = re.search(
        r"([0-9.]+)\s*C\s*\(\s*([0-9.]+)\s*%\s*\)\s*-\s*([0-9.]+)\s*C?",
        policy_str,
    )
    if match:
        try:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        except ValueError:
            pass
    return np.nan, np.nan, np.nan


class FeatureEngineer:
    """Extracts initial susceptibility, snapshot, individual trajectory,

    and differential voltage curve features strictly within the observation window.
    """

    def __init__(self, datasets_dir: str = "Datasets"):
        self.datasets_dir = Path(datasets_dir)
        self._h5_cache: Dict[str, h5py.File] = {}

    def _get_h5_file(self, batch_key: str) -> h5py.File:
        """Retrieves or opens an HDF5 file handle."""
        if batch_key not in self._h5_cache:
            fname = BATCH_FILES[batch_key]
            fpath = self.datasets_dir / fname
            self._h5_cache[batch_key] = h5py.File(fpath, "r")
        return self._h5_cache[batch_key]

    def close(self):
        """Closes all cached HDF5 file handles."""
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def extract_curve_features(
        self,
        batch_key: str,
        batch_cell_index: int,
        cutoff: int,
        baseline_cycle: int = 10,
    ) -> Dict[str, float]:
        """Extracts summary statistics of Delta Q_{cutoff - baseline}(V)

        using linearly interpolated discharge curves (Qdlin) on the standard 1000-point Vdlin grid.
        Formula: Delta Q(V) = Q_cutoff(V) - Q_10(V).
        """
        try:
            h5_file = self._get_h5_file(batch_key)
            batch_grp = h5_file["batch"]
            cyc_grp = h5_file[batch_grp["cycles"][batch_cell_index, 0]]
            qdlin_ds = cyc_grp["Qdlin"]

            # Ensure both cycles are within available range
            n_avail = qdlin_ds.shape[0]
            if baseline_cycle > n_avail or cutoff > n_avail:
                return {
                    "curve_delta_q_var_log10": np.nan,
                    "curve_delta_q_min": np.nan,
                    "curve_delta_q_mean": np.nan,
                    "curve_delta_q_skew": np.nan,
                    "curve_delta_q_kurtosis": np.nan,
                }

            # Dereference 1000-point arrays (cycle indices are 1-based, array 0-based)
            ref_base = qdlin_ds[baseline_cycle - 1, 0]
            ref_cut = qdlin_ds[cutoff - 1, 0]

            q_base = np.asarray(h5_file[ref_base]).ravel()
            q_cut = np.asarray(h5_file[ref_cut]).ravel()

            if len(q_base) == 0 or len(q_cut) == 0:
                raise ValueError("Empty Qdlin curve")

            delta_q = q_cut - q_base
            var_val = float(np.var(delta_q))
            log10_var = float(np.log10(var_val)) if var_val > 0 else np.nan

            return {
                "curve_delta_q_var_log10": log10_var,
                "curve_delta_q_min": float(np.min(delta_q)),
                "curve_delta_q_mean": float(np.mean(delta_q)),
                "curve_delta_q_skew": float(skew(delta_q)),
                "curve_delta_q_kurtosis": float(kurtosis(delta_q)),
            }
        except Exception as err:
            logger.debug(
                "Error extracting curve features for %s index %d at cutoff %d: %s",
                batch_key,
                batch_cell_index,
                cutoff,
                err,
            )
            return {
                "curve_delta_q_var_log10": np.nan,
                "curve_delta_q_min": np.nan,
                "curve_delta_q_mean": np.nan,
                "curve_delta_q_skew": np.nan,
                "curve_delta_q_kurtosis": np.nan,
            }

    @staticmethod
    def extract_cell_features_at_cutoff(
        cell_summary_df: pd.DataFrame,
        cell_meta: pd.Series,
        cutoff: int,
        curve_features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Calculates initial condition, snapshot, and individual trajectory features

        for one battery up to the observation cutoff.
        """
        # Filter summary strictly up to the cutoff
        history = cell_summary_df[
            (cell_summary_df["cycle"] >= 1) & (cell_summary_df["cycle"] <= cutoff)
        ].sort_values("cycle")

        # Initial condition is evaluated at cycle 2 (cycle 1 was unmonitored diagnostic)
        rec_init = history[history["cycle"] == 2]
        if rec_init.empty:
            # Fallback to the lowest available cycle >= 1 if cycle 2 missing
            rec_init = history.head(1)

        # Snapshot is evaluated at cutoff cycle (or latest observed cycle <= cutoff)
        rec_snap = history[history["cycle"] == cutoff]
        if rec_snap.empty:
            rec_snap = history.tail(1)

        # Parse fast charging policy parameters
        c1, soc, c2 = parse_charging_policy(str(cell_meta.get("policy", "")))

        # 1. INITIAL-CONDITION FEATURES (Group init_)
        q2 = float(rec_init["QDischarge"].values[0]) if not rec_init.empty else np.nan
        qc2 = float(rec_init["QCharge"].values[0]) if not rec_init.empty else np.nan
        ir2 = float(rec_init["IR"].values[0]) if not rec_init.empty else np.nan
        ce2 = (q2 / qc2) if (qc2 > 0 and not np.isnan(qc2) and not np.isnan(q2)) else np.nan

        init_features = {
            "init_q_discharge": q2,
            "init_q_charge": qc2,
            "init_coulombic_eff": ce2,
            "init_internal_resistance": ir2,
            "init_temp_max": (
                float(rec_init["Tmax"].values[0]) if not rec_init.empty else np.nan
            ),
            "init_temp_avg": (
                float(rec_init["Tavg"].values[0]) if not rec_init.empty else np.nan
            ),
            "init_temp_min": (
                float(rec_init["Tmin"].values[0]) if not rec_init.empty else np.nan
            ),
            "init_charge_time": (
                float(rec_init["chargetime"].values[0]) if not rec_init.empty else np.nan
            ),
            "init_policy_c1": c1,
            "init_policy_soc": soc,
            "init_policy_c2": c2,
        }

        # 2. SNAPSHOT FEATURES AT CUTOFF (Group snap_)
        q_cut = float(rec_snap["QDischarge"].values[0]) if not rec_snap.empty else np.nan
        qc_cut = float(rec_snap["QCharge"].values[0]) if not rec_snap.empty else np.nan
        ir_cut = float(rec_snap["IR"].values[0]) if not rec_snap.empty else np.nan
        ce_cut = (
            (q_cut / qc_cut) if (qc_cut > 0 and not np.isnan(qc_cut) and not np.isnan(q_cut)) else np.nan
        )

        snap_features = {
            "snap_q_discharge": q_cut,
            "snap_q_charge": qc_cut,
            "snap_coulombic_eff": ce_cut,
            "snap_internal_resistance": ir_cut,
            "snap_temp_max": (
                float(rec_snap["Tmax"].values[0]) if not rec_snap.empty else np.nan
            ),
            "snap_temp_avg": (
                float(rec_snap["Tavg"].values[0]) if not rec_snap.empty else np.nan
            ),
            "snap_charge_time": (
                float(rec_snap["chargetime"].values[0]) if not rec_snap.empty else np.nan
            ),
        }

        # 3. INDIVIDUAL TRAJECTORY FEATURES OVER [2, cutoff] (Group traj_)
        # Use strictly cycles >= 2 to exclude cycle 1 diagnostic zero
        traj_df = history[history["cycle"] >= 2].copy()
        n_pts = len(traj_df)

        if n_pts >= 3:
            cyc_arr = traj_df["cycle"].values.astype(np.float64)
            q_arr = traj_df["QDischarge"].values.astype(np.float64)
            ir_arr = traj_df["IR"].values.astype(np.float64)

            # Linear slope dQ/dn
            p_q1 = np.polyfit(cyc_arr, q_arr, 1)
            q_slope = float(p_q1[0])

            # Quadratic curvature d^2Q/dn^2
            p_q2 = np.polyfit(cyc_arr, q_arr, 2)
            q_curvature = float(p_q2[0])  # Acceleration coefficient

            # IR slope d(IR)/dn
            valid_ir = np.isfinite(ir_arr) & (ir_arr > 0)
            if valid_ir.sum() >= 3:
                p_ir1 = np.polyfit(cyc_arr[valid_ir], ir_arr[valid_ir], 1)
                ir_slope = float(p_ir1[0])
            else:
                ir_slope = np.nan

            q_fade = q_cut - q2
            q_retention = (q_cut / q2) if (q2 > 0 and not np.isnan(q2)) else np.nan
            q_std = float(np.std(q_arr))
            q_min = float(np.min(q_arr))
            ir_change = (ir_cut - ir2) if (np.isfinite(ir_cut) and np.isfinite(ir2)) else np.nan

            tmax_init = init_features["init_temp_max"]
            tmax_cut = snap_features["snap_temp_max"]
            temp_max_change = (
                (tmax_cut - tmax_init)
                if (np.isfinite(tmax_cut) and np.isfinite(tmax_init))
                else np.nan
            )
            temp_avg_mean = float(np.mean(traj_df["Tavg"].values))

            ct_init = init_features["init_charge_time"]
            ct_cut = snap_features["snap_charge_time"]
            charge_time_change = (
                (ct_cut - ct_init)
                if (np.isfinite(ct_cut) and np.isfinite(ct_init))
                else np.nan
            )
        else:
            q_fade = np.nan
            q_retention = np.nan
            q_slope = np.nan
            q_curvature = np.nan
            q_std = np.nan
            q_min = np.nan
            ir_change = np.nan
            ir_slope = np.nan
            temp_max_change = np.nan
            temp_avg_mean = np.nan
            charge_time_change = np.nan

        traj_features = {
            "traj_q_fade": q_fade,
            "traj_q_retention": q_retention,
            "traj_q_slope": q_slope,
            "traj_q_curvature": q_curvature,
            "traj_q_std": q_std,
            "traj_q_min": q_min,
            "traj_ir_change": ir_change,
            "traj_ir_slope": ir_slope,
            "traj_temp_max_change": temp_max_change,
            "traj_temp_avg_mean": temp_avg_mean,
            "traj_charge_time_change": charge_time_change,
        }

        # Combine all groups
        all_features = {}
        all_features.update(init_features)
        all_features.update(snap_features)
        all_features.update(traj_features)
        all_features.update(curve_features)
        return all_features
