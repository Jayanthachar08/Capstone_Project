"""
Module: dashboard.server
Author: Capstone Project Team
Description: Lightweight Python HTTP data server for the Battery Degradation Screening Dashboard.
             Uses Python's standard library http.server to safely serve real CSV data and static assets.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import mimetypes
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
SPLITS_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_METADATA = {
    "M1": {
        "name": "Condition Baseline",
        "feature_count": 18,
        "groups": ["Initial Condition (11)", "Snapshot at Cutoff (7)"],
        "desc": "Baseline model using only initial health state and current cycle snapshot measurements.",
    },
    "M2": {
        "name": "Individual Trajectory",
        "feature_count": 34,
        "groups": ["Initial Condition (11)", "Snapshot (7)", "Historical Trajectory (11)", "Differential Curves (5)"],
        "desc": "Individual degradation model adding historical trajectory slopes, curvature, and differential curve statistics.",
    },
    "M3": {
        "name": "Peer-Relative",
        "feature_count": 42,
        "groups": ["Initial Condition", "Snapshot", "Trajectory", "Differential Curves", "Peer-Relative Trajectory (8)"],
        "desc": "Peer-relative model combining individual features with relative trajectory deviations from matched peers.",
    },
    "M4": {
        "name": "Confidence-Aware Peer-Relative",
        "feature_count": 46,
        "groups": ["Initial Condition", "Snapshot", "Trajectory", "Differential Curves", "Peer Trajectory", "Peer-Match Reliability PMRI (4)"],
        "desc": "Full proposed model incorporating peer-relative trajectory deviations and peer-match reliability metrics.",
    },
}

MODEL_THRESHOLDS = {
    40: {"M1": 0.5691, "M2": 0.7284, "M3": 0.5486, "M4": 0.5267},
    70: {"M1": 0.6656, "M2": 0.5957, "M3": 0.6423, "M4": 0.6066},
    100: {"M1": 0.6079, "M2": 0.5305, "M3": 0.5950, "M4": 0.5790},
}

FEATURE_METADATA = {
    # Group 1: Initial Condition (11)
    "init_q_discharge": {
        "group": "Initial Condition",
        "name": "Initial Discharge Capacity",
        "unit": "Ah",
        "desc": "Baseline discharge capacity measured at Cycle 2 (Cycle 1 was diagnostic formation).",
    },
    "init_q_charge": {
        "group": "Initial Condition",
        "name": "Initial Charge Capacity",
        "unit": "Ah",
        "desc": "Baseline charge capacity measured at Cycle 2.",
    },
    "init_coulombic_eff": {
        "group": "Initial Condition",
        "name": "Initial Coulombic Efficiency",
        "unit": "Fraction",
        "desc": "Ratio of discharge to charge capacity at Cycle 2 (Q_d / Q_c).",
    },
    "init_internal_resistance": {
        "group": "Initial Condition",
        "name": "Initial Internal Resistance",
        "unit": "Ω",
        "desc": "Baseline high-frequency internal resistance at Cycle 2.",
    },
    "init_temp_max": {
        "group": "Initial Condition",
        "name": "Initial Max Temperature",
        "unit": "°C",
        "desc": "Peak cell surface temperature recorded during Cycle 2.",
    },
    "init_temp_avg": {
        "group": "Initial Condition",
        "name": "Initial Avg Temperature",
        "unit": "°C",
        "desc": "Mean cell surface temperature during Cycle 2.",
    },
    "init_temp_min": {
        "group": "Initial Condition",
        "name": "Initial Min Temperature",
        "unit": "°C",
        "desc": "Minimum cell surface temperature recorded during Cycle 2.",
    },
    "init_charge_time": {
        "group": "Initial Condition",
        "name": "Initial Charge Duration",
        "unit": "min",
        "desc": "Duration required to charge the cell during Cycle 2.",
    },
    "init_policy_c1": {
        "group": "Initial Condition",
        "name": "Nominal Fast-Charge C1",
        "unit": "C",
        "desc": "First-stage charging current rate specified in protocol.",
    },
    "init_policy_soc": {
        "group": "Initial Condition",
        "name": "SOC Switch Point",
        "unit": "%",
        "desc": "State-of-charge percentage where charging steps down to C2.",
    },
    "init_policy_c2": {
        "group": "Initial Condition",
        "name": "Secondary Charge C2",
        "unit": "C",
        "desc": "Second-stage charging current rate specified in protocol.",
    },
    # Group 2: Instantaneous Snapshot (7)
    "snap_q_discharge": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Discharge Capacity",
        "unit": "Ah",
        "desc": "Discharge capacity measured at the observation cutoff cycle.",
    },
    "snap_q_charge": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Charge Capacity",
        "unit": "Ah",
        "desc": "Charge capacity measured at the observation cutoff cycle.",
    },
    "snap_coulombic_eff": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Coulombic Efficiency",
        "unit": "Fraction",
        "desc": "Coulombic efficiency at the observation cutoff cycle.",
    },
    "snap_internal_resistance": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Internal Resistance",
        "unit": "Ω",
        "desc": "Internal resistance at the observation cutoff cycle.",
    },
    "snap_temp_max": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Max Temperature",
        "unit": "°C",
        "desc": "Peak temperature at the observation cutoff cycle.",
    },
    "snap_temp_avg": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Avg Temperature",
        "unit": "°C",
        "desc": "Mean temperature at the observation cutoff cycle.",
    },
    "snap_charge_time": {
        "group": "Instantaneous Snapshot",
        "name": "Snapshot Charge Duration",
        "unit": "min",
        "desc": "Charging duration at the observation cutoff cycle.",
    },
    # Group 3: Emerging Individual Trajectory (11)
    "traj_q_fade": {
        "group": "Emerging Trajectory",
        "name": "Net Capacity Fade",
        "unit": "Ah",
        "desc": "Change in capacity from Cycle 2 to cutoff (Q_cutoff - Q_2).",
    },
    "traj_q_retention": {
        "group": "Emerging Trajectory",
        "name": "Capacity Retention",
        "unit": "Fraction",
        "desc": "Ratio of cutoff capacity to baseline (Q_cutoff / Q_2).",
    },
    "traj_q_slope": {
        "group": "Emerging Trajectory",
        "name": "Degradation Rate (dQ/dn)",
        "unit": "Ah/cycle",
        "desc": "Linear regression slope of capacity fade over [2, cutoff].",
    },
    "traj_q_curvature": {
        "group": "Emerging Trajectory",
        "name": "Degradation Curvature (d²Q/dn²)",
        "unit": "Ah/cycle²",
        "desc": "Quadratic polynomial acceleration coefficient of capacity fade.",
    },
    "traj_q_std": {
        "group": "Emerging Trajectory",
        "name": "Capacity Trajectory Volatility",
        "unit": "Ah",
        "desc": "Standard deviation of discharge capacity over observed window.",
    },
    "traj_q_min": {
        "group": "Emerging Trajectory",
        "name": "Minimum Observed Capacity",
        "unit": "Ah",
        "desc": "Lowest discharge capacity recorded over [2, cutoff].",
    },
    "traj_ir_change": {
        "group": "Emerging Trajectory",
        "name": "Resistance Growth (ΔIR)",
        "unit": "Ω",
        "desc": "Net increase in internal resistance from Cycle 2 to cutoff.",
    },
    "traj_ir_slope": {
        "group": "Emerging Trajectory",
        "name": "Resistance Growth Rate (dIR/dn)",
        "unit": "Ω/cycle",
        "desc": "Linear regression slope of resistance growth over observed window.",
    },
    "traj_temp_max_change": {
        "group": "Emerging Trajectory",
        "name": "Thermal Rise Drift",
        "unit": "°C",
        "desc": "Change in maximum temperature from Cycle 2 to cutoff.",
    },
    "traj_temp_avg_mean": {
        "group": "Emerging Trajectory",
        "name": "Mean Window Operating Temp",
        "unit": "°C",
        "desc": "Mean temperature across all cycles in the observation window.",
    },
    "traj_charge_time_change": {
        "group": "Emerging Trajectory",
        "name": "Charge Duration Drift",
        "unit": "min",
        "desc": "Change in charging duration from Cycle 2 to cutoff.",
    },
    # Group 4: Differential Discharge Curves (5)
    "curve_delta_q_var_log10": {
        "group": "Differential Curves",
        "name": "Log10 Variance of ΔQ(V)",
        "unit": "log10(Ah²)",
        "desc": "Log10 variance of ΔQ_{cutoff-10}(V) curve (canonical Severson feature).",
    },
    "curve_delta_q_min": {
        "group": "Differential Curves",
        "name": "Min Differential Capacity",
        "unit": "Ah",
        "desc": "Minimum value along the ΔQ_{cutoff-10}(V) discharge curve.",
    },
    "curve_delta_q_mean": {
        "group": "Differential Curves",
        "name": "Mean Differential Capacity",
        "unit": "Ah",
        "desc": "Average voltage-averaged shift of the ΔQ_{cutoff-10}(V) curve.",
    },
    "curve_delta_q_skew": {
        "group": "Differential Curves",
        "name": "Differential Curve Skewness",
        "unit": "Dim-less",
        "desc": "Asymmetry of the ΔQ_{cutoff-10}(V) differential capacity curve.",
    },
    "curve_delta_q_kurtosis": {
        "group": "Differential Curves",
        "name": "Differential Curve Kurtosis",
        "unit": "Dim-less",
        "desc": "Peak sharpness / tail heaviness of ΔQ_{cutoff-10}(V) curve.",
    },
}


class BatteryDataManager:
    """In-memory cache and query engine for processed battery dataset artifacts."""

    def __init__(self):
        self.cells_meta: Dict[str, Dict[str, Any]] = {}
        self.modeling_data: Dict[str, Dict[int, Dict[str, Any]]] = {}  # cell_id -> cutoff -> row
        self.cycle_summaries: Dict[str, List[Dict[str, Any]]] = {}  # cell_id -> list of cycle dicts
        self.peer_features: Dict[str, Dict[int, Dict[str, Any]]] = {}  # cell_id -> cutoff -> peer row
        self.model_metrics: List[Dict[str, Any]] = []
        self.model_predictions: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.splits_summary: Dict[str, Any] = {}
        self.research_summary: Dict[str, Any] = {}
        self.research_differences: List[Dict[str, Any]] = []
        self.research_robustness: List[Dict[str, Any]] = []
        self.load_all_data()

    def load_all_data(self):
        """Loads and indexes the processed CSV artifacts."""
        # 1. Load cells metadata
        meta_csv = DATA_DIR / "cells_metadata.csv"
        if not meta_csv.exists():
            raise FileNotFoundError(f"Missing {meta_csv}")

        with open(meta_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["cell_id"]
                row["is_benchmark_eligible"] = row["is_benchmark_eligible"].lower() == "true"
                row["raw_cycle_life"] = (
                    float(row["raw_cycle_life"]) if row["raw_cycle_life"] else None
                )
                row["benchmark_cycle_life"] = (
                    float(row["benchmark_cycle_life"]) if row["benchmark_cycle_life"] else None
                )
                row["num_summary_cycles"] = int(row["num_summary_cycles"])
                row["channel_id"] = int(row["channel_id"]) if row.get("channel_id") else -1
                self.cells_meta[cid] = row

        logger.info("Loaded metadata for %d cells.", len(self.cells_meta))

        # 2. Load modeling dataset
        model_csv = DATA_DIR / "modeling_dataset.csv"
        if model_csv.exists():
            with open(model_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row["cell_id"]
                    cutoff = int(row["cutoff"])
                    if cid not in self.modeling_data:
                        self.modeling_data[cid] = {}
                    # Parse numeric floats
                    typed_row = {}
                    for k, v in row.items():
                        if k in ("cell_id", "batch", "policy", "benchmark_partition", "exclusion_reason", "status_fail_within_300", "status_fail_within_500", "status_fail_by_700"):
                            typed_row[k] = v
                        elif k in ("cutoff", "channel_id"):
                            typed_row[k] = int(v) if v else None
                        elif k == "is_benchmark_eligible":
                            typed_row[k] = v.lower() == "true"
                        else:
                            try:
                                typed_row[k] = float(v) if v not in ("", "nan", "None", None) else None
                            except ValueError:
                                typed_row[k] = v
                    self.modeling_data[cid][cutoff] = typed_row

            logger.info("Loaded modeling records for %d cells across cutoffs.", len(self.modeling_data))

        # 3. Load cycle summaries
        summaries_csv = DATA_DIR / "cycle_summaries.csv"
        summaries_gz = DATA_DIR / "cycle_summaries.csv.gz"

        open_fn = gzip.open if (not summaries_csv.exists() and summaries_gz.exists()) else open
        path_to_open = summaries_gz if (not summaries_csv.exists() and summaries_gz.exists()) else summaries_csv

        logger.info("Loading cycle summaries from %s...", path_to_open.name)
        with open_fn(path_to_open, mode="rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["cell_id"]
                if cid not in self.cycle_summaries:
                    self.cycle_summaries[cid] = []

                cyc = int(row["cycle"])
                qd = float(row["QDischarge"]) if row["QDischarge"] else 0.0
                qc = float(row["QCharge"]) if row["QCharge"] else 0.0
                ir = float(row["IR"]) if row["IR"] else 0.0
                tavg = float(row["Tavg"]) if row["Tavg"] else 0.0
                tmax = float(row["Tmax"]) if row["Tmax"] else 0.0
                tmin = float(row["Tmin"]) if row["Tmin"] else 0.0
                ct = float(row["chargetime"]) if row["chargetime"] else 0.0

                self.cycle_summaries[cid].append({
                    "cycle": cyc,
                    "QDischarge": qd,
                    "QCharge": qc,
                    "IR": ir,
                    "Tavg": tavg,
                    "Tmax": tmax,
                    "Tmin": tmin,
                    "chargetime": ct,
                })

        logger.info("Loaded cycle series for %d cells.", len(self.cycle_summaries))

        # 4. Load splits JSON
        splits_json = SPLITS_DIR / "splits_benchmark.json"
        if splits_json.exists():
            with open(splits_json, mode="r", encoding="utf-8") as f:
                self.splits_summary = json.load(f)

        # 5. Load peer features if present (Phase 4)
        peers_csv = DATA_DIR / "peer_features.csv"
        if peers_csv.exists():
            with open(peers_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row["cell_id"]
                    cutoff = int(row["cutoff"])
                    if cid not in self.peer_features:
                        self.peer_features[cid] = {}
                    typed_row = {}
                    for k, v in row.items():
                        if k in ("cell_id", "batch", "policy", "benchmark_partition", "peer_match_status", "peer_ids", "peer_distances", "reliability_warning", "matching_method_version"):
                            typed_row[k] = v
                        elif k in ("cutoff", "channel_id", "peer_count"):
                            typed_row[k] = int(v) if v else None
                        elif k == "is_benchmark_eligible":
                            typed_row[k] = v.lower() == "true"
                        else:
                            try:
                                typed_row[k] = float(v) if v not in ("", "nan", "None", None) else None
                            except ValueError:
                                typed_row[k] = v
                    self.peer_features[cid][cutoff] = typed_row
            logger.info("Loaded peer features for %d cells across cutoffs.", len(self.peer_features))

        # 6. Load Phase 5 model metrics and predictions if present
        metrics_csv = BASE_DIR / "results" / "phase5_model_metrics.csv"
        if metrics_csv.exists():
            with open(metrics_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    typed = {}
                    for k, v in row.items():
                        try:
                            typed[k] = float(v) if v not in ("", "nan", "None", None) else None
                        except ValueError:
                            typed[k] = v
                    self.model_metrics.append(typed)
            logger.info("Loaded %d model metric evaluations.", len(self.model_metrics))

        preds_csv = BASE_DIR / "results" / "phase5_predictions.csv"
        if preds_csv.exists():
            with open(preds_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row["cell_id"]
                    cutoff = int(row["cutoff"])
                    if cid not in self.model_predictions:
                        self.model_predictions[cid] = {}
                    typed = {}
                    for k, v in row.items():
                        try:
                            typed[k] = float(v) if v not in ("", "nan", "None", None) else None
                        except ValueError:
                            typed[k] = v
                    self.model_predictions[cid][cutoff] = typed
            logger.info("Loaded predictions for %d cells across cutoffs.", len(self.model_predictions))

        # 7. Load Phase 6 research summary, differences, and robustness if present
        summary_json = RESULTS_DIR / "phase6_hypothesis_summary.json"
        if summary_json.exists():
            with open(summary_json, mode="r", encoding="utf-8") as f:
                self.research_summary = json.load(f)
            logger.info("Loaded Phase 6 hypothesis summary.")

        diff_csv = RESULTS_DIR / "phase6_model_differences.csv"
        if diff_csv.exists():
            with open(diff_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    typed = {}
                    for k, v in row.items():
                        try:
                            typed[k] = float(v) if v not in ("", "nan", "None", None) else None
                        except ValueError:
                            typed[k] = v
                    self.research_differences.append(typed)
            logger.info("Loaded %d Phase 6 model difference evaluations.", len(self.research_differences))

        rob_csv = RESULTS_DIR / "phase6_robustness_results.csv"
        if rob_csv.exists():
            with open(rob_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    typed = {}
                    for k, v in row.items():
                        try:
                            typed[k] = float(v) if v not in ("", "nan", "None", None) else None
                        except ValueError:
                            typed[k] = v
                    self.research_robustness.append(typed)
            logger.info("Loaded %d Phase 6 robustness evaluations.", len(self.research_robustness))

    def get_overview(self) -> Dict[str, Any]:
        """Returns high-level statistics about the dataset, partitions, models, and predictions."""
        total_cells = len(self.cells_meta)
        eligible_cells = sum(1 for m in self.cells_meta.values() if m["is_benchmark_eligible"])
        excluded_cells = total_cells - eligible_cells

        batches = {}
        for m in self.cells_meta.values():
            b = m["batch"]
            batches[b] = batches.get(b, 0) + 1

        partitions = {}
        for m in self.cells_meta.values():
            p = m["benchmark_partition"]
            partitions[p] = partitions.get(p, 0) + 1

        # Calculate elevated risk counts per cutoff and model
        elevated_risk_counts: Dict[int, Dict[str, int]] = {40: {}, 70: {}, 100: {}}
        for cutoff in (40, 70, 100):
            for model_key in ("M1", "M2", "M3", "M4"):
                th = MODEL_THRESHOLDS[cutoff][model_key]
                count = 0
                for cell_dict in self.model_predictions.values():
                    row = cell_dict.get(cutoff)
                    if row:
                        prob = row.get(f"prob_{model_key}")
                        if prob is not None and prob >= th:
                            count += 1
                elevated_risk_counts[cutoff][model_key] = count

        total_predictions = sum(len(c_dict) for c_dict in self.model_predictions.values())

        return {
            "total_cells": total_cells,
            "eligible_cells": eligible_cells,
            "excluded_cells": excluded_cells,
            "batches": batches,
            "partitions": partitions,
            "cutoffs": [40, 70, 100],
            "total_cycle_records": sum(len(c) for c in self.cycle_summaries.values()),
            "nominal_capacity_ah": 1.1,
            "eol_threshold_ah": 0.88,
            "total_models": 4,
            "models": ["M1", "M2", "M3", "M4"],
            "model_metadata": MODEL_METADATA,
            "model_thresholds": MODEL_THRESHOLDS,
            "total_predictions": total_predictions,
            "prediction_target": "target_fail_within_500",
            "prediction_horizon_cycles": 500,
            "elevated_risk_counts": elevated_risk_counts,
        }

    def get_batteries_list(self) -> List[Dict[str, Any]]:
        """Returns sorted list of batteries with basic metadata."""
        batteries = []
        for cid, meta in sorted(self.cells_meta.items()):
            batteries.append({
                "cell_id": cid,
                "batch": meta["batch"],
                "policy": meta["policy"],
                "channel_id": meta["channel_id"],
                "is_benchmark_eligible": meta["is_benchmark_eligible"],
                "benchmark_partition": meta["benchmark_partition"],
                "benchmark_cycle_life": meta["benchmark_cycle_life"],
                "num_summary_cycles": meta["num_summary_cycles"],
                "exclusion_reason": meta["exclusion_reason"],
            })
        return batteries

    def get_battery_detail(self, cell_id: str, cutoff: Optional[int] = 100) -> Optional[Dict[str, Any]]:
        """Returns comprehensive data for a single battery: metadata,

        summary cards, cycle series up to cutoff, engineered features, and targets.
        """
        if cell_id not in self.cells_meta:
            return None

        meta = self.cells_meta[cell_id]
        all_cycles = self.cycle_summaries.get(cell_id, [])

        # Filter cycle measurements
        if cutoff and cutoff > 0:
            active_cycles = [c for c in all_cycles if c["cycle"] <= cutoff]
        else:
            active_cycles = all_cycles

        # Calculate robust summary card statistics
        # Cycle 1 was an unmonitored diagnostic cycle with QDischarge = 0.0 Ah
        # Baseline capacity starts strictly from Cycle 2
        cyc2_records = [c for c in all_cycles if c["cycle"] == 2]
        if cyc2_records:
            q_start = cyc2_records[0]["QDischarge"]
            ir_start = cyc2_records[0]["IR"]
        elif all_cycles:
            # Fallback to earliest cycle > 1 with valid capacity
            valid_early = [c for c in all_cycles if c["cycle"] > 1 and c["QDischarge"] > 0.5]
            q_start = valid_early[0]["QDischarge"] if valid_early else all_cycles[0]["QDischarge"]
            ir_start = valid_early[0]["IR"] if valid_early else all_cycles[0]["IR"]
        else:
            q_start = None
            ir_start = None

        # Current observed values at the cutoff
        if active_cycles:
            last_record = active_cycles[-1]
            q_latest = last_record["QDischarge"]
            ir_latest = last_record["IR"]
            tmax_latest = last_record["Tmax"]
            tavg_latest = last_record["Tavg"]
            ct_latest = last_record["chargetime"]
            max_observed_cycle = last_record["cycle"]
        else:
            q_latest = None
            ir_latest = None
            tmax_latest = None
            tavg_latest = None
            ct_latest = None
            max_observed_cycle = 0

        # Capacity change in the observed window
        if q_start is not None and q_latest is not None and q_start > 0:
            q_change = q_latest - q_start
            q_change_pct = (q_change / q_start) * 100.0
            retention_pct = (q_latest / q_start) * 100.0
        else:
            q_change = None
            q_change_pct = None
            retention_pct = None

        if ir_start is not None and ir_latest is not None and ir_start > 0:
            ir_change = ir_latest - ir_start
            ir_change_pct = (ir_change / ir_start) * 100.0
        else:
            ir_change = None
            ir_change_pct = None

        # Retrieve modeled features for the selected cutoff
        features_grouped = {
            "Initial Condition": [],
            "Instantaneous Snapshot": [],
            "Emerging Trajectory": [],
            "Differential Curves": [],
        }

        cell_model_cutoffs = self.modeling_data.get(cell_id, {})
        target_info = {}
        target_cutoff = cutoff if (cutoff in (40, 70, 100)) else 100

        if target_cutoff in cell_model_cutoffs:
            m_row = cell_model_cutoffs[target_cutoff]
            for col_name, meta_f in FEATURE_METADATA.items():
                val = m_row.get(col_name)
                features_grouped[meta_f["group"]].append({
                    "column": col_name,
                    "name": meta_f["name"],
                    "value": val,
                    "unit": meta_f["unit"],
                    "description": meta_f["desc"],
                })

            target_info = {
                "target_eol_cycle": m_row.get("target_eol_cycle"),
                "target_rul": m_row.get("target_rul"),
                "target_fail_within_500": m_row.get("target_fail_within_500"),
                "status_fail_within_500": m_row.get("status_fail_within_500"),
                "target_fail_within_300": m_row.get("target_fail_within_300"),
                "status_fail_within_300": m_row.get("status_fail_within_300"),
                "target_fail_by_700": m_row.get("target_fail_by_700"),
                "status_fail_by_700": m_row.get("status_fail_by_700"),
            }

        # Data-grounded explainability summary (strictly factual)
        observations = []
        if q_start and q_latest:
            if q_change is not None and q_change < 0:
                observations.append(
                    f"Observed capacity fade: Capacity dropped from {q_start:.4f} Ah (at Cycle 2) to {q_latest:.4f} Ah "
                    f"by Cycle {max_observed_cycle} ({abs(q_change):.4f} Ah reduction, {abs(q_change_pct):.2f}% loss; {retention_pct:.2f}% retention)."
                )
            else:
                observations.append(
                    f"Stable early capacity: Initial capacity remained steady at {q_latest:.4f} Ah through Cycle {max_observed_cycle}."
                )

        if ir_start and ir_latest:
            if ir_change is not None and ir_change > 0:
                observations.append(
                    f"Impedance rise: Internal resistance grew from {ir_start:.5f} Ω to {ir_latest:.5f} Ω (+{ir_change_pct:.2f}% increase), "
                    f"reflecting solid electrolyte interphase (SEI) growth during cycling."
                )
            else:
                observations.append(f"Stable internal resistance: Internal resistance measured {ir_latest:.5f} Ω at Cycle {max_observed_cycle}.")

        if tmax_latest:
            observations.append(
                f"Thermal response: Peak surface temperature reached {tmax_latest:.1f} °C (average: {tavg_latest:.1f} °C) under the {meta['policy']} protocol."
            )

        peer_info = self.get_peer_info(cell_id, target_cutoff)
        model_preds = self.get_model_predictions_for_cell(cell_id, target_cutoff)

        return {
            "metadata": meta,
            "cutoff_requested": cutoff,
            "summary_cards": {
                "cell_id": cell_id,
                "total_life_cycles": meta["benchmark_cycle_life"],
                "observed_cycles_in_view": len(active_cycles),
                "total_cycles_recorded": len(all_cycles),
                "q_start_ah": q_start,
                "q_latest_ah": q_latest,
                "q_change_ah": q_change,
                "q_change_pct": q_change_pct,
                "retention_pct": retention_pct,
                "ir_start_ohm": ir_start,
                "ir_latest_ohm": ir_latest,
                "ir_change_pct": ir_change_pct,
                "tmax_latest_c": tmax_latest,
                "tavg_latest_c": tavg_latest,
                "ct_latest_min": ct_latest,
                "cycle_1_zero_note": "Cycle 1 is an unmonitored diagnostic cycle (Qd = 0.0 Ah). Physical baseline capacity begins at Cycle 2.",
            },
            "cycles": active_cycles,
            "features_grouped": features_grouped,
            "target_info": target_info,
            "peer_info": peer_info,
            "model_predictions": model_preds,
            "factual_observations": observations,
        }

    def get_peer_info(self, cell_id: str, cutoff: int = 100) -> Optional[Dict[str, Any]]:
        """Retrieves peer matching details, relative features, and PMRI for a battery cell."""
        if cell_id not in self.peer_features:
            return None
        cell_cutoffs = self.peer_features[cell_id]
        if cutoff not in cell_cutoffs:
            return None
        return cell_cutoffs[cutoff]

    def get_model_metrics(self) -> List[Dict[str, Any]]:
        """Retrieves Phase 5 model evaluation metrics across cutoffs and models."""
        return self.model_metrics

    def get_model_predictions_for_cell(self, cell_id: str, cutoff: int = 100) -> Optional[Dict[str, Any]]:
        """Retrieves model prediction probabilities for a cell at a specific cutoff,
        enriched with threshold classifications, metadata, and validation status."""
        if cell_id not in self.model_predictions:
            return None
        raw_pred = self.model_predictions[cell_id].get(cutoff)
        if not raw_pred:
            return None

        enriched = dict(raw_pred)
        cutoff_th = MODEL_THRESHOLDS.get(cutoff, {})

        model_details = {}
        for m in ("M1", "M2", "M3", "M4"):
            p_val = raw_pred.get(f"prob_{m}")
            th_val = cutoff_th.get(m, 0.50)
            is_elevated = (p_val >= th_val) if (p_val is not None) else None
            model_details[m] = {
                "name": MODEL_METADATA[m]["name"],
                "probability": p_val,
                "threshold": th_val,
                "is_elevated_risk": is_elevated,
                "risk_category": "ELEVATED_RISK" if is_elevated else "NOMINAL_RISK" if (is_elevated is not None) else "UNKNOWN",
                "feature_count": MODEL_METADATA[m]["feature_count"],
                "groups": MODEL_METADATA[m]["groups"],
                "description": MODEL_METADATA[m]["desc"],
            }

        enriched["model_details"] = model_details
        enriched["target_name"] = "target_fail_within_500"
        enriched["target_horizon_cycles"] = 500
        enriched["validation_status"] = (
            "VALIDATED_IN_DISTRIBUTION" if raw_pred.get("benchmark_partition") == "primary_test"
            else "VALIDATED_DISTRIBUTION_SHIFT" if raw_pred.get("benchmark_partition") == "secondary_test"
            else "TRAINING_SAMPLE"
        )
        return enriched

    def get_peer_trajectories(self, cell_id: str, cutoff: int = 100) -> Optional[Dict[str, Any]]:
        """Extracts aligned trajectories for the target cell and its matched peers,
        computing mean peer trajectory and relative deviation curves."""
        if cell_id not in self.peer_features:
            return None
        cell_cutoffs = self.peer_features[cell_id]
        if cutoff not in cell_cutoffs:
            return None
        peer_row = cell_cutoffs[cutoff]

        # Extract peer IDs and distances
        p_ids_str = str(peer_row.get("peer_ids", "") or "")
        p_dists_str = str(peer_row.get("peer_distances", "") or "")
        peer_ids = [p.strip() for p in p_ids_str.split(",") if p.strip()]
        peer_dists = []
        for d in p_dists_str.split(","):
            d_clean = d.strip()
            if d_clean:
                try:
                    peer_dists.append(float(d_clean))
                except ValueError:
                    pass

        # Target cell trajectory (cycles >= 2 up to cutoff)
        target_cycles = self.cycle_summaries.get(cell_id, [])
        target_pts = [{"x": c["cycle"], "y": c["QDischarge"]} for c in target_cycles if 2 <= c["cycle"] <= cutoff]

        # Peer trajectories
        peer_series = []
        peer_cycles_map: Dict[int, List[float]] = {}
        for idx, pid in enumerate(peer_ids):
            p_dist = peer_dists[idx] if idx < len(peer_dists) else 0.0
            p_cycles = self.cycle_summaries.get(pid, [])
            p_pts = [{"x": c["cycle"], "y": c["QDischarge"]} for c in p_cycles if 2 <= c["cycle"] <= cutoff]
            peer_series.append({
                "cell_id": pid,
                "distance": p_dist,
                "points": p_pts,
            })
            for pt in p_pts:
                cyc = pt["x"]
                qd = pt["y"]
                if cyc not in peer_cycles_map:
                    peer_cycles_map[cyc] = []
                peer_cycles_map[cyc].append(qd)

        # Aggregate / mean peer trajectory
        mean_peer_pts = []
        deviation_pts = []
        target_map = {pt["x"]: pt["y"] for pt in target_pts}

        for cyc in sorted(peer_cycles_map.keys()):
            q_list = peer_cycles_map[cyc]
            avg_q = sum(q_list) / len(q_list)
            mean_peer_pts.append({"x": cyc, "y": round(avg_q, 5)})
            if cyc in target_map:
                dev = target_map[cyc] - avg_q
                deviation_pts.append({"x": cyc, "y": round(dev, 5)})

        return {
            "cell_id": cell_id,
            "cutoff": cutoff,
            "pmri": peer_row.get("pmri"),
            "peer_count": peer_row.get("peer_count", len(peer_ids)),
            "mean_peer_distance": peer_row.get("mean_peer_distance"),
            "peer_similarity_score": peer_row.get("peer_similarity_score"),
            "peer_agreement_score": peer_row.get("peer_agreement_score"),
            "peer_availability_score": peer_row.get("peer_availability_score"),
            "peer_feature_completeness": peer_row.get("peer_feature_completeness"),
            "reliability_warning": peer_row.get("reliability_warning", "NONE"),
            "target_trajectory": target_pts,
            "peers": peer_series,
            "mean_peer_trajectory": mean_peer_pts,
            "deviation_trajectory": deviation_pts,
            "relative_capacity_deviation": peer_row.get("relative_capacity_deviation"),
            "relative_slope_deviation": peer_row.get("relative_slope_deviation"),
            "relative_fade_deviation": peer_row.get("relative_fade_deviation"),
            "relative_ir_deviation": peer_row.get("relative_ir_deviation"),
        }

    def get_comparison(
        self, cell_ids: List[str], metric: str = "QDischarge", cutoff: Optional[int] = 100
    ) -> Dict[str, Any]:
        """Returns aligned trajectory series for multiple batteries for side-by-side exploratory comparison."""
        series_data = []

        valid_metrics = {"QDischarge", "IR", "Tavg", "Tmax", "chargetime"}
        if metric not in valid_metrics:
            metric = "QDischarge"

        for cid in cell_ids:
            if cid not in self.cells_meta:
                continue
            meta = self.cells_meta[cid]
            all_cycles = self.cycle_summaries.get(cid, [])
            if cutoff and cutoff > 0:
                filtered = [c for c in all_cycles if c["cycle"] <= cutoff]
            else:
                filtered = all_cycles

            # Exclude cycle 1 if plotting QDischarge to remove the diagnostic 0
            if metric == "QDischarge":
                pts = [{"x": c["cycle"], "y": c[metric]} for c in filtered if c["cycle"] >= 2]
            else:
                pts = [{"x": c["cycle"], "y": c[metric]} for c in filtered if c[metric] > 0]

            series_data.append({
                "cell_id": cid,
                "batch": meta["batch"],
                "policy": meta["policy"],
                "benchmark_cycle_life": meta["benchmark_cycle_life"],
                "points": pts,
            })

        return {
            "metric": metric,
            "cutoff": cutoff,
            "series": series_data,
            "disclaimer": (
                "Exploratory battery comparison only. Visualizing user-selected cells together. "
                "Validated scientific peer-matching algorithms (evaluating condition similarity, "
                "cohort size k, and trajectory agreement) will be computed in Phase 5."
            ),
        }


# Global data manager singleton
DATA_MANAGER = BatteryDataManager()


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler serving static dashboard assets and JSON API routes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        """Routes API requests or delegates to static file handler."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # Serve static figures from results/figures/
        if path.startswith("/results/figures/") or path.startswith("/api/figures/"):
            filename = os.path.basename(path)
            file_path = FIGURES_DIR / filename
            if file_path.exists() and file_path.suffix.lower() == ".png":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(file_path.stat().st_size))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error_response(404, f"Figure '{filename}' not found.")
                return

        # Route API endpoints
        if path.startswith("/api/"):
            self.handle_api(path, query)
        else:
            # Serve index.html as root
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def handle_api(self, path: str, query: Dict[str, List[str]]):
        """Processes API endpoints and sends JSON responses."""
        try:
            if path == "/api/overview":
                data = DATA_MANAGER.get_overview()
                self.send_json_response(data)

            elif path == "/api/batteries":
                data = DATA_MANAGER.get_batteries_list()
                self.send_json_response(data)

            elif path.startswith("/api/battery/"):
                cell_id = path.replace("/api/battery/", "").strip("/")
                cutoff_str = query.get("cutoff", ["100"])[0]
                cutoff = None if cutoff_str.lower() == "all" else int(cutoff_str)

                data = DATA_MANAGER.get_battery_detail(cell_id, cutoff)
                if data is None:
                    self.send_error_response(404, f"Battery '{cell_id}' not found.")
                else:
                    self.send_json_response(data)

            elif path == "/api/compare":
                ids_param = query.get("ids", [""])[0]
                metric = query.get("metric", ["QDischarge"])[0]
                cutoff_str = query.get("cutoff", ["100"])[0]
                cutoff = None if cutoff_str.lower() == "all" else int(cutoff_str)

                cell_ids = [i.strip() for i in ids_param.split(",") if i.strip()]
                data = DATA_MANAGER.get_comparison(cell_ids, metric, cutoff)
                self.send_json_response(data)

            elif path == "/api/peers":
                cid = query.get("id", [""])[0]
                cutoff_str = query.get("cutoff", ["100"])[0]
                cutoff = int(cutoff_str) if cutoff_str.isdigit() else 100

                data = DATA_MANAGER.get_peer_info(cid, cutoff)
                if data is None:
                    self.send_error_response(404, f"Peer data for '{cid}' at cutoff {cutoff} not found.")
                else:
                    self.send_json_response(data)

            elif path == "/api/peers/trajectories":
                cid = query.get("id", [""])[0]
                cutoff_str = query.get("cutoff", ["100"])[0]
                cutoff = int(cutoff_str) if cutoff_str.isdigit() else 100

                data = DATA_MANAGER.get_peer_trajectories(cid, cutoff)
                if data is None:
                    self.send_error_response(404, f"Peer trajectories for '{cid}' at cutoff {cutoff} not found.")
                else:
                    self.send_json_response(data)

            elif path == "/api/models/metrics":
                data = DATA_MANAGER.get_model_metrics()
                self.send_json_response(data)

            elif path == "/api/models/predictions":
                cid = query.get("id", [""])[0]
                cutoff_str = query.get("cutoff", ["100"])[0]
                cutoff = int(cutoff_str) if cutoff_str.isdigit() else 100

                data = DATA_MANAGER.get_model_predictions_for_cell(cid, cutoff)
                if data is None:
                    self.send_error_response(404, f"Model predictions for '{cid}' at cutoff {cutoff} not found.")
                else:
                    self.send_json_response(data)

            elif path == "/api/models/predict":
                cid = query.get("id", [""])[0]
                cutoff_str = query.get("cutoff", ["100"])[0]
                model_name = query.get("model", ["M4"])[0].upper()
                cutoff = int(cutoff_str) if cutoff_str.isdigit() else 100

                if cid not in DATA_MANAGER.cells_meta:
                    self.send_error_response(404, f"Battery '{cid}' not found.")
                    return
                if cutoff not in (40, 70, 100):
                    self.send_error_response(400, f"Unsupported observation cutoff: {cutoff}. Allowed: 40, 70, 100.")
                    return
                if model_name not in ("M1", "M2", "M3", "M4"):
                    self.send_error_response(400, f"Unsupported model '{model_name}'. Allowed: M1, M2, M3, M4.")
                    return

                cell_preds = DATA_MANAGER.get_model_predictions_for_cell(cid, cutoff)
                if cell_preds is None:
                    self.send_error_response(404, f"No predictions available for '{cid}' at cutoff {cutoff} (Cell may be excluded from benchmark).")
                    return

                details = cell_preds.get("model_details", {}).get(model_name)
                res = {
                    "cell_id": cid,
                    "cutoff": cutoff,
                    "model": model_name,
                    "model_name": details["name"],
                    "probability": details["probability"],
                    "threshold": details["threshold"],
                    "risk_category": details["risk_category"],
                    "is_elevated_risk": details["is_elevated_risk"],
                    "target_name": cell_preds.get("target_name"),
                    "target_horizon_cycles": cell_preds.get("target_horizon_cycles"),
                    "feature_count": details["feature_count"],
                    "feature_groups": details["groups"],
                    "pmri": cell_preds.get("pmri"),
                    "reliability_warning": cell_preds.get("reliability_warning"),
                    "validation_status": cell_preds.get("validation_status"),
                    "source": "SAVED_VALIDATED_PREDICTION",
                }
                self.send_json_response(res)

            elif path == "/api/research/summary":
                self.send_json_response(DATA_MANAGER.research_summary)

            elif path == "/api/research/differences":
                self.send_json_response(DATA_MANAGER.research_differences)

            elif path == "/api/research/robustness":
                self.send_json_response(DATA_MANAGER.research_robustness)

            else:
                self.send_error_response(404, f"API route '{path}' not found.")

        except Exception as err:
            logger.exception("Error handling API request: %s", err)
            self.send_error_response(500, f"Internal server error: {err}")

    def send_json_response(self, data: Any, status: int = 200):
        """Serializes and sends a JSON payload."""
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def send_error_response(self, status: int, message: str):
        """Sends a JSON-formatted error response."""
        self.send_json_response({"error": message, "status": status}, status=status)

    def log_message(self, format: str, *args: Any):
        """Clean HTTP access logging."""
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}\n")


def run_server(port: int = 8000, host: str = "127.0.0.1"):
    """Starts the local dashboard server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    print("\n" + "=" * 65)
    print("  BATTERY DEGRADATION SCREENING DASHBOARD")
    print("  Confidence-Aware Peer-Relative Degradation Analysis")
    print("=" * 65)
    print(f"  Server running at: http://{host}:{port}/")
    print(f"  Serving dashboard from: {DASHBOARD_DIR}")
    print("  Press Ctrl+C to terminate the server gracefully.")
    print("=" * 65 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
        httpd.server_close()
        print("Server stopped.")


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port=port)
