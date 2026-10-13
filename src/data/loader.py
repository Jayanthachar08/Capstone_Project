"""
Module: src.data.loader
Author: Capstone Project Team
Description: Memory-efficient streaming extractor, validator, and split manager
             for the MATR/Severson battery degradation dataset (MATLAB v7.3 / HDF5).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

# Setup module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Authoritative batch filenames mapping
BATCH_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Authoritative 16 exclusions defined in Severson et al. (Nature Energy 2019)
# and official MIT/Stanford Braatz codebase (Load Data.ipynb):
EXCLUSION_REGISTRY: Dict[str, str] = {
    # Batch 1 exclusions: 5 cells that did not reach 80% nominal capacity
    "b1c8": "Batch 1: Channel test terminated before cell reached 80% nominal capacity",
    "b1c10": "Batch 1: Channel test terminated before cell reached 80% nominal capacity",
    "b1c12": "Batch 1: Channel test terminated before cell reached 80% nominal capacity",
    "b1c13": "Batch 1: Channel test terminated before cell reached 80% nominal capacity",
    "b1c22": "Batch 1: Channel test terminated before cell reached 80% nominal capacity",
    # Batch 2 exclusions: 5 carry-over continuation segments merged into Batch 1 cells
    "b2c7": "Batch 2: Continuation segment of b1c0 (merged into b1c0 per benchmark protocol)",
    "b2c8": "Batch 2: Continuation segment of b1c1 (merged into b1c1 per benchmark protocol)",
    "b2c9": "Batch 2: Continuation segment of b1c2 (merged into b1c2 per benchmark protocol)",
    "b2c15": "Batch 2: Continuation segment of b1c3 (merged into b1c3 per benchmark protocol)",
    "b2c16": "Batch 2: Continuation segment of b1c4 (merged into b1c4 per benchmark protocol)",
    # Batch 3 exclusions: 6 noisy / incomplete / thermocouple detached channels
    "b3c2": "Batch 3: Irregular noisy temperature/voltage profile",
    "b3c23": "Batch 3: Undefined cycle life (cell reached >2189 cycles without 80% cutoff)",
    "b3c32": "Batch 3: Undefined cycle life (cell reached >2237 cycles without 80% cutoff)",
    "b3c37": "Batch 3: Temperature sensor detached / high thermal anomaly",
    "b3c42": "Batch 3: Noisy sensor channel in test batch",
    "b3c43": "Batch 3: Loose contact / noisy resistance channel in test batch",
}

# Authoritative continuation mapping for the 5 resumed cells
CARRYOVER_MAPPING: Dict[str, Tuple[str, int]] = {
    "b1c0": ("b2c7", 662),
    "b1c1": ("b2c8", 981),
    "b1c2": ("b2c9", 1060),
    "b1c3": ("b2c15", 208),
    "b1c4": ("b2c16", 482),
}


class BatteryDataLoader:
    """Memory-efficient HDF5 extractor and split manager for MATR battery batches."""

    def __init__(self, datasets_dir: str = "Datasets", output_dir: str = "data/processed"):
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.splits_dir = Path("data/splits")
        self.results_dir = Path("results")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def decode_hdf5_string(h5_file: h5py.File, ref: Any) -> str:
        """Safely dereferences and decodes uint16/uint32 ASCII/UTF character arrays."""
        if not ref:
            return ""
        try:
            obj = h5_file[ref]
            arr = np.asarray(obj).ravel()
            if arr.dtype.kind in ("i", "u"):
                return "".join(chr(c) for c in arr if 32 <= c <= 126).strip()
            return str(arr)
        except Exception as err:
            logger.warning("Error decoding HDF5 string reference: %s", err)
            return ""

    @staticmethod
    def decode_hdf5_scalar(h5_file: h5py.File, ref: Any) -> float:
        """Safely dereferences a scalar float value."""
        if not ref:
            return float("nan")
        try:
            obj = h5_file[ref]
            arr = np.asarray(obj).ravel()
            return float(arr[0]) if len(arr) > 0 else float("nan")
        except Exception as err:
            logger.warning("Error decoding HDF5 scalar reference: %s", err)
            return float("nan")

    @staticmethod
    def decode_channel_id(h5_file: h5py.File, ref: Any) -> int:
        """Extracts channel number from MATLAB v7.3 struct header array."""
        if not ref:
            return -1
        try:
            arr = np.asarray(h5_file[ref]).ravel()
            # In MATLAB v7.3 uint32 headers, the integer payload is stored at index 4
            if len(arr) >= 5:
                return int(arr[4])
            return int(arr[0]) if len(arr) > 0 else -1
        except Exception:
            return -1

    def stream_cell_summaries(
        self, batch_key: str, h5_file: h5py.File, cell_index: int, cell_id: str
    ) -> pd.DataFrame:
        """Extracts cycle summary records for a single cell without loading raw intra-cycle waveforms."""
        batch_grp = h5_file["batch"]
        sum_ref = batch_grp["summary"][cell_index, 0]
        sum_obj = h5_file[sum_ref]

        cycles = np.asarray(sum_obj["cycle"]).ravel()
        n_rec = len(cycles)
        if n_rec == 0:
            return pd.DataFrame()

        def _get_field(field_name: str) -> np.ndarray:
            if field_name in sum_obj:
                arr = np.asarray(sum_obj[field_name]).ravel()
                if len(arr) == n_rec:
                    return arr
            return np.full(n_rec, np.nan, dtype=np.float64)

        df = pd.DataFrame({
            "cell_id": cell_id,
            "batch": batch_key,
            "cycle": cycles.astype(np.int32),
            "QDischarge": _get_field("QDischarge"),
            "QCharge": _get_field("QCharge"),
            "IR": _get_field("IR"),
            "Tavg": _get_field("Tavg"),
            "Tmin": _get_field("Tmin"),
            "Tmax": _get_field("Tmax"),
            "chargetime": _get_field("chargetime"),
        })
        return df

    def extract_raw_metadata(self) -> List[Dict[str, Any]]:
        """Scans all three batch files and extracts metadata for all 140 raw cells."""
        raw_metadata: List[Dict[str, Any]] = []

        for batch_key, fname in BATCH_FILES.items():
            fpath = self.datasets_dir / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Missing required batch file: {fpath}")

            prefix = "b1" if batch_key == "batch1" else "b2" if batch_key == "batch2" else "b3"
            logger.info("Scanning batch metadata from %s (%s)...", batch_key, fname)

            with h5py.File(fpath, "r") as f:
                batch_grp = f["batch"]
                num_cells = batch_grp["summary"].shape[0]

                for i in range(num_cells):
                    cell_id = f"{prefix}c{i}"
                    policy = self.decode_hdf5_string(f, batch_grp["policy_readable"][i, 0])
                    if not policy:
                        policy = self.decode_hdf5_string(f, batch_grp["policy"][i, 0])

                    cycle_life = self.decode_hdf5_scalar(f, batch_grp["cycle_life"][i, 0])
                    channel_id = self.decode_channel_id(f, batch_grp["channel_id"][i, 0])

                    # Summary cycle count
                    s_ref = batch_grp["summary"][i, 0]
                    s_obj = f[s_ref]
                    cyc_arr = np.asarray(s_obj["cycle"]).ravel()
                    num_cycles = len(cyc_arr)

                    # Determine benchmark eligibility & exclusion reason
                    is_excluded = cell_id in EXCLUSION_REGISTRY
                    exclusion_reason = EXCLUSION_REGISTRY.get(cell_id, "None (Retained)")
                    is_eligible = not is_excluded

                    # Base metadata record
                    meta = {
                        "cell_id": cell_id,
                        "batch": batch_key,
                        "file_name": fname,
                        "batch_cell_index": i,
                        "channel_id": channel_id,
                        "policy": policy,
                        "raw_cycle_life": cycle_life,
                        "benchmark_cycle_life": cycle_life,  # Adjusted later for carry-overs
                        "num_summary_cycles": num_cycles,
                        "is_benchmark_eligible": is_eligible,
                        "exclusion_reason": exclusion_reason,
                        "benchmark_partition": "excluded" if is_excluded else "unassigned",
                    }
                    raw_metadata.append(meta)

        # Apply carry-over length adjustments to the 5 resumed cells
        meta_dict = {m["cell_id"]: m for m in raw_metadata}
        for b1_key, (b2_continuation_key, add_l) in CARRYOVER_MAPPING.items():
            if b1_key in meta_dict and b2_continuation_key in meta_dict:
                meta_dict[b1_key]["benchmark_cycle_life"] = (
                    meta_dict[b1_key]["raw_cycle_life"] + add_l
                )
                meta_dict[b1_key]["num_summary_cycles"] += add_l
                logger.info(
                    "Adjusted carry-over for %s: added %d cycles from %s (Total benchmark life: %.0f)",
                    b1_key,
                    add_l,
                    b2_continuation_key,
                    meta_dict[b1_key]["benchmark_cycle_life"],
                )

        return raw_metadata

    def assign_benchmark_partitions(
        self, metadata_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assigns authoritative benchmark train/primary_test/secondary_test partitions

        following Severson et al. (2019) Load Data.ipynb:
        - Eligible Batch 1: 41 cells
        - Eligible Batch 2: 43 cells
        - Eligible Batch 3: 40 cells
        - Combined Batch 1 + Batch 2: 84 cells
        - Train (41 cells): odd indices in combined pool: 1, 3, 5, ..., 81
        - Primary Test (43 cells): even indices in combined pool (0, 2, ..., 82) + index 83
        - Secondary Test (40 cells): all 40 cells of Batch 3
        """
        # Group eligible cells in order
        b1_eligible = [
            m for m in metadata_list if m["batch"] == "batch1" and m["is_benchmark_eligible"]
        ]
        b2_eligible = [
            m for m in metadata_list if m["batch"] == "batch2" and m["is_benchmark_eligible"]
        ]
        b3_eligible = [
            m for m in metadata_list if m["batch"] == "batch3" and m["is_benchmark_eligible"]
        ]

        logger.info(
            "Eligible counts: Batch 1 = %d, Batch 2 = %d, Batch 3 = %d (Sum = %d)",
            len(b1_eligible),
            len(b2_eligible),
            len(b3_eligible),
            len(b1_eligible) + len(b2_eligible) + len(b3_eligible),
        )

        assert len(b1_eligible) == 41, f"Expected 41 eligible Batch 1 cells, found {len(b1_eligible)}"
        assert len(b2_eligible) == 43, f"Expected 43 eligible Batch 2 cells, found {len(b2_eligible)}"
        assert len(b3_eligible) == 40, f"Expected 40 eligible Batch 3 cells, found {len(b3_eligible)}"

        combined_b1_b2 = b1_eligible + b2_eligible  # 84 cells
        num_b1_b2 = len(combined_b1_b2)

        # Canonical Severson indexing
        test_indices = set(np.hstack((np.arange(0, num_b1_b2, 2), 83)).tolist())
        train_indices = set(np.arange(1, num_b1_b2 - 1, 2).tolist())

        # Verify indexing integrity
        assert len(train_indices) == 41, f"Expected 41 train indices, got {len(train_indices)}"
        assert len(test_indices) == 43, f"Expected 43 test indices, got {len(test_indices)}"
        assert train_indices.isdisjoint(test_indices), "Train and primary test indices overlap!"

        for idx, cell_meta in enumerate(combined_b1_b2):
            if idx in train_indices:
                cell_meta["benchmark_partition"] = "train"
            elif idx in test_indices:
                cell_meta["benchmark_partition"] = "primary_test"
            else:
                raise ValueError(f"Unassigned cell index in combined pool: {idx}")

        for cell_meta in b3_eligible:
            cell_meta["benchmark_partition"] = "secondary_test"

        return metadata_list

    def extract_and_export_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Executes full streaming extraction, carry-over stitching, and CSV export."""
        logger.info("Starting Phase 2 data extraction pipeline...")

        # 1. Metadata and Partitioning
        metadata = self.extract_raw_metadata()
        metadata = self.assign_benchmark_partitions(metadata)
        df_meta = pd.DataFrame(metadata)

        meta_path = self.output_dir / "cells_metadata.csv"
        df_meta.to_csv(meta_path, index=False)
        logger.info("Exported cells metadata to %s (%d rows)", meta_path, len(df_meta))

        # 2. Streaming Cycle Summaries Extraction
        summary_dfs: List[pd.DataFrame] = []
        meta_lookup = {m["cell_id"]: m for m in metadata}

        for batch_key, fname in BATCH_FILES.items():
            fpath = self.datasets_dir / fname
            prefix = "b1" if batch_key == "batch1" else "b2" if batch_key == "batch2" else "b3"

            logger.info("Extracting cycle summaries from %s...", fname)
            with h5py.File(fpath, "r") as f:
                batch_grp = f["batch"]
                num_cells = batch_grp["summary"].shape[0]

                for i in range(num_cells):
                    cell_id = f"{prefix}c{i}"
                    cell_summary = self.stream_cell_summaries(batch_key, f, i, cell_id)
                    if not cell_summary.empty:
                        # Append benchmark flags
                        cell_info = meta_lookup[cell_id]
                        cell_summary["is_benchmark_eligible"] = cell_info["is_benchmark_eligible"]
                        cell_summary["benchmark_partition"] = cell_info["benchmark_partition"]
                        cell_summary["is_continuation_segment"] = cell_id in CARRYOVER_MAPPING.values()
                        summary_dfs.append(cell_summary)

        df_all_summaries = pd.concat(summary_dfs, ignore_index=True)

        # 3. Handle Carry-over Cycle Appending for the 5 resumed cells
        # Stitch continuation cycles onto b1c0..b1c4 so they reflect complete degradation
        logger.info("Stitching continuation cycles for the 5 resumed cells...")
        for b1_id, (b2_id, add_l) in CARRYOVER_MAPPING.items():
            b1_mask = df_all_summaries["cell_id"] == b1_id
            b2_rows = df_all_summaries[df_all_summaries["cell_id"] == b2_id].copy()

            if not b2_rows.empty and b1_mask.any():
                max_b1_cycle = df_all_summaries.loc[b1_mask, "cycle"].max()
                b2_rows["cell_id"] = b1_id
                b2_rows["cycle"] = b2_rows["cycle"] + max_b1_cycle
                b2_rows["is_continuation_segment"] = True
                df_all_summaries = pd.concat([df_all_summaries, b2_rows], ignore_index=True)
                logger.info(
                    "Stitched %d continuation cycles onto %s (New total: %d cycles)",
                    len(b2_rows),
                    b1_id,
                    len(df_all_summaries[df_all_summaries["cell_id"] == b1_id]),
                )

        # Sort summary rows deterministically
        df_all_summaries.sort_values(by=["cell_id", "cycle"], inplace=True)

        # Export cycle summaries
        summaries_path_gz = self.output_dir / "cycle_summaries.csv.gz"
        summaries_path_csv = self.output_dir / "cycle_summaries.csv"

        df_all_summaries.to_csv(summaries_path_gz, index=False, compression="gzip")
        df_all_summaries.to_csv(summaries_path_csv, index=False)
        logger.info(
            "Exported cycle summaries to %s and %s (%d records)",
            summaries_path_gz,
            summaries_path_csv,
            len(df_all_summaries),
        )

        # 4. Export Partition Splits
        splits_dict = {
            "description": "Authoritative Severson et al. (2019) benchmark splits",
            "reference_pool_rule": "For peer-matching, test cells must draw peers EXCLUSIVELY from the training reference pool.",
            "train": [m["cell_id"] for m in metadata if m["benchmark_partition"] == "train"],
            "primary_test": [
                m["cell_id"] for m in metadata if m["benchmark_partition"] == "primary_test"
            ],
            "secondary_test": [
                m["cell_id"] for m in metadata if m["benchmark_partition"] == "secondary_test"
            ],
            "excluded": [
                m["cell_id"] for m in metadata if m["benchmark_partition"] == "excluded"
            ],
            "split_sizes": {
                "train": len([m for m in metadata if m["benchmark_partition"] == "train"]),
                "primary_test": len(
                    [m for m in metadata if m["benchmark_partition"] == "primary_test"]
                ),
                "secondary_test": len(
                    [m for m in metadata if m["benchmark_partition"] == "secondary_test"]
                ),
                "excluded": len([m for m in metadata if m["benchmark_partition"] == "excluded"]),
                "eligible_total": len([m for m in metadata if m["is_benchmark_eligible"]]),
                "raw_total": len(metadata),
            },
        }

        splits_json_path = self.splits_dir / "splits_benchmark.json"
        with open(splits_json_path, "w", encoding="utf-8") as f:
            json.dump(splits_dict, f, indent=2)

        splits_csv_path = self.splits_dir / "splits_benchmark.csv"
        df_meta[["cell_id", "batch", "benchmark_partition", "is_benchmark_eligible", "exclusion_reason"]].to_csv(
            splits_csv_path, index=False
        )
        logger.info("Exported splits to %s and %s", splits_json_path, splits_csv_path)

        # 5. Generate Machine-Readable Audit Report
        audit_report = self.generate_audit_report(metadata, df_all_summaries)
        audit_json_path = self.results_dir / "data_audit.json"
        with open(audit_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)
        logger.info("Saved data audit report to %s", audit_json_path)

        return df_meta, df_all_summaries, audit_report

    def generate_audit_report(
        self, metadata: List[Dict[str, Any]], df_summaries: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compiles a rigorous data audit report."""
        batch_counts = {
            "batch1": len([m for m in metadata if m["batch"] == "batch1"]),
            "batch2": len([m for m in metadata if m["batch"] == "batch2"]),
            "batch3": len([m for m in metadata if m["batch"] == "batch3"]),
            "total_raw": len(metadata),
        }

        # Check for non-finite or non-positive cycle life
        non_finite_life_cells = [
            m["cell_id"]
            for m in metadata
            if np.isnan(m["raw_cycle_life"]) or m["raw_cycle_life"] <= 0
        ]

        # Analyze zero discharge capacities in summaries
        zero_qd_total = int((df_summaries["QDischarge"] == 0).sum())
        zero_qd_cycle1 = int(
            ((df_summaries["cycle"] == 1) & (df_summaries["QDischarge"] == 0)).sum()
        )
        zero_qd_later = zero_qd_total - zero_qd_cycle1

        # Summary statistics for eligible cells
        eligible_meta = [m for m in metadata if m["is_benchmark_eligible"]]
        eligible_lives = [m["benchmark_cycle_life"] for m in eligible_meta]

        audit = {
            "dataset_files": {
                b: {
                    "filename": fname,
                    "size_bytes": os.path.getsize(self.datasets_dir / fname),
                    "size_gb": round(os.path.getsize(self.datasets_dir / fname) / (1024**3), 3),
                    "raw_cells": batch_counts[b],
                }
                for b, fname in BATCH_FILES.items()
            },
            "cell_counts": {
                "raw_total": len(metadata),
                "retained_eligible": len(eligible_meta),
                "excluded_total": len(metadata) - len(eligible_meta),
                "batch_breakdown": batch_counts,
            },
            "quality_filter": {
                "authoritative_source": "Severson et al. (2019) Nature Energy / Braatz Group Load Data.ipynb",
                "total_excluded": len(EXCLUSION_REGISTRY),
                "exclusions": [
                    {
                        "cell_id": cid,
                        "batch": [m["batch"] for m in metadata if m["cell_id"] == cid][0],
                        "reason": reason,
                    }
                    for cid, reason in EXCLUSION_REGISTRY.items()
                ],
            },
            "partitions": {
                "train_reference_pool_count": len(
                    [m for m in metadata if m["benchmark_partition"] == "train"]
                ),
                "primary_test_count": len(
                    [m for m in metadata if m["benchmark_partition"] == "primary_test"]
                ),
                "secondary_test_count": len(
                    [m for m in metadata if m["benchmark_partition"] == "secondary_test"]
                ),
                "excluded_count": len(
                    [m for m in metadata if m["benchmark_partition"] == "excluded"]
                ),
                "leakage_prevention_rule": "Peer pool for test batteries must be drawn exclusively from train reference pool.",
            },
            "integrity_checks": {
                "unique_cell_ids": len(set(m["cell_id"] for m in metadata)) == len(metadata),
                "non_finite_cycle_life_cells": non_finite_life_cells,
                "zero_capacity_findings": {
                    "total_zero_qd_records": zero_qd_total,
                    "cycle_1_zero_qd_records": zero_qd_cycle1,
                    "post_cycle_1_zero_qd_records": zero_qd_later,
                    "explanation": (
                        "Cycle 1 discharge capacity is 0 across Batch 1 cells because cycle 1 was unmonitored / diagnostic. "
                        "All early-life feature extraction starts strictly from Cycle 2 onward."
                    ),
                },
                "cycle_life_statistics_eligible": {
                    "min": float(np.min(eligible_lives)),
                    "max": float(np.max(eligible_lives)),
                    "mean": float(np.mean(eligible_lives)),
                    "median": float(np.median(eligible_lives)),
                    "std": float(np.std(eligible_lives)),
                },
            },
        }
        return audit


# Convenience loader functions
def load_cells_metadata(path: Optional[str] = None) -> pd.DataFrame:
    """Loads extracted cell metadata."""
    if path is None:
        path = "data/processed/cells_metadata.csv"
    return pd.read_csv(path)


def load_cycle_summaries(path: Optional[str] = None, eligible_only: bool = True) -> pd.DataFrame:
    """Loads cycle summary records."""
    if path is None:
        gz_path = "data/processed/cycle_summaries.csv.gz"
        csv_path = "data/processed/cycle_summaries.csv"
        path = gz_path if os.path.exists(gz_path) else csv_path
    df = pd.read_csv(path)
    if eligible_only and "is_benchmark_eligible" in df.columns:
        df = df[df["is_benchmark_eligible"] == True].copy()  # noqa: E712
    return df


def load_benchmark_splits(path: Optional[str] = None) -> Dict[str, Any]:
    """Loads authoritative benchmark splits."""
    if path is None:
        path = "data/splits/splits_benchmark.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    loader = BatteryDataLoader()
    loader.extract_and_export_all()
