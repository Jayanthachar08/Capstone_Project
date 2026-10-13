"""
Module: src.features.modeling_dataset
Author: Capstone Project Team
Description: Assembles, validates, and exports the model-ready dataset
             combining metadata, initial conditions, snapshots, trajectories,
             differential curves, and prediction targets across observation cutoffs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.features.feature_engineering import FeatureEngineer
from src.features.target_builder import TargetBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelingDatasetBuilder:
    """Builds and exports the master multi-cutoff modeling dataset."""

    DEFAULT_CUTOFFS = [40, 70, 100]

    def __init__(
        self,
        datasets_dir: str = "Datasets",
        processed_dir: str = "data/processed",
        cutoffs: Optional[List[int]] = None,
    ):
        self.datasets_dir = Path(datasets_dir)
        self.processed_dir = Path(processed_dir)
        self.cutoffs = cutoffs or self.DEFAULT_CUTOFFS

    def build_dataset(self) -> pd.DataFrame:
        """Constructs the complete modeling dataset across all cutoffs and batteries."""
        meta_path = self.processed_dir / "cells_metadata.csv"
        summaries_path_gz = self.processed_dir / "cycle_summaries.csv.gz"
        summaries_path_csv = self.processed_dir / "cycle_summaries.csv"

        if not meta_path.exists():
            raise FileNotFoundError(f"Missing master metadata: {meta_path}")

        df_meta = pd.read_csv(meta_path)
        sum_path = summaries_path_gz if summaries_path_gz.exists() else summaries_path_csv
        if not sum_path.exists():
            raise FileNotFoundError(f"Missing cycle summaries: {sum_path}")

        logger.info("Loading cycle summaries from %s...", sum_path)
        df_summaries = pd.read_csv(sum_path)

        rows: List[Dict[str, Any]] = []

        with FeatureEngineer(datasets_dir=str(self.datasets_dir)) as engineer:
            for cutoff in self.cutoffs:
                logger.info("Processing features and targets for cutoff %d cycles...", cutoff)

                for _, meta_row in df_meta.iterrows():
                    cid = meta_row["cell_id"]
                    batch_key = meta_row["batch"]
                    batch_cell_idx = int(meta_row["batch_cell_index"])
                    is_eligible = bool(meta_row["is_benchmark_eligible"])
                    cl = float(meta_row["benchmark_cycle_life"])
                    n_cyc = int(meta_row["num_summary_cycles"])

                    # 1. Target computation
                    targets = TargetBuilder.compute_targets_for_cell(
                        cell_id=cid,
                        cutoff=cutoff,
                        cycle_life=cl,
                        is_eligible=is_eligible,
                        num_summary_cycles=n_cyc,
                    )

                    # 2. Differential voltage curve features
                    curve_feats = engineer.extract_curve_features(
                        batch_key=batch_key,
                        batch_cell_index=batch_cell_idx,
                        cutoff=cutoff,
                        baseline_cycle=10,
                    )

                    # 3. Summary-based features
                    cell_sums = df_summaries[df_summaries["cell_id"] == cid]
                    cell_feats = engineer.extract_cell_features_at_cutoff(
                        cell_summary_df=cell_sums,
                        cell_meta=meta_row,
                        cutoff=cutoff,
                        curve_features=curve_feats,
                    )

                    # 4. Row composition
                    row_data = {
                        "cell_id": cid,
                        "batch": batch_key,
                        "cutoff": cutoff,
                        "channel_id": meta_row.get("channel_id", -1),
                        "policy": meta_row.get("policy", ""),
                        "is_benchmark_eligible": is_eligible,
                        "benchmark_partition": meta_row.get("benchmark_partition", "unassigned"),
                        "exclusion_reason": meta_row.get("exclusion_reason", ""),
                    }
                    row_data.update(cell_feats)
                    row_data.update(targets)

                    rows.append(row_data)

        df_model = pd.DataFrame(rows)

        # Deterministic sort
        df_model.sort_values(by=["cutoff", "cell_id"], inplace=True)
        return df_model

    def export_dataset(self, df_model: pd.DataFrame) -> Tuple[Path, Optional[Path]]:
        """Saves modeling dataset to CSV, CSV.gz, and Parquet (if pyarrow available)."""
        csv_path = self.processed_dir / "modeling_dataset.csv"
        csv_gz_path = self.processed_dir / "modeling_dataset.csv.gz"
        parquet_path = self.processed_dir / "modeling_dataset.parquet"

        logger.info("Exporting modeling dataset to %s and %s...", csv_path, csv_gz_path)
        df_model.to_csv(csv_path, index=False)
        df_model.to_csv(csv_gz_path, index=False, compression="gzip")

        exported_parquet = None
        try:
            df_model.to_parquet(parquet_path, index=False)
            logger.info("Exported modeling dataset to %s", parquet_path)
            exported_parquet = parquet_path
        except Exception as err:
            logger.info("Parquet export skipped (optional engine not installed): %s", err)

        return csv_path, exported_parquet


# Helper inspection functions
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Returns list of predictive feature column names."""
    return [
        col
        for col in df.columns
        if col.startswith(("init_", "snap_", "traj_", "curve_"))
    ]


def get_target_columns(df: pd.DataFrame) -> List[str]:
    """Returns list of target-related column names."""
    return [col for col in df.columns if col.startswith(("target_", "status_"))]


if __name__ == "__main__":
    builder = ModelingDatasetBuilder()
    df_built = builder.build_dataset()
    builder.export_dataset(df_built)
    logger.info("Phase 3 modeling dataset successfully created: %s rows", len(df_built))
