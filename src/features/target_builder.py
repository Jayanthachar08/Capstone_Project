"""
Module: src.features.target_builder
Author: Capstone Project Team
Description: Builds, validates, and labels continuous and horizon-specific future EOL
             prediction targets across early-life observation cutoffs with strict censoring handling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class TargetBuilder:
    """Computes and validates future End-of-Life (EOL) prediction targets.

    EOL Criterion:
        Cycle number at which the lithium-ion cell's discharge capacity first degrades
        to 80% of nominal capacity (1.1 Ah nominal -> 0.88 Ah threshold), following
        the Severson et al. (Nature Energy 2019) benchmark definition.
    """

    HORIZON_SHORT = 300
    HORIZON_PRIMARY = 500
    EARLY_FAILURE_THRESHOLD = 700.0

    @classmethod
    def compute_targets_for_cell(
        cls,
        cell_id: str,
        cutoff: int,
        cycle_life: float,
        is_eligible: bool,
        num_summary_cycles: int,
    ) -> Dict[str, Optional[float | str | int]]:
        """Computes target labels and auditing status for a single cell at a given cutoff.

        Labeling rules:
        - Positive (1): Cell reaches documented EOL event within the specified horizon.
        - Negative (0): Observed follow-up reliably establishes event did not occur in horizon.
        - Censored (NaN / 'censored'): Available record insufficient or unverified.
        """
        # Censored or unverified non-benchmark cell
        if not is_eligible or np.isnan(cycle_life) or cycle_life <= 0:
            return {
                "cell_id": cell_id,
                "cutoff": cutoff,
                "target_eol_cycle": np.nan,
                "target_rul": np.nan,
                "target_fail_within_300": np.nan,
                "status_fail_within_300": "censored",
                "target_fail_within_500": np.nan,
                "status_fail_within_500": "censored",
                "target_fail_by_700": np.nan,
                "status_fail_by_700": "censored",
            }

        # For eligible benchmark cells, cycle_life is uncensored observed cycle to 80% capacity
        eol_cycle = float(cycle_life)
        rul = float(eol_cycle - cutoff)

        # 1. Horizon 300: failure within cutoff + 300 cycles
        limit_300 = cutoff + cls.HORIZON_SHORT
        if eol_cycle <= limit_300:
            y_300 = 1
            s_300 = "positive"
        else:
            y_300 = 0
            s_300 = "negative"

        # 2. Horizon 500 (Primary Target): failure within cutoff + 500 cycles
        limit_500 = cutoff + cls.HORIZON_PRIMARY
        if eol_cycle <= limit_500:
            y_500 = 1
            s_500 = "positive"
        else:
            y_500 = 0
            s_500 = "negative"

        # 3. Absolute Early Failure: failure before cycle 700
        if eol_cycle <= cls.EARLY_FAILURE_THRESHOLD:
            y_700 = 1
            s_700 = "positive"
        else:
            y_700 = 0
            s_700 = "negative"

        return {
            "cell_id": cell_id,
            "cutoff": cutoff,
            "target_eol_cycle": eol_cycle,
            "target_rul": rul,
            "target_fail_within_300": y_300,
            "status_fail_within_300": s_300,
            "target_fail_within_500": y_500,
            "status_fail_within_500": s_500,
            "target_fail_by_700": y_700,
            "status_fail_by_700": s_700,
        }

    @classmethod
    def build_targets_dataframe(
        cls, df_meta: pd.DataFrame, cutoffs: List[int] = [40, 70, 100]
    ) -> pd.DataFrame:
        """Constructs target labels across all cells and specified observation cutoffs."""
        records: List[Dict[str, Optional[float | str | int]]] = []

        for cutoff in cutoffs:
            for _, row in df_meta.iterrows():
                target_dict = cls.compute_targets_for_cell(
                    cell_id=row["cell_id"],
                    cutoff=cutoff,
                    cycle_life=row["benchmark_cycle_life"],
                    is_eligible=bool(row["is_benchmark_eligible"]),
                    num_summary_cycles=int(row["num_summary_cycles"]),
                )
                records.append(target_dict)

        return pd.DataFrame(records)
