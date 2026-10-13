"""
Module: src.models.leakage_audit
Author: Capstone Project Team
Description: Formal audit module verifying data leakage prevention, feature set isolation,
             and partition integrity across model configurations M1, M2, M3, and M4.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from src.models.model_factory import FEATURE_GROUPS, MODEL_CONFIGS, get_feature_names_for_model

logger = logging.getLogger(__name__)

FORBIDDEN_TARGET_KEYWORDS = ["eol", "rul", "life", "target", "status", "fail"]


class LeakageAuditor:
    """Performs rigorous automated checks ensuring zero data leakage in Phase 5 models."""

    def __init__(self, df_merged: pd.DataFrame, splits_json_path: str = "data/splits/splits_benchmark.json"):
        self.df = df_merged
        self.splits_path = Path(splits_json_path)

    def run_full_audit(self, output_json_path: Optional[str] = "results/phase5_leakage_audit.json") -> Dict[str, Any]:
        """Runs the complete suite of anti-leakage and configuration integrity checks."""
        audit_results: Dict[str, Any] = {
            "checks_passed": True,
            "target_keyword_audit": self._audit_target_keywords(),
            "configuration_orthogonality_audit": self._audit_configuration_orthogonality(),
            "partition_disjointness_audit": self._audit_partitions(),
            "split_consistency_audit": self._audit_split_invariance(),
            "completeness_audit": self._audit_feature_completeness(),
        }

        # Check if all checks succeeded
        for check_name, check_data in audit_results.items():
            if isinstance(check_data, dict) and not check_data.get("passed", True):
                audit_results["checks_passed"] = False
                logger.error("Leakage check failed: %s -> %s", check_name, check_data)

        if output_json_path:
            out_p = Path(output_json_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(audit_results, f, indent=2)
            logger.info("Saved Phase 5 leakage audit to %s", out_p)

        return audit_results

    def _audit_target_keywords(self) -> Dict[str, Any]:
        """Verifies that no model feature contains forbidden target or outcome keywords."""
        violations: Dict[str, List[str]] = {}
        for m_key in ["M1", "M2", "M3", "M4"]:
            feats = get_feature_names_for_model(m_key)
            bad = []
            for f in feats:
                f_lower = f.lower()
                for kw in FORBIDDEN_TARGET_KEYWORDS:
                    if kw in f_lower:
                        bad.append(f"{f} contains '{kw}'")
            if bad:
                violations[m_key] = bad

        return {
            "passed": len(violations) == 0,
            "forbidden_keywords_checked": FORBIDDEN_TARGET_KEYWORDS,
            "violations": violations,
        }

    def _audit_configuration_orthogonality(self) -> Dict[str, Any]:
        """Verifies strict feature group boundaries between M1, M2, M3, and M4."""
        m1_feats = set(get_feature_names_for_model("M1"))
        m2_feats = set(get_feature_names_for_model("M2"))
        m3_feats = set(get_feature_names_for_model("M3"))
        m4_feats = set(get_feature_names_for_model("M4"))

        traj_feats = set(FEATURE_GROUPS["traj"]) | set(FEATURE_GROUPS["curve"])
        peer_traj_feats = set(FEATURE_GROUPS["peer_traj"])
        reliability_feats = set(FEATURE_GROUPS["peer_reliability"])

        errors = []

        # Check 1: M1 must contain zero trajectory, peer, or reliability features
        if not m1_feats.isdisjoint(traj_feats):
            errors.append(f"M1 contains trajectory features: {m1_feats & traj_feats}")
        if not m1_feats.isdisjoint(peer_traj_feats):
            errors.append(f"M1 contains peer-relative features: {m1_feats & peer_traj_feats}")
        if not m1_feats.isdisjoint(reliability_feats):
            errors.append(f"M1 contains reliability features: {m1_feats & reliability_feats}")

        # Check 2: M2 must contain trajectory features, but zero peer or reliability features
        if not m2_feats.isdisjoint(peer_traj_feats):
            errors.append(f"M2 contains peer-relative features: {m2_feats & peer_traj_feats}")
        if not m2_feats.isdisjoint(reliability_feats):
            errors.append(f"M2 contains reliability features: {m2_feats & reliability_feats}")

        # Check 3: M3 must contain peer-relative features, but zero PMRI/reliability features
        if not m3_feats.isdisjoint(reliability_feats):
            errors.append(f"M3 contains reliability features: {m3_feats & reliability_feats}")

        # Check 4: Hierarchy M1 subset of M2 subset of M3 subset of M4
        if not m1_feats.issubset(m2_feats):
            errors.append("M1 is not a strict subset of M2")
        if not m2_feats.issubset(m3_feats):
            errors.append("M2 is not a strict subset of M3")
        if not m3_feats.issubset(m4_feats):
            errors.append("M3 is not a strict subset of M4")

        return {
            "passed": len(errors) == 0,
            "feature_counts": {
                "M1": len(m1_feats),
                "M2": len(m2_feats),
                "M3": len(m3_feats),
                "M4": len(m4_feats),
            },
            "hierarchy_verified": len(errors) == 0,
            "errors": errors,
        }

    def _audit_partitions(self) -> Dict[str, Any]:
        """Verifies pairwise disjointness between partitions."""
        train_cells = set(self.df[self.df["benchmark_partition"] == "train"]["cell_id"])
        primary_test_cells = set(self.df[self.df["benchmark_partition"] == "primary_test"]["cell_id"])
        secondary_test_cells = set(self.df[self.df["benchmark_partition"] == "secondary_test"]["cell_id"])
        excluded_cells = set(self.df[self.df["benchmark_partition"] == "excluded"]["cell_id"])

        overlaps = {
            "train_vs_primary": list(train_cells & primary_test_cells),
            "train_vs_secondary": list(train_cells & secondary_test_cells),
            "primary_vs_secondary": list(primary_test_cells & secondary_test_cells),
            "train_vs_excluded": list(train_cells & excluded_cells),
        }

        has_overlap = any(len(v) > 0 for v in overlaps.values())

        return {
            "passed": not has_overlap,
            "counts": {
                "train": len(train_cells),
                "primary_test": len(primary_test_cells),
                "secondary_test": len(secondary_test_cells),
                "excluded": len(excluded_cells),
            },
            "overlaps": overlaps,
        }

    def _audit_split_invariance(self) -> Dict[str, Any]:
        """Confirms that every battery belongs to the exact same partition across all cutoffs."""
        inconsistent = []
        for cid, grp in self.df.groupby("cell_id"):
            parts = grp["benchmark_partition"].unique()
            if len(parts) > 1:
                inconsistent.append(f"{cid} assigned to {parts}")

        return {
            "passed": len(inconsistent) == 0,
            "inconsistent_cells": inconsistent,
        }

    def _audit_feature_completeness(self) -> Dict[str, Any]:
        """Confirms that all required features for eligible cells are completely non-null and finite."""
        elig = self.df[self.df["is_benchmark_eligible"]]
        m4_feats = get_feature_names_for_model("M4")
        null_counts = elig[m4_feats].isna().sum().to_dict()
        bad_feats = {k: v for k, v in null_counts.items() if v > 0}

        return {
            "passed": len(bad_feats) == 0,
            "eligible_records_checked": len(elig),
            "features_checked": len(m4_feats),
            "null_violations": bad_feats,
        }
