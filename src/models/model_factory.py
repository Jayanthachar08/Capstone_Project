"""
Module: src.models.model_factory
Author: Capstone Project Team
Description: Defines the feature group configurations (M1, M2, M3, M4) and constructs
             reproducible scikit-learn machine learning pipelines for battery EOL risk prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

# Feature Group Definitions
FEATURE_GROUPS: Dict[str, List[str]] = {
    # 1. Initial Condition (11 features at Cycle 2)
    "init": [
        "init_q_discharge",
        "init_q_charge",
        "init_coulombic_eff",
        "init_internal_resistance",
        "init_temp_max",
        "init_temp_avg",
        "init_temp_min",
        "init_charge_time",
        "init_policy_c1",
        "init_policy_soc",
        "init_policy_c2",
    ],
    # 2. Instantaneous Snapshot (7 features at cutoff N_obs)
    "snap": [
        "snap_q_discharge",
        "snap_q_charge",
        "snap_coulombic_eff",
        "snap_internal_resistance",
        "snap_temp_max",
        "snap_temp_avg",
        "snap_charge_time",
    ],
    # 3. Emerging Individual Trajectory (11 features over [2, N_obs])
    "traj": [
        "traj_q_fade",
        "traj_q_retention",
        "traj_q_slope",
        "traj_q_curvature",
        "traj_q_std",
        "traj_q_min",
        "traj_ir_change",
        "traj_ir_slope",
        "traj_temp_max_change",
        "traj_temp_avg_mean",
        "traj_charge_time_change",
    ],
    # 4. Differential Voltage Curve Statistics (5 features from Delta Q_{cutoff-10}(V))
    "curve": [
        "curve_delta_q_var_log10",
        "curve_delta_q_min",
        "curve_delta_q_mean",
        "curve_delta_q_skew",
        "curve_delta_q_kurtosis",
    ],
    # 5. Peer-Relative Trajectory Features (8 features from Phase 4)
    "peer_traj": [
        "relative_capacity_deviation",
        "relative_slope_deviation",
        "relative_capacity_ratio",
        "relative_fade_deviation",
        "relative_ir_deviation",
        "peer_trajectory_dispersion",
        "peer_slope_dispersion",
        "relative_z_score_capacity",
    ],
    # 6. Peer-Match Reliability Features (4 features from Phase 4)
    "peer_reliability": [
        "pmri",
        "peer_similarity_score",
        "peer_agreement_score",
        "mean_peer_distance",
    ],
}

# Four Formal Model Feature Configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "M1": {
        "name": "Condition Baseline",
        "description": "Baseline model using only initial health state and current cycle snapshot measurements.",
        "groups": ["init", "snap"],
        "expected_feature_count": 18,
    },
    "M2": {
        "name": "Individual Trajectory",
        "description": "Individual degradation model adding historical trajectory slopes, curvature, and differential curve statistics.",
        "groups": ["init", "snap", "traj", "curve"],
        "expected_feature_count": 34,
    },
    "M3": {
        "name": "Peer-Relative",
        "description": "Peer-relative model combining individual features with relative trajectory deviations from matched peers.",
        "groups": ["init", "snap", "traj", "curve", "peer_traj"],
        "expected_feature_count": 42,
    },
    "M4": {
        "name": "Confidence-Aware Peer-Relative",
        "description": "Full proposed model incorporating peer-relative trajectory deviations and peer-match reliability metrics.",
        "groups": ["init", "snap", "traj", "curve", "peer_traj", "peer_reliability"],
        "expected_feature_count": 46,
    },
}


def get_feature_names_for_model(model_key: str) -> List[str]:
    """Retrieves the list of feature column names for a given model configuration.

    Args:
        model_key: Key in ('M1', 'M2', 'M3', 'M4').

    Returns:
        List of feature column names.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{model_key}'. Permitted: {list(MODEL_CONFIGS.keys())}")

    groups = MODEL_CONFIGS[model_key]["groups"]
    features: List[str] = []
    for g in groups:
        features.extend(FEATURE_GROUPS[g])
    return features


class ModelFactory:
    """Factory creating scikit-learn classification pipelines for the four model configurations."""

    @staticmethod
    def build_logistic_regression(
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> Pipeline:
        """Constructs a standardized Logistic Regression pipeline.

        Args:
            C: Inverse regularization strength (default: 1.0).
            class_weight: Class balancing strategy (default: 'balanced').
            max_iter: Maximum solver iterations (default: 1000).
            random_state: Deterministic random seed (default: 42).
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(C),
                    class_weight=class_weight,
                    solver="lbfgs",
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                ),
            ),
        ])

    @staticmethod
    def build_random_forest(
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 4,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> Pipeline:
        """Constructs a secondary Random Forest benchmark pipeline."""
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(n_estimators),
                    max_depth=max_depth,
                    min_samples_split=int(min_samples_split),
                    class_weight=class_weight,
                    random_state=int(random_state),
                ),
            ),
        ])

    @staticmethod
    def build_rbf_svm(
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> Pipeline:
        """Constructs a standardized RBF Support Vector Machine pipeline with Platt probability calibration."""
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=float(C),
                    kernel="rbf",
                    gamma=gamma,
                    probability=True,
                    class_weight=class_weight,
                    random_state=int(random_state),
                ),
            ),
        ])

    @staticmethod
    def build_hist_gradient_boosting(
        max_iter: int = 100,
        max_depth: Optional[int] = 3,
        learning_rate: float = 0.05,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ) -> Pipeline:
        """Constructs a standardized Histogram Gradient Boosting Classifier pipeline."""
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=int(max_iter),
                    max_depth=max_depth,
                    learning_rate=float(learning_rate),
                    class_weight=class_weight,
                    random_state=int(random_state),
                ),
            ),
        ])

    @classmethod
    def build_pipeline(
        cls,
        algorithm: str = "logistic_regression",
        random_state: int = 42,
        **kwargs,
    ) -> Pipeline:
        """Shared modular interface creating an ML pipeline for any supported algorithm."""
        algo = algorithm.lower().replace("-", "_").replace(" ", "_")
        if algo in ("logistic_regression", "lr"):
            return cls.build_logistic_regression(random_state=random_state, **kwargs)
        elif algo in ("random_forest", "rf"):
            return cls.build_random_forest(random_state=random_state, **kwargs)
        elif algo in ("rbf_svm", "svm", "svc"):
            return cls.build_rbf_svm(random_state=random_state, **kwargs)
        elif algo in ("hist_gradient_boosting", "gradient_boosting", "hgb", "gbm"):
            return cls.build_hist_gradient_boosting(random_state=random_state, **kwargs)
        else:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Supported: "
                "['logistic_regression', 'random_forest', 'rbf_svm', 'hist_gradient_boosting']"
            )


SUPPORTED_ALGORITHMS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "name": "Standardized Logistic Regression",
        "family": "Linear",
        "description": "Baseline linear classifier with L2 regularization and Platt probability calibration.",
        "params": {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced"},
    },
    "random_forest": {
        "name": "Random Forest Ensemble",
        "family": "Tree Ensemble",
        "description": "Nonlinear bagging ensemble with 100 shallow decision trees.",
        "params": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4, "class_weight": "balanced"},
    },
    "rbf_svm": {
        "name": "RBF Support Vector Machine",
        "family": "Kernel Method",
        "description": "Radial basis function kernel SVM with training-only Platt scaling.",
        "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
    },
    "hist_gradient_boosting": {
        "name": "Histogram Gradient Boosting",
        "family": "Boosting",
        "description": "Lightweight gradient boosted decision trees with histogram binning.",
        "params": {"max_iter": 100, "max_depth": 3, "learning_rate": 0.05, "class_weight": "balanced"},
    },
}
