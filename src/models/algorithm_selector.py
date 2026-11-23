"""
Module: src.models.algorithm_selector
Author: Capstone Project Team
Description: Centralized machine-learning algorithm selection and registry engine.
             Provides a global algorithm selector that ensures the selected algorithm
             (Logistic Regression, RBF-SVM, or TabPFN 2.5) is consistently instantiated
             across all four feature configurations (M1, M2, M3, M4).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from sklearn.pipeline import Pipeline

from src.models.model_factory import ModelFactory, SUPPORTED_ALGORITHMS, MODEL_CONFIGS

logger = logging.getLogger(__name__)

DEFAULT_ALGORITHM = "logistic_regression"


class AlgorithmSelector:
    """Manages global algorithm selection and pipeline instantiation across M1-M4."""

    _active_algorithm: str = DEFAULT_ALGORITHM

    @classmethod
    def get_active_algorithm(cls) -> str:
        """Returns the currently active global algorithm identifier."""
        return cls._active_algorithm

    @classmethod
    def set_active_algorithm(cls, algorithm: str) -> str:
        """Sets the global active algorithm.

        Args:
            algorithm: Algorithm key in ('logistic_regression', 'rbf_svm', 'tabpfn_2_5').

        Returns:
            Normalized algorithm key.
        """
        norm = cls.normalize_algorithm_key(algorithm)
        if norm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Permitted: {list(SUPPORTED_ALGORITHMS.keys())}"
            )
        cls._active_algorithm = norm
        logger.info("Global algorithm selector set to: %s (%s)", norm, SUPPORTED_ALGORITHMS[norm]["name"])
        return cls._active_algorithm

    @classmethod
    def normalize_algorithm_key(cls, key: str) -> str:
        """Normalizes algorithm names and aliases."""
        k = key.lower().strip().replace("-", "_").replace(" ", "_").replace(".", "_")
        if k in ("logistic_regression", "lr", "logistic"):
            return "logistic_regression"
        elif k in ("rbf_svm", "svm", "svc", "rbf"):
            return "rbf_svm"
        elif k in ("tabpfn_2_5", "tabpfn_2", "tabpfn", "tab_pfn"):
            return "tabpfn_2_5"
        return k

    @classmethod
    def list_algorithms(cls) -> List[Dict[str, Any]]:
        """Returns metadata for all supported algorithms."""
        result = []
        for k, meta in SUPPORTED_ALGORITHMS.items():
            result.append({
                "id": k,
                "name": meta["name"],
                "family": meta["family"],
                "description": meta["description"],
                "params": meta["params"],
                "is_active": (k == cls._active_algorithm),
            })
        return result

    @classmethod
    def get_algorithm_metadata(cls, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Retrieves metadata dictionary for the specified or active algorithm."""
        algo = cls.normalize_algorithm_key(algorithm) if algorithm else cls._active_algorithm
        if algo not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algo}'. Permitted: {list(SUPPORTED_ALGORITHMS.keys())}")
        return SUPPORTED_ALGORITHMS[algo]

    @classmethod
    def build_pipeline_for_model(
        cls,
        model_key: str,
        algorithm: Optional[str] = None,
        random_state: int = 42,
        **kwargs: Any,
    ) -> Pipeline:
        """Constructs a pipeline for the given feature configuration (M1-M4)

        using either the specified algorithm or the global active algorithm.
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key '{model_key}'. Permitted: {list(MODEL_CONFIGS.keys())}")

        algo = cls.normalize_algorithm_key(algorithm) if algorithm else cls._active_algorithm
        return ModelFactory.build_pipeline(algo, random_state=random_state, **kwargs)
