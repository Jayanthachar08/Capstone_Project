"""
Package: src.models
Author: Capstone Project Team
Description: Training, evaluation, calibration, and leakage auditing subsystem
             for the four lithium-ion battery EOL risk prediction model configurations:
             M1 (Condition baseline), M2 (Individual trajectory), M3 (Peer-relative),
             and M4 (Confidence-aware peer-relative).
"""

from src.models.model_factory import (
    MODEL_CONFIGS,
    ModelFactory,
    get_feature_names_for_model,
)
from src.models.evaluate_models import ModelEvaluator
from src.models.leakage_audit import LeakageAuditor
from src.models.calibration import ProbabilityCalibrator

__all__ = [
    "MODEL_CONFIGS",
    "ModelFactory",
    "get_feature_names_for_model",
    "ModelEvaluator",
    "LeakageAuditor",
    "ProbabilityCalibrator",
]
