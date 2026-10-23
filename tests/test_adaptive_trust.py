"""
Module: tests.test_adaptive_trust
Author: Capstone Project Team
Description: Unit tests for the Adaptive Peer-Trust Engine (M5).
             Verifies mathematical bounds, abstention logic, parameter tuning, and ablation modes.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.models.adaptive_peer_trust import AdaptivePeerTrustEngine


class TestAdaptivePeerTrust(unittest.TestCase):
    """Test suite verifying mathematical bounds, trust equations, and abstention policies."""

    def setUp(self):
        self.engine = AdaptivePeerTrustEngine(
            lambda_param=3.0,
            pmri_abstention_threshold=0.50,
            instability_abstention_threshold=0.15,
            trust_abstention_threshold=0.30,
            random_seed=42,
        )

    def test_trust_score_bounds(self):
        """Rule 1: Trust score T must strictly lie in [0.0, 1.0]."""
        test_cases = [
            (1.0, 0.0, 3.0),
            (0.0, 0.5, 3.0),
            (0.8, 0.2, 5.0),
            (0.5, 10.0, 1.0),
            (np.nan, 0.1, 3.0),
            (0.7, np.nan, 3.0),
        ]
        for pmri, u, lam in test_cases:
            t, penalty = self.engine.calculate_trust(pmri, u, lam)
            self.assertTrue(0.0 <= t <= 1.0, f"Trust {t} out of bounds for R={pmri}, U={u}")
            self.assertTrue(0.0 <= penalty <= 1.0)

    def test_blended_probability_bounds(self):
        """Rule 2: Blended M5 probability must strictly lie in [0.0, 1.0]."""
        test_cases = [
            (0.2, 0.8, 0.5),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.4, 0.6, 0.3),
            (np.nan, 0.7, 0.5),
            (0.3, np.nan, 0.5),
        ]
        for ph, pp, t in test_cases:
            p_m5 = self.engine.blend_probability(ph, pp, t)
            self.assertTrue(0.0 <= p_m5 <= 1.0, f"Probability {p_m5} out of bounds for ph={ph}, pp={pp}, t={t}")

    def test_ablation_modes(self):
        """Rule 3: Ablation modes must behave exactly according to formal mathematical definitions."""
        pmri = 0.80
        u = 0.10
        lam = 2.0

        # Full: T = R * exp(-lambda * U)
        t_full, _ = self.engine.calculate_trust(pmri, u, lam, ablation_mode="full")
        expected_full = pmri * np.exp(-lam * u)
        self.assertAlmostEqual(t_full, expected_full, places=5)

        # No PMRI: T = exp(-lambda * U)
        t_no_pmri, _ = self.engine.calculate_trust(pmri, u, lam, ablation_mode="no_pmri")
        expected_no_pmri = np.exp(-lam * u)
        self.assertAlmostEqual(t_no_pmri, expected_no_pmri, places=5)

        # No Instability: T = R
        t_no_u, _ = self.engine.calculate_trust(pmri, u, lam, ablation_mode="no_instability")
        self.assertAlmostEqual(t_no_u, pmri, places=5)

        # Fixed Blend: T = 0.5
        t_fixed, _ = self.engine.calculate_trust(pmri, u, lam, ablation_mode="fixed_blend")
        self.assertAlmostEqual(t_fixed, 0.5, places=5)

    def test_abstention_reasons(self):
        """Rule 4: Abstentions must be triggered with documented, specific reasons."""
        # 1. Low PMRI
        abstained, reason = self.engine.evaluate_abstention(pmri=0.40, instability=0.05, trust_score=0.35)
        self.assertTrue(abstained)
        self.assertIn("INSUFFICIENT_REFERENCE_COVERAGE", reason)

        # 2. Excessive Instability
        abstained, reason = self.engine.evaluate_abstention(pmri=0.85, instability=0.25, trust_score=0.45)
        self.assertTrue(abstained)
        self.assertIn("EXCESSIVE_COUNTERFACTUAL_INSTABILITY", reason)

        # 3. Low Adaptive Trust
        abstained, reason = self.engine.evaluate_abstention(pmri=0.60, instability=0.12, trust_score=0.25)
        self.assertTrue(abstained)
        self.assertIn("LOW_ADAPTIVE_TRUST", reason)

        # 4. Nominal Trust
        abstained, reason = self.engine.evaluate_abstention(pmri=0.85, instability=0.05, trust_score=0.75)
        self.assertFalse(abstained)
        self.assertEqual(reason, "NOMINAL_TRUST")

    def test_training_only_lambda_selection(self):
        """Rule 5: Lambda selection must optimize objective and select non-negative parameter."""
        train_y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
        train_ph = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.2, 0.1])
        train_pp = np.array([0.4, 0.1, 0.6, 0.9, 0.6, 0.8, 0.5, 0.3])
        train_r = np.array([0.8, 0.7, 0.9, 0.8, 0.5, 0.9, 0.7, 0.6])
        train_u = np.array([0.05, 0.02, 0.15, 0.08, 0.25, 0.04, 0.12, 0.03])

        best_lam = self.engine.tune_lambda_on_training_cv(
            train_y, train_ph, train_pp, train_r, train_u, lambda_candidates=[0.0, 1.0, 5.0, 10.0]
        )
        self.assertIn(best_lam, [0.0, 1.0, 5.0, 10.0])
        self.assertTrue(self.engine.is_tuned)


if __name__ == "__main__":
    unittest.main()
