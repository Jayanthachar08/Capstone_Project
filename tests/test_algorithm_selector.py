"""
Unit tests for the centralized AlgorithmSelector and ModelFactory.
Verifies that Logistic Regression, RBF-SVM, and TabPFN 2.5 can be dynamically selected
and applied consistently across all four model configurations (M1, M2, M3, M4).
"""

import unittest
import numpy as np
from src.models.algorithm_selector import AlgorithmSelector
from src.models.model_factory import MODEL_CONFIGS, get_feature_names_for_model


class TestAlgorithmSelector(unittest.TestCase):
    """Test suite for the centralized AlgorithmSelector engine."""

    def setUp(self):
        # Reset to default
        AlgorithmSelector.set_active_algorithm("logistic_regression")

    def test_default_active_algorithm(self):
        """Verify default algorithm is logistic_regression."""
        self.assertEqual(AlgorithmSelector.get_active_algorithm(), "logistic_regression")

    def test_normalize_algorithm_key(self):
        """Verify normalization of algorithm aliases."""
        self.assertEqual(AlgorithmSelector.normalize_algorithm_key("lr"), "logistic_regression")
        self.assertEqual(AlgorithmSelector.normalize_algorithm_key("LOGISTIC-REGRESSION"), "logistic_regression")
        self.assertEqual(AlgorithmSelector.normalize_algorithm_key("svm"), "rbf_svm")
        self.assertEqual(AlgorithmSelector.normalize_algorithm_key("rbf-svm"), "rbf_svm")
        self.assertEqual(AlgorithmSelector.normalize_algorithm_key("tabpfn"), "tabpfn_2_5")
        self.assertEqual(AlgorithmSelector.normalize_algorithm_key("TabPFN 2.5"), "tabpfn_2_5")

    def test_set_active_algorithm(self):
        """Verify setting active algorithm and handling invalid names."""
        AlgorithmSelector.set_active_algorithm("rbf_svm")
        self.assertEqual(AlgorithmSelector.get_active_algorithm(), "rbf_svm")

        AlgorithmSelector.set_active_algorithm("tabpfn_2_5")
        self.assertEqual(AlgorithmSelector.get_active_algorithm(), "tabpfn_2_5")

        with self.assertRaises(ValueError):
            AlgorithmSelector.set_active_algorithm("invalid_algo_name")

    def test_list_algorithms(self):
        """Verify all three algorithms are registered with full metadata."""
        algos = AlgorithmSelector.list_algorithms()
        ids = {a["id"] for a in algos}
        self.assertEqual(ids, {"logistic_regression", "rbf_svm", "tabpfn_2_5"})

    def test_pipeline_instantiation_for_all_models(self):
        """Verify that all three algorithms build valid pipelines for M1, M2, M3, and M4."""
        for algo in ["logistic_regression", "rbf_svm", "tabpfn_2_5"]:
            AlgorithmSelector.set_active_algorithm(algo)
            for m in ["M1", "M2", "M3", "M4"]:
                pipe = AlgorithmSelector.build_pipeline_for_model(m)
                self.assertIsNotNone(pipe)
                self.assertTrue(hasattr(pipe, "fit"))
                self.assertTrue(hasattr(pipe, "predict_proba"))

    def test_feature_counts_preserved(self):
        """Verify M1=18, M2=34, M3=42, M4=46 feature configurations are strictly preserved."""
        self.assertEqual(len(get_feature_names_for_model("M1")), 18)
        self.assertEqual(len(get_feature_names_for_model("M2")), 34)
        self.assertEqual(len(get_feature_names_for_model("M3")), 42)
        self.assertEqual(len(get_feature_names_for_model("M4")), 46)

    def test_fit_and_predict_probabilities(self):
        """Verify all three algorithms fit and produce bounded probabilities on sample data."""
        np.random.seed(42)
        X = np.random.randn(30, 18)
        y = np.random.choice([0, 1], size=30)

        for algo in ["logistic_regression", "rbf_svm", "tabpfn_2_5"]:
            pipe = AlgorithmSelector.build_pipeline_for_model("M1", algorithm=algo)
            pipe.fit(X, y)
            probs = pipe.predict_proba(X[:5])
            self.assertEqual(probs.shape, (5, 2))
            self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
