"""
Unit tests for models and evaluation metrics module.
"""
import unittest
import numpy as np
import pandas as pd
from src.models import (
    HeuristicCivilDefenseModel,
    create_logistic_baseline,
    create_lightgbm_model,
)
from src.evaluate import compute_metrics


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        self.X = pd.DataFrame({
            "inner_ring_active_count": np.random.randint(0, 5, size=n),
            "adjacent_active_count": np.random.randint(0, 4, size=n),
            "national_wave_speed_30m": np.random.randint(-2, 5, size=n),
            "hour_sin": np.random.uniform(-1, 1, size=n),
            "hour_cos": np.random.uniform(-1, 1, size=n),
        })
        self.y = pd.Series(np.random.randint(0, 2, size=n))

    def test_heuristic_model_predict_proba(self):
        """Verify heuristic model outputs probabilities in [0, 1]."""
        model = HeuristicCivilDefenseModel()
        model.fit(self.X, self.y)
        probs = model.predict_proba(self.X)
        self.assertEqual(probs.shape, (len(self.X), 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_logistic_baseline_pipeline(self):
        """Verify logistic regression fits and predicts probabilities."""
        pipe = create_logistic_baseline()
        pipe.fit(self.X, self.y)
        probs = pipe.predict_proba(self.X)
        self.assertEqual(probs.shape, (len(self.X), 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_lightgbm_model(self):
        """Verify LightGBM model fits and produces probabilities."""
        lgb_model = create_lightgbm_model(n_estimators=10, min_child_samples=5)
        lgb_model.fit(self.X, self.y)
        probs = lgb_model.predict_proba(self.X)
        self.assertEqual(probs.shape, (len(self.X), 2))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())

    def test_compute_metrics_keys(self):
        """Verify compute_metrics returns all required diagnostic metrics."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.4])
        metrics = compute_metrics(y_true, y_prob)
        expected_keys = {"roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1"}
        self.assertEqual(set(metrics.keys()), expected_keys)
        self.assertGreaterEqual(metrics["roc_auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
