"""
Machine learning models module for SirenCast UA.
Implements:
1. Heuristic Civil Defense Rule-based Baseline
2. L2-Regularized Logistic Regression Baseline
3. Tuned Regularized LightGBM Classifier with probability calibration
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HeuristicCivilDefenseModel(BaseEstimator, ClassifierMixin):
    """
    Expert Rule-Based Baseline mimicking operational civil defense intuition:
    - High probability (0.75) if >= 2 raions in Kyivska oblast are active,
      OR if >= 2 adjacent oblasts are active and nationwide wave is expanding.
    - Low probability (0.03) during quiet times.
    """
    def __init__(self, high_risk_prob: float = 0.75, base_prob: float = 0.03):
        self.high_risk_prob = high_risk_prob
        self.base_prob = base_prob
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        probs = np.full(n, self.base_prob, dtype=np.float32)

        inner_cond = (X.get("inner_ring_active_count", pd.Series(0, index=X.index)) >= 2)
        adj_cond = (
            (X.get("adjacent_active_count", pd.Series(0, index=X.index)) >= 2)
            & (X.get("national_wave_speed_30m", pd.Series(0, index=X.index)) > 0)
        )
        high_risk_mask = inner_cond | adj_cond
        probs[high_risk_mask] = self.high_risk_prob

        return np.column_stack([1.0 - probs, probs])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def create_logistic_baseline(C: float = 0.1) -> Pipeline:
    """
    Builds standard scaled L2 logistic regression pipeline with class weighting.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def create_lightgbm_model(
    n_estimators: int = 250,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    max_depth: int = 6,
    min_child_samples: int = 50,
    reg_alpha: float = 0.5,
    reg_lambda: float = 1.0,
    scale_pos_weight: Optional[float] = None,
) -> lgb.LGBMClassifier:
    """
    Builds regularized LightGBM classifier tuned to prevent overfitting on sparse lags.
    """
    kwargs = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": 42,
        "verbose": -1,
    }
    if scale_pos_weight is not None:
        kwargs["scale_pos_weight"] = scale_pos_weight

    return lgb.LGBMClassifier(**kwargs)
