"""Baseline machine learning models for comparison against PyTorch LSTM.

Uses Logistic Regression (with balanced class weights) for viability classification
and Ridge / Gradient Boosting for log-duration regression.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


class BaselineModel:
    """Benchmark baseline models operating on temporal mean-pooled features."""

    def __init__(self):
        self.clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        self.reg = Ridge(alpha=1.0)

    def _pool_features(self, X: np.ndarray) -> np.ndarray:
        """Flattens or pools sequence: uses final step + sequence mean.
        
        X shape: (N, seq_len, num_features) -> (N, num_features * 2)
        """
        if X.ndim == 3:
            last_step = X[:, -1, :]
            mean_step = np.mean(X, axis=1)
            return np.hstack([last_step, mean_step])
        return X

    def fit(self, X_train: np.ndarray, y_cls_train: np.ndarray, y_reg_train: np.ndarray, censored_train: np.ndarray) -> None:
        X_pooled = self._pool_features(X_train)
        # Train classifier
        self.clf.fit(X_pooled, y_cls_train)

        # Train regressor on uncensored samples
        uncensored_mask = ~censored_train
        if np.sum(uncensored_mask) > 5:
            self.reg.fit(X_pooled[uncensored_mask], y_reg_train[uncensored_mask])
        else:
            self.reg.fit(X_pooled, y_reg_train)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (pred_viability_probs, pred_log_durations)."""
        X_pooled = self._pool_features(X_test)
        # Viability probability for class 1
        probs = self.clf.predict_proba(X_pooled)[:, 1]
        log_durs = self.reg.predict(X_pooled)
        return probs, log_durs
