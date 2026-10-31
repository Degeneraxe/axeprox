"""Dataset handling and walk-forward chronological splitting with purge gaps.

Implements Amendments 5 & 6:
- Strict chronological event ordering (no random shuffle)
- Purge gaps between Train, Validation, and Test sets to eliminate autocorrelation leakage
- Feature standardization fit strictly on the training fold
- Produces tensors for PyTorch LSTM training (sequence, viability label, log_duration target, censored mask)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


class FeatureScaler:
    """Standardizes feature vectors using mean and standard deviation computed strictly on train set."""

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """X shape: (N, seq_len, num_features) or (N, num_features)."""
        flat = X.reshape(-1, X.shape[-1])
        self.means = np.mean(flat, axis=0)
        self.stds = np.std(flat, axis=0)
        # Prevent division by zero
        self.stds[self.stds < 1e-6] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            raise ValueError("Scaler must be fit before transform!")
        return (X - self.means) / self.stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


@dataclass
class DatasetSplits:
    """Holds chronological train, validation, and test arrays."""
    X_train: np.ndarray
    y_cls_train: np.ndarray
    y_reg_train: np.ndarray
    censored_train: np.ndarray

    X_val: np.ndarray
    y_cls_val: np.ndarray
    y_reg_val: np.ndarray
    censored_val: np.ndarray

    X_test: np.ndarray
    y_cls_test: np.ndarray
    y_reg_test: np.ndarray
    censored_test: np.ndarray

    scaler: FeatureScaler


def create_walk_forward_splits(
    samples_X: List[np.ndarray],
    labels_cls: List[int],
    targets_reg: List[float],
    censored_flags: List[bool],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap: int = 5,
) -> DatasetSplits:
    """Partitions chronological sequence of events with purge gaps (Amendment 6)."""
    n_total = len(samples_X)
    if n_total < 10:
        raise ValueError(f"Insufficient samples for walk-forward split (found {n_total})")

    # Strict chronological partition
    n_train = int(n_total * train_ratio)
    val_start = n_train + purge_gap
    n_val = int(n_total * val_ratio)
    val_end = min(n_total, val_start + n_val)
    test_start = min(n_total, val_end + purge_gap)

    # Indices
    idx_train = list(range(0, n_train))
    idx_val = list(range(val_start, val_end))
    idx_test = list(range(test_start, n_total))

    # Convert to numpy arrays
    X_all = np.array(samples_X, dtype=np.float32)
    y_cls_all = np.array(labels_cls, dtype=np.float32)
    y_reg_all = np.array(targets_reg, dtype=np.float32)
    cens_all = np.array(censored_flags, dtype=np.bool_)

    X_tr = X_all[idx_train]
    y_cls_tr = y_cls_all[idx_train]
    y_reg_tr = y_reg_all[idx_train]
    cens_tr = cens_all[idx_train]

    X_va = X_all[idx_val]
    y_cls_va = y_cls_all[idx_val]
    y_reg_va = y_reg_all[idx_val]
    cens_va = cens_all[idx_val]

    X_te = X_all[idx_test]
    y_cls_te = y_cls_all[idx_test]
    y_reg_te = y_reg_all[idx_test]
    cens_te = cens_all[idx_test]

    # Scaler fit STRICTLY on train fold (zero data leakage)
    scaler = FeatureScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    X_va_norm = scaler.transform(X_va)
    X_te_norm = scaler.transform(X_te)

    return DatasetSplits(
        X_train=X_tr_norm,
        y_cls_train=y_cls_tr,
        y_reg_train=y_reg_tr,
        censored_train=cens_tr,
        X_val=X_va_norm,
        y_cls_val=y_cls_va,
        y_reg_val=y_reg_va,
        censored_val=cens_va,
        X_test=X_te_norm,
        y_cls_test=y_cls_te,
        y_reg_test=y_reg_te,
        censored_test=cens_te,
        scaler=scaler,
    )
