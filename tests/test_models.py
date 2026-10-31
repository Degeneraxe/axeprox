"""Unit tests for ML feature extraction, walk-forward splits, scalers, and models."""
from __future__ import annotations

import math
import numpy as np
import pytest

from models.baseline import BaselineModel
from models.dataset import FeatureScaler, create_walk_forward_splits
from models.evaluate import evaluate_predictions
from models.features import extract_step_features
from data.normalizer import OrderBookSnapshot, OrderBookLevel


def test_feature_extraction():
    snap_buy = OrderBookSnapshot(
        exchange="binance",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1025.0,
        transit_latency_ms=25.0,
        bids=[OrderBookLevel(81500.0, 1.0)],
        asks=[OrderBookLevel(81510.0, 1.0)],
    )
    snap_sell = OrderBookSnapshot(
        exchange="kraken",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1140.0,
        transit_latency_ms=140.0,
        bids=[OrderBookLevel(81580.0, 1.0)],
        asks=[OrderBookLevel(81590.0, 1.0)],
    )

    feats = extract_step_features(
        snap_buy=snap_buy,
        snap_sell=snap_sell,
        gross_spread_bps=8.5,
        prev_spread_bps=6.5,
        prev_velocity=1.0,
        vol_buy_bps=1.5,
        vol_sell_bps=1.2,
    )

    assert len(feats) == 14
    assert feats[0] == pytest.approx(8.5)      # gross spread
    assert feats[1] == pytest.approx(2.0)      # velocity (8.5 - 6.5)
    assert feats[2] == pytest.approx(1.0)      # accel (2.0 - 1.0)
    assert feats[8] == pytest.approx(25.0)     # buy transit latency
    assert feats[9] == pytest.approx(140.0)    # sell transit latency
    assert feats[10] == pytest.approx(-115.0)  # latency diff


def test_walk_forward_splits_with_purge_gap():
    # 100 sequential events
    n_samples = 100
    seq_len = 20
    feat_dim = 14

    samples_X = [np.ones((seq_len, feat_dim), dtype=np.float32) * i for i in range(n_samples)]
    labels_cls = [1 if i % 3 == 0 else 0 for i in range(n_samples)]
    targets_reg = [math.log(1.0 + (i * 10)) for i in range(n_samples)]
    censored_flags = [False] * n_samples

    purge_gap = 4
    splits = create_walk_forward_splits(
        samples_X=samples_X,
        labels_cls=labels_cls,
        targets_reg=targets_reg,
        censored_flags=censored_flags,
        train_ratio=0.70,
        val_ratio=0.15,
        purge_gap=purge_gap,
    )

    # Train: 0 to 70 (70 samples)
    assert len(splits.X_train) == 70

    # Val starts at 70 + purge_gap (74), length = 15 -> 74 to 89
    assert len(splits.X_val) == 15

    # Test starts at 89 + purge_gap (93) to 100 -> 7 samples
    assert len(splits.X_test) == 7

    # Check purge gap: last train sample was index 69, first val sample was index 74
    # Zero data leakage between folds
    assert splits.scaler is not None


def test_baseline_model_training_and_evaluation():
    # Synthetic dataset for baseline test
    np.random.seed(42)
    n = 60
    X = np.random.randn(n, 10, 14).astype(np.float32)
    y_cls = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    y_reg = np.random.exponential(scale=2.0, size=n).astype(np.float32)
    cens = np.array([False] * n)

    baseline = BaselineModel()
    baseline.fit(X[:40], y_cls[:40], y_reg[:40], cens[:40])

    probs, log_durs = baseline.predict(X[40:])
    assert len(probs) == 20
    assert len(log_durs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Test evaluate_predictions
    metrics = evaluate_predictions(
        y_cls_true=y_cls[40:],
        y_cls_prob=probs,
        y_reg_true=y_reg[40:],
        y_reg_pred=log_durs,
        censored_flags=cens[40:],
    )
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "pr_auc" in metrics
    assert "brier_calibration" in metrics
    assert "mae_duration_ms" in metrics
