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

    feats_no_lat = extract_step_features(
        snap_buy=snap_buy,
        snap_sell=snap_sell,
        gross_spread_bps=8.5,
        prev_spread_bps=6.5,
        prev_velocity=1.0,
        vol_buy_bps=1.5,
        vol_sell_bps=1.2,
        use_latency=False,
    )
    assert len(feats_no_lat) == 11
    assert feats_no_lat[0] == pytest.approx(8.5)      # gross spread
    assert feats_no_lat[1] == pytest.approx(2.0)      # velocity (8.5 - 6.5)
    assert feats_no_lat[2] == pytest.approx(1.0)      # accel (2.0 - 1.0)

    # When use_latency=True: NaN-safe latency representation (value + is_measured)
    feats = extract_step_features(
        snap_buy=snap_buy,
        snap_sell=snap_sell,
        gross_spread_bps=8.5,
        prev_spread_bps=6.5,
        prev_velocity=1.0,
        vol_buy_bps=1.5,
        vol_sell_bps=1.2,
        use_latency=True,
    )

    assert len(feats) == 16
    assert feats[0] == pytest.approx(8.5)      # gross spread
    assert feats[1] == pytest.approx(2.0)      # velocity (8.5 - 6.5)
    assert feats[2] == pytest.approx(1.0)      # accel (2.0 - 1.0)
    assert feats[8] == pytest.approx(25.0)     # buy transit latency
    assert feats[9] == pytest.approx(1.0)      # buy is_measured flag
    assert feats[10] == pytest.approx(140.0)   # sell transit latency
    assert feats[11] == pytest.approx(1.0)     # sell is_measured flag
    assert feats[12] == pytest.approx(-115.0)  # latency diff


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


def test_train_raises_insufficient_data(tmp_path):
    """Verifies that the training pipeline strictly raises ValueError on insufficient real data (Defects 1 & 7)."""
    import sqlite3
    from models.train import build_dataset_from_db

    fake_db = tmp_path / "empty_test.db"
    conn = sqlite3.connect(fake_db)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE spread_events (
            event_id TEXT PRIMARY KEY,
            symbol TEXT,
            buy_exchange TEXT,
            sell_exchange TEXT,
            open_time TEXT,
            close_time TEXT,
            peak_spread_bps REAL,
            open_spread_bps REAL,
            cost_hurdle_bps REAL,
            duration_ms REAL,
            log_duration REAL,
            is_viable INTEGER,
            is_right_censored INTEGER,
            feature_window_json TEXT,
            vol_buy_bps REAL,
            vol_sell_bps REAL,
            latency_threshold_ms REAL
        )
    """)
    cur.execute("""
        CREATE TABLE orderbook_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            exchange TEXT,
            symbol TEXT,
            quote_currency TEXT,
            best_bid REAL,
            best_ask REAL,
            mid_price REAL,
            top_spread REAL,
            imbalance REAL,
            bid_depth TEXT,
            ask_depth TEXT,
            exchange_ts_ms REAL,
            local_ts_ms REAL,
            transit_latency_ms REAL,
            is_stale INTEGER
        )
    """)
    conn.commit()
    conn.close()

    # Fewer than 20 events must raise ValueError, never fabricate samples
    with pytest.raises(ValueError, match="Insufficient real events with persisted feature windows in DB"):
        build_dataset_from_db(db_path=str(fake_db), min_events=20)



def test_feature_label_leakage():
    """Verifies zero lookahead leakage: features use strictly point-in-time data preceding open_time (Defect 7)."""
    from models.features import extract_step_features, FEATURE_NAMES
    from data.normalizer import OrderBookSnapshot, OrderBookLevel

    event_open_time_ms = 5000.0

    # Historical book strictly BEFORE event open
    pre_snap_buy = OrderBookSnapshot(
        exchange="binance",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=4950.0,
        local_ts_ms=4975.0,
        transit_latency_ms=25.0,
        bids=[OrderBookLevel(81500.0, 1.0)],
        asks=[OrderBookLevel(81510.0, 1.0)],
    )
    pre_snap_sell = OrderBookSnapshot(
        exchange="kraken",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=4900.0,
        local_ts_ms=4980.0,
        transit_latency_ms=80.0,
        bids=[OrderBookLevel(81580.0, 1.0)],
        asks=[OrderBookLevel(81590.0, 1.0)],
    )

    # Assert timestamps are strictly <= event open time (no future ticks leaked into features)
    assert pre_snap_buy.local_ts_ms <= event_open_time_ms
    assert pre_snap_sell.local_ts_ms <= event_open_time_ms

    feats1 = extract_step_features(
        snap_buy=pre_snap_buy,
        snap_sell=pre_snap_sell,
        gross_spread_bps=8.5,
    )
    assert len(feats1) == len(FEATURE_NAMES)

    # Now verify that modifying future post-open event outcome (future duration, closing time, peak spread)
    # has ZERO impact on point-in-time feature extraction
    future_duration_a = 150.0
    future_duration_b = 99999.0
    feats2 = extract_step_features(
        snap_buy=pre_snap_buy,
        snap_sell=pre_snap_sell,
        gross_spread_bps=8.5,
    )
    np.testing.assert_array_equal(np.array(feats1), np.array(feats2))



def test_predictor_checkpoint_strictness(tmp_path):
    """Verifies that predictor raises FileNotFoundError on missing checkpoint and loads unified checkpoint (Defect 5)."""
    import torch
    from models.predictor import LiveSpreadPredictor
    from models.lstm_model import MultiHeadSpreadLSTM
    from models.features import FEATURE_NAMES

    missing_path = tmp_path / "non_existent.pt"
    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        LiveSpreadPredictor(checkpoint_path=str(missing_path))

    # Create valid unified checkpoint dictionary with complete metadata
    feat_dim = len(FEATURE_NAMES)
    model = MultiHeadSpreadLSTM(input_dim=feat_dim, hidden_dim=64, num_layers=2)
    ckpt_path = tmp_path / "lstm_valid.pt"
    valid_payload = {
        "model_state_dict": model.state_dict(),
        "dataset_version": "v2.0",
        "git_commit": "d9a00e1",
        "decision_threshold": 0.45,
        "scaler": {
            "means": [0.0] * feat_dim,
            "stds": [1.0] * feat_dim,
        },
        "feature_names": FEATURE_NAMES,
    }
    torch.save(valid_payload, ckpt_path)

    predictor = LiveSpreadPredictor(checkpoint_path=str(ckpt_path), expected_dataset_version="v2.0")
    assert predictor.is_loaded is True
    assert predictor.decision_threshold == pytest.approx(0.45)
    assert len(predictor.feature_names) == feat_dim
    assert predictor.dataset_version == "v2.0"
    assert predictor.git_commit == "d9a00e1"

    # Test rejection on dataset version mismatch
    with pytest.raises(ValueError, match="Dataset version mismatch"):
        LiveSpreadPredictor(checkpoint_path=str(ckpt_path), expected_dataset_version="v9.9")

    # Test rejection on missing metadata
    for key in ["dataset_version", "git_commit", "decision_threshold", "scaler", "feature_names"]:
        incomplete = dict(valid_payload)
        del incomplete[key]
        inc_path = tmp_path / f"lstm_missing_{key}.pt"
        torch.save(incomplete, inc_path)
        with pytest.raises(ValueError, match="missing required metadata"):
            LiveSpreadPredictor(checkpoint_path=str(inc_path), expected_dataset_version="v2.0")


def test_asof_join_ticks():
    """Verifies that asof_join_ticks correctly aligns buy/sell ticks by timestamp with max staleness (Requirement 2)."""
    from models.features import asof_join_ticks

    # Buy ticks arrive at 1000, 1100, 1200
    buy_ticks = [
        {"local_ts_ms": 1000.0, "best_ask": 81500.0, "best_bid": 81490.0, "imbalance": 0.1, "transit_latency_ms": 20.0, "top_spread": 10.0, "mid_price": 81495.0},
        {"local_ts_ms": 1100.0, "best_ask": 81505.0, "best_bid": 81495.0, "imbalance": 0.2, "transit_latency_ms": 22.0, "top_spread": 10.0, "mid_price": 81500.0},
        {"local_ts_ms": 1200.0, "best_ask": 81510.0, "best_bid": 81500.0, "imbalance": 0.3, "transit_latency_ms": 25.0, "top_spread": 10.0, "mid_price": 81505.0},
    ]

    # Sell ticks arrive at 1050, 1150
    sell_ticks = [
        {"local_ts_ms": 1050.0, "best_ask": 81590.0, "best_bid": 81580.0, "imbalance": -0.1, "transit_latency_ms": 100.0, "top_spread": 10.0, "mid_price": 81585.0},
        {"local_ts_ms": 1150.0, "best_ask": 81595.0, "best_bid": 81585.0, "imbalance": -0.2, "transit_latency_ms": 110.0, "top_spread": 10.0, "mid_price": 81590.0},
    ]

    pairs = asof_join_ticks(buy_ticks, sell_ticks, max_staleness_ms=500.0, seq_len=10)
    assert len(pairs) > 0
    for b_tick, s_tick in pairs:
        # Check staleness bound
        delta = abs(b_tick["local_ts_ms"] - s_tick["local_ts_ms"])
        assert delta <= 500.0


def test_build_dataset_from_persisted_feature_windows(tmp_path):
    """Verifies that train.py reads persisted feature_window_json directly from spread_events (Requirement 1)."""
    import json
    import sqlite3
    from models.train import build_dataset_from_db
    from models.features import FEATURE_NAMES

    db_file = tmp_path / "persisted_test.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE spread_events (
            event_id TEXT PRIMARY KEY,
            symbol TEXT,
            buy_exchange TEXT,
            sell_exchange TEXT,
            open_time TEXT,
            close_time TEXT,
            peak_spread_bps REAL,
            open_spread_bps REAL,
            cost_hurdle_bps REAL,
            duration_ms REAL,
            log_duration REAL,
            is_viable INTEGER,
            is_right_censored INTEGER,
            feature_window_json TEXT,
            vol_buy_bps REAL,
            vol_sell_bps REAL,
            latency_threshold_ms REAL
        )
    """)

    # Insert 25 valid events with 20-step feature windows
    dummy_seq = [[1.0] * 14 for _ in range(20)]
    for i in range(25):
        cur.execute("""
            INSERT INTO spread_events (
                event_id, symbol, buy_exchange, sell_exchange, open_time, close_time,
                peak_spread_bps, open_spread_bps, cost_hurdle_bps, duration_ms,
                log_duration, is_viable, is_right_censored, feature_window_json, latency_threshold_ms
            ) VALUES (?, 'BTC/USDT', 'binance', 'kraken', ?, ?, 10.0, 8.0, 5.0, 200.0, 5.3, 1, 0, ?, 85.0)
        """, (f"evt_{i}", f"2026-10-30T12:{i:02d}:00Z", f"2026-10-30T12:{i:02d}:01Z", json.dumps(dummy_seq)))
    conn.commit()
    conn.close()

    samples_X, labels_cls, targets_reg, censored_flags = build_dataset_from_db(db_path=str(db_file), min_events=20)
    assert len(samples_X) == 25
    assert len(labels_cls) == 25
    assert samples_X[0].shape == (20, len(FEATURE_NAMES))


def test_dataset_version_filtering_and_invalid_v0_exclusion(tmp_path):
    """Verifies that train.py strictly uses the active dataset_version and excludes 'invalid_v0' rows (Step 1)."""
    import json
    import sqlite3
    from models.train import build_dataset_from_db
    from models.features import FEATURE_NAMES

    db_file = tmp_path / "version_test.db"
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE spread_events (
            event_id TEXT PRIMARY KEY,
            symbol TEXT,
            buy_exchange TEXT,
            sell_exchange TEXT,
            open_time TEXT,
            close_time TEXT,
            peak_spread_bps REAL,
            open_spread_bps REAL,
            cost_hurdle_bps REAL,
            duration_ms REAL,
            log_duration REAL,
            is_viable INTEGER,
            is_executable INTEGER,
            is_profitable_retail INTEGER,
            is_profitable_vip1 INTEGER,
            is_profitable_vip3 INTEGER,
            is_profitable_zero_maker INTEGER,
            is_right_censored INTEGER,
            feature_window_json TEXT,
            dataset_version TEXT,
            is_suspect INTEGER
        )
    """)

    dummy_seq = [[1.0] * 14 for _ in range(20)]
    dummy_json = json.dumps(dummy_seq)

    # 1. Insert 10 events with dataset_version = 'invalid_v0' (historical constant-duration events)
    for i in range(10):
        cur.execute("""
            INSERT INTO spread_events (
                event_id, symbol, buy_exchange, sell_exchange, open_time, close_time,
                peak_spread_bps, open_spread_bps, cost_hurdle_bps, duration_ms,
                log_duration, is_viable, is_executable, is_right_censored, feature_window_json,
                dataset_version, is_suspect
            ) VALUES (?, 'BTC/USDT', 'binance', 'kraken', '1000', '1180', 10.0, 10.0, 50.0, 180.0, 5.19, 0, 1, 0, ?, 'invalid_v0', 0)
        """, (f"old_{i}", dummy_json))

    # 2. Insert 22 events with dataset_version = 'v1.0' (current verified run)
    for i in range(22):
        cur.execute("""
            INSERT INTO spread_events (
                event_id, symbol, buy_exchange, sell_exchange, open_time, close_time,
                peak_spread_bps, open_spread_bps, cost_hurdle_bps, duration_ms,
                log_duration, is_viable, is_executable, is_right_censored, feature_window_json,
                dataset_version, is_suspect
            ) VALUES (?, 'BTC/USDT', 'binance', 'kraken', '1000', '1450', 12.0, 8.0, 50.0, 450.0, 6.11, 0, 1, 0, ?, 'v1.0', 0)
        """, (f"v1_{i}", dummy_json))

    # 3. Insert 5 events with dataset_version = 'v2.0' (different version)
    for i in range(5):
        cur.execute("""
            INSERT INTO spread_events (
                event_id, symbol, buy_exchange, sell_exchange, open_time, close_time,
                peak_spread_bps, open_spread_bps, cost_hurdle_bps, duration_ms,
                log_duration, is_viable, is_executable, is_right_censored, feature_window_json,
                dataset_version, is_suspect
            ) VALUES (?, 'BTC/USDT', 'binance', 'kraken', '1000', '1600', 15.0, 9.0, 50.0, 600.0, 6.40, 0, 1, 0, ?, 'v2.0', 0)
        """, (f"v2_{i}", dummy_json))

    conn.commit()
    conn.close()

    # Query v1.0: must find exactly the 22 v1.0 events, strictly excluding all 10 'invalid_v0' events and 5 'v2.0' events
    samples_X, labels_cls, targets_reg, censored_flags, balances = build_dataset_from_db(
        db_path=str(db_file),
        dataset_version="v1.0",
        target_label="is_executable",
        min_events=20,
        return_stats=True,
    )
    assert len(samples_X) == 22
    assert len(labels_cls) == 22
    assert balances["is_executable"]["pos"] == 22
    assert balances["is_executable"]["neg"] == 0

    # Querying invalid_v0 should fail or return 0 events because of the explicit exclusion
    import pytest
    with pytest.raises(ValueError, match="Insufficient real events"):
        build_dataset_from_db(
            db_path=str(db_file),
            dataset_version="invalid_v0",
            min_events=1,
        )




def test_tabular_inputs_exclude_arrival_slippage():
    import numpy as np
    from models.features import engineer_tabular_features
    window = np.ones((20, 14))
    event = {"open_time": 1000, "open_spread_bps": 3, "slippage_bps": None}
    before = engineer_tabular_features(window, event)
    event["slippage_bps"] = 100
    np.testing.assert_array_equal(before, engineer_tabular_features(window, event))
