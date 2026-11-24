"""Unit and integration tests for ArbSense hardening and refactor items."""
import json
import sqlite3
import pytest
import numpy as np
from pathlib import Path

from config.settings import AppConfig, get_git_commit
from data.normalizer import OrderBookNormalizer, OrderBookSnapshot, OrderBookLevel
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel
from engine.simulator import ExecutionSimulator
from engine.tracker import SpreadTracker, SpreadCandidate
from models.features import extract_step_features, engineer_tabular_features
from models.walk_forward import create_non_overlapping_folds, compute_metrics, tune_mcc_threshold
from scripts.propose_latency_threshold import analyze_latencies


def test_ts_source_normalizer_and_batch_writer(tmp_path):
    """Item 3: Binance and Coinbase snapshot marked 'estimated'; Kraken marked 'exchange'."""
    norm = OrderBookNormalizer()
    
    # Binance snapshot -> estimated
    binance_snap = norm.update_book(
        exchange="binance",
        raw_symbol="btcusdt",
        bids=[(50000.0, 1.0)],
        asks=[(50010.0, 1.0)],
        exchange_ts_ms=1000.0,
        local_ts_ms=1050.0,
    )
    assert binance_snap.ts_source == "estimated"

    # Kraken snapshot -> exchange
    kraken_snap = norm.update_book(
        exchange="kraken",
        raw_symbol="XBT/USDT",
        bids=[(50000.0, 1.0)],
        asks=[(50010.0, 1.0)],
        exchange_ts_ms=1000.0,
        local_ts_ms=1050.0,
    )
    assert kraken_snap.ts_source == "exchange"


def test_ts_source_db_filtering_and_empirical_latency(tmp_path):
    """Item 3: Latency stats filter strictly for ts_source='exchange'."""
    db_file = tmp_path / "test_ts_src.db"
    db = DatabaseManager(sqlite_path=str(db_file))
    db.initialize()

    # Insert exchange and estimated ticks
    ticks = [
        {
            "time": "2026-11-24T12:00:00Z",
            "exchange": "kraken",
            "symbol": "BTC/USDT",
            "quote_currency": "USDT",
            "best_bid": 50000.0,
            "best_ask": 50010.0,
            "mid_price": 50005.0,
            "top_spread": 10.0,
            "imbalance": 0.0,
            "bid_depth": 1.0,
            "ask_depth": 1.0,
            "exchange_ts_ms": 1000.0,
            "local_ts_ms": 1080.0,
            "transit_latency_ms": 80.0,
            "is_stale": 0,
            "ts_source": "exchange",
        },
        {
            "time": "2026-11-24T12:00:01Z",
            "exchange": "binance",
            "symbol": "BTC/USDT",
            "quote_currency": "USDT",
            "best_bid": 50000.0,
            "best_ask": 50010.0,
            "mid_price": 50005.0,
            "top_spread": 10.0,
            "imbalance": 0.0,
            "bid_depth": 1.0,
            "ask_depth": 1.0,
            "exchange_ts_ms": 1000.0,
            "local_ts_ms": 1500.0,
            "transit_latency_ms": 500.0,  # Artificial estimated arrival latency
            "is_stale": 0,
            "ts_source": "estimated",
        },
    ]
    db.insert_ticks_batch(ticks)

    counts = db.get_tick_counts_by_source()
    assert counts.get("exchange", 0) == 1
    assert counts.get("estimated", 0) == 1

    stats = db.get_empirical_latency_stats()
    # Stats should only include the 1 exchange tick (80.0 ms), not 500.0 ms
    assert stats["count"] == 1
    assert stats["mean_ms"] == 80.0
    assert stats["min_ms"] == 80.0
    assert stats["max_ms"] == 80.0
    db.close()


def test_clock_offsets_table_and_run_metadata(tmp_path):
    """Item 4: Clock offsets stored in clock_offsets table and run_metadata."""
    db_file = tmp_path / "test_ntp.db"
    db = DatabaseManager(sqlite_path=str(db_file))
    db.initialize()

    db.record_clock_offset(offset_ms=12.45, time_str="2026-11-24T12:00:00Z")
    db.record_clock_offset(offset_ms=-3.10, time_str="2026-11-24T12:05:00Z")

    cur = db._sqlite_conn.cursor()
    cur.execute("SELECT * FROM clock_offsets ORDER BY id ASC")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0]["offset_ms"] == 12.45
    assert rows[1]["offset_ms"] == -3.10

    # Test run_metadata recording clock_offset_ms
    db.check_and_record_run_metadata(
        dataset_version="v2.1",
        config_hash="test_hash_123",
        git_commit="test_git_commit",
        latency_threshold_ms=85.0,
        clock_offset_ms=12.45,
    )
    cur.execute("SELECT * FROM run_metadata WHERE dataset_version = 'v2.1'")
    meta = cur.fetchone()
    assert meta["clock_offset_ms"] == 12.45
    db.close()


def test_simulator_single_frozen_latency_threshold(tmp_path):
    """Item 5: Simulator strictly uses viability_latency_threshold_ms."""
    cfg = AppConfig.load()
    sim = ExecutionSimulator(db_manager=None, config=cfg)
    expected_threshold = float(cfg.raw.get("dataset", {}).get("viability_latency_threshold_ms", 85.0))
    assert sim.viability_latency_threshold_ms == expected_threshold
    assert sim.latency_mean_ms == expected_threshold


def test_propose_latency_threshold_script(tmp_path):
    """Item 5: Propose latency threshold script runs and computes percentiles correctly."""
    db_file = tmp_path / "test_propose.db"
    db = DatabaseManager(sqlite_path=str(db_file))
    db.initialize()

    # Insert 20 exchange ticks with known latencies
    ticks = []
    for i in range(20):
        ticks.append({
            "time": f"2026-11-24T12:{i:02d}:00Z",
            "exchange": "kraken",
            "symbol": "BTC/USDT",
            "quote_currency": "USDT",
            "best_bid": 50000.0,
            "best_ask": 50010.0,
            "mid_price": 50005.0,
            "top_spread": 10.0,
            "imbalance": 0.0,
            "bid_depth": 1.0,
            "ask_depth": 1.0,
            "exchange_ts_ms": 1000.0,
            "local_ts_ms": 1000.0 + (i + 1) * 10,
            "transit_latency_ms": float((i + 1) * 10),
            "is_stale": 0,
            "ts_source": "exchange",
        })
    db.insert_ticks_batch(ticks)
    db.close()

    res = analyze_latencies(str(db_file))
    assert res["counts_by_source"]["exchange"] == 20
    assert res["overall"]["count"] == 20
    assert res["overall"]["min"] == 10.0
    assert res["overall"]["max"] == 200.0
    assert 90.0 <= res["overall"]["p50"] <= 110.0


def test_walk_forward_non_overlapping_folds_and_held_out():
    """Item 2: Folds are strictly non-overlapping (stride >= test_window) + held-out period."""
    # Synthetic sequence of 500 events ordered chronologically
    events = []
    for i in range(500):
        t_ms = 1_000_000.0 + i * 2000.0  # 2 seconds between events
        events.append({
            "event_id": f"evt_{i:04d}",
            "symbol": "BTC/USDT",
            "direction": "binance->kraken",
            "open_time": t_ms,
            "duration_ms": 1000.0,
            "log_duration": np.log1p(1000.0),
            "is_censored": False,
            "y_cls": i % 2,
            "seq": np.zeros((20, 14), dtype=np.float32),
            "tabular": np.zeros(27, dtype=np.float32),
        })

    fold_fractions = [
        {"train_ratio": 0.40, "val_ratio": 0.15, "test_ratio": 0.15},
        {"train_ratio": 0.55, "val_ratio": 0.15, "test_ratio": 0.15},
        {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15},
    ]

    folds, held_out = create_non_overlapping_folds(
        events=events,
        fold_fractions=fold_fractions,
        held_out_ratio=0.15,
        purge_gap_sec=60.0,
    )

    # Held out is exactly 15% (75 events out of 500)
    assert len(held_out) == 75
    assert held_out[0]["open_time"] > folds[-1]["test"][-1]["open_time"]

    # Verify each fold has non-overlapping test window with subsequent folds
    for i in range(len(folds) - 1):
        f_curr = folds[i]
        f_next = folds[i + 1]
        # Current test end time must be <= next test start time
        curr_test_end = f_curr["test"][-1]["open_time"]
        next_test_start = f_next["test"][0]["open_time"]
        assert curr_test_end <= next_test_start, f"Overlap detected between Fold {i+1} and Fold {i+2}"

    # Verify purge gap > 60s
    for f in folds:
        assert f["purge1_gap_s"] >= 60.0 or len(f["val"]) == 0
        assert f["purge2_gap_s"] >= 60.0 or len(f["test"]) == 0


def test_dashboard_read_only_mode(tmp_path):
    """Item 7: Dashboard DB manager opens in mode=ro and raises on write."""
    db_file = tmp_path / "test_dashboard_ro.db"
    
    # Initialize schema first with standard manager
    setup_db = DatabaseManager(sqlite_path=str(db_file))
    setup_db.initialize()
    setup_db.insert_ticks_batch([{
        "time": "2026-11-24T12:00:00Z",
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "quote_currency": "USDT",
        "best_bid": 50000.0,
        "best_ask": 50010.0,
        "mid_price": 50005.0,
        "top_spread": 10.0,
        "imbalance": 0.0,
        "bid_depth": 1.0,
        "ask_depth": 1.0,
        "exchange_ts_ms": 1000.0,
        "local_ts_ms": 1050.0,
        "transit_latency_ms": 50.0,
        "is_stale": 0,
        "ts_source": "estimated",
    }])
    setup_db.close()

    # Open read-only manager
    ro_db = DatabaseManager(sqlite_path=str(db_file), read_only=True)
    ro_db.initialize()

    # Read operations must succeed
    ticks = ro_db.get_recent_ticks("binance", "BTC/USDT")
    assert len(ticks) == 1
    assert ticks[0]["best_bid"] == 50000.0

    # Write operations must fail with RuntimeError
    with pytest.raises(RuntimeError, match="read-only"):
        ro_db.insert_ticks_batch([{"dummy": 1}])

    with pytest.raises(RuntimeError, match="read-only"):
        ro_db.insert_spread_event({"dummy": 1})

    with pytest.raises(RuntimeError, match="read-only"):
        ro_db.update_spread_event("dummy", {"dummy": 1})

    with pytest.raises(RuntimeError, match="read-only"):
        ro_db.prune_old_ticks()

    ro_db.close()


def test_config_parameters_and_version():
    """Item 6 & 10: Config contains v2.1 and non-hardcoded values."""
    cfg = AppConfig.load()
    assert cfg.raw["dataset"]["version"] == "v2.1"
    assert "execution_latency_assumption_ms" not in cfg.raw.get("labeling", {})
    assert cfg.raw["dataset"]["viability_latency_threshold_ms"] == 85.0
    assert cfg.raw["storage"]["tick_prune_keep_count"] == 1500
    assert cfg.raw["tracking"]["max_staleness_ms"] == 1500.0
    assert cfg.raw["training"]["held_out_ratio"] == 0.15
    assert len(cfg.raw["training"]["fold_fractions"]) >= 3
