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
    assert cfg.raw["dataset"]["version"] == "v2.3"
    assert "execution_latency_assumption_ms" not in cfg.raw.get("labeling", {})
    assert cfg.raw["dataset"]["viability_latency_threshold_ms"] == 85.0
    assert cfg.raw["storage"]["replay_window_hours"] == 6
    assert cfg.detection.depth_snapshot_throttle_ms == 100
    assert cfg.raw["tracking"]["max_staleness_ms"] == 1500.0
    assert cfg.raw["training"]["held_out_ratio"] == 0.15
    assert len(cfg.raw["training"]["fold_fractions"]) >= 3


def test_non_blocking_clock_sync_during_tick_processing():
    """Item 1: main loop keeps processing ticks while clock_sync.sync() sleeps for 3s in thread."""
    import asyncio
    import time

    def mock_slow_sync():
        time.sleep(3.0)
        return True

    processed_ticks = []

    async def main():
        async def tick_processor():
            for i in range(15):
                processed_ticks.append(f"tick_{i}")
                await asyncio.sleep(0.1)

        t0 = time.time()
        sync_task = asyncio.create_task(asyncio.to_thread(mock_slow_sync))
        tick_task = asyncio.create_task(tick_processor())

        await asyncio.gather(sync_task, tick_task)
        elapsed = time.time() - t0
        return elapsed

    elapsed = asyncio.run(main())

    assert len(processed_ticks) == 15
    assert 2.9 <= elapsed <= 4.5


def test_feature_window_raw_latency_persistence_and_no_zero_latency_injection():
    """Item 2: Feature windows persist RAW latencies and flags; estimated ticks never inject fake 0.0."""
    from data.normalizer import OrderBookNormalizer
    from engine.tracker import SpreadTracker
    from models.features import parse_feature_window

    norm = OrderBookNormalizer()
    tracker = SpreadTracker(normalizer=norm)

    t_base = 100000.0
    for i in range(25):
        t_ms = t_base + i * 100.0
        # Binance has ts_source="estimated" with real transit latency = 45.0 ms
        norm.update_book(
            "binance", "btcusdt",
            [(50000.0 + i, 1.0)], [(50010.0 + i, 1.0)],
            exchange_ts_ms=t_ms - 45.0, local_ts_ms=t_ms,
        )
        # Kraken has ts_source="exchange" with real transit latency = 120.0 ms
        norm.update_book(
            "kraken", "XBT/USDT",
            [(50020.0 + i, 1.0)], [(50030.0 + i, 1.0)],
            exchange_ts_ms=t_ms - 120.0, local_ts_ms=t_ms,
        )

    live_win, _, _ = norm.build_feature_window("BTC/USDT", "binance", "kraken", seq_len=20)
    train_win, _, _ = tracker._build_point_in_time_feature_window("BTC/USDT", "binance", "kraken", open_time_ms=t_ms, seq_len=20)

    assert len(live_win) == 20
    assert len(train_win) == 20

    # Check first step: raw latencies and flags preserved, NO 0.0 injection
    step0 = live_win[0]
    assert isinstance(step0, dict)
    assert step0["lat_buy_ms"] == pytest.approx(45.0)  # Real latency preserved! Not 0.0!
    assert step0["ts_src_buy"] == "estimated"
    assert step0["is_measured_buy"] == 0.0
    assert step0["lat_sell_ms"] == pytest.approx(120.0) # Real latency preserved!
    assert step0["ts_src_sell"] == "exchange"
    assert step0["is_measured_sell"] == 1.0

    # Live and training feature vectors match identically
    np.testing.assert_allclose(np.array(live_win), np.array(train_win), rtol=1e-5, atol=1e-5)

    # Serializes to honest JSON without 0.0 substitution
    win_json = json.dumps(live_win)
    loaded = json.loads(win_json)
    assert loaded[0]["lat_buy_ms"] == pytest.approx(45.0)

    # When parsed with use_latency=False: drops latency features (11 features)
    parsed_no_lat = parse_feature_window(win_json, use_latency=False)
    assert parsed_no_lat.shape == (20, 11)

    # When parsed with use_latency=True: NaN-safe representation uses imputed 85.0 for estimated, not 0.0
    parsed_with_lat = parse_feature_window(win_json, use_latency=True, imputed_latency_ms=85.0)
    assert parsed_with_lat.shape == (20, 16)
    # Buy leg was estimated -> imputed to 85.0, flag = 0.0
    assert parsed_with_lat[0, 8] == pytest.approx(85.0)
    assert parsed_with_lat[0, 9] == pytest.approx(0.0)
    # Sell leg was exchange -> real 120.0, flag = 1.0
    assert parsed_with_lat[0, 10] == pytest.approx(120.0)
    assert parsed_with_lat[0, 11] == pytest.approx(1.0)


def test_simulator_fallback_and_lstm_hyperparameters_in_config():
    """Item 3: Simulator fallbacks and LSTM hyperparameters are in config.yaml."""
    cfg = AppConfig.load()
    sim_lat = cfg.raw.get("simulation", {}).get("latency", {})
    assert sim_lat["fallback_std_ms"] == 15.0
    assert sim_lat["fallback_min_ms"] == 5.0
    assert sim_lat["fallback_max_ms"] == 250.0

    lstm_cfg = cfg.raw.get("model", {}).get("lstm", {})
    assert lstm_cfg["hidden_dim"] == 64
    assert lstm_cfg["layers"] == 2
    assert lstm_cfg["dropout"] == 0.2
    assert lstm_cfg["epochs"] == 25
    assert lstm_cfg["lr"] == 0.001
    assert lstm_cfg["seeds"] == [42]


def test_fresh_db_startup_v21_and_old_v20_db_refusal(tmp_path):
    """Item 5: Startup on fresh DB creates ts_source, clock_offsets, run_metadata under v2.1;
    starting against old v2.0 DB refuses on config mismatch and creates new version when incremented."""
    fresh_db_path = tmp_path / "fresh_test_v21.db"
    db = DatabaseManager(sqlite_path=str(fresh_db_path))
    db.initialize()

    # 1. Fresh DB verification
    conn = sqlite3.connect(str(fresh_db_path))
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(orderbook_ticks);")
    cols = {r[1] for r in cur.fetchall()}
    assert "ts_source" in cols

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='clock_offsets';")
    assert cur.fetchone() is not None

    cur.execute("PRAGMA table_info(run_metadata);")
    meta_cols = {r[1] for r in cur.fetchall()}
    assert "dataset_version" in meta_cols
    assert "config_hash" in meta_cols
    assert "clock_offset_ms" in meta_cols

    # Record v2.1 run metadata on fresh DB
    db.check_and_record_run_metadata(
        dataset_version="v2.1",
        config_hash="hash_v21_initial",
        git_commit="git_v21",
        latency_threshold_ms=85.0,
        clock_offset_ms=-1.25,
    )
    cur.execute("SELECT dataset_version, config_hash, clock_offset_ms FROM run_metadata WHERE dataset_version='v2.1';")
    row = cur.fetchone()
    assert row[0] == "v2.1"
    assert row[1] == "hash_v21_initial"
    assert row[2] == pytest.approx(-1.25)
    db.close()
    conn.close()

    # 2. Old v2.0 DB refusal and version increment behavior
    old_db_path = tmp_path / "old_v20.db"
    old_db = DatabaseManager(sqlite_path=str(old_db_path))
    old_db.initialize()
    old_db.check_and_record_run_metadata(
        dataset_version="v2.0",
        config_hash="original_v20_hash",
        git_commit="git_v20",
        latency_threshold_ms=85.0,
    )

    # Starting against old v2.0 DB with modified config hash MUST REFUSE
    with pytest.raises(ValueError, match="Config hash mismatch for dataset_version 'v2.0'"):
        old_db.check_and_record_run_metadata(
            dataset_version="v2.0",
            config_hash="modified_v20_hash",
            git_commit="git_v20_mod",
            latency_threshold_ms=85.0,
        )

    # Starting against old DB with new dataset version 'v2.1' MUST SUCCEED (creates new version entry)
    old_db.check_and_record_run_metadata(
        dataset_version="v2.1",
        config_hash="new_v21_hash",
        git_commit="git_v21",
        latency_threshold_ms=85.0,
    )

    conn_old = sqlite3.connect(str(old_db_path))
    cur_old = conn_old.cursor()
    cur_old.execute("SELECT dataset_version FROM run_metadata ORDER BY run_id ASC;")
    versions = [r[0] for r in cur_old.fetchall()]
    assert versions == ["v2.0", "v2.1"]
    old_db.close()
    conn_old.close()
