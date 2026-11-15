"""Tests for long-run freeze, basis staleness guard, and deployment utilities."""
import os
import sqlite3
import time
import pytest
from pathlib import Path

from config.settings import AppConfig, get_git_commit
from data.normalizer import OrderBookNormalizer, OrderBookSnapshot
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel
from engine.detector import CrossExchangeDetector
from scripts.break_even_analysis import run_break_even_analysis
from scripts.backup_db import backup_sqlite, prune_old_backups


def test_basis_staleness_guard():
    config = AppConfig.load()
    normalizer = OrderBookNormalizer(max_quote_age_ms=10000.0)
    cost_model = CostModel(config)
    detector = CrossExchangeDetector(normalizer=normalizer, cost_model=cost_model, config=config)

    # Populate Binance and Coinbase books
    t = time.time()
    normalizer.update_book(
        "binance", "BTC/USDT",
        bids=[[100000.0, 1.0]], asks=[[100001.0, 1.0]],
        exchange_ts_ms=t * 1000, local_ts_ms=t * 1000
    )
    normalizer.update_book(
        "coinbase", "BTC-USD",
        bids=[[100100.0, 1.0]], asks=[[100101.0, 1.0]],
        exchange_ts_ms=t * 1000, local_ts_ms=t * 1000
    )

    # 1. When basis was NEVER updated or is > 300s old:
    assert normalizer.is_basis_stale(300.0) is True
    candidates_stale = detector.scan_symbol("BTC/USDT", current_time_ms=t * 1000)
    # Any candidate involving Coinbase should be skipped
    assert not any("coinbase" in c.buy_exchange.lower() or "coinbase" in c.sell_exchange.lower() for c in candidates_stale)

    # 2. Update basis with fresh rate
    normalizer.set_usdt_usd_basis(1.0001)
    assert normalizer.is_basis_stale(300.0) is False
    assert normalizer.get_basis_age_sec() < 2.0

    candidates_fresh = detector.scan_symbol("BTC/USDT", current_time_ms=t * 1000)
    # Coinbase candidate should now be scanned
    cb_found = any("coinbase" in c.buy_exchange.lower() or "coinbase" in c.sell_exchange.lower() for c in candidates_fresh)
    assert cb_found is True


def test_run_metadata_freeze(tmp_path):
    db_file = tmp_path / "test_freeze.db"
    db_mgr = DatabaseManager(sqlite_path=str(db_file))
    db_mgr.initialize()

    # Initial run record
    db_mgr.check_and_record_run_metadata(
        dataset_version="v2.0",
        config_hash="hash_alpha",
        git_commit="commit_1",
        latency_threshold_ms=85.0,
        config_json='{"version": "v2.0"}',
    )

    # Same config_hash allows repeated runs under the frozen version
    db_mgr.check_and_record_run_metadata(
        dataset_version="v2.0",
        config_hash="hash_alpha",
        git_commit="commit_2",
        latency_threshold_ms=85.0,
        config_json='{"version": "v2.0"}',
    )

    # Different config_hash under the same frozen version MUST raise ValueError
    with pytest.raises(ValueError) as excinfo:
        db_mgr.check_and_record_run_metadata(
            dataset_version="v2.0",
            config_hash="hash_beta_modified",
            git_commit="commit_3",
            latency_threshold_ms=85.0,
            config_json='{"version": "v2.0", "modified": true}',
        )
    assert "Config hash mismatch" in str(excinfo.value)
    assert "Freeze rule" in str(excinfo.value)

    # New dataset_version allows a different config hash
    db_mgr.check_and_record_run_metadata(
        dataset_version="v3.0",
        config_hash="hash_beta_modified",
        git_commit="commit_3",
        latency_threshold_ms=50.0,
        config_json='{"version": "v3.0"}',
    )
    db_mgr.close()


def test_break_even_analysis_execution():
    res = run_break_even_analysis(db_path="data/arbsense.db", version="v2.0", exclude_censored=True)
    assert res["total_events"] == 198
    assert res["mean_bps"] > 0.0
    assert 0.0 in res["fractions"]
    assert 5.0 in res["fractions"]
    assert 10.0 in res["fractions"]
    assert 20.0 in res["fractions"]
    # Check that fraction at 0 bps is high (>90%) and fraction at 20 bps is 0%
    assert res["fractions"][0.0][1] > 90.0
    assert res["fractions"][20.0][1] == 0.0


def test_backup_utility(tmp_path):
    # Create sample sqlite DB
    db_file = tmp_path / "sample.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE t (id INT);")
    conn.execute("INSERT INTO t VALUES (1), (2), (3);")
    conn.commit()
    conn.close()

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    backup_path = backup_sqlite(db_file, backup_dir)
    assert backup_path.exists()

    # Verify backup is readable
    conn_b = sqlite3.connect(str(backup_path))
    cnt = conn_b.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    conn_b.close()
    assert cnt == 3

    # Pruning with 0 days should prune nothing
    pruned = prune_old_backups(backup_dir, retention_days=0)
    assert pruned == 0
