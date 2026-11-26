"""Unit tests for storage engine, migrations, batch writer, and TimescaleDB probe."""
from __future__ import annotations

import os
import tempfile
import pytest
from pathlib import Path

from data.normalizer import OrderBookSnapshot, OrderBookLevel
from data.storage.db_manager import DatabaseManager
from data.storage.batch_writer import BatchWriter


def test_sqlite_initialization_and_migrations():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = DatabaseManager(sqlite_path=db_path)
        backend = db.initialize()
        assert backend == "sqlite"
        assert db.is_connected is True
        assert "Operating on SQLite WAL engine" in db.connection_diagnostic

        # Verify tables exist
        cur = db._sqlite_conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        assert "orderbook_ticks" in tables
        assert "spread_events" in tables
        assert "simulated_trades" in tables
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_timescaledb_probe_diagnostic():
    # Probes invalid/offline port to verify resilient failure handling
    db = DatabaseManager(db_url="postgresql://postgres:postgres@localhost:59999/arbsense")
    ok, diag = db._probe_timescaledb()
    assert ok is False
    assert "failed" in diag.lower()


def test_batch_insert_and_query():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = DatabaseManager(sqlite_path=db_path)
        db.initialize()

        sample_ticks = [
            {
                "time": "2026-10-29T15:30:00.000Z",
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "quote_currency": "USDT",
                "best_bid": 81500.0,
                "best_ask": 81501.0,
                "mid_price": 81500.5,
                "top_spread": 1.0,
                "imbalance": 0.25,
                "bid_depth": [{"p": 81500.0, "a": 1.5}],
                "ask_depth": [{"p": 81501.0, "a": 0.8}],
                "exchange_ts_ms": 1700000000000.0,
                "local_ts_ms": 1700000000025.0,
                "transit_latency_ms": 25.0,
                "is_stale": False,
            },
            {
                "time": "2026-10-29T15:30:00.100Z",
                "exchange": "kraken",
                "symbol": "BTC/USDT",
                "quote_currency": "USDT",
                "best_bid": 81505.0,
                "best_ask": 81506.0,
                "mid_price": 81505.5,
                "top_spread": 1.0,
                "imbalance": -0.15,
                "bid_depth": [{"p": 81505.0, "a": 0.5}],
                "ask_depth": [{"p": 81506.0, "a": 1.2}],
                "exchange_ts_ms": 1700000000000.0,
                "local_ts_ms": 1700000000140.0,
                "transit_latency_ms": 140.0,
                "is_stale": False,
            },
        ]

        inserted = db.insert_ticks_batch(sample_ticks)
        assert inserted == 2
        assert db.get_tick_count() == 2

        recent = db.get_recent_ticks(limit=5)
        assert len(recent) == 2
        # Most recent first
        assert recent[0]["exchange"] == "kraken"
        assert recent[0]["best_bid"] == 81505.0
        assert recent[1]["exchange"] == "binance"
        assert recent[1]["best_bid"] == 81500.0
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_spread_events_and_trades_insert():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        db = DatabaseManager(sqlite_path=db_path)
        db.initialize()

        event = {
            "event_id": "evt-001",
            "symbol": "BTC/USDT",
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "open_time": "2026-10-29T15:30:00Z",
            "open_spread_bps": 6.5,
            "cost_hurdle_bps": 5.0,
            "peak_spread_bps": 8.2,
        }
        db.insert_spread_event(event)

        # Update upon closure
        db.update_spread_event("evt-001", {
            "close_time": "2026-10-29T15:30:01Z",
            "peak_spread_bps": 9.1,
            "duration_ms": 1000.0,
            "log_duration": 6.908,
            "is_viable": True,
            "is_right_censored": False,
        })

        # Insert trade
        trade = {
            "trade_id": "trd-001",
            "event_id": "evt-001",
            "timestamp": "2026-10-29T15:30:00.100Z",
            "symbol": "BTC/USDT",
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "quantity": 0.05,
            "buy_vwap": 81500.0,
            "sell_vwap": 81550.0,
            "buy_fee": 4.075,
            "sell_fee": 16.31,
            "gross_pnl": 2.50,
            "net_pnl": 1.48,
            "latency_ms": 85.0,
            "status": "FILLED",
        }
        db.insert_simulated_trade(trade)

        cur = db._sqlite_conn.cursor()
        cur.execute("SELECT * FROM spread_events WHERE event_id = 'evt-001'")
        evt_row = dict(cur.fetchone())
        assert evt_row["is_viable"] == 1
        assert evt_row["duration_ms"] == 1000.0
        assert evt_row["peak_spread_bps"] == 9.1

        cur.execute("SELECT * FROM simulated_trades WHERE trade_id = 'trd-001'")
        trd_row = dict(cur.fetchone())
        assert trd_row["status"] == "FILLED"
        assert trd_row["net_pnl"] == 1.48
    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_depth_archive_survives_backpressure_and_stop():
    import asyncio
    from unittest.mock import Mock

    async def exercise():
        saved = []
        db = Mock()
        db.insert_ticks_batch.side_effect = lambda batch: saved.extend(batch)
        writer = BatchWriter(db, batch_size=100, flush_interval_sec=60, depth_throttle_ms=0)
        writer.queue = asyncio.Queue(maxsize=2)
        await writer.start()
        for stamp in range(8):
            writer.enqueue_snapshot(OrderBookSnapshot(
                "kraken", "BTC/USDT", "USDT", stamp, stamp, 0,
                [OrderBookLevel(99, 2)], [OrderBookLevel(100, 2)]))
        await asyncio.sleep(0.01)
        await writer.stop()
        assert [tick["local_ts_ms"] for tick in saved] == list(range(8))
        assert all(tick["bid_depth"] and tick["ask_depth"] for tick in saved)

    asyncio.run(exercise())
