"""Unit tests for basis direction/conversion, hysteresis, and persistence labels (Step 6)."""
from __future__ import annotations

import pytest
import sqlite3
import json
from config.settings import AppConfig
from data.normalizer import OrderBookLevel, OrderBookNormalizer, OrderBookSnapshot
from data.storage.db_manager import DatabaseManager
from engine.detector import CrossExchangeDetector, SpreadCandidate
from engine.tracker import SpreadTracker, ActiveSpreadEvent


def test_basis_conversion_direction_and_magnitude():
    """Verifies that price_in_USDT = price_in_USD / usdt_usd strictly raises the USDT-equivalent price

    when usdt_usd < 1.0, eliminating the artificial 12 bps Coinbase asymmetry.
    """
    norm = OrderBookNormalizer()
    snap_usd = norm.update_book(
        exchange="coinbase",
        raw_symbol="BTC-USD",
        bids=[(65000.0, 1.0)],
        asks=[(65005.0, 1.0)],
        exchange_ts_ms=1000.0,
        local_ts_ms=1020.0,
    )
    assert snap_usd.quote_currency == "USD"

    # Given live rate: 1 USDT = 0.9993 USD
    usdt_usd = 0.9993
    norm.set_usdt_usd_basis(usdt_usd)

    # Convert Coinbase USD ask to USDT:
    # 65005.0 USD / 0.9993 = 65050.535 USDT (an increase of ~7 bps in USDT terms)
    adj_ask_usdt = snap_usd.get_adjusted_price(snap_usd.best_ask.price, target_quote="USDT", basis_rate=usdt_usd)
    expected_usdt = 65005.0 / 0.9993
    assert adj_ask_usdt == pytest.approx(expected_usdt)
    assert adj_ask_usdt > snap_usd.best_ask.price

    # Verify inverse conversion (USDT to USD): price_in_USD = price_in_USDT * usdt_usd
    snap_usdt = OrderBookSnapshot(
        exchange="binance",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1020.0,
        transit_latency_ms=20.0,
        asks=[OrderBookLevel(price=expected_usdt, amount=1.0)],
    )
    adj_usd = snap_usdt.get_adjusted_price(expected_usdt, target_quote="USD", basis_rate=usdt_usd)
    assert adj_usd == pytest.approx(65005.0)

    # Parity check: when basis rate == 1.0, no shift occurs
    assert snap_usd.get_adjusted_price(65005.0, target_quote="USDT", basis_rate=1.0) == 65005.0


def test_hysteresis_and_min_open_threshold(tmp_path):
    """Verifies that open hurdle enforces hysteresis_bps + min_open_spread_bps,

    and closure threshold enforces hurdle - hysteresis_bps to prevent microtick churn.
    """
    db_path = str(tmp_path / "hysteresis_test.db")
    db = DatabaseManager(sqlite_path=db_path)
    db.initialize()

    tracker = SpreadTracker(db_manager=db)

    # Warm up ring buffers
    for i in range(25):
        t = 800.0 + i * 5.0
        tracker.record_tick(OrderBookSnapshot("binance", "BTC/USDT", "USDT", t, t, 5.0, [OrderBookLevel(65000.0, 1.0)], [OrderBookLevel(65010.0, 1.0)]))
        tracker.record_tick(OrderBookSnapshot("kraken", "BTC/USDT", "USDT", t, t, 5.0, [OrderBookLevel(65080.0, 1.0)], [OrderBookLevel(65090.0, 1.0)]))

    # Open hurdle = 5.0 bps hurdle + 1.0 bps hysteresis = 6.0 bps
    # Candidate at 5.5 bps: exceeds hurdle (5.0) but below open_hurdle with hysteresis (6.0)
    # Open an event at 10.0 bps (exceeds hurdle + hysteresis)
    cand_open = SpreadCandidate(
        symbol="BTC/USDT",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_ask=65000.0,
        sell_bid=65065.0,
        gross_spread_bps=10.0,
        net_spread_bps=5.0,
        total_hurdle_bps=5.0,
        fee_bps=5.0,
        vol_buffer_bps=0.0,
        is_gross_candidate=True,
        is_net_profitable=True,
        timestamp_ms=1000.0,
        quote_age_buy_ms=5.0,
        quote_age_sell_ms=5.0,
    )
    evt = tracker.process_candidate(cand_open)
    assert evt is not None
    assert evt.is_closed is False

    # Spread drops to 4.5 bps: below cost hurdle (5.0 bps),
    # BUT close hurdle with hysteresis = hurdle (5.0) - hysteresis (1.0) = 4.0 bps!
    # Therefore, 4.5 bps must NOT close the event yet (hysteresis buffer active)!
    cand_inside_hysteresis = SpreadCandidate(
        symbol="BTC/USDT",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_ask=65000.0,
        sell_bid=65029.25,
        gross_spread_bps=4.5,
        net_spread_bps=-0.5,
        total_hurdle_bps=5.0,
        fee_bps=5.0,
        vol_buffer_bps=0.0,
        is_gross_candidate=True,
        is_net_profitable=False,
        timestamp_ms=1100.0,
        quote_age_buy_ms=5.0,
        quote_age_sell_ms=5.0,
    )
    still_open = tracker.process_candidate(cand_inside_hysteresis)
    assert still_open is not None
    assert still_open.is_closed is False

    # Spread drops to 3.5 bps: below 4.0 bps (hurdle - hysteresis) -> event closes cleanly!
    cand_below_hysteresis = SpreadCandidate(
        symbol="BTC/USDT",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_ask=65000.0,
        sell_bid=65022.75,
        gross_spread_bps=3.5,
        net_spread_bps=-1.5,
        total_hurdle_bps=5.0,
        fee_bps=5.0,
        vol_buffer_bps=0.0,
        is_gross_candidate=True,
        is_net_profitable=False,
        timestamp_ms=1200.0,
        quote_age_buy_ms=5.0,
        quote_age_sell_ms=5.0,
    )
    closed = tracker.process_candidate(cand_below_hysteresis)
    assert closed is not None
    assert closed.is_closed is True
    assert closed.duration_ms == pytest.approx(200.0)
    db.close()


def test_persistence_labels_derivation_and_storage(tmp_path):
    """Verifies is_persistent_500ms, is_persistent_2000ms, and is_persistent_5000ms

    classification labels at various durations.
    """
    db_path = str(tmp_path / "persistence_test.db")
    db = DatabaseManager(sqlite_path=db_path)
    db.initialize()

    tracker = SpreadTracker(db_manager=db)

    # Warm up ring buffers
    for i in range(25):
        t = 800.0 + i * 5.0
        tracker.record_tick(OrderBookSnapshot("binance", "BTC/USDT", "USDT", t, t, 5.0, [OrderBookLevel(65000.0, 1.0)], [OrderBookLevel(65010.0, 1.0)]))
        tracker.record_tick(OrderBookSnapshot("kraken", "BTC/USDT", "USDT", t, t, 5.0, [OrderBookLevel(65080.0, 1.0)], [OrderBookLevel(65090.0, 1.0)]))

    # Case 1: duration = 300 ms (< 500 ms)
    # Open at 1000, close at 1300
    c_open1 = SpreadCandidate("BTC/USDT", "binance", "kraken", 65000.0, 65065.0, 10.0, 5.0, 5.0, 5.0, 0.0, True, True, 1000.0, 5.0, 5.0)
    c_close1 = SpreadCandidate("BTC/USDT", "binance", "kraken", 65000.0, 65010.0, 1.5, -3.5, 5.0, 5.0, 0.0, False, False, 1300.0, 5.0, 5.0)
    tracker.process_candidate(c_open1)
    evt1 = tracker.process_candidate(c_close1)
    assert evt1.duration_ms == pytest.approx(300.0)
    assert evt1.is_persistent_500ms is False
    assert evt1.is_persistent_2000ms is False
    assert evt1.is_persistent_5000ms is False

    # Case 2: duration = 2200 ms (>= 500 ms and >= 2000 ms, but < 5000 ms)
    # Open at 2000, close at 4200
    c_open2 = SpreadCandidate("BTC/USDT", "binance", "kraken", 65000.0, 65065.0, 10.0, 5.0, 5.0, 5.0, 0.0, True, True, 2000.0, 5.0, 5.0)
    c_close2 = SpreadCandidate("BTC/USDT", "binance", "kraken", 65000.0, 65010.0, 1.5, -3.5, 5.0, 5.0, 0.0, False, False, 4200.0, 5.0, 5.0)
    tracker.process_candidate(c_open2)
    evt2 = tracker.process_candidate(c_close2)
    assert evt2.duration_ms == pytest.approx(2200.0)
    assert evt2.is_persistent_500ms is True
    assert evt2.is_persistent_2000ms is True
    assert evt2.is_persistent_5000ms is False

    # Case 3: duration = 6000 ms (>= 5000 ms)
    # Open at 5000, close at 11000
    c_open3 = SpreadCandidate("BTC/USDT", "binance", "kraken", 65000.0, 65065.0, 10.0, 5.0, 5.0, 5.0, 0.0, True, True, 5000.0, 5.0, 5.0)
    c_close3 = SpreadCandidate("BTC/USDT", "binance", "kraken", 65000.0, 65010.0, 1.5, -3.5, 5.0, 5.0, 0.0, False, False, 11000.0, 5.0, 5.0)
    tracker.process_candidate(c_open3)
    evt3 = tracker.process_candidate(c_close3)
    assert evt3.duration_ms == pytest.approx(6000.0)
    assert evt3.is_persistent_500ms is True
    assert evt3.is_persistent_2000ms is True
    assert evt3.is_persistent_5000ms is True

    # Verify persisted values in database
    cur = db._sqlite_conn.cursor()
    cur.execute("SELECT event_id, is_persistent_500ms, is_persistent_2000ms, is_persistent_5000ms FROM spread_events ORDER BY open_time ASC")
    rows = cur.fetchall()
    assert len(rows) == 3
    # Case 1
    assert rows[0][1] == 0 and rows[0][2] == 0 and rows[0][3] == 0
    # Case 2
    assert rows[1][1] == 1 and rows[1][2] == 1 and rows[1][3] == 0
    # Case 3
    assert rows[2][1] == 1 and rows[2][2] == 1 and rows[2][3] == 1
    db.close()
