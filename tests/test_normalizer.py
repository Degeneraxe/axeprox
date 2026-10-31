"""Unit tests for OrderBookNormalizer, snapshots, depth-walking, and stale filtering."""
from __future__ import annotations

import pytest
import time
from data.normalizer import OrderBookNormalizer, OrderBookLevel, OrderBookSnapshot


def test_symbol_normalization():
    norm = OrderBookNormalizer()
    assert norm.normalize_symbol("binance", "btcusdt") == "BTC/USDT"
    assert norm.normalize_symbol("binance", "ethusdt") == "ETH/USDT"
    assert norm.normalize_symbol("coinbase", "BTC-USDT") == "BTC/USDT"
    assert norm.normalize_symbol("coinbase", "BTC-USD") == "BTC/USD"
    assert norm.normalize_symbol("kraken", "XBT/USDT") == "BTC/USDT"
    assert norm.normalize_symbol("kraken", "ETH/USDT") == "ETH/USDT"


def test_order_book_sorting_and_spread():
    norm = OrderBookNormalizer()
    # Unsorted input bids & asks
    bids = [(68000.0, 1.0), (68010.0, 0.5), (67990.0, 2.0)]
    asks = [(68030.0, 1.5), (68020.0, 0.8), (68040.0, 3.0)]

    snap = norm.update_book(
        exchange="binance",
        raw_symbol="btcusdt",
        bids=bids,
        asks=asks,
        exchange_ts_ms=1000.0,
        local_ts_ms=1050.0,
    )

    # Best bid must be highest price (68010.0)
    assert snap.best_bid.price == 68010.0
    assert snap.best_bid.amount == 0.5

    # Best ask must be lowest price (68020.0)
    assert snap.best_ask.price == 68020.0
    assert snap.best_ask.amount == 0.8

    # Top spread = Ask - Bid = 10.0
    assert snap.top_spread == pytest.approx(10.0)
    assert snap.mid_price == pytest.approx(68015.0)
    assert snap.transit_latency_ms == pytest.approx(50.0)


def test_order_book_imbalance():
    # Symmetric book -> imbalance = 0.0
    snap = OrderBookSnapshot(
        exchange="binance",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1050.0,
        transit_latency_ms=50.0,
        bids=[OrderBookLevel(68000.0, 10.0)],
        asks=[OrderBookLevel(68010.0, 10.0)],
    )
    assert snap.compute_imbalance(1) == pytest.approx(0.0)

    # 3x more bids than asks: (30 - 10) / (30 + 10) = 20 / 40 = 0.5
    snap.bids[0].amount = 30.0
    assert snap.compute_imbalance(1) == pytest.approx(0.5)


def test_depth_walking_vwap():
    snap = OrderBookSnapshot(
        exchange="binance",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1050.0,
        transit_latency_ms=50.0,
        bids=[
            OrderBookLevel(68000.0, 1.0),
            OrderBookLevel(67990.0, 2.0),
        ],
        asks=[
            OrderBookLevel(68010.0, 1.0),
            OrderBookLevel(68020.0, 2.0),
        ],
    )

    # Buying 0.5 BTC: fits within first ask level (68010.0)
    vwap, filled, fully_filled = snap.compute_vwap("buy", 0.5)
    assert fully_filled is True
    assert filled == pytest.approx(0.5)
    assert vwap == pytest.approx(68010.0)

    # Buying 2.0 BTC: 1.0 at 68010 and 1.0 at 68020 -> VWAP = 68015.0
    vwap, filled, fully_filled = snap.compute_vwap("buy", 2.0)
    assert fully_filled is True
    assert filled == pytest.approx(2.0)
    expected_vwap = (1.0 * 68010.0 + 1.0 * 68020.0) / 2.0
    assert vwap == pytest.approx(expected_vwap)


def test_stale_quote_filter():
    norm = OrderBookNormalizer(max_quote_age_ms=1500.0)
    now = 10_000.0

    # Book updated at 9_500.0 (age = 500ms, not stale)
    norm.update_book(
        exchange="kraken",
        raw_symbol="XBT/USDT",
        bids=[(68000.0, 1.0)],
        asks=[(68010.0, 1.0)],
        exchange_ts_ms=9_450.0,
        local_ts_ms=9_500.0,
    )
    snap = norm.get_snapshot("kraken", "BTC/USDT", current_time_ms=now)
    assert snap is not None
    assert snap.is_stale is False
    assert snap.quote_age_ms == pytest.approx(500.0)

    # Query at 12_000.0 (age = 2500ms > 1500ms -> STALE!)
    stale_snap = norm.get_snapshot("kraken", "BTC/USDT", current_time_ms=12_000.0)
    assert stale_snap is not None
    assert stale_snap.is_stale is True
    assert stale_snap.quote_age_ms == pytest.approx(2500.0)
