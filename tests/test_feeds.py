"""Unit tests for WebSocket feed parsers (Binance, Coinbase, Kraken)."""
from __future__ import annotations

import json
import pytest
from data.feeds.binance import BinanceFeed
from data.feeds.coinbase import CoinbaseFeed
from data.feeds.kraken import KrakenFeed


def test_binance_parse_message():
    feed = BinanceFeed(symbols=["btcusdt"])
    raw_payload = json.dumps({
        "stream": "btcusdt@depth10@100ms",
        "data": {
            "lastUpdateId": 1234567,
            "E": 1700000000100,
            "bids": [["68000.50", "1.25"]],
            "asks": [["68001.00", "0.85"]],
        }
    })

    result = feed.parse_message(raw_payload, local_ts_ms=1700000000150.0)
    assert result is not None
    symbol, bids, asks, exch_ts = result
    assert symbol == "btcusdt"
    assert bids == [(68000.50, 1.25)]
    assert asks == [(68001.00, 0.85)]
    assert exch_ts == 1700000000100.0


def test_coinbase_parse_ticker_message():
    feed = CoinbaseFeed(symbols=["BTC-USDT"])
    raw_payload = json.dumps({
        "type": "ticker",
        "product_id": "BTC-USDT",
        "best_bid": "67995.00",
        "best_bid_size": "2.5",
        "best_ask": "68005.00",
        "best_ask_size": "1.8",
        "time": "2026-10-29T15:20:00.500000Z",
    })

    result = feed.parse_message(raw_payload, local_ts_ms=1700000000000.0)
    assert result is not None
    symbol, bids, asks, exch_ts = result
    assert symbol == "BTC-USDT"
    assert bids[0] == (67995.00, 2.5)
    assert asks[0] == (68005.00, 1.8)
    assert exch_ts > 0.0


def test_kraken_parse_book_message():
    feed = KrakenFeed(symbols=["XBT/USDT"])
    # Kraken snapshot message: [channelID, {"as": [...], "bs": [...]}, "book-10", "XBT/USDT"]
    raw_payload = json.dumps([
        42,
        {
            "as": [["68010.00", "0.75", "1700000000.250"]],
            "bs": [["68000.00", "1.10", "1700000000.245"]],
        },
        "book-10",
        "XBT/USDT",
    ])

    result = feed.parse_message(raw_payload, local_ts_ms=1700000000300.0)
    assert result is not None
    symbol, bids, asks, exch_ts = result
    assert symbol == "XBT/USDT"
    assert bids[0] == (68000.00, 1.10)
    assert asks[0] == (68010.00, 0.75)
    assert exch_ts == pytest.approx(1700000000250.0)
