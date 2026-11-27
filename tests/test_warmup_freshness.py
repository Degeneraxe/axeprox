from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from data.feeds.base_feed import ConnectionState, feeds_ready
from data.normalizer import OrderBookNormalizer
from data.storage import DatabaseManager
from engine.detector import SpreadCandidate
from engine.tracker import SpreadTracker
from scripts.propose_latency_threshold import analyze_latencies


def test_all_connections_and_reconnect_gate():
    feeds = [SimpleNamespace(state=ConnectionState.CONNECTED, connected_since_ms=t)
             for t in (1000, 2000, 3000)]
    assert not feeds_ready([], 63000, 60)
    assert not feeds_ready(feeds, 62999, 60)
    assert feeds_ready(feeds, 63000, 60)
    feeds[0].state = ConnectionState.RECONNECTING
    assert not feeds_ready(feeds, 63000, 60)
    feeds[0].state = ConnectionState.CONNECTED
    feeds[0].connected_since_ms = 64000
    assert not feeds_ready(feeds, 123999, 60)
    assert feeds_ready(feeds, 124000, 60)


def test_warmup_suppresses_open_and_reconnect_blocks_pending_fill():
    feeds = [SimpleNamespace(state=ConnectionState.CONNECTED, connected_since_ms=0)]
    books = OrderBookNormalizer()
    db = Mock()
    tracker = SpreadTracker(db_manager=db, normalizer=books,
                            ready_at=lambda t: feeds_ready(feeds, t, 60))
    for t in range(59980, 60000):
        for ex in ("binance", "kraken"):
            books.update_book(ex, "BTC/USDT", [(99, 1)], [(101, 1)], t, t)
    candidate = SpreadCandidate("BTC/USDT", "binance", "kraken", 100, 101, 100,
                                80, 20, 20, 0, True, True, 59999, 0, 0)
    assert tracker.process_candidate(candidate) is None
    db.insert_spread_event.assert_not_called()
    candidate.timestamp_ms = 60000
    event = tracker.process_candidate(candidate)
    assert event is not None
    feeds[0].connected_since_ms = 60040
    tracker.check_timeouts(60085)
    assert not event.is_executable
    assert db.record_arrival.call_args.args[1]["missing_cause"] == "connection_warmup"


@pytest.mark.parametrize("delay,flag", [(499, False), (500, False), (501, True)])
def test_exchange_age_at_arrival_and_receipt_delay(delay, flag):
    from engine.simulator import inspect_arrival
    books = OrderBookNormalizer()
    buy = books.update_book("coinbase", "BTC-USD", [(99, 1)], [(100, 1)], 1000, 1490)
    sell = books.update_book("binance", "BTC/USDT", [(101, 1)], [(102, 1)], 1490, 1490)
    buy.transit_latency_ms = 490
    pair, ev = inspect_arrival(books.ring_buffers, buy.symbol, sell.symbol, "coinbase", "binance", 1000+delay)
    assert pair == (buy, sell)
    assert ev["buy"]["corrected_receipt_delay_ms"] == 490
    assert ev["buy"]["exchange_age_ms"] == delay
    assert ev["buy"]["exchange_age_over_500ms"] is flag
    assert ev["sell"]["exchange_age_ms"] is None


def test_latency_report_excludes_warmup_and_unmarked_ticks(tmp_path):
    path = tmp_path / "latency.db"
    db = DatabaseManager(sqlite_path=str(path))
    db.initialize(sqlite_only=True)
    books = OrderBookNormalizer()
    for source, ready, delay in [("exchange", False, 999), ("exchange", None, 777),
                                 ("estimated", True, 555), ("exchange", True, -2)]:
        snap = books.update_book("kraken", "BTC/USDT", [(99, 1)], [(100, 1)], 1000,
                                 1000+delay, ts_source=source)
        snap.warmup_complete = ready
        snap.transit_latency_ms = delay
        db.insert_ticks_batch([snap])
    db.record_clock_offset(1, method="test", rtt_ms=4, error_bound_ms=3)
    db.close()
    result = analyze_latencies(str(path))
    assert result["per_exchange"]["kraken"]["p99"] == -2
    assert result["overall"]["count"] == 1
    assert result["bounds"][0][2] == 3


def test_usd_leg_records_exchange_age_without_pricing():
    from engine.tracker import ActiveSpreadEvent
    db = Mock()
    norm = OrderBookNormalizer()
    tracker = SpreadTracker(normalizer=norm, db_manager=db)
    book = norm.update_book("coinbase", "BTC-USD", [(99, 1)], [(100, 1)], 400, 1075)
    book.transit_latency_ms = 675
    norm.update_book("kraken", "BTC/USDT", [(99, 1)], [(100, 1)], 1060, 1070)
    event = ActiveSpreadEvent("id", "BTC/USDT", "coinbase", "kraken", 1000, 5, 2, 5, 1000,
                              native_usdt_route=False, buy_symbol="BTC/USD", sell_symbol="BTC/USDT")
    tracker.pending_arrivals["id"] = event
    tracker.observe_arrivals(1085)
    ev = db.record_arrival.call_args.args[1]
    assert ev["status"] == "conversion_required"
    assert ev["buy"]["exchange_age_ms"] == 685
    assert ev["buy"]["exchange_age_over_500ms"]
    assert ev["sell"]["exchange_age_over_500ms"] is False
    assert event.net_peak_vip3 is None
