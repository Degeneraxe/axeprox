import pytest

from data.normalizer import OrderBookNormalizer
from data.ntp_sync import ClockSynchronizer
from engine.tracker import ActiveSpreadEvent, SpreadTracker


def pending(tracker, opened=1000, sell="kraken"):
    event = ActiveSpreadEvent("evt", "BTC/USDT", "binance", sell, opened, 3, 2, 3,
                              opened, native_usdt_route=sell != "coinbase")
    tracker.pending_arrivals[event.event_id] = event
    return event


@pytest.mark.parametrize("buy_time,sell_time,volume,venue,status", [
    (1086, 1085, 1, "kraken", "missing_books"),
    (834, 834, 1, "kraken", "missing_books"),
    (984, 1085, 1, "kraken", "missing_books"),
    (1084, 1085, .001, "kraken", "insufficient_depth"),
    (1084, 1085, 1, "coinbase", "conversion_required"),
    (1084, 1085, 1, "kraken", "priced"),
])
def test_arrival_outcome(buy_time, sell_time, volume, venue, status):
    books = OrderBookNormalizer()
    tracker = SpreadTracker(normalizer=books)
    for exchange, ts in [("binance", buy_time), (venue, sell_time)]:
        books.update_book(exchange, "BTC/USDT", [(99, 1)], [(101, volume)], ts, ts)
    event = pending(tracker, sell=venue)
    tracker.observe_arrivals(1100)
    assert event.execution_status == status
    if status == "priced":
        assert event.is_executable and event.net_peak_vip3 < 0


def test_capture_ended():
    tracker = SpreadTracker()
    event = pending(tracker)
    tracker.finish(1084)
    assert event.execution_status == "capture_ended"


def test_offset_steps_do_not_move_arrival_or_duration(monkeypatch):
    import time
    tick = 0
    monkeypatch.setattr(time, "monotonic_ns", lambda: tick * 1000000)
    clock = ClockSynchronizer()
    tracker = SpreadTracker()
    event = pending(tracker, clock.event_time_ms())
    tracker.active_events[(event.symbol, event.buy_exchange, event.sell_exchange)] = event
    for tick, offset in [(84, 5000), (85, -5000)]:
        monkeypatch.setattr(clock, "_sample_server", lambda _: (1, offset, .5, "test"))
        clock.sync()
        tracker.observe_arrivals(clock.event_time_ms())
        assert event.execution_status == ("pending" if tick == 84 else "missing_books")
    tick = 101
    tracker.finish(clock.event_time_ms())
    assert event.duration_ms == pytest.approx(101)


def test_slow_rest_keeps_good_measurement(monkeypatch):
    clock = ClockSynchronizer()
    monkeypatch.setattr(clock, "_sample_server", lambda _: (5, -4, 3, "NTP:test"))
    clock.sync()
    previous = clock.measurement()
    monkeypatch.setattr(clock, "_sample_server", lambda _: None)
    monkeypatch.setattr(clock, "_query_rest", lambda: (500, 100))
    clock.sync()
    assert clock.measurement() == previous



def test_rolling_window_only_removes_raw_ticks(tmp_path):
    from data.ntp_sync import clock_sync
    from data.storage import DatabaseManager
    db = DatabaseManager(sqlite_path=str(tmp_path / "replay.db"))
    db.initialize(sqlite_only=True)
    now = clock_sync.event_time_ms()
    books = OrderBookNormalizer()
    ticks = [books.update_book("kraken", "BTC/USDT", [(99, 1)], [(100, 1)], ts, ts)
             for ts in [now - 7*3600000, now - 5*3600000]]
    db.insert_ticks_batch(ticks)
    db.insert_spread_event(dict(event_id="evt", symbol="BTC/USDT", buy_exchange="binance",
                                sell_exchange="kraken", open_time=ticks[0].local_ts_ms,
                                feature_window_json="[1]", lifetime_ticks_json="[2]"))
    db.record_arrival(pending(SpreadTracker()), {"status": "missing_books"})
    assert db.prune_old_ticks() == 1
    assert db.get_tick_count() == 1 and db.get_cumulative_tick_count() == 2
    row = db._sqlite_conn.execute(
        "SELECT feature_window_json, lifetime_ticks_json, execution_evidence_json FROM spread_events"
    ).fetchone()
    assert tuple(row) == ("[1]", "[2]", '{"status": "missing_books"}')
    db.close()


def test_filled_loss_is_persisted(tmp_path):
    from data.storage import DatabaseManager
    from engine.simulator import ExecutionSimulator
    db = DatabaseManager(sqlite_path=str(tmp_path / "loss.db"))
    db.initialize(sqlite_only=True)
    norm = OrderBookNormalizer()
    buy, sell = [norm.update_book(v, "BTC/USDT", [(99, 1)], [(101, 1)], 1000, 1000)
                 for v in ("binance", "kraken")]
    sim = ExecutionSimulator(db_manager=db)
    trade = sim.execute_arbitrage("BTC/USDT", buy, sell, quantity=.05, fee_scenario="vip3")
    stored = db._sqlite_conn.execute("SELECT status, net_pnl FROM simulated_trades").fetchone()
    assert stored["status"] == "FILLED"
    assert stored["net_pnl"] == trade.net_pnl_usdt < 0
    assert sim.virtual_balance == pytest.approx(sim.initial_capital + trade.net_pnl_usdt)
    db.close()


def test_dirty_long_run_requires_override(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "git_dirty", lambda: True)
    for duration in (None, 601):
        with pytest.raises(ValueError, match="--allow-dirty"):
            settings.check_run_provenance(duration)
        assert settings.check_run_provenance(duration, True)
    assert settings.check_run_provenance(600)

