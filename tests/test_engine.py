"""Unit tests for Detection, Cost Hurdle, Lifecycle Tracking, and Execution Simulation."""
from __future__ import annotations

import pytest
from data.normalizer import OrderBookNormalizer, OrderBookSnapshot, OrderBookLevel
from engine.cost_model import CostModel
from engine.detector import CrossExchangeDetector
from engine.tracker import SpreadTracker
from engine.simulator import ExecutionSimulator


def test_cost_model_scenarios_and_volatility():
    cost_model = CostModel()
    # Test retail scenario
    hurdle, fees, vol_buf = cost_model.get_hurdle_bps("binance", "kraken", "BTC/USDT", scenario="retail")
    # Binance taker = 10 bps, Kraken taker = 40 bps -> total fees = 50 bps
    assert fees == pytest.approx(50.0)
    assert hurdle > 50.0  # Fees + volatility buffer

    # Test VIP3 scenario
    _, fees_vip, _ = cost_model.get_hurdle_bps("binance", "kraken", "BTC/USDT", scenario="vip3")
    # Binance taker = 6 bps, Kraken taker = 16 bps -> 22 bps
    assert fees_vip == pytest.approx(22.0)


def test_cross_exchange_detector_spread_calculation():
    norm = OrderBookNormalizer()
    now_ms = 1000.0

    # Binance: Bid 81500, Ask 81510
    norm.update_book("binance", "btcusdt", [(81500.0, 1.0)], [(81510.0, 1.0)], now_ms, now_ms)
    # Kraken: Bid 81580, Ask 81590
    norm.update_book("kraken", "XBT/USDT", [(81580.0, 1.0)], [(81590.0, 1.0)], now_ms, now_ms)

    detector = CrossExchangeDetector(normalizer=norm)
    candidates = detector.scan_symbol("BTC/USDT", current_time_ms=now_ms)

    # Opportunity: Buy on Binance (Ask 81510), Sell on Kraken (Bid 81580)
    # Gross Spread = (81580 - 81510) / 81510 * 10000 = 70 / 81510 * 10000 ~= 8.588 bps
    binance_to_kraken = next(
        (c for c in candidates if c.buy_exchange == "binance" and c.sell_exchange == "kraken"), None
    )
    assert binance_to_kraken is not None
    assert binance_to_kraken.gross_spread_bps == pytest.approx((70.0 / 81510.0) * 10000.0, rel=1e-3)
    assert binance_to_kraken.is_gross_candidate is True


def test_spread_tracker_lifecycle_and_labeling():
    tracker = SpreadTracker()
    now_ms = 5000.0

    # Populate ring buffers with required >= 20 ticks so tracker can open events
    for i in range(25):
        t = 1000.0 + i * 100.0
        tracker.record_tick(OrderBookSnapshot("binance", "BTC/USDT", "USDT", t, t, 10.0, [OrderBookLevel(81490.0, 1.0)], [OrderBookLevel(81500.0, 1.0)]))
        tracker.record_tick(OrderBookSnapshot("kraken", "BTC/USDT", "USDT", t, t, 10.0, [OrderBookLevel(81560.0, 1.0)], [OrderBookLevel(81570.0, 1.0)]))

    from engine.detector import SpreadCandidate
    cand = SpreadCandidate(
        symbol="BTC/USDT",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_ask=81500.0,
        sell_bid=81560.0,
        gross_spread_bps=7.36,
        net_spread_bps=2.36,
        total_hurdle_bps=5.00,
        fee_bps=5.00,
        vol_buffer_bps=0.00,
        is_gross_candidate=True,
        is_net_profitable=True,
        timestamp_ms=now_ms,
        quote_age_buy_ms=10.0,
        quote_age_sell_ms=15.0,
    )

    # 1. Open event
    evt = tracker.process_candidate(cand)
    assert evt is not None
    assert evt.is_closed is False
    assert evt.open_spread_bps == pytest.approx(7.36)

    # 2. Advance time and update peak spread
    cand.gross_spread_bps = 9.20
    cand.timestamp_ms = 5150.0  # 150ms later (> 85ms execution hurdle)
    tracker.process_candidate(cand)

    # 3. Close event (spread drops to negative)
    cand.gross_spread_bps = -1.0
    cand.timestamp_ms = 5200.0  # Total duration = 200ms
    closed_evt = tracker.process_candidate(cand)

    assert closed_evt is not None
    assert closed_evt.is_closed is True
    assert closed_evt.duration_ms == pytest.approx(200.0)
    assert closed_evt.peak_spread_bps == pytest.approx(9.20)
    assert closed_evt.is_viable is True  # 200ms > 85ms and peak 9.20 > hurdle 5.00
    assert closed_evt.log_duration == pytest.approx(pytest.approx(5.3033, rel=1e-2))
    assert len(closed_evt.lifetime_ticks) == 3
    assert closed_evt.lifetime_ticks[0]["gross_spread_bps"] == pytest.approx(7.36)
    assert closed_evt.lifetime_ticks[1]["gross_spread_bps"] == pytest.approx(9.20)
    assert closed_evt.lifetime_ticks[2]["gross_spread_bps"] == pytest.approx(-1.0)
    assert closed_evt.is_suspect is False


def test_execution_simulator_vwap_and_pnl():
    simulator = ExecutionSimulator()

    # Buy venue (Binance) asks
    buy_snap = OrderBookSnapshot(
        exchange="binance",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1025.0,
        transit_latency_ms=25.0,
        asks=[
            OrderBookLevel(81500.0, 0.03),  # 0.03 at 81500
            OrderBookLevel(81510.0, 0.05),  # remaining 0.02 at 81510
        ],
    )

    # Sell venue (Kraken) bids
    sell_snap = OrderBookSnapshot(
        exchange="kraken",
        symbol="BTC/USDT",
        quote_currency="USDT",
        exchange_ts_ms=1000.0,
        local_ts_ms=1140.0,
        transit_latency_ms=140.0,
        bids=[
            OrderBookLevel(81580.0, 0.04),  # 0.04 at 81580
            OrderBookLevel(81570.0, 0.05),  # remaining 0.01 at 81570
        ],
    )

    # Target trade size = 0.05 BTC
    result = simulator.execute_arbitrage(
        symbol="BTC/USDT",
        buy_snap=buy_snap,
        sell_snap=sell_snap,
        opportunity_duration_ms=500.0,  # Long-lived opportunity (not fleeting)
        quantity=0.05,
        fee_scenario="vip3",  # Lower fee scenario to see clean positive net profit
    )

    assert result.status == "FILLED"
    assert result.filled_quantity == pytest.approx(0.05)

    # Buy VWAP: (0.03 * 81500 + 0.02 * 81510) / 0.05 = 81504.0
    assert result.buy_vwap == pytest.approx(81504.0)

    # Sell VWAP: (0.04 * 81580 + 0.01 * 81570) / 0.05 = 81578.0
    assert result.sell_vwap == pytest.approx(81578.0)

    # Gross P&L: (81578.0 - 81504.0) * 0.05 = 74.0 * 0.05 = 3.70 USDT
    assert result.gross_pnl_usdt == pytest.approx(3.70)
    # Net P&L = Gross P&L - Fees: exactly matches accounting
    expected_fees = result.buy_fee_usdt + result.sell_fee_usdt
    assert result.net_pnl_usdt == pytest.approx(3.70 - expected_fees)

    # Now test with a spread that exceeds the 22 bps fee hurdle (e.g. Kraken at 81800 -> ~36 bps)
    sell_snap.bids[0].price = 81800.0
    sell_snap.bids[1].price = 81790.0
    prof_result = simulator.execute_arbitrage(
        symbol="BTC/USDT",
        buy_snap=buy_snap,
        sell_snap=sell_snap,
        opportunity_duration_ms=500.0,
        quantity=0.05,
        fee_scenario="vip3",
    )
    assert prof_result.status == "FILLED"
    assert prof_result.net_pnl_usdt > 0.0

    # Also verify honest recording of losing trade when fees exceed gross alpha (Defect 3)
    bal_before = simulator.virtual_balance
    losing_result = simulator.execute_arbitrage(
        symbol="BTC/USDT",
        buy_snap=buy_snap,
        sell_snap=sell_snap,
        opportunity_duration_ms=500.0,
        quantity=0.05,
        fee_scenario="retail",  # Retail taker fees (~40-60 bps) exceed the ~36 bps spread
    )
    assert losing_result.status == "FILLED"
    assert losing_result.net_pnl_usdt < 0.0
    assert simulator.virtual_balance == pytest.approx(bal_before + losing_result.net_pnl_usdt)


def test_engine_importable_without_torch():
    """Verifies that engine modules can be imported and instantiated without torch (Requirement 5)."""
    import sys

    # Simulate environment where torch is completely absent
    orig_torch = sys.modules.get("torch")
    sys.modules["torch"] = None
    try:
        import importlib
        import engine.cost_model
        import engine.detector
        import engine.simulator
        import engine.tracker

        importlib.reload(engine.cost_model)
        importlib.reload(engine.detector)
        importlib.reload(engine.simulator)
        importlib.reload(engine.tracker)

        cm = engine.cost_model.CostModel()
        sim = engine.simulator.ExecutionSimulator()
        tracker = engine.tracker.SpreadTracker()
        assert cm is not None
        assert sim is not None
        assert tracker is not None
    finally:
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)


def test_live_and_training_feature_vectors_equal():
    """Verifies that the live feature window equals the training feature window for the exact same ticks (Item 1)."""
    import numpy as np
    from data.normalizer import OrderBookNormalizer
    from engine.tracker import SpreadTracker

    norm = OrderBookNormalizer()
    tracker = SpreadTracker(normalizer=norm)

    # Generate 25 sequential ticks for Binance and Kraken
    for i in range(25):
        t_ms = 1000.0 + (i * 100.0)
        p_b_ask = 81500.0 + i
        p_b_bid = 81490.0 + i
        p_k_ask = 81590.0 + i
        p_k_bid = 81580.0 + i

        norm.update_book(
            "binance", "btcusdt",
            [(p_b_bid, 1.0)], [(p_b_ask, 1.0)],
            exchange_ts_ms=t_ms - 25.0, local_ts_ms=t_ms,
        )
        norm.update_book(
            "kraken", "XBT/USDT",
            [(p_k_bid, 1.0)], [(p_k_ask, 1.0)],
            exchange_ts_ms=t_ms - 80.0, local_ts_ms=t_ms,
        )

    # 1. Live feature window built via normalizer
    live_win, live_vol_b, live_vol_s = norm.build_feature_window("BTC/USDT", "binance", "kraken", seq_len=20)

    # 2. Training feature window built via tracker
    train_win, train_vol_b, train_vol_s = tracker._build_point_in_time_feature_window(
        "BTC/USDT", "binance", "kraken", open_time_ms=t_ms, seq_len=20
    )

    assert len(live_win) == 20
    assert len(train_win) == 20
    assert live_vol_b == pytest.approx(train_vol_b)
    assert live_vol_s == pytest.approx(train_vol_s)

    # Live feature vector equals training feature vector for the same ticks
    np.testing.assert_allclose(np.array(live_win), np.array(train_win), rtol=1e-5, atol=1e-5)


def test_no_event_persisted_without_complete_window(tmp_path):
    """Verifies that no event is persisted without a complete 20-step feature window (Task 2 & 5)."""
    import json
    from data.normalizer import OrderBookNormalizer
    from data.storage.db_manager import DatabaseManager
    from engine.detector import SpreadCandidate
    from engine.tracker import SpreadTracker

    db_file = tmp_path / "test_window.db"
    db = DatabaseManager(sqlite_path=str(db_file), migrations_dir="data/storage/migrations")
    db.initialize()

    norm = OrderBookNormalizer()
    tracker = SpreadTracker(db_manager=db, normalizer=norm)

    candidate = SpreadCandidate(
        symbol="BTC/USDT",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_ask=81500.0,
        sell_bid=81600.0,
        gross_spread_bps=12.27,
        net_spread_bps=5.0,
        total_hurdle_bps=7.27,
        fee_bps=7.0,
        vol_buffer_bps=0.27,
        is_gross_candidate=True,
        is_net_profitable=True,
        timestamp_ms=1000.0,
        quote_age_buy_ms=10.0,
        quote_age_sell_ms=10.0,
    )

    # 1. Under-populated buffer: only 5 ticks (less than seq_len=20)
    for i in range(5):
        norm.update_book("binance", "btcusdt", [(81490.0, 1.0)], [(81500.0, 1.0)], 1000.0 + i, 1000.0 + i)
        norm.update_book("kraken", "XBT/USDT", [(81600.0, 1.0)], [(81610.0, 1.0)], 1000.0 + i, 1000.0 + i)

    res = tracker.process_candidate(candidate)
    assert res is None  # Must NOT open event
    cur = db._sqlite_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM spread_events")
    assert cur.fetchone()[0] == 0  # Nothing persisted

    # 2. Add 20 more ticks so ring buffers have >= 20 ticks
    for i in range(5, 25):
        t = 1000.0 + i * 10.0
        norm.update_book("binance", "btcusdt", [(81490.0, 1.0)], [(81500.0, 1.0)], t, t)
        norm.update_book("kraken", "XBT/USDT", [(81600.0, 1.0)], [(81610.0, 1.0)], t, t)

    candidate.timestamp_ms = 1250.0
    res2 = tracker.process_candidate(candidate)
    assert res2 is not None
    assert len(res2.feature_window) == 20

    cur.execute("SELECT event_id, feature_window_json FROM spread_events")
    row = cur.fetchone()
    assert row is not None
    assert row[1] is not None
    parsed_win = json.loads(row[1])
    assert len(parsed_win) == 20
    db.close()


def test_minimum_depth_filter():
    """Verifies that candidates with insufficient top-N depth liquidity are filtered out (Task 3 & 5)."""
    from data.normalizer import OrderBookNormalizer
    from engine.detector import CrossExchangeDetector

    norm = OrderBookNormalizer()
    detector = CrossExchangeDetector(normalizer=norm)

    now_ms = 10000.0
    # Configured trade size for BTC/USDT is 0.05 BTC (~$4,000 notional)
    # 1. Thin book: Binance has only 0.001 BTC on ask (insufficient depth)
    norm.update_book("binance", "btcusdt", [(81490.0, 1.0)], [(81500.0, 0.001)], now_ms - 10, now_ms)
    norm.update_book("kraken", "XBT/USDT", [(81600.0, 1.0)], [(81610.0, 1.0)], now_ms - 10, now_ms)

    candidates = detector.scan_symbol("BTC/USDT", now_ms)
    # The spread 81600 vs 81500 is ~12.27 bps gross, but depth is only 0.001 BTC -> must be filtered
    assert len(candidates) == 0

    # 2. Liquid book: Binance has 0.10 BTC on ask (exceeds 0.05 BTC requirement)
    norm.update_book("binance", "btcusdt", [(81490.0, 1.0)], [(81500.0, 0.10)], now_ms - 5, now_ms)
    candidates_liquid = detector.scan_symbol("BTC/USDT", now_ms)
    assert len(candidates_liquid) == 1
    assert candidates_liquid[0].buy_exchange == "binance"
    assert candidates_liquid[0].sell_exchange == "kraken"


def test_cumulative_tick_counter(tmp_path):
    """Verifies that cumulative tick counter survives pruning and tracks total ingested ticks (Task 4 & 5)."""
    from data.normalizer import OrderBookLevel, OrderBookSnapshot
    from data.storage.db_manager import DatabaseManager

    db_file = tmp_path / "test_cumul.db"
    db = DatabaseManager(sqlite_path=str(db_file), migrations_dir="data/storage/migrations")
    db.initialize()

    # Ingest 10 ticks
    ticks = []
    for i in range(10):
        snap = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            quote_currency="USDT",
            exchange_ts_ms=1000.0 + i,
            local_ts_ms=1010.0 + i,
            transit_latency_ms=10.0,
            bids=[OrderBookLevel(81500.0, 1.0)],
            asks=[OrderBookLevel(81510.0, 1.0)],
        )
        ticks.append(snap)
    db.insert_ticks_batch(ticks)

    assert db.get_tick_count() == 10
    assert db.get_cumulative_tick_count() == 10

    # Prune down to 3 ticks
    pruned = db.prune_old_ticks(keep_latest_per_symbol=3)
    assert pruned == 7
    # Active buffer decreased to 3
    assert db.get_tick_count() == 3
    # Cumulative counter survives and remains 10
    assert db.get_cumulative_tick_count() == 10

    # Ingest 5 more ticks
    more_ticks = []
    for i in range(10, 15):
        snap = OrderBookSnapshot(
            exchange="binance",
            symbol="BTC/USDT",
            quote_currency="USDT",
            exchange_ts_ms=1000.0 + i,
            local_ts_ms=1010.0 + i,
            transit_latency_ms=10.0,
            bids=[OrderBookLevel(81500.0, 1.0)],
            asks=[OrderBookLevel(81510.0, 1.0)],
        )
        more_ticks.append(snap)
    db.insert_ticks_batch(more_ticks)

    assert db.get_tick_count() == 8  # 3 retained + 5 new
    assert db.get_cumulative_tick_count() == 15  # 15 cumulative
    db.close()


def test_spread_tracker_suspect_gate_and_lifetime_ticks(tmp_path):
    """Verifies that spreads exceeding max_plausible_spread_bps are flagged as suspect and lifetime ticks are persisted."""
    import json
    from data.storage.db_manager import DatabaseManager
    from engine.detector import SpreadCandidate

    db_file = tmp_path / "test_suspect.db"
    db = DatabaseManager(sqlite_path=str(db_file), migrations_dir="data/storage/migrations")
    db.initialize()

    tracker = SpreadTracker(db_manager=db)
    now_ms = 10000.0

    for i in range(25):
        t = 1000.0 + i * 100.0
        tracker.record_tick(OrderBookSnapshot("binance", "BTC/USDT", "USDT", t, t, 10.0, [OrderBookLevel(80000.0, 1.0)], [OrderBookLevel(80010.0, 1.0)]))
        tracker.record_tick(OrderBookSnapshot("kraken", "BTC/USDT", "USDT", t, t, 10.0, [OrderBookLevel(81500.0, 1.0)], [OrderBookLevel(81510.0, 1.0)]))

    # Candidate with 150 bps spread (> 80 bps max plausible)
    suspect_cand = SpreadCandidate(
        symbol="BTC/USDT",
        buy_exchange="binance",
        sell_exchange="kraken",
        buy_ask=80010.0,
        sell_bid=81210.0,
        gross_spread_bps=150.0,
        net_spread_bps=100.0,
        total_hurdle_bps=50.0,
        fee_bps=50.0,
        vol_buffer_bps=0.0,
        is_gross_candidate=True,
        is_net_profitable=True,
        timestamp_ms=now_ms,
        quote_age_buy_ms=5.0,
        quote_age_sell_ms=5.0,
    )

    evt = tracker.process_candidate(suspect_cand)
    assert evt is not None
    assert evt.is_suspect is True

    # Advance tick
    suspect_cand.timestamp_ms = 10200.0
    suspect_cand.gross_spread_bps = 160.0
    tracker.process_candidate(suspect_cand)

    # Close tick
    suspect_cand.timestamp_ms = 10400.0
    suspect_cand.gross_spread_bps = 20.0  # falls below 50 bps hurdle
    closed_evt = tracker.process_candidate(suspect_cand)

    assert closed_evt is not None
    assert closed_evt.is_closed is True
    assert closed_evt.is_suspect is True
    assert closed_evt.duration_ms == pytest.approx(400.0)
    assert closed_evt.peak_spread_bps == pytest.approx(160.0)
    assert len(closed_evt.lifetime_ticks) == 3

    # Check persistence in database
    cur = db._sqlite_conn.cursor()
    cur.execute("SELECT is_suspect, lifetime_ticks_json FROM spread_events WHERE event_id = ?", (closed_evt.event_id,))
    row = cur.fetchone()
    assert row is not None
    assert row[0] == 1  # is_suspect = True
    ticks_saved = json.loads(row[1])
    assert len(ticks_saved) == 3
    assert ticks_saved[0]["gross_spread_bps"] == 150.0
    assert ticks_saved[1]["gross_spread_bps"] == 160.0
    assert ticks_saved[2]["gross_spread_bps"] == 20.0
    db.close()





