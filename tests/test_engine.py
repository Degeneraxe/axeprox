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


