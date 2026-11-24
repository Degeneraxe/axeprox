"""Phase 3 Verification Script: Cross-Exchange Spread Detection & Event Tracking.

Streams live market data, executes cross-exchange spread detection across
Binance, Coinbase, and Kraken, applies dynamic volatility-adjusted cost hurdles,
and tracks spread lifecycles to generate labeled ground-truth events.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from data.feeds import BinanceFeed, CoinbaseFeed, KrakenFeed
from data.normalizer import OrderBookNormalizer
from data.ntp_sync import clock_sync
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel
from engine.detector import CrossExchangeDetector
from engine.tracker import ActiveSpreadEvent, SpreadTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3Verifier")


async def run_phase3_verification(duration_sec: int = 15):
    print("=" * 80)
    print("ArbSense Phase 3 Verification: Live Spread Detection & Ground-Truth Labeling")
    print("=" * 80)

    config = AppConfig.load()
    clock_sync.sync()
    db = DatabaseManager(sqlite_path="data/arbsense.db")
    db.initialize()

    normalizer = OrderBookNormalizer(
        max_quote_age_ms=config.time_sync.max_quote_age_ms,
        default_quote=config.markets.primary_quote,
    )
    cost_model = CostModel(config)
    detector = CrossExchangeDetector(normalizer=normalizer, cost_model=cost_model, config=config)

    detected_candidates = []
    labeled_events = []

    def on_event_closed(evt: ActiveSpreadEvent):
        labeled_events.append(evt)

    tracker = SpreadTracker(db_manager=db, config=config, on_event_closed=on_event_closed)

    def on_book_update(exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        snap = normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        now_ms = clock_sync.get_corrected_time_ms()
        # Scan spreads across pairs
        for sym in ["BTC/USDT", "ETH/USDT"]:
            cands = detector.scan_symbol(sym, now_ms)
            for c in cands:
                detected_candidates.append(c)
                tracker.process_candidate(c)
        tracker.check_timeouts(now_ms)

    # Launch feeds
    feeds = [
        BinanceFeed(symbols=["btcusdt", "ethusdt"], on_book_update=on_book_update),
        CoinbaseFeed(symbols=["BTC-USDT", "ETH-USDT", "BTC-USD", "ETH-USD"], on_book_update=on_book_update),
        KrakenFeed(symbols=["XBT/USDT", "ETH/USDT", "XBT/USD", "ETH/USD"], on_book_update=on_book_update),
    ]

    for f in feeds:
        await f.start()

    print(f"[*] Scanning live spreads across Binance, Coinbase, and Kraken for {duration_sec}s...")
    start_t = time.time()
    while time.time() - start_t < duration_sec:
        await asyncio.sleep(2.0)
        elapsed = int(time.time() - start_t)
        print(f"    [Elapsed {elapsed:02d}s] Detected Candidates: {len(detected_candidates)} | Labeled Events: {len(labeled_events)}")

    for f in feeds:
        await f.stop()

    # Flush remaining open events
    tracker.check_timeouts(clock_sync.get_corrected_time_ms() + 5000.0)

    print("\n" + "=" * 80)
    print("EVIDENCE: Sample Real Detected Spread Opportunities")
    print("=" * 80)
    sample_cands = detected_candidates[-5:] if len(detected_candidates) >= 5 else detected_candidates
    for c in sample_cands:
        net_tag = "NET PROFITABLE" if c.is_net_profitable else "SUB-HURDLE"
        print(f"Pair: {c.buy_exchange.upper()} -> {c.sell_exchange.upper()} ({c.symbol})")
        print(f"     Buy Ask: ${c.buy_ask:.2f} | Sell Bid: ${c.sell_bid:.2f}")
        print(f"     Gross Spread: {c.gross_spread_bps:+.2f} bps | Fee Hurdle: {c.fee_bps:.2f} bps | Vol Buffer: {c.vol_buffer_bps:.2f} bps")
        print(f"     Net Spread:   {c.net_spread_bps:+.2f} bps | Evaluation: {net_tag}")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("EVIDENCE: Ground-Truth Labeled Spread Persistence Events")
    print("=" * 80)
    sample_events = tracker.completed_events[-5:] if len(tracker.completed_events) >= 5 else tracker.completed_events
    for evt in sample_events:
        via_tag = "VIABLE" if evt.is_viable else "NON-VIABLE / FLEETING"
        cens_tag = "RIGHT-CENSORED" if evt.is_right_censored else "COMPLETED"
        print(f"Event ID: {evt.event_id} | Route: {evt.buy_exchange.upper()} -> {evt.sell_exchange.upper()} ({evt.symbol})")
        print(f"     Duration: {evt.duration_ms:.1f}ms | Log(Duration): {evt.log_duration:.3f} | Status: {cens_tag}")
        print(f"     Peak Spread: {evt.peak_spread_bps:.2f} bps | Cost Hurdle: {evt.cost_hurdle_bps:.2f} bps")
        print(f"     Ground-Truth Label: [{via_tag}]")
        print("-" * 80)

    db.close()
    print("\nPhase 3 verification complete.")


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    asyncio.run(run_phase3_verification(dur))
