"""End-to-end execution pipeline for ArbSense.

Orchestrates:
1. System clock offset synchronization (NTP)
2. Database initialization and migrations
3. Live multi-exchange data collection across Binance, Coinbase, and Kraken
4. Cross-exchange spread detection and dynamic volatility hurdle evaluation
5. Spread persistence tracking and ground-truth duration/viability labeling
6. Paper execution simulation with depth-walking and empirical latency injection
7. Model training and benchmark evaluation (PyTorch LSTM vs Baseline)
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
from data.storage import BatchWriter, DatabaseManager
from engine.cost_model import CostModel
from engine.detector import CrossExchangeDetector
from engine.simulator import ExecutionSimulator
from engine.tracker import ActiveSpreadEvent, SpreadTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")


async def run_pipeline(collection_duration_sec: int = 15):
    print("=" * 80)
    print("ArbSense End-to-End Pipeline Execution")
    print("=" * 80)

    # 1. Config & Clock Sync
    config = AppConfig.load()
    offset_ms = clock_sync.sync()
    print(f"[*] Clock synchronized: offset = {offset_ms:+.2f} ms via {clock_sync.method_used}")

    # 2. Database
    db = DatabaseManager(sqlite_path="data/arbsense.db")
    backend = db.initialize()
    print(f"[*] Storage initialized: {backend.upper()} ({db.connection_diagnostic})")

    # 3. Engines
    normalizer = OrderBookNormalizer(
        max_quote_age_ms=config.time_sync.max_quote_age_ms,
        default_quote=config.markets.primary_quote,
    )
    cost_model = CostModel(config)
    detector = CrossExchangeDetector(normalizer, cost_model, config)
    simulator = ExecutionSimulator(db_manager=db, cost_model=cost_model, config=config)

    executed_trades = []

    def on_spread_closed(evt: ActiveSpreadEvent):
        # Execute paper trade if opportunity is candidate
        buy_snap = normalizer.get_snapshot(evt.buy_exchange, evt.symbol, clock_sync.get_corrected_time_ms())
        sell_snap = normalizer.get_snapshot(evt.sell_exchange, evt.symbol, clock_sync.get_corrected_time_ms())
        if buy_snap and sell_snap and not buy_snap.is_stale and not sell_snap.is_stale:
            res = simulator.execute_arbitrage(
                symbol=evt.symbol,
                buy_snap=buy_snap,
                sell_snap=sell_snap,
                event_id=evt.event_id,
                opportunity_duration_ms=evt.duration_ms,
            )
            executed_trades.append(res)

    tracker = SpreadTracker(db_manager=db, config=config, on_event_closed=on_spread_closed)
    batch_writer = BatchWriter(db_manager=db, batch_size=50, flush_interval_sec=0.5)
    await batch_writer.start()

    # 4. Ingestion Callbacks
    def on_book_update(exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        snap = normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        batch_writer.enqueue_snapshot(snap)

        now_ms = clock_sync.get_corrected_time_ms()
        for sym in ["BTC/USDT", "ETH/USDT"]:
            cands = detector.scan_symbol(sym, now_ms)
            for c in cands:
                tracker.process_candidate(c)
        tracker.check_timeouts(now_ms)

    # 5. Launch Feeds
    feeds = [
        BinanceFeed(symbols=["btcusdt", "ethusdt"], on_book_update=on_book_update),
        CoinbaseFeed(symbols=["BTC-USDT", "ETH-USDT", "BTC-USD", "ETH-USD"], on_book_update=on_book_update),
        KrakenFeed(symbols=["XBT/USDT", "ETH/USDT", "XBT/USD", "ETH/USD"], on_book_update=on_book_update),
    ]

    for f in feeds:
        await f.start()

    print(f"\n[*] Streaming live market ticks for {collection_duration_sec} seconds...")
    start_time = time.time()
    while time.time() - start_time < collection_duration_sec:
        await asyncio.sleep(3.0)
        print(f"    - Elapsed {int(time.time() - start_time)}s | Stored Ticks: {db.get_tick_count()} | Completed Events: {len(tracker.completed_events)} | Trades: {len(executed_trades)}")

    # Stop Feeds
    for f in feeds:
        await f.stop()
    tracker.check_timeouts(clock_sync.get_corrected_time_ms() + 5000.0)
    await batch_writer.stop()

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Total Ticks Stored in Database:   {db.get_tick_count():,}")
    print(f"Total Spread Windows Tracked:     {len(tracker.completed_events)}")
    print(f"Simulated Paper Executions:       {len(executed_trades)}")
    print(f"Simulator Net P&L:                ${simulator.virtual_balance - simulator.initial_capital:+.2f} USDT")

    db.close()
    print("\nPipeline execution complete.")


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    asyncio.run(run_pipeline(dur))
