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

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig, check_run_provenance
from data.feeds.base_feed import feeds_ready
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


async def run_pipeline(collection_duration_sec: int = 15, allow_dirty=False):
    dirty = check_run_provenance(collection_duration_sec, allow_dirty)
    print("=" * 80)
    print("ArbSense End-to-End Pipeline Execution")
    print("=" * 80)

    # 1. Config & Clock Sync
    config = AppConfig.load()
    offset_ms = clock_sync.sync()
    print(f"[*] Clock synchronized: offset = {offset_ms:+.2f} ms via {clock_sync.method_used}")

    # 2. Database
    db = DatabaseManager(sqlite_path="data/arbsense.db")
    backend = db.initialize(sqlite_only=True)
    print(f"[*] Storage initialized: {backend.upper()} ({db.connection_diagnostic})")

    db.record_capture_start(config, dirty, clock_sync)

    # 3. Engines
    normalizer = OrderBookNormalizer(
        max_quote_age_ms=config.time_sync.max_quote_age_ms,
        default_quote=config.markets.primary_quote,
    )
    cost_model = CostModel(config)
    detector = CrossExchangeDetector(normalizer, cost_model, config)
    simulator = ExecutionSimulator(db_manager=db, cost_model=cost_model, config=config)

    executed_trades = []

    def on_arrival(evt, buy_snap, sell_snap):
        result = simulator.execute_arbitrage(evt.symbol, buy_snap, sell_snap, event_id=evt.event_id,
                                             arrival_latency_ms=evt.latency_threshold_ms)
        executed_trades.append(result)

    def ready(at_ms):
        return feeds_ready(feeds, at_ms, config.warmup_seconds)

    tracker = SpreadTracker(ready_at=ready, db_manager=db, config=config, normalizer=normalizer, on_arrival=on_arrival)
    batch_writer = BatchWriter(db_manager=db, batch_size=50, flush_interval_sec=0.5,
                               depth_throttle_ms=config.detection.depth_snapshot_throttle_ms)
    await batch_writer.start()

    accepting_updates = True

    # 4. Ingestion Callbacks
    def on_book_update(exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        if not accepting_updates:
            return
        snap = normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        snap.warmup_complete = ready(loc_ts)
        batch_writer.enqueue_snapshot(snap)

        now_ms = clock_sync.event_time_ms()
        for sym in ["BTC/USDT", "ETH/USDT"]:
            cands = detector.scan_symbol(sym, now_ms, active_pairs=set(tracker.active_events))
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
    start_time = time.monotonic()
    last_prune = start_time
    while time.monotonic() - start_time < collection_duration_sec:
        await asyncio.sleep(3.0)
        if time.monotonic() - last_prune >= 60:
            db.prune_old_ticks(replay_window_hours=config.raw["storage"].get("replay_window_hours", 6))
            last_prune = time.monotonic()
        print(f"    - Elapsed {int(time.monotonic() - start_time)}s | Stored Ticks: {db.get_tick_count()} | Completed Events: {len(tracker.completed_events)} | Trades: {len(executed_trades)}")

    # End the capture before the first disconnection closes the warm-up gate.
    accepting_updates = False
    tracker.finish(clock_sync.event_time_ms())
    for f in feeds:
        await f.stop()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("duration", nargs="?", type=int, default=15)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.duration, args.allow_dirty))
