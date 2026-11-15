"""ArbSense Unified Live Ingestion, Detection, and Event Tracking Service.

Task 2: One unified, robust long-running service that:
1. Ingests all 3 exchange feeds (Binance, Coinbase, Kraken) in real-time.
2. Normalizes Level 2 order books and maintains in-memory ring buffers.
3. Dynamically synchronizes USDT/USD basis from live market depth.
4. Continuously runs CrossExchangeDetector (with minimum depth liquidity filters)
   and SpreadTracker.
5. Persists detected spread events at open with complete 20-step feature windows,
   rolling volatility, and fixed dataset latency thresholds.
6. Periodically checks event timeouts and right-censoring.
7. Periodically prunes old raw ticks (keeping latest 1500 per symbol) while permanently
   retaining all spread events and feature windows.
8. Logs a clean, comprehensive status line every 60 seconds.
9. Operates with auto-reconnect and graceful shutdown (SIGINT / SIGTERM).
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from data.feeds import BinanceFeed, CoinbaseFeed, KrakenFeed
from data.normalizer import OrderBookNormalizer, OrderBookSnapshot
from data.ntp_sync import clock_sync
from data.storage import BatchWriter, DatabaseManager
from engine.cost_model import CostModel
from engine.detector import CrossExchangeDetector
from engine.tracker import SpreadTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveCollectorTracker")


class UnifiedLiveService:
    def __init__(self, sqlite_path: str = "data/arbsense.db"):
        self.config = AppConfig.load()
        self.db = DatabaseManager(
            sqlite_path=sqlite_path,
            migrations_dir="data/storage/migrations",
        )
        self.normalizer = OrderBookNormalizer(
            max_quote_age_ms=self.config.time_sync.max_quote_age_ms,
            default_quote=self.config.markets.primary_quote,
        )
        st_cfg = self.config.raw.get("storage", {})
        self.batch_writer = BatchWriter(
            db_manager=self.db,
            batch_size=int(st_cfg.get("batch_size", 50)),
            flush_interval_sec=float(st_cfg.get("flush_interval_ms", 500)) / 1000.0,
            depth_throttle_ms=self.config.detection.depth_snapshot_throttle_ms,
        )
        self.cost_model = CostModel(self.config)
        self.detector = CrossExchangeDetector(
            normalizer=self.normalizer,
            cost_model=self.cost_model,
            config=self.config,
        )
        self.tracker = SpreadTracker(
            db_manager=self.db,
            config=self.config,
            normalizer=self.normalizer,
        )

        self.cumulative_ticks = 0
        self.events_opened_session = 0
        self.events_closed_session = 0
        self.tick_counts = {"binance": 0, "coinbase": 0, "kraken": 0}
        self.feeds: List[Any] = []
        self._running = False

    def on_book_update(
        self,
        exchange: str,
        raw_symbol: str,
        bids: list,
        asks: list,
        exch_ts: float,
        loc_ts: float,
    ) -> None:
        """Callback executed on every incoming order book depth update."""
        snap = self.normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        self.cumulative_ticks += 1
        self.tick_counts[exchange] = self.tick_counts.get(exchange, 0) + 1
        self.batch_writer.enqueue_snapshot(snap)

        # Dynamic USDT/USD basis synchronization from Kraken USDT/USD ticker
        if exchange.lower() == "kraken" and snap.symbol == "USDT/USD" and snap.mid_price > 0:
            self.normalizer.set_usdt_usd_basis(snap.mid_price)

        # Trigger detection for this symbol if it's one of our supported pairs
        base_symbol = snap.symbol.replace("/USD", "/USDT")
        if base_symbol in self.config.markets.supported_pairs:
            candidates = self.detector.scan_symbol(base_symbol, loc_ts)
            for cand in candidates:
                evt = self.tracker.process_candidate(cand)
                if evt and evt.tick_count == 1:
                    self.events_opened_session += 1

    async def start(self) -> None:
        logger.info("=" * 80)
        logger.info("ArbSense Unified Live Ingestion, Detection & Spread Tracking Engine")
        logger.info("=" * 80)

        # 1. Sync Clock via NTP / REST fallback
        clock_sync.sync()
        logger.info(
            "Clock sync verified: offset=%.2fms via %s (transit latency correction active)",
            clock_sync.offset_ms,
            clock_sync.method_used,
        )

        # 2. Init Storage Layer
        backend = self.db.initialize()
        logger.info("Database initialized: backend=%s (%s)", backend, self.db.connection_diagnostic)

        # 3. Start High-Throughput Batch Writer
        await self.batch_writer.start()

        # 4. Initialize Multi-Exchange Feeds
        binance_symbols = ["btcusdt", "ethusdt", "xrpusdt", "adausdt", "solusdt", "dogeusdt"]
        coinbase_symbols = [
            "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD",
            "BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT",
        ]
        kraken_symbols = [
            "XBT/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", "DOGE/USDT",
            "XBT/USD", "ETH/USD", "XRP/USD", "ADA/USD", "SOL/USD", "USDT/USD",
        ]

        self.feeds = [
            BinanceFeed(symbols=binance_symbols, on_book_update=self.on_book_update),
            CoinbaseFeed(symbols=coinbase_symbols, on_book_update=self.on_book_update),
            KrakenFeed(symbols=kraken_symbols, on_book_update=self.on_book_update),
        ]

        for feed in self.feeds:
            await feed.start()

        self._running = True
        logger.info("All feeds streaming live. Real-time arbitrage detection & tracking online.")

        start_time = time.time()
        last_log_time = time.time()
        last_prune_time = time.time()

        try:
            while self._running:
                await asyncio.sleep(0.5)
                now_ms = clock_sync.get_corrected_time_ms()

                # Periodically check timeouts for active spread windows (1500ms inactivity / 30s right-censoring)
                closed_events = self.tracker.check_timeouts(now_ms)
                if closed_events:
                    self.events_closed_session += len(closed_events)

                now_sec = time.time()

                # 60-Second Periodic Tick Pruning (raw ticks only; events/windows permanently preserved)
                if (now_sec - last_prune_time) >= 60.0:
                    pruned = self.db.prune_old_ticks(keep_latest_per_symbol=1500)
                    last_prune_time = now_sec

                # 60-Second Periodic Status Line Logging (Task 2 Requirement)
                if (now_sec - last_log_time) >= 60.0:
                    uptime_sec = int(now_sec - start_time)
                    reconnect_count = sum(getattr(f, "reconnect_count", 0) for f in self.feeds)

                    # Query database stats in place
                    cur = self.db._sqlite_conn.cursor() if self.db._sqlite_conn else None
                    if cur:
                        cur.execute("""
                            SELECT 
                                COUNT(*),
                                SUM(feature_window_json IS NOT NULL),
                                SUM(is_viable = 1),
                                SUM(is_right_censored = 1)
                            FROM spread_events
                        """)
                        row = cur.fetchone()
                        tot_ev = row[0] or 0
                        with_win = row[1] or 0
                        viable_cnt = row[2] or 0
                        censored_cnt = row[3] or 0
                    else:
                        tot_ev = len(self.tracker.completed_events) + len(self.tracker.active_events)
                        with_win = sum(1 for e in self.tracker.completed_events if e.feature_window)
                        viable_cnt = sum(1 for e in self.tracker.completed_events if e.is_viable)
                        censored_cnt = sum(1 for e in self.tracker.completed_events if e.is_right_censored)

                    cumul_ticks = self.db.get_cumulative_tick_count() if self.db else self.cumulative_ticks
                    active_buffer = self.db.get_tick_count() if self.db else 0

                    logger.info(
                        "[STATUS 60s] Uptime %02dm%02ds | Ticks Ingested: %d (Active Buffer: %d) | "
                        "Events Opened: %d, Closed: %d | Total DB Events: %d (With Windows: %d) | "
                        "Viable: %d, Censored: %d | Reconnects: %d | Clock Offset: %.2fms",
                        uptime_sec // 60,
                        uptime_sec % 60,
                        cumul_ticks,
                        active_buffer,
                        self.events_opened_session,
                        self.events_closed_session,
                        tot_ev,
                        with_win,
                        viable_cnt,
                        censored_cnt,
                        reconnect_count,
                        clock_sync.offset_ms,
                    )
                    last_log_time = now_sec

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        logger.info("Initiating graceful shutdown of feeds and storage queues...")
        for feed in self.feeds:
            await feed.stop()
        await self.batch_writer.stop()
        self.db.close()
        logger.info("Unified Live Service stopped cleanly.")


def main():
    service = UnifiedLiveService(sqlite_path="data/arbsense.db")

    def _handle_signal(*_):
        logger.info("Signal received. Triggering graceful shutdown...")
        service._running = False

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass

    asyncio.run(service.start())


if __name__ == "__main__":
    main()
