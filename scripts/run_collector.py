"""ArbSense Real-Time Background Market Data Collector Service.

Continuously ingests public L2 order book feeds from Binance, Coinbase, and Kraken,
normalizes microstructural data, and writes to database storage via BatchWriter.

Runs as an independent background service to accumulate real tick data.
"""
from __future__ import annotations

import asyncio
import logging
import signal
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CollectorService")


class MarketDataCollector:
    def __init__(self):
        self.config = AppConfig.load()
        self.db = DatabaseManager(
            sqlite_path="data/arbsense.db",
            migrations_dir="data/storage/migrations",
        )
        self.normalizer = OrderBookNormalizer(
            max_quote_age_ms=self.config.time_sync.max_quote_age_ms,
            default_quote=self.config.markets.primary_quote,
        )
        self.batch_writer = BatchWriter(
            db_manager=self.db,
            batch_size=50,
            flush_interval_sec=0.5,
            depth_throttle_ms=self.config.detection.depth_snapshot_throttle_ms,
        )
        self.tick_counts = {"binance": 0, "coinbase": 0, "kraken": 0}
        self.feeds = []
        self._running = False

    def on_book_update(self, exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        snap = self.normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        self.tick_counts[exchange] = self.tick_counts.get(exchange, 0) + 1
        self.batch_writer.enqueue_snapshot(snap)

    async def start(self):
        logger.info("=" * 70)
        logger.info("Starting ArbSense Continuous Live Data Collector Service")
        logger.info("=" * 70)

        # 1. Sync Clock
        clock_sync.sync()
        logger.info("Clock sync completed: offset=%.2fms via %s", clock_sync.offset_ms, clock_sync.method_used)

        # 2. Init Storage
        backend = self.db.initialize()
        logger.info("Database initialized with backend: %s (%s)", backend, self.db.connection_diagnostic)

        # 3. Start Batch Writer
        await self.batch_writer.start()

        # 4. Init Feeds with full multi-asset coverage
        self.feeds = [
            BinanceFeed(
                symbols=["btcusdt", "ethusdt", "xrpusdt", "adausdt", "solusdt"],
                on_book_update=self.on_book_update,
            ),
            CoinbaseFeed(
                symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT", "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD"],
                on_book_update=self.on_book_update,
            ),
            KrakenFeed(
                symbols=["XBT/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", "XBT/USD", "ETH/USD", "XRP/USD", "ADA/USD", "SOL/USD"],
                on_book_update=self.on_book_update,
            ),
        ]

        for feed in self.feeds:
            await feed.start()

        self._running = True
        logger.info("All feeds online. Collector streaming live market depth to %s...", backend)

        start_time = time.time()
        last_prune_time = time.time()
        # Monitoring loop
        try:
            while self._running:
                await asyncio.sleep(10.0)
                uptime_sec = int(time.time() - start_time)

                # Periodic rolling tick pruner: keep recent 1500 ticks per symbol
                if (time.time() - last_prune_time) >= 60.0:
                    self.db.prune_old_ticks(keep_latest_per_symbol=1500)
                    last_prune_time = time.time()

                db_count = self.db.get_tick_count()
                q_size = self.batch_writer.queue.qsize()
                logger.info(
                    "[Uptime %02dm%02ds] Ingested -> B: %d | C: %d | K: %d | Stored Ticks: %d (Queue: %d)",
                    uptime_sec // 60, uptime_sec % 60,
                    self.tick_counts["binance"], self.tick_counts["coinbase"], self.tick_counts["kraken"],
                    db_count, q_size,
                )
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        logger.info("Stopping feeds and draining storage queues...")
        for feed in self.feeds:
            await feed.stop()
        await self.batch_writer.stop()
        self.db.close()
        logger.info("Collector service shut down cleanly.")


def main():
    collector = MarketDataCollector()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(collector.start())
    except KeyboardInterrupt:
        logger.info("Received interrupt. Exiting...")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
