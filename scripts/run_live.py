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
7. Periodically prunes old raw ticks outside the replay window while permanently
   retaining all spread events and feature windows.
8. Logs a clean, comprehensive status line every 60 seconds.
9. Operates with auto-reconnect and graceful shutdown (SIGINT / SIGTERM).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig, check_run_provenance
from data.feeds.base_feed import feeds_ready
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
            ready_at=self.ready,
            db_manager=self.db,
            config=self.config,
            normalizer=self.normalizer,
        )

        self.accepting_updates = True
        self.cumulative_ticks = 0
        self.events_opened_session = 0
        self.events_closed_session = 0
        self.basis_poll_failures = 0
        self.tick_counts = {"binance": 0, "coinbase": 0, "kraken": 0}
        self.feeds: List[Any] = []
        self._running = False

    def ready(self, at_ms):
        return feeds_ready(self.feeds, at_ms, self.config.warmup_seconds)

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
        if not self.accepting_updates:
            return
        snap = self.normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        self.cumulative_ticks += 1
        self.tick_counts[exchange] = self.tick_counts.get(exchange, 0) + 1
        snap.warmup_complete = self.ready(loc_ts)
        self.batch_writer.enqueue_snapshot(snap)
        self.tracker.observe_arrivals(loc_ts)

        # Dynamic USDT/USD basis synchronization from Kraken or Coinbase USDT/USD ticker (Fix A)
        if snap.symbol == "USDT/USD" and snap.mid_price > 0:
            self.normalizer.set_usdt_usd_basis(snap.mid_price)

        # Trigger detection for this symbol if it's one of our supported pairs
        base_symbol = snap.symbol.removesuffix("/USD") + "/USDT" if snap.symbol.endswith("/USD") else snap.symbol
        if base_symbol in self.config.markets.supported_pairs:
            active_pairs = set(self.tracker.active_events.keys())
            candidates = self.detector.scan_symbol(base_symbol, loc_ts, active_pairs=active_pairs)
            for cand in candidates:
                evt = self.tracker.process_candidate(cand)
                if evt:
                    if evt.is_closed:
                        self.events_closed_session += 1
                    elif evt.tick_count == 1:
                        self.events_opened_session += 1

    async def _periodic_basis_updater(self) -> None:
        """Periodically polls REST USDT/USD rate as fallback and logs current basis rate (Fix A & Requirement 1)."""
        import urllib.request
        import json
        def fetch_rates():
            rates = []
            try:
                req_cb = urllib.request.Request(
                    "https://api.exchange.coinbase.com/products/USDT-USD/ticker",
                    headers={"User-Agent": "ArbSenseBasis/1.0"}
                )
                with urllib.request.urlopen(req_cb, timeout=3.0) as resp:
                    d = json.loads(resp.read().decode())
                    if "bid" in d and "ask" in d:
                        rates.append((float(d["bid"]) + float(d["ask"])) / 2.0)
            except Exception as e:
                self.basis_poll_failures += 1
                logger.warning("[BasisUpdater] Failed to poll Coinbase USDT-USD ticker: %s (failure count: %d)", e, self.basis_poll_failures)

            try:
                req_kr = urllib.request.Request(
                    "https://api.kraken.com/0/public/Ticker?pair=USDTUSD",
                    headers={"User-Agent": "ArbSenseBasis/1.0"}
                )
                with urllib.request.urlopen(req_kr, timeout=3.0) as resp:
                    d = json.loads(resp.read().decode())
                    res = d.get("result", {})
                    k = list(res.keys())[0] if res else None
                    if k:
                        rates.append((float(res[k]["b"][0]) + float(res[k]["a"][0])) / 2.0)
            except Exception as e:
                self.basis_poll_failures += 1
                logger.warning("[BasisUpdater] Failed to poll Kraken USDTUSD ticker: %s (failure count: %d)", e, self.basis_poll_failures)
            return rates

        while self._running:
            try:
                rates = await asyncio.to_thread(fetch_rates)
                if rates:
                    mean_rate = float(sum(rates) / len(rates))
                    self.normalizer.set_usdt_usd_basis(mean_rate)
                    bps_offset = (1.0 / mean_rate - 1.0) * 10000.0
                    logger.info("[BasisUpdater] Active USDT/USD rate: %.5f (%+.2f bps offset, poll failures: %d)", mean_rate, bps_offset, self.basis_poll_failures)
                else:
                    self.basis_poll_failures += 1
                    logger.warning(
                        "[BasisUpdater] Failed to fetch USDT/USD basis from all venues (total failures: %d, current basis age: %.1fs)",
                        self.basis_poll_failures,
                        self.normalizer.get_basis_age_sec(),
                    )
            except Exception as e:
                self.basis_poll_failures += 1
                logger.warning("[BasisUpdater] Unexpected poll cycle exception: %s (total failures: %d)", e, self.basis_poll_failures)
            await asyncio.sleep(60.0)

    async def start(self, duration_sec: Optional[int] = None, allow_dirty: bool = False) -> None:
        dirty = check_run_provenance(duration_sec, allow_dirty)
        logger.info("=" * 80)
        logger.info("ArbSense Unified Live Ingestion, Detection & Spread Tracking Engine")
        if duration_sec:
            logger.info("Configured Run Duration: %d seconds (%.1f minutes)", duration_sec, duration_sec / 60.0)
        logger.info("=" * 80)

        # 1. Sync Clock via NTP / REST fallback (non-blocking in thread)
        await asyncio.to_thread(clock_sync.sync)
        logger.info(
            "Clock reference: offset=%.2fms via %s",
            clock_sync.offset_ms,
            clock_sync.method_used,
        )

        # 2. Init Storage Layer
        backend = self.db.initialize(sqlite_only=True)
        logger.info("Database initialized: backend=%s (%s)", backend, self.db.connection_diagnostic)

        dataset_ver = self.config.raw["dataset"]["version"]
        self.db.record_capture_start(self.config, dirty, clock_sync)

        # 4. Start High-Throughput Batch Writer
        await self.batch_writer.start()

        mapping = self.config.raw["markets"]["exchange_symbols"]
        pairs = self.config.markets.supported_pairs
        def subscriptions(venue, extras=()):
            symbols = [mapping[venue][pair] for pair in pairs if pair in mapping[venue]]
            return list(dict.fromkeys([*symbols, *extras]))

        binance_symbols = subscriptions("binance")
        coinbase_symbols = subscriptions("coinbase", ["USDT-USD"])
        kraken_symbols = subscriptions("kraken", ["XBT/USD", "ETH/USD", "USDT/USD"])

        self.feeds = [
            BinanceFeed(symbols=binance_symbols, on_book_update=self.on_book_update),
            CoinbaseFeed(symbols=coinbase_symbols, on_book_update=self.on_book_update),
            KrakenFeed(symbols=kraken_symbols, on_book_update=self.on_book_update),
        ]

        for feed in self.feeds:
            await feed.start()

        # Initial basis fetch and periodic polling (Fix A)
        asyncio.create_task(self._periodic_basis_updater())

        self._running = True
        logger.info("All feeds streaming live. Real-time arbitrage detection & tracking online.")

        start_time = time.monotonic()
        last_log_time = time.monotonic()
        last_prune_time = time.monotonic()
        last_ntp_sync_time = time.monotonic()

        try:
            while self._running:
                if duration_sec is not None and (time.monotonic() - start_time) >= duration_sec:
                    logger.info("Target duration of %ds reached. Stopping live session cleanly...", duration_sec)
                    break

                await asyncio.sleep(0.5)
                now_ms = clock_sync.event_time_ms()

                # Periodically check timeouts for active spread windows (inactivity / right-censoring)
                closed_events = self.tracker.check_timeouts(now_ms)
                if closed_events:
                    self.events_closed_session += len(closed_events)

                now_sec = time.monotonic()

                # 5-Minute (300s) Periodic NTP Re-sync & Offset Recording (Item 4 & Non-blocking)
                if (now_sec - last_ntp_sync_time) >= 300.0:
                    await asyncio.to_thread(clock_sync.sync)
                    self.db.record_clock_offset(**clock_sync.measurement())
                    last_ntp_sync_time = now_sec

                # 60-Second Periodic Tick Pruning (raw ticks only; events/windows permanently preserved)
                if (now_sec - last_prune_time) >= 60.0:
                    self.db.prune_old_ticks(replay_window_hours=self.config.raw["storage"].get("replay_window_hours", 6))
                    last_prune_time = now_sec

                # 60-Second Periodic Status Line Logging (Task 2 & 10 Requirement)
                if (now_sec - last_log_time) >= 60.0:
                    uptime_sec = int(now_sec - start_time)

                    # Query database stats in place
                    cur = self.db._sqlite_conn.cursor() if self.db._sqlite_conn else None
                    if cur:
                        cur.execute("""
                            SELECT 
                                COUNT(*),
                                SUM(feature_window_json IS NOT NULL),
                                SUM(is_viable = 1),
                                SUM(is_right_censored = 1),
                                SUM(is_executable = 1),
                                SUM(is_persistent_500ms = 1),
                                SUM(is_persistent_2000ms = 1),
                                SUM(is_persistent_5000ms = 1)
                            FROM spread_events
                            WHERE dataset_version = ?
                        """, (dataset_ver,))
                        row = cur.fetchone()
                        tot_ev = row[0] or 0
                        with_win = row[1] or 0
                        viable_cnt = row[2] or 0
                        censored_cnt = row[3] or 0
                        executable_cnt = row[4] or 0
                        p500_cnt = row[5] or 0
                        p2000_cnt = row[6] or 0
                        p5000_cnt = row[7] or 0
                    else:
                        tot_ev = len(self.tracker.completed_events) + len(self.tracker.active_events)
                        with_win = sum(1 for e in self.tracker.completed_events if e.feature_window)
                        viable_cnt = sum(1 for e in self.tracker.completed_events if e.is_viable)
                        censored_cnt = sum(1 for e in self.tracker.completed_events if e.is_right_censored)
                        executable_cnt = sum(1 for e in self.tracker.completed_events if getattr(e, "is_executable", 0))
                        p500_cnt = sum(1 for e in self.tracker.completed_events if getattr(e, "is_persistent_500ms", 0))
                        p2000_cnt = sum(1 for e in self.tracker.completed_events if getattr(e, "is_persistent_2000ms", 0))
                        p5000_cnt = sum(1 for e in self.tracker.completed_events if getattr(e, "is_persistent_5000ms", 0))

                    cumul_ticks = self.db.get_cumulative_tick_count() if self.db else self.cumulative_ticks
                    active_buffer = self.db.get_tick_count() if self.db else 0
                    cur_basis = getattr(self.normalizer, "usdt_usd_basis_rate", 1.0)
                    basis_age = self.normalizer.get_basis_age_sec() if hasattr(self.normalizer, "get_basis_age_sec") else 0.0

                    # Reconnects per exchange
                    reconn_details = []
                    kraken_feed_obj = None
                    coinbase_feed_obj = None
                    for feed in self.feeds:
                        exch = getattr(feed, "exchange_name", feed.__class__.__name__.replace("Feed", "")).upper()
                        reconn_cnt = getattr(feed, "reconnect_count", 0)
                        reconn_details.append(f"{exch}:{reconn_cnt}")
                        if "KRAKEN" in exch:
                            kraken_feed_obj = feed
                        elif "COINBASE" in exch:
                            coinbase_feed_obj = feed

                    # Kraken checksum failure rate
                    kr_fails = getattr(kraken_feed_obj, "checksum_failure_count", 0) if kraken_feed_obj else 0
                    kr_checks = getattr(kraken_feed_obj, "checksum_checks_count", 0) if kraken_feed_obj else 0
                    kr_fail_rate_str = f"{(kr_fails / max(1, kr_checks)) * 100.0:.2f}% ({kr_fails}/{kr_checks})"

                    # Coinbase resync count
                    cb_resync_cnt = getattr(coinbase_feed_obj, "crossed_book_count", 0) if coinbase_feed_obj else 0

                    logger.info(
                        "[STATUS 60s] Uptime %02dm%02ds | Ticks: %d (Buf: %d) | "
                        "Events: %d opened / %d closed (DB Total: %d, With Windows: %d, Censored: %d) | "
                        "Persistence: [500ms: %d, 2s: %d, 5s: %d] | "
                        "Reconnects: [%s] | Kraken CRC Fail Rate: %s | CB Resyncs: %d | "
                        "Basis: %.5f (Age: %.1fs) | ClockOffset: %+.2fms",
                        uptime_sec // 60,
                        uptime_sec % 60,
                        cumul_ticks,
                        active_buffer,
                        self.events_opened_session,
                        self.events_closed_session,
                        tot_ev,
                        with_win,
                        censored_cnt,
                        p500_cnt,
                        p2000_cnt,
                        p5000_cnt,
                        ", ".join(reconn_details),
                        kr_fail_rate_str,
                        cb_resync_cnt,
                        cur_basis,
                        basis_age,
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
        self.accepting_updates = False
        self.tracker.finish(clock_sync.event_time_ms())
        for feed in self.feeds:
            await feed.stop()
        await self.batch_writer.stop()
        self.db.close()
        logger.info("Unified Live Service stopped cleanly.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("duration", type=int, nargs="?")
    parser.add_argument("database", nargs="?", default="data/arbsense.db")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    service = UnifiedLiveService(sqlite_path=args.database)

    def _handle_signal(*_):
        logger.info("Signal received. Triggering graceful shutdown...")
        service._running = False

    try:
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
    except Exception:
        pass

    asyncio.run(service.start(duration_sec=args.duration, allow_dirty=args.allow_dirty))


if __name__ == "__main__":
    main()
