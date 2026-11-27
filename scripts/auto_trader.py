"""Autonomous Paper Trading Bot for ArbSense.

Continuously monitors live public WebSocket streams from Binance, Coinbase, and Kraken.
When a cross-exchange spread passes the fee-adjusted depth check, it simulates a
paired paper fill against the latest eligible Level 2 books after a configured delay:
- Computes depth-walking realized VWAP
- Uses an explicit order-arrival delay assumption
- Deducts exchange taker fees
- Records executed trades into the database ledger (simulated_trades)
- Updates virtual portfolio capital in real time.
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
from data.feeds.base_feed import feeds_ready
from data.feeds import BinanceFeed, CoinbaseFeed, KrakenFeed
from data.normalizer import OrderBookNormalizer
from data.ntp_sync import clock_sync
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel
from engine.detector import CrossExchangeDetector
from engine.simulator import ExecutionSimulator, arrival_books

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutoTrader")


class AutonomousPaperTrader:
    def __init__(self, fee_scenario: str = "vip3", trade_cooldown_sec: float = 2.0):
        self.config = AppConfig.load()
        self.fee_scenario = fee_scenario
        self.trade_cooldown_sec = trade_cooldown_sec
        self.last_trade_time: dict[str, float] = {}

        if fee_scenario not in self.config.raw["fee_scenarios"]["scenarios"]:
            raise ValueError(f"Unknown fee scenario: {fee_scenario}")
        self.pending: dict[tuple[str, str, str], asyncio.Task] = {}
        self.db = DatabaseManager(sqlite_path=str(PROJECT_ROOT / "data" / f"paper_{fee_scenario}.db"))
        self.db.initialize(sqlite_only=True)

        self.normalizer = OrderBookNormalizer(
            max_quote_age_ms=self.config.time_sync.max_quote_age_ms,
            default_quote=self.config.markets.primary_quote,
        )
        self.cost_model = CostModel(self.config)
        self.detector = CrossExchangeDetector(self.normalizer, self.cost_model, self.config)
        self.simulator = ExecutionSimulator(db_manager=self.db, cost_model=self.cost_model, config=self.config)


        self.trade_count = 0
        self.rejected_count = 0
        self.total_net_pnl = 0.0
        self.feeds = []
        self._running = False

    def ready(self, at_ms):
        return feeds_ready(self.feeds, at_ms, self.config.warmup_seconds)

    def on_book_update(self, exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        snap = self.normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        if not self.ready(loc_ts) or snap.symbol not in self.config.markets.supported_pairs:
            return
        now_ms = clock_sync.event_time_ms()
        for candidate in self.detector.scan_symbol(snap.symbol, now_ms, paper_usdt_only=True):
            route = (candidate.buy_exchange, candidate.sell_exchange, candidate.symbol)
            pair_key = "_".join(route)
            now_sec = time.monotonic()
            if route in self.pending or now_sec - self.last_trade_time.get(pair_key, -float("inf")) < self.trade_cooldown_sec:
                continue
            books = self._fresh_books(route)
            if books is None:
                continue
            buy, sell = books
            buy_rate, sell_rate = self.simulator.taker_rates(buy, sell, self.fee_scenario)
            quantity = 100.0 / buy.best_ask.price
            buy_vwap, _, buy_complete = buy.compute_vwap("buy", quantity)
            sell_vwap, _, sell_complete = sell.compute_vwap("sell", quantity)
            if not buy_complete or not sell_complete:
                self.rejected_count += 1
                continue
            expected_net_bps = 10000 * (sell_vwap * (1 - sell_rate) / buy_vwap - 1 - buy_rate)
            if expected_net_bps <= 2.5:
                self.rejected_count += 1
                continue
            self.last_trade_time[pair_key] = now_sec
            self.pending[route] = asyncio.create_task(self._paper_fill(route, quantity, clock_sync.event_time_ms()))

    def _fresh_books(self, route):
        buy_exchange, sell_exchange, symbol = route
        now_ms = clock_sync.event_time_ms()
        buy = self.normalizer.get_snapshot(buy_exchange, symbol, now_ms)
        sell = self.normalizer.get_snapshot(sell_exchange, symbol, now_ms)
        if buy is None or sell is None or buy.best_ask is None or sell.best_bid is None:
            return None
        if any(book.quote_currency != "USDT" or not 0 <= now_ms - book.local_ts_ms <= 250
               for book in (buy, sell)):
            return None
        if abs(buy.local_ts_ms - sell.local_ts_ms) > 100:
            return None
        return buy, sell

    async def _paper_fill(self, route, quantity, decision_ms):
        try:
            # Explicit stress assumption. Public feed transit measures a different path.
            delay_ms = self.simulator.viability_latency_threshold_ms
            await asyncio.sleep(delay_ms / 1000)
            if not self.ready(decision_ms + delay_ms):
                self.rejected_count += 1
                return
            books = arrival_books(self.normalizer.ring_buffers, route[2], route[0], route[1],
                                  decision_ms + delay_ms)
            if books is None:
                self.rejected_count += 1
                logger.info("Paper attempt had no fresh arrival books for %s", route)
                return
            result = self.simulator.execute_arbitrage(
                route[2], *books, quantity=quantity, fee_scenario=self.fee_scenario,
                arrival_latency_ms=delay_ms,
            )
            if result.status == "FILLED":
                self.trade_count += 1
                self.total_net_pnl += result.net_pnl_usdt
            logger.info("Paper %s %s | net P&L %+.4f USDT", result.status, route, result.net_pnl_usdt)
        except Exception:
            logger.exception("Paper attempt failed for %s", route)
        finally:
            self.pending.pop(route, None)

    async def start(self):
        logger.info("=" * 70)
        logger.info("Starting ArbSense Autonomous Paper Trading Bot")
        logger.info("Fee Scenario: %s | Pre-Funded Balance: $%.2f USDT", self.fee_scenario, self.simulator.initial_capital)
        logger.info("Pairs Tracked: %s", ", ".join(self.config.markets.supported_pairs))
        logger.info("=" * 70)

        await asyncio.to_thread(clock_sync.sync)
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

        for f in self.feeds:
            await f.start()

        self._running = True
        try:
            while self._running:
                await asyncio.sleep(5.0)

                logger.info("[Storage Optimizer] Pruned %d old ticks (disk preserved)", pruned)

                logger.info(
                    "[Auto-Trader Active] Executed: %d trades | Filtered False Positives: %d | Portfolio: $%.2f USDT (Net Realized Alpha: %+.2f USDT)",
                    self.trade_count,
                    self.rejected_count,
                    self.simulator.virtual_balance,
                    self.total_net_pnl,
                )
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        for f in self.feeds:
            await f.stop()
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.db.close()
        logger.info("Auto-Trader stopped.")


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "vip3"
    trader = AutonomousPaperTrader(fee_scenario=scenario)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(trader.start())
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
