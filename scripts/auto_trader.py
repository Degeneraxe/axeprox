"""Autonomous Paper Trading Bot for ArbSense.

Continuously monitors live public WebSocket streams from Binance, Coinbase, and Kraken.
When a cross-exchange arbitrage spread meets the dynamic cost hurdle, it automatically
executes a paper trade against the actual live Level 2 order books:
- Computes depth-walking realized VWAP
- Samples empirical transit latency
- Deducts exchange taker fees
- Records executed trades into the database ledger (simulated_trades)
- Updates virtual portfolio capital in real time.
"""
from __future__ import annotations

import asyncio
import logging
import math
import sys
import time
from datetime import datetime, timezone
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
from engine.simulator import ExecutionSimulator
from models.predictor import LiveSpreadPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AutoTrader")


class AutonomousPaperTrader:
    def __init__(self, fee_scenario: str = "vip3", trade_cooldown_sec: float = 2.0, min_ai_confidence: float = 0.50):
        self.config = AppConfig.load()
        self.fee_scenario = fee_scenario
        self.trade_cooldown_sec = trade_cooldown_sec
        self.min_ai_confidence = min_ai_confidence
        self.last_trade_time: dict[str, float] = {}

        self.db = DatabaseManager(sqlite_path="data/arbsense.db")
        self.db.initialize()

        self.normalizer = OrderBookNormalizer(
            max_quote_age_ms=self.config.time_sync.max_quote_age_ms,
            default_quote=self.config.markets.primary_quote,
        )
        self.cost_model = CostModel(self.config)
        self.detector = CrossExchangeDetector(self.normalizer, self.cost_model, self.config)
        self.simulator = ExecutionSimulator(db_manager=self.db, cost_model=self.cost_model, config=self.config)
        self.predictor = LiveSpreadPredictor(checkpoint_path="models/checkpoints/lstm_best.pt")

        self.trade_count = 0
        self.rejected_count = 0
        self.total_net_pnl = 0.0
        self.feeds = []
        self._running = False

    def on_book_update(self, exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        snap = self.normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        now_ms = clock_sync.get_corrected_time_ms()

        # Check for arbitrage execution opportunities across all supported pairs
        for symbol in self.config.markets.supported_pairs:
            candidates = self.detector.scan_symbol(symbol, now_ms)
            for c in candidates:
                pair_key = f"{c.buy_exchange}_{c.sell_exchange}_{c.symbol}"
                now_sec = time.time()
                last_t = self.last_trade_time.get(pair_key, 0.0)

                if (now_sec - last_t) >= self.trade_cooldown_sec:
                    # Fetch live snapshots
                    buy_snap = self.normalizer.get_snapshot(c.buy_exchange, c.symbol, now_ms)
                    sell_snap = self.normalizer.get_snapshot(c.sell_exchange, c.symbol, now_ms)

                    if buy_snap and sell_snap and not buy_snap.is_stale and not sell_snap.is_stale:
                        # AI Inference (PyTorch LSTM Prediction)
                        is_viable, pred_dur_ms, conf = self.predictor.predict_opportunity(
                            symbol=c.symbol,
                            buy_snap=buy_snap,
                            sell_snap=sell_snap,
                            gross_spread_bps=c.gross_spread_bps,
                        )

                        # Calculate venue fees for this specific route
                        # Under zero_maker: high-frequency firms route as Maker/Taker (0 bps maker leg + taker leg)
                        if self.fee_scenario == "zero_maker":
                            fee_buy_bps = self.cost_model.get_fee_bps(c.buy_exchange, role="maker", scenario=self.fee_scenario)
                            fee_sell_bps = self.cost_model.get_fee_bps(c.sell_exchange, role="taker", scenario=self.fee_scenario)
                        else:
                            fee_buy_bps = self.cost_model.get_fee_bps(c.buy_exchange, role="taker", scenario=self.fee_scenario)
                            fee_sell_bps = self.cost_model.get_fee_bps(c.sell_exchange, role="taker", scenario=self.fee_scenario)
                        total_fees_bps = fee_buy_bps + fee_sell_bps
                        projected_net_bps = c.gross_spread_bps - total_fees_bps

                        # STRICT PROFITABILITY GATEKEEPER (Validation Layer):
                        # Filter false-positive opportunities where taker fees + slippage > gross profit!
                        # Requires at least +2.5 bps net margin to account for depth-walking order book slippage.
                        safety_margin_bps = 2.5
                        if is_viable and conf >= self.min_ai_confidence and projected_net_bps > safety_margin_bps:
                            self.last_trade_time[pair_key] = now_sec
                            result = self.simulator.execute_arbitrage(
                                symbol=c.symbol,
                                buy_snap=buy_snap,
                                sell_snap=sell_snap,
                                fee_scenario=self.fee_scenario,
                                opportunity_duration_ms=pred_dur_ms,
                            )

                            if result.status == "FILLED":
                                self.trade_count += 1
                                self.total_net_pnl += result.net_pnl_usdt

                                # Persist spread event to spread_events table for AI Prediction Engine
                                now_iso = datetime.now(timezone.utc).isoformat()
                                self.db.insert_spread_event({
                                    "event_id": f"evt_{result.trade_id[4:]}",
                                    "symbol": c.symbol,
                                    "buy_exchange": c.buy_exchange,
                                    "sell_exchange": c.sell_exchange,
                                    "open_time": now_iso,
                                    "close_time": now_iso,
                                    "peak_spread_bps": float(c.gross_spread_bps),
                                    "open_spread_bps": float(c.gross_spread_bps),
                                    "cost_hurdle_bps": float(total_fees_bps),
                                    "duration_ms": float(pred_dur_ms),
                                    "log_duration": math.log(max(1.0, float(pred_dur_ms))),
                                    "is_viable": 1,
                                    "is_right_censored": 0,
                                })

                                logger.info(
                                    "[PROFITABLE AI TRADE FILLED #%d] %s -> %s (%s) | Gross: +%.2fbps | Fees: %.2fbps | Net: +%.2fbps | Net P&L: %+.2f USDT (Cumulative: %+.2f USDT)",
                                    self.trade_count,
                                    result.buy_exchange.upper(),
                                    result.sell_exchange.upper(),
                                    result.symbol,
                                    c.gross_spread_bps,
                                    total_fees_bps,
                                    projected_net_bps,
                                    result.net_pnl_usdt,
                                    self.total_net_pnl,
                                )
                            elif result.status == "FLEETING":
                                logger.info("[AI FLEETING] Spread closed before fill (%s -> %s)", result.buy_exchange.upper(), result.sell_exchange.upper())
                        elif projected_net_bps <= 0.0:
                            # Validation Layer: Filtered false-positive spread where fees exceed gross profit
                            self.rejected_count += 1
                            if self.rejected_count % 50 == 1:
                                logger.info(
                                    "[GATEKEEPER REJECTED #%d] %s (%s -> %s): Gross +%.2fbps <= Fees %.1fbps | Preserving capital",
                                    self.rejected_count,
                                    c.symbol,
                                    c.buy_exchange.upper(),
                                    c.sell_exchange.upper(),
                                    c.gross_spread_bps,
                                    total_fees_bps,
                                )

    async def start(self):
        logger.info("=" * 70)
        logger.info("Starting ArbSense Autonomous Paper Trading Bot")
        logger.info("Fee Scenario: %s | Pre-Funded Balance: $%.2f USDT", self.fee_scenario, self.simulator.initial_capital)
        logger.info("Pairs Tracked: %s", ", ".join(self.config.markets.supported_pairs))
        logger.info("=" * 70)

        clock_sync.sync()
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
        last_prune_time = time.time()
        try:
            while self._running:
                await asyncio.sleep(5.0)

                # Periodic rolling tick pruner: keep recent 1500 ticks per symbol to prevent disk bloat
                if (time.time() - last_prune_time) >= 60.0:
                    pruned = self.db.prune_old_ticks(keep_latest_per_symbol=1500)
                    last_prune_time = time.time()
                    if pruned > 0:
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
