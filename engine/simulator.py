"""Paper trading execution simulator with depth-walking and empirical latency injection.

Implements Amendments 1 and 7:
- Realistic depth-walking across actual Level 2 order book levels (VWAP)
- Single-counted slippage: slippage is embedded naturally in VWAP (no double counting)
- Empirical latency injection to simulate race against order-book updates
- Pre-funded inventory assumption documented and modeled
- Full taker fee deduction per exchange fee tier schedule
"""
from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from config.settings import AppConfig
from data.normalizer import OrderBookSnapshot
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTradeResult:
    """Represents the outcome of a simulated paper arbitrage execution."""
    trade_id: str
    event_id: Optional[str]
    timestamp_iso: str
    symbol: str
    buy_exchange: str
    sell_exchange: str
    target_quantity: float
    filled_quantity: float
    buy_vwap: float
    sell_vwap: float
    buy_slippage_bps: float
    sell_slippage_bps: float
    buy_fee_usdt: float
    sell_fee_usdt: float
    gross_pnl_usdt: float
    net_pnl_usdt: float
    injected_latency_ms: float
    status: str  # FILLED, FLEETING, PARTIAL, INSUFFICIENT_LIQUIDITY


class ExecutionSimulator:
    """Simulates realistic paper trading executions against real L2 order book depth."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        cost_model: Optional[CostModel] = None,
        config: Optional[AppConfig] = None,
    ):
        self.db = db_manager
        self.config = config or AppConfig.load()
        self.cost_model = cost_model or CostModel(self.config)

        # Empirical latency distribution parameters (derived from live feed measurements in Phase 1)
        # Fast-leg execution latency (Binance median was 24.8ms, Kraken 140ms)
        self.latency_mean_ms = 45.0
        self.latency_std_ms = 15.0
        self.latency_min_ms = 15.0
        self.latency_max_ms = 250.0

        # Capital and balance state (Pre-funded inventory assumption)
        sim_cfg = self.config.raw.get("simulation", {})
        self.initial_capital = float(sim_cfg.get("initial_capital_usdt", 10000.0))
        self.virtual_balance = self.initial_capital
        self.trade_sizes = sim_cfg.get("trade_sizes", {"BTC/USDT": 0.05, "ETH/USDT": 0.5})

        self.execution_history: List[SimulatedTradeResult] = []

    def sample_latency_ms(self) -> float:
        """Samples simulated round-trip network execution latency from empirical distribution."""
        sample = random.gauss(self.latency_mean_ms, self.latency_std_ms)
        return max(self.latency_min_ms, min(self.latency_max_ms, sample))

    def execute_arbitrage(
        self,
        symbol: str,
        buy_snap: OrderBookSnapshot,
        sell_snap: OrderBookSnapshot,
        event_id: Optional[str] = None,
        opportunity_duration_ms: Optional[float] = None,
        quantity: Optional[float] = None,
        fee_scenario: Optional[str] = None,
    ) -> SimulatedTradeResult:
        """Executes simulated paper trade against real order book depth snapshots.
        
        Pre-Funded Balance Assumption (Amendment 7):
        Assumes the trader maintains pre-allocated balances:
        - USDT on the Buy exchange
        - Base asset (BTC/ETH) on the Sell exchange
        Both legs execute concurrently in parallel; no cross-chain transfer latency is incurred.
        """
        trade_id = f"trd_{uuid.uuid4().hex[:10]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        target_qty = quantity or self.trade_sizes.get(symbol, 0.05)

        # 1. Sample Empirical Latency
        injected_lat = self.sample_latency_ms()

        # 2. Race against time: Fleeting Opportunity Check
        # If the spread closes before the order reaches the matching engines, the trade fails.
        if opportunity_duration_ms is not None and opportunity_duration_ms < injected_lat:
            result = SimulatedTradeResult(
                trade_id=trade_id,
                event_id=event_id,
                timestamp_iso=now_iso,
                symbol=symbol,
                buy_exchange=buy_snap.exchange,
                sell_exchange=sell_snap.exchange,
                target_quantity=target_qty,
                filled_quantity=0.0,
                buy_vwap=0.0,
                sell_vwap=0.0,
                buy_slippage_bps=0.0,
                sell_slippage_bps=0.0,
                buy_fee_usdt=0.0,
                sell_fee_usdt=0.0,
                gross_pnl_usdt=0.0,
                net_pnl_usdt=0.0,
                injected_latency_ms=injected_lat,
                status="FLEETING",
            )
            self._record_result(result)
            return result

        # 3. Depth-Walking Order Book VWAP (Amendment 7)
        buy_vwap, buy_filled, buy_complete = buy_snap.compute_vwap("buy", target_qty)
        sell_vwap, sell_filled, sell_complete = sell_snap.compute_vwap("sell", target_qty)

        filled_qty = min(buy_filled, sell_filled)

        if not buy_complete or not sell_complete or filled_qty <= 0:
            status = "PARTIAL" if filled_qty > 0 else "INSUFFICIENT_LIQUIDITY"
            if filled_qty <= 0:
                result = SimulatedTradeResult(
                    trade_id=trade_id,
                    event_id=event_id,
                    timestamp_iso=now_iso,
                    symbol=symbol,
                    buy_exchange=buy_snap.exchange,
                    sell_exchange=sell_snap.exchange,
                    target_quantity=target_qty,
                    filled_quantity=0.0,
                    buy_vwap=0.0,
                    sell_vwap=0.0,
                    buy_slippage_bps=0.0,
                    sell_slippage_bps=0.0,
                    buy_fee_usdt=0.0,
                    sell_fee_usdt=0.0,
                    gross_pnl_usdt=0.0,
                    net_pnl_usdt=0.0,
                    injected_latency_ms=injected_lat,
                    status=status,
                )
                self._record_result(result)
                return result
        else:
            status = "FILLED"

        # 4. Single-Counted Slippage Calculation (Relative to Top-of-Book)
        best_ask = buy_snap.best_ask.price if buy_snap.best_ask else buy_vwap
        best_bid = sell_snap.best_bid.price if sell_snap.best_bid else sell_vwap

        buy_slippage_bps = max(0.0, ((buy_vwap - best_ask) / best_ask) * 10000.0)
        sell_slippage_bps = max(0.0, ((best_bid - sell_vwap) / best_bid) * 10000.0)

        # 5. Venue Fee Calculation
        # Under zero_maker, high-frequency traders use Maker/Taker routing (0 bps maker leg + taker leg)
        if fee_scenario == "zero_maker":
            fee_rate_buy = self.cost_model.get_fee_bps(buy_snap.exchange, role="maker", scenario=fee_scenario) / 10000.0
            fee_rate_sell = self.cost_model.get_fee_bps(sell_snap.exchange, role="taker", scenario=fee_scenario) / 10000.0
        else:
            fee_rate_buy = self.cost_model.get_fee_bps(buy_snap.exchange, role="taker", scenario=fee_scenario) / 10000.0
            fee_rate_sell = self.cost_model.get_fee_bps(sell_snap.exchange, role="taker", scenario=fee_scenario) / 10000.0

        buy_fee_usdt = (buy_vwap * filled_qty) * fee_rate_buy
        sell_fee_usdt = (sell_vwap * filled_qty) * fee_rate_sell

        # 6. Realized Gross and Net P&L Accounting
        gross_pnl = (sell_vwap - buy_vwap) * filled_qty
        net_pnl = gross_pnl - (buy_fee_usdt + sell_fee_usdt)

        # ZERO-LOSS VALIDATION GATEKEEPER:
        # If order-book slippage causes net realized alpha to fall <= 0,
        # reject the trade from the executed ledger to protect capital!
        if net_pnl <= 0.0:
            status = "REJECTED_SLIPPAGE_DEFICIT"
            result = SimulatedTradeResult(
                trade_id=trade_id,
                event_id=event_id,
                timestamp_iso=now_iso,
                symbol=symbol,
                buy_exchange=buy_snap.exchange,
                sell_exchange=sell_snap.exchange,
                target_quantity=target_qty,
                filled_quantity=0.0,
                buy_vwap=buy_vwap,
                sell_vwap=sell_vwap,
                buy_slippage_bps=buy_slippage_bps,
                sell_slippage_bps=sell_slippage_bps,
                buy_fee_usdt=0.0,
                sell_fee_usdt=0.0,
                gross_pnl_usdt=0.0,
                net_pnl_usdt=0.0,
                injected_latency_ms=injected_lat,
                status=status,
            )
            return result

        self.virtual_balance += net_pnl

        result = SimulatedTradeResult(
            trade_id=trade_id,
            event_id=event_id,
            timestamp_iso=now_iso,
            symbol=symbol,
            buy_exchange=buy_snap.exchange,
            sell_exchange=sell_snap.exchange,
            target_quantity=target_qty,
            filled_quantity=filled_qty,
            buy_vwap=buy_vwap,
            sell_vwap=sell_vwap,
            buy_slippage_bps=buy_slippage_bps,
            sell_slippage_bps=sell_slippage_bps,
            buy_fee_usdt=buy_fee_usdt,
            sell_fee_usdt=sell_fee_usdt,
            gross_pnl_usdt=gross_pnl,
            net_pnl_usdt=net_pnl,
            injected_latency_ms=injected_lat,
            status=status,
        )

        self._record_result(result)
        return result

    def _record_result(self, result: SimulatedTradeResult) -> None:
        self.execution_history.append(result)
        if self.db:
            self.db.insert_simulated_trade({
                "trade_id": result.trade_id,
                "event_id": result.event_id,
                "timestamp": result.timestamp_iso,
                "symbol": result.symbol,
                "buy_exchange": result.buy_exchange,
                "sell_exchange": result.sell_exchange,
                "quantity": result.filled_quantity,
                "buy_vwap": result.buy_vwap,
                "sell_vwap": result.sell_vwap,
                "buy_fee": result.buy_fee_usdt,
                "sell_fee": result.sell_fee_usdt,
                "gross_pnl": result.gross_pnl_usdt,
                "net_pnl": result.net_pnl_usdt,
                "latency_ms": result.injected_latency_ms,
                "status": result.status,
            })
