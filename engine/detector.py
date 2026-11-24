"""Cross-exchange spread detection and anomaly evaluation engine.

Implements Amendments 1, 2, and 4:
- Directional spread matrix computation across all exchange pairs (Binance, Coinbase, Kraken)
- Enforces strict stale quote filtering on both legs
- Explicit quote currency basis handling
- Evaluates low gross spread detection and calculates net viability per fee tier
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config.settings import AppConfig
from data.normalizer import OrderBookNormalizer, OrderBookSnapshot
from engine.cost_model import CostModel

logger = logging.getLogger(__name__)


@dataclass
class SpreadCandidate:
    """Represents a directional cross-exchange arbitrage spread."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_ask: float
    sell_bid: float
    gross_spread_bps: float
    net_spread_bps: float
    total_hurdle_bps: float
    fee_bps: float
    vol_buffer_bps: float
    is_gross_candidate: bool
    is_net_profitable: bool
    timestamp_ms: float
    quote_age_buy_ms: float
    quote_age_sell_ms: float
    native_usdt_route: bool = True


class CrossExchangeDetector:
    """Detects spatial arbitrage opportunities across synchronized order books."""

    def __init__(
        self,
        normalizer: OrderBookNormalizer,
        cost_model: Optional[CostModel] = None,
        config: Optional[AppConfig] = None,
    ):
        self.normalizer = normalizer
        self.config = config or AppConfig.load()
        self.cost_model = cost_model or CostModel(self.config)
        self.gross_threshold_bps = float(self.config.detection.gross_spread_threshold_bps)
        self.exchanges = ["binance", "coinbase", "kraken"]

    def scan_symbol(
        self,
        symbol: str,
        current_time_ms: float,
        active_pairs: Optional[set] = None,
        *,
        paper_usdt_only: bool = False,
    ) -> List[SpreadCandidate]:
        """Scans all directed exchange pairs for a symbol and returns detected spreads.
        
        If active_pairs is provided, ongoing active events for this symbol are always
        returned so the tracker receives every tick during their lifetime.
        """
        candidates: List[SpreadCandidate] = []

        # Retrieve snapshots for all venues
        snaps: Dict[str, OrderBookSnapshot] = {}
        for exch in self.exchanges:
            s = self.normalizer.get_snapshot(exch, symbol, current_time_ms)
            if (not s or s.is_stale) and not paper_usdt_only:
                # Fallback to USD quote if USDT is not directly present or is stale
                s_usd = self.normalizer.get_snapshot(exch, symbol.replace("USDT", "USD"), current_time_ms)
                if s_usd and not s_usd.is_stale:
                    s = s_usd
            if s and not s.is_stale and s.best_bid and s.best_ask:
                snaps[exch] = s
                # Update volatility tracker with mid price
                self.cost_model.update_price(symbol, s.mid_price)

        # Iterate all directed pairs (A -> B: Buy on A, Sell on B)
        for buy_exch in self.exchanges:
            for sell_exch in self.exchanges:
                if buy_exch == sell_exch:
                    continue
                if buy_exch not in snaps or sell_exch not in snaps:
                    continue

                snap_buy = snaps[buy_exch]
                snap_sell = snaps[sell_exch]

                # Stale quote check (Amendment 2)
                if snap_buy.is_stale or snap_sell.is_stale:
                    continue

                p_buy_ask = snap_buy.best_ask.price
                p_sell_bid = snap_sell.best_bid.price

                if p_buy_ask <= 0:
                    continue

                # Minimum depth liquidity filter (Task 3):
                # Top-N-levels notional/quantity must exceed configured trade size for both legs
                det_cfg = self.config.raw.get("detection", {})
                if not paper_usdt_only and det_cfg.get("min_depth_filter_enabled", True):
                    top_n = int(det_cfg.get("min_depth_levels", 5))
                    trade_sizes = self.config.raw.get("simulation", {}).get("trade_sizes", {})
                    min_qty = float(trade_sizes.get(symbol, 0.0))
                    if min_qty > 0:
                        buy_qty = snap_buy.compute_top_quantity("buy", depth=top_n)
                        sell_qty = snap_sell.compute_top_quantity("sell", depth=top_n)
                        buy_notional = snap_buy.compute_top_notional("buy", depth=top_n)
                        sell_notional = snap_sell.compute_top_notional("sell", depth=top_n)
                        min_buy_notional = min_qty * p_buy_ask
                        min_sell_notional = min_qty * p_sell_bid

                        if buy_qty < min_qty or sell_qty < min_qty or buy_notional < min_buy_notional or sell_notional < min_sell_notional:
                            # Discard illiquid/thin candidate
                            continue

                # Basis staleness guard (Item 1): flag Coinbase books unusable if basis > 5 min stale
                if not paper_usdt_only and (buy_exch == "coinbase" or sell_exch == "coinbase"):
                    if hasattr(self.normalizer, "is_basis_stale") and self.normalizer.is_basis_stale(300.0):
                        if (current_time_ms - getattr(self, "_last_stale_warn_ms", 0.0)) >= 30000.0:
                            logger.warning(
                                "[BasisStalenessGuard] USDT/USD basis is stale (>300s). Flagging Coinbase books as unusable; skipping candidate %s %s->%s.",
                                symbol, buy_exch, sell_exch
                            )
                            self._last_stale_warn_ms = current_time_ms
                        continue

                # Defect 8 & Task 3: Map Coinbase / USD pairs with explicit basis adjustment or exclude them
                if snap_buy.quote_currency != snap_sell.quote_currency:
                    basis_cfg = self.config.raw.get("markets", {}).get("basis_tracking", {})
                    if not basis_cfg.get("enabled", True):
                        # If basis tracking is disabled, strictly exclude cross-currency quote comparisons
                        continue
                    # Explicit basis adjustment: adjust USD quotes to USDT equivalent
                    rate = getattr(self.normalizer, "usdt_usd_basis_rate", 1.0000)
                    if snap_buy.quote_currency == "USD":
                        p_buy_ask = snap_buy.get_adjusted_price(p_buy_ask, target_quote="USDT", basis_rate=rate)
                    if snap_sell.quote_currency == "USD":
                        p_sell_bid = snap_sell.get_adjusted_price(p_sell_bid, target_quote="USDT", basis_rate=rate)

                # Gross Spread in basis points: (Bid_B - Ask_A) / Ask_A * 10,000
                gross_spread_bps = ((p_sell_bid - p_buy_ask) / p_buy_ask) * 10000.0

                # Compute dynamic cost hurdle and net spread (Amendment 1)
                hurdle_bps, fee_bps, vol_buf_bps = self.cost_model.get_hurdle_bps(
                    buy_exch, sell_exch, symbol
                )
                net_spread_bps = gross_spread_bps - fee_bps

                # Fix D: Hysteresis + Minimum Open Threshold
                hysteresis_bps = float(det_cfg.get("hysteresis_bps", 1.0))
                min_open_bps = float(det_cfg.get("min_open_spread_bps", 2.0))
                open_hurdle_bps = max(min_open_bps, hurdle_bps + hysteresis_bps)

                is_gross = gross_spread_bps >= max(float(self.gross_threshold_bps), min_open_bps)
                is_net = gross_spread_bps >= open_hurdle_bps

                candidate = SpreadCandidate(
                    symbol=symbol,
                    buy_exchange=buy_exch,
                    sell_exchange=sell_exch,
                    buy_ask=p_buy_ask,
                    sell_bid=p_sell_bid,
                    gross_spread_bps=gross_spread_bps,
                    net_spread_bps=net_spread_bps,
                    total_hurdle_bps=hurdle_bps,
                    fee_bps=fee_bps,
                    vol_buffer_bps=vol_buf_bps,
                    is_gross_candidate=is_gross,
                    is_net_profitable=is_net,
                    timestamp_ms=current_time_ms,
                    quote_age_buy_ms=snap_buy.quote_age_ms,
                    quote_age_sell_ms=snap_sell.quote_age_ms,
                    native_usdt_route=(snap_buy.quote_currency == snap_sell.quote_currency == "USDT"),
                )

                is_active = (active_pairs is not None) and ((symbol, buy_exch, sell_exch) in active_pairs)
                if is_gross or is_active:
                    candidates.append(candidate)

        return candidates
