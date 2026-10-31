"""Order book normalization, data structures, and microstructure utilities.

Implements:
- Standardized Level 2 order book snapshot representations
- Stale quote filtering (Amendment 2)
- Explicit USD/USDT quote basis handling (Amendment 4)
- Depth-walking VWAP calculations for execution simulation (Amendment 7)
- Order book imbalance (OBI) and microstructural metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrderBookLevel:
    """Represents a single price-volume level in an order book."""
    price: float
    amount: float


@dataclass
class OrderBookSnapshot:
    """Standardized snapshot of an exchange order book with dual timestamps."""
    exchange: str
    symbol: str                     # Normalized symbol: e.g. "BTC/USDT", "ETH/USDT"
    quote_currency: str             # "USDT" or "USD"
    exchange_ts_ms: float           # Server event timestamp from exchange payload
    local_ts_ms: float              # Local UTC timestamp (corrected via NTP/offset)
    transit_latency_ms: float       # local_ts_ms - exchange_ts_ms
    bids: List[OrderBookLevel] = field(default_factory=list)  # Sorted descending by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # Sorted ascending by price
    is_stale: bool = False
    quote_age_ms: float = 0.0

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2.0
        return 0.0

    @property
    def top_spread(self) -> float:
        """Returns the venue's internal top-of-book spread (Ask - Bid)."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0

    def compute_imbalance(self, depth: int = 5) -> float:
        """Computes Order Book Imbalance (OBI) across top k levels.
        
        OBI = (V_bid - V_ask) / (V_bid + V_ask)
        Range: [-1.0, 1.0]. Positive means excess bid pressure (buying pressure).
        """
        bid_vol = sum(lvl.amount for lvl in self.bids[:depth])
        ask_vol = sum(lvl.amount for lvl in self.asks[:depth])
        total_vol = bid_vol + ask_vol
        if total_vol <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total_vol

    def compute_vwap(self, side: str, target_volume: float) -> Tuple[float, float, bool]:
        """Performs order-book depth walking to calculate volume-weighted average price (VWAP).
        
        Args:
            side: 'buy' (walks asks) or 'sell' (walks bids)
            target_volume: total base asset quantity to fill
            
        Returns:
            Tuple of (vwap_price, filled_volume, fully_filled)
        """
        levels = self.asks if side.lower() == "buy" else self.bids
        if not levels or target_volume <= 0:
            return 0.0, 0.0, False

        remaining = target_volume
        total_cost = 0.0
        filled = 0.0

        for lvl in levels:
            fill_amount = min(remaining, lvl.amount)
            total_cost += fill_amount * lvl.price
            filled += fill_amount
            remaining -= fill_amount
            if remaining <= 1e-9:
                break

        fully_filled = (remaining <= 1e-9)
        vwap = (total_cost / filled) if filled > 0 else 0.0
        return vwap, filled, fully_filled


class OrderBookNormalizer:
    """Maintains synchronized, normalized order books across multiple venues."""

    # Standard symbol translation tables
    EXCHANGE_SYMBOL_MAP = {
        "binance": {
            "btcusdt": "BTC/USDT",
            "ethusdt": "ETH/USDT",
            "xrpusdt": "XRP/USDT",
            "adausdt": "ADA/USDT",
            "solusdt": "SOL/USDT",
            "dogeusdt": "DOGE/USDT",
        },
        "coinbase": {
            "BTC-USDT": "BTC/USDT",
            "ETH-USDT": "ETH/USDT",
            "SOL-USDT": "SOL/USDT",
            "ADA-USDT": "ADA/USDT",
            "DOGE-USDT": "DOGE/USDT",
            "BTC-USD": "BTC/USD",
            "ETH-USD": "ETH/USD",
            "XRP-USD": "XRP/USD",
            "ADA-USD": "ADA-USD",
            "SOL-USD": "SOL-USD",
            "DOGE-USD": "DOGE-USD",
        },
        "kraken": {
            "XBT/USDT": "BTC/USDT",
            "ETH/USDT": "ETH/USDT",
            "BTC/USDT": "BTC/USDT",
            "XRP/USDT": "XRP/USDT",
            "ADA/USDT": "ADA/USDT",
            "SOL/USDT": "SOL/USDT",
            "DOGE/USDT": "DOGE/USDT",
            "XBT/USD": "BTC/USD",
            "ETH/USD": "ETH/USD",
            "XRP/USD": "XRP/USD",
            "ADA/USD": "ADA/USD",
            "SOL/USD": "SOL/USD",
            "DOGE/USD": "DOGE/USD",
        },
    }

    def __init__(self, max_quote_age_ms: float = 1500.0, default_quote: str = "USDT"):
        self.max_quote_age_ms = max_quote_age_ms
        self.default_quote = default_quote
        # State: (exchange, symbol) -> OrderBookSnapshot
        self.books: Dict[Tuple[str, str], OrderBookSnapshot] = {}
        # Quote basis tracking: e.g. "USDT/USD" parity
        self.usdt_usd_basis_rate: float = 1.0000

    def normalize_symbol(self, exchange: str, raw_symbol: str) -> str:
        """Translates raw exchange-specific ticker to standardized format."""
        exch = exchange.lower()
        mapping = self.EXCHANGE_SYMBOL_MAP.get(exch, {})
        if raw_symbol in mapping:
            return mapping[raw_symbol]
        # Fallback heuristic
        clean = raw_symbol.upper().replace("-", "/").replace("_", "/")
        if "XBT" in clean:
            clean = clean.replace("XBT", "BTC")
        return clean

    def update_book(
        self,
        exchange: str,
        raw_symbol: str,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        exchange_ts_ms: float,
        local_ts_ms: float,
    ) -> OrderBookSnapshot:
        """Updates and sorts order book snapshot for an exchange pair."""
        norm_symbol = self.normalize_symbol(exchange, raw_symbol)
        quote_curr = "USD" if norm_symbol.endswith("/USD") else "USDT"

        # Sort bids descending, asks ascending
        sorted_bids = [OrderBookLevel(p, a) for p, a in sorted(bids, key=lambda x: x[0], reverse=True)]
        sorted_asks = [OrderBookLevel(p, a) for p, a in sorted(asks, key=lambda x: x[0])]

        transit_latency = max(0.0, local_ts_ms - exchange_ts_ms)

        snapshot = OrderBookSnapshot(
            exchange=exchange.lower(),
            symbol=norm_symbol,
            quote_currency=quote_curr,
            exchange_ts_ms=exchange_ts_ms,
            local_ts_ms=local_ts_ms,
            transit_latency_ms=transit_latency,
            bids=sorted_bids,
            asks=sorted_asks,
            is_stale=False,
            quote_age_ms=0.0,
        )

        self.books[(exchange.lower(), norm_symbol)] = snapshot
        return snapshot

    def get_snapshot(
        self, exchange: str, symbol: str, current_time_ms: float
    ) -> Optional[OrderBookSnapshot]:
        """Retrieves order book snapshot and enforces the stale quote filter (Amendment 2)."""
        key = (exchange.lower(), symbol)
        if key not in self.books:
            return None

        snap = self.books[key]
        quote_age = current_time_ms - snap.local_ts_ms
        snap.quote_age_ms = max(0.0, quote_age)
        snap.is_stale = quote_age > self.max_quote_age_ms

        return snap

    def set_usdt_usd_basis(self, rate: float) -> None:
        """Sets the active USDT/USD exchange rate to allow cross-basis spread normalization."""
        self.usdt_usd_basis_rate = rate
