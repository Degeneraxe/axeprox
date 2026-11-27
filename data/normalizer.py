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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

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
    local_ts_ms: float              # Epoch-anchored monotonic receipt time
    transit_latency_ms: float       # Clock-adjusted receipt time minus venue timestamp
    bids: List[OrderBookLevel] = field(default_factory=list)  # Sorted descending by price
    asks: List[OrderBookLevel] = field(default_factory=list)  # Sorted ascending by price
    is_stale: bool = False
    quote_age_ms: float = 0.0
    is_crossed: bool = False
    ts_source: str = "exchange"
    warmup_complete: Optional[bool] = None

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

    def get_adjusted_price(self, price: float, target_quote: str = "USDT", basis_rate: float = 1.0) -> float:
        """Applies explicit basis adjustment when converting between USD and USDT.
        
        Formula: price_in_USDT = price_in_USD / usdt_usd (where basis_rate is USD per USDT, e.g. 0.9993)
                 price_in_USD = price_in_USDT * usdt_usd
        """
        if self.quote_currency == target_quote:
            return price
        if self.quote_currency == "USD" and target_quote == "USDT":
            return (price / basis_rate) if basis_rate > 0 else price
        if self.quote_currency == "USDT" and target_quote == "USD":
            return price * basis_rate
        return price

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

    def compute_top_notional(self, side: str, depth: int = 5) -> float:
        """Computes total depth notional (price * amount) across top k levels in quote units."""
        levels = self.asks[:depth] if side.lower() == "buy" else self.bids[:depth]
        return sum(lvl.price * lvl.amount for lvl in levels)

    def compute_top_quantity(self, side: str, depth: int = 5) -> float:
        """Computes total base quantity across top k levels."""
        levels = self.asks[:depth] if side.lower() == "buy" else self.bids[:depth]
        return sum(lvl.amount for lvl in levels)

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
            "ADA-USD": "ADA/USD",
            "SOL-USD": "SOL/USD",
            "DOGE-USD": "DOGE/USD",
            "USDT-USD": "USDT/USD",
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
            "USDT/USD": "USDT/USD",
        },
    }

    def __init__(self, max_quote_age_ms: float = 1500.0, default_quote: str = "USDT"):
        self.max_quote_age_ms = max_quote_age_ms
        self.default_quote = default_quote
        # State: (exchange, symbol) -> OrderBookSnapshot
        self.books: Dict[Tuple[str, str], OrderBookSnapshot] = {}
        # In-memory per-(exchange, symbol) ring buffers for live inference & tracker (Requirement 3)
        self.ring_buffers: Dict[Tuple[str, str], Deque[OrderBookSnapshot]] = defaultdict(lambda: deque(maxlen=100))
        # Quote basis tracking: e.g. "USDT/USD" parity
        self.usdt_usd_basis_rate: float = 1.0000
        self.last_basis_update_ts: float = 0.0
        # Book integrity monitor: crossed books counter
        self.crossed_books_count: int = 0

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
        ts_source: Optional[str] = None,
    ) -> OrderBookSnapshot:
        """Updates and sorts order book snapshot for an exchange pair."""
        norm_symbol = self.normalize_symbol(exchange, raw_symbol)
        quote_curr = "USD" if norm_symbol.endswith("/USD") else "USDT"

        # Determine timestamp source
        if ts_source is None:
            if exchange.lower() == "binance":
                ts_source = "estimated"
            elif exchange_ts_ms == local_ts_ms or exchange_ts_ms <= 0:
                ts_source = "estimated"
            else:
                ts_source = "exchange"

        # Sort bids descending, asks ascending
        sorted_bids = [OrderBookLevel(p, a) for p, a in sorted(bids, key=lambda x: x[0], reverse=True)]
        sorted_asks = [OrderBookLevel(p, a) for p, a in sorted(asks, key=lambda x: x[0])]

        # Book integrity monitor: check if book is crossed (best_bid >= best_ask)
        is_crossed = bool(sorted_bids and sorted_asks and sorted_bids[0].price >= sorted_asks[0].price)
        if is_crossed:
            self.crossed_books_count += 1
            logger.warning("[BookIntegrity] Crossed book on %s %s: bid=%.4f >= ask=%.4f", exchange, norm_symbol, sorted_bids[0].price, sorted_asks[0].price)

        # Transit latency honesty (Fix B): maintain true measurement without artificial flooring
        from data.ntp_sync import clock_sync
        transit_latency = clock_sync.transit_ms(local_ts_ms, exchange_ts_ms) if ts_source == "exchange" else local_ts_ms - exchange_ts_ms

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
            is_crossed=is_crossed,
            ts_source=ts_source,
        )

        self.books[(exchange.lower(), norm_symbol)] = snapshot
        self.ring_buffers[(exchange.lower(), norm_symbol)].append(snapshot)
        return snapshot

    def get_ring_buffer(self, exchange: str, symbol: str) -> List[OrderBookSnapshot]:
        """Returns the in-memory ring buffer of recent order book snapshots for (exchange, symbol)."""
        norm_symbol = self.normalize_symbol(exchange, symbol)
        return list(self.ring_buffers.get((exchange.lower(), norm_symbol), []))

    def build_feature_window(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        seq_len: int = 20,
        max_staleness_ms: float = 1500.0,
        use_latency: Optional[bool] = None,
        imputed_latency_ms: Optional[float] = None,
    ) -> Tuple[List[Any], float, float]:
        """Builds a 20-step LSTM feature window from in-memory ring buffers using asof_join_ticks (Requirement 1 & 3)."""
        from models.features import asof_join_ticks, compute_rolling_volatility_bps, build_feature_step
        from config.settings import AppConfig

        if use_latency is None:
            cfg = getattr(self, "config", None) or AppConfig.load()
            use_latency = bool(cfg.raw.get("features", {}).get("use_latency", False))
        if imputed_latency_ms is None:
            cfg = getattr(self, "config", None) or AppConfig.load()
            imputed_latency_ms = float(cfg.raw.get("dataset", {}).get("viability_latency_threshold_ms", 85.0))

        norm_sym = self.normalize_symbol(buy_exchange, symbol)
        b_ticks = self.get_ring_buffer(buy_exchange, norm_sym)
        s_ticks = self.get_ring_buffer(sell_exchange, norm_sym)

        # 50-tick rolling volatility per exchange
        b_prices = [t.mid_price for t in b_ticks[-50:]] if b_ticks else []
        s_prices = [t.mid_price for t in s_ticks[-50:]] if s_ticks else []
        vol_b = compute_rolling_volatility_bps(b_prices) if b_prices else 1.0
        vol_s = compute_rolling_volatility_bps(s_prices) if s_prices else 1.0

        # As-of join by timestamp with max staleness
        aligned_pairs = asof_join_ticks(b_ticks, s_ticks, max_staleness_ms=max_staleness_ms, seq_len=seq_len)
        if len(aligned_pairs) < seq_len:
            return [], vol_b, vol_s

        seq: List[Any] = []
        prev_spread = None
        prev_vel = None
        basis_rate = getattr(self, "usdt_usd_basis_rate", 1.0000)

        for b_snap, s_snap in aligned_pairs:
            ask_buy = float(b_snap.best_ask.price) if b_snap.best_ask else 0.0
            bid_sell = float(s_snap.best_bid.price) if s_snap.best_bid else 0.0
            if getattr(b_snap, "quote_currency", "USDT") != "USDT":
                ask_buy = b_snap.get_adjusted_price(ask_buy, target_quote="USDT", basis_rate=basis_rate)
            if getattr(s_snap, "quote_currency", "USDT") != "USDT":
                bid_sell = s_snap.get_adjusted_price(bid_sell, target_quote="USDT", basis_rate=basis_rate)
            gross_bps = ((bid_sell - ask_buy) / ask_buy * 10000.0) if ask_buy > 0 else 0.0

            vel = (gross_bps - prev_spread) if prev_spread is not None else 0.0
            acc = (vel - prev_vel) if prev_vel is not None else 0.0
            prev_spread = gross_bps
            prev_vel = vel

            obi_b = float(b_snap.compute_imbalance(5))
            obi_s = float(s_snap.compute_imbalance(5))
            b_ts_src = str(getattr(b_snap, "ts_source", "exchange"))
            s_ts_src = str(getattr(s_snap, "ts_source", "exchange"))
            lat_b = float(b_snap.transit_latency_ms)
            lat_s = float(s_snap.transit_latency_ms)
            top_b = float(b_snap.top_spread)
            top_s = float(s_snap.top_spread)
            mp_b = float(b_snap.mid_price)
            mp_s = float(s_snap.mid_price)
            mp_diff = ((mp_s - mp_b) / mp_b * 10000.0) if mp_b > 0 else 0.0

            step = build_feature_step(
                gross_bps=gross_bps,
                vel=vel,
                acc=acc,
                obi_b=obi_b,
                obi_s=obi_s,
                vol_b=vol_b,
                vol_s=vol_s,
                top_b=top_b,
                top_s=top_s,
                mp_diff=mp_diff,
                raw_lat_b=lat_b,
                raw_lat_s=lat_s,
                ts_src_b=b_ts_src,
                ts_src_s=s_ts_src,
                use_latency=use_latency,
                imputed_latency_ms=imputed_latency_ms,
            )
            seq.append(step)

        return seq, vol_b, vol_s


    def get_snapshot(
        self, exchange: str, symbol: str, current_time_ms: float
    ) -> Optional[OrderBookSnapshot]:
        """Retrieves order book snapshot and enforces the stale quote filter (Amendment 2)."""
        key = (exchange.lower(), symbol)
        if key not in self.books:
            return None

        snap = self.books[key]
        if getattr(snap, "is_crossed", False):
            return None  # Strictly exclude corrupt / crossed books from arbitrage detection

        quote_age = current_time_ms - snap.local_ts_ms
        snap.quote_age_ms = max(0.0, quote_age)
        snap.is_stale = quote_age > self.max_quote_age_ms

        return snap

    def set_usdt_usd_basis(self, rate: float) -> None:
        """Sets the active USDT/USD exchange rate to allow cross-basis spread normalization."""
        import time
        self.usdt_usd_basis_rate = rate
        self.last_basis_update_ts = time.time()

    def is_basis_stale(self, max_staleness_sec: float = 300.0) -> bool:
        """Returns True if no basis update occurred within max_staleness_sec (default 5 minutes)."""
        import time
        if self.last_basis_update_ts <= 0:
            return True
        return (time.time() - self.last_basis_update_ts) > max_staleness_sec

    def get_basis_age_sec(self) -> float:
        """Returns elapsed seconds since last basis rate update."""
        import time
        if self.last_basis_update_ts <= 0:
            return 9999.0
        return max(0.0, time.time() - self.last_basis_update_ts)
