"""Coinbase Exchange public WebSocket order book feed.

Connects to:
- Endpoint: wss://ws-feed.exchange.coinbase.com
- Channels: level2_batch (L2 order book snapshots & diffs) and ticker
- Parses ISO 8601 server event timestamps (with microsecond precision) into UTC epoch ms.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from data.feeds.base_feed import BaseExchangeFeed

logger = logging.getLogger(__name__)


def _parse_iso_to_epoch_ms(iso_str: str) -> float:
    """Converts ISO 8601 timestamp (e.g. 2026-10-29T15:20:00.123456Z) to UTC epoch milliseconds."""
    try:
        clean = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        return dt.timestamp() * 1000.0
    except Exception:
        return 0.0


class CoinbaseFeed(BaseExchangeFeed):
    """Coinbase Exchange Public WebSocket Feed."""

    DEFAULT_WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_book_update: Optional[Callable[[str, str, List, List, float, float], None]] = None,
        heartbeat_interval_sec: float = 15.0,
    ):
        target_symbols = symbols or [
            "BTC-USDT", "ETH-USDT", "SOL-USDT", "ADA-USDT",
            "BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "SOL-USD"
        ]
        super().__init__(
            exchange_name="coinbase",
            ws_url=self.DEFAULT_WS_URL,
            symbols=target_symbols,
            on_book_update=on_book_update,
            heartbeat_interval_sec=heartbeat_interval_sec,
        )
        # In-memory order book cache: product_id -> {bids: {p: a}, asks: {p: a}}
        self._books: Dict[str, Dict[str, Dict[float, float]]] = {}
        for s in self.symbols:
            self._books[s] = {"bids": {}, "asks": {}}

    def build_subscription_payload(self) -> Dict[str, Any]:
        return {
            "type": "subscribe",
            "product_ids": self.symbols,
            "channels": ["level2_batch", "ticker"],
        }

    def parse_message(self, raw_msg: str, local_ts_ms: float) -> Optional[Tuple[str, List, List, float]]:
        data = json.loads(raw_msg)
        msg_type = data.get("type")
        product_id = data.get("product_id")

        if not product_id or product_id not in self._books:
            return None

        book = self._books[product_id]

        if msg_type == "snapshot":
            # Initial L2 snapshot
            bids_raw = data.get("bids", [])
            asks_raw = data.get("asks", [])
            book["bids"] = {float(p): float(a) for p, a in bids_raw}
            book["asks"] = {float(p): float(a) for p, a in asks_raw}
            exchange_ts_ms = local_ts_ms

            sorted_bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:10]
            sorted_asks = sorted(book["asks"].items(), key=lambda x: x[0])[:10]
            return product_id, sorted_bids, sorted_asks, exchange_ts_ms

        elif msg_type == "l2update":
            # Incremental updates
            time_str = data.get("time")
            exchange_ts_ms = _parse_iso_to_epoch_ms(time_str) if time_str else local_ts_ms
            changes = data.get("changes", [])

            for side, price_str, size_str in changes:
                price = float(price_str)
                size = float(size_str)
                target_side = "bids" if side == "buy" else "asks"
                if size <= 0:
                    book[target_side].pop(price, None)
                else:
                    book[target_side][price] = size

            sorted_bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:10]
            sorted_asks = sorted(book["asks"].items(), key=lambda x: x[0])[:10]
            return product_id, sorted_bids, sorted_asks, exchange_ts_ms

        elif msg_type == "ticker":
            time_str = data.get("time")
            exchange_ts_ms = _parse_iso_to_epoch_ms(time_str) if time_str else local_ts_ms

            best_bid = data.get("best_bid")
            best_bid_size = data.get("best_bid_size", "1.0")
            best_ask = data.get("best_ask")
            best_ask_size = data.get("best_ask_size", "1.0")

            if best_bid and best_ask:
                p_bid = float(best_bid)
                p_ask = float(best_ask)
                book["bids"][p_bid] = float(best_bid_size)
                book["asks"][p_ask] = float(best_ask_size)

                sorted_bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:10]
                sorted_asks = sorted(book["asks"].items(), key=lambda x: x[0])[:10]
                return product_id, sorted_bids, sorted_asks, exchange_ts_ms

        return None
