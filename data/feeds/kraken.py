"""Kraken public WebSocket order book feed.

Connects to:
- Endpoint: wss://ws.kraken.com
- Channel: book (depth 10)
- Handles Kraken's L2 snapshots ("as"/"bs") and incremental book diffs ("a"/"b")
- Extracts sub-millisecond exchange timestamps from entry tuples [price, volume, timestamp].
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from data.feeds.base_feed import BaseExchangeFeed

logger = logging.getLogger(__name__)


class KrakenFeed(BaseExchangeFeed):
    """Kraken Public WebSocket Feed."""

    DEFAULT_WS_URL = "wss://ws.kraken.com"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_book_update: Optional[Callable[[str, str, List, List, float, float], None]] = None,
        heartbeat_interval_sec: float = 15.0,
    ):
        target_symbols = symbols or [
            "XBT/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT",
            "XBT/USD", "ETH/USD", "XRP/USD", "ADA/USD", "SOL/USD"
        ]
        super().__init__(
            exchange_name="kraken",
            ws_url=self.DEFAULT_WS_URL,
            symbols=target_symbols,
            on_book_update=on_book_update,
            heartbeat_interval_sec=heartbeat_interval_sec,
        )
        self._books: Dict[str, Dict[str, Dict[float, float]]] = {}
        for s in self.symbols:
            self._books[s] = {"bids": {}, "asks": {}}

    def build_subscription_payload(self) -> Dict[str, Any]:
        return {
            "event": "subscribe",
            "pair": self.symbols,
            "subscription": {
                "name": "book",
                "depth": 10,
            },
        }

    def parse_message(self, raw_msg: str, local_ts_ms: float) -> Optional[Tuple[str, List, List, float]]:
        data = json.loads(raw_msg)

        # Skip control events (heartbeats, subscription status)
        if isinstance(data, dict):
            return None

        # Data messages are lists: [channel_id, payload, channel_name, pair]
        if isinstance(data, list) and len(data) >= 4:
            payload = data[1]
            pair = data[3]

            if pair not in self._books:
                # Find matching book
                matched = next((k for k in self._books if k.replace("/", "") == pair.replace("/", "")), None)
                if not matched:
                    return None
                pair = matched

            book = self._books[pair]
            latest_exch_ts_sec = 0.0

            # 1. Snapshot handling (contains "as" for asks snapshot and "bs" for bids snapshot)
            if "as" in payload and "bs" in payload:
                book["asks"] = {}
                book["bids"] = {}
                for entry in payload["as"]:
                    p, v, t = float(entry[0]), float(entry[1]), float(entry[2])
                    book["asks"][p] = v
                    latest_exch_ts_sec = max(latest_exch_ts_sec, t)
                for entry in payload["bs"]:
                    p, v, t = float(entry[0]), float(entry[1]), float(entry[2])
                    book["bids"][p] = v
                    latest_exch_ts_sec = max(latest_exch_ts_sec, t)

            # 2. Incremental updates (payload may contain "a" and/or "b", or data[2] can contain the other side)
            payloads_to_process = [payload]
            if len(data) >= 5 and isinstance(data[2], dict):
                payloads_to_process.append(data[2])

            for p_dict in payloads_to_process:
                if "a" in p_dict:
                    for entry in p_dict["a"]:
                        p, v, t = float(entry[0]), float(entry[1]), float(entry[2])
                        latest_exch_ts_sec = max(latest_exch_ts_sec, t)
                        if v <= 0.0:
                            book["asks"].pop(p, None)
                        else:
                            book["asks"][p] = v
                if "b" in p_dict:
                    for entry in p_dict["b"]:
                        p, v, t = float(entry[0]), float(entry[1]), float(entry[2])
                        latest_exch_ts_sec = max(latest_exch_ts_sec, t)
                        if v <= 0.0:
                            book["bids"].pop(p, None)
                        else:
                            book["bids"][p] = v

            exchange_ts_ms = (latest_exch_ts_sec * 1000.0) if latest_exch_ts_sec > 0 else local_ts_ms

            sorted_bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:10]
            sorted_asks = sorted(book["asks"].items(), key=lambda x: x[0])[:10]

            return pair, sorted_bids, sorted_asks, exchange_ts_ms

        return None
