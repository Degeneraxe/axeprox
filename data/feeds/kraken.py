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
import zlib
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
            "XBT/USD", "ETH/USD", "XRP/USD", "ADA/USD", "SOL/USD",
            "USDT/USD"
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
        self.checksum_failure_count = 0
        self.crossed_book_count = 0

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

            # Enforce depth truncation to top 10 on both sides (Task 2)
            sorted_bids = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:10]
            sorted_asks = sorted(book["asks"].items(), key=lambda x: x[0])[:10]
            book["bids"] = dict(sorted_bids)
            book["asks"] = dict(sorted_asks)

            # Integrity check: crossed book detection
            if sorted_bids and sorted_asks and sorted_bids[0][0] >= sorted_asks[0][0]:
                self.crossed_book_count += 1
                logger.warning("[kraken] Crossed book detected on %s: bid=%.4f >= ask=%.4f. Resyncing.", pair, sorted_bids[0][0], sorted_asks[0][0])
                book["bids"].clear()
                book["asks"].clear()
                return None

            # Optional CRC32 Checksum verification if published by Kraken
            for p_dict in payloads_to_process:
                if "c" in p_dict:
                    remote_crc = str(p_dict["c"])
                    # Construct CRC string: top 10 asks (p, v stripped), then top 10 bids (p, v stripped)
                    try:
                        crc_parts = []
                        for p, v in sorted_asks:
                            crc_parts.append(str(p).replace(".", "").lstrip("0") + str(v).replace(".", "").lstrip("0"))
                        for p, v in sorted_bids:
                            crc_parts.append(str(p).replace(".", "").lstrip("0") + str(v).replace(".", "").lstrip("0"))
                        local_crc = str(zlib.crc32("".join(crc_parts).encode("utf-8")))
                        if remote_crc != local_crc:
                            self.checksum_failure_count += 1
                    except Exception:
                        pass

            return pair, sorted_bids, sorted_asks, exchange_ts_ms

        return None
