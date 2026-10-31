"""Binance public WebSocket order book feed.

Connects to Binance public streams:
- Endpoint: wss://stream.binance.com:9443/stream
- Subscribes to combined streams: <symbol>@depth10@100ms and <symbol>@bookTicker
- Extracts real exchange timestamps (E), Level 2 depth (bids/asks), and provides microsecond resolution.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from data.feeds.base_feed import BaseExchangeFeed
from data.ntp_sync import clock_sync

logger = logging.getLogger(__name__)


class BinanceFeed(BaseExchangeFeed):
    """Binance Public WebSocket Market Feed."""

    DEFAULT_WS_URL = "wss://stream.binance.com:9443/stream"

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_book_update: Optional[Callable[[str, str, List, List, float, float], None]] = None,
        heartbeat_interval_sec: float = 15.0,
    ):
        target_symbols = symbols or ["btcusdt", "ethusdt", "xrpusdt", "adausdt", "solusdt"]
        # Format for Binance streams (lowercase without slashes)
        self.clean_symbols = [s.lower().replace("/", "").replace("-", "") for s in target_symbols]
        # Subscribes to partial depth and bookTicker streams to capture both depth levels and event time (E/T)
        streams = []
        for s in self.clean_symbols:
            streams.append(f"{s}@depth10@100ms")
            streams.append(f"{s}@bookTicker")
        ws_url = f"{self.DEFAULT_WS_URL}?streams={'/'.join(streams)}"

        super().__init__(
            exchange_name="binance",
            ws_url=ws_url,
            symbols=self.clean_symbols,
            on_book_update=on_book_update,
            heartbeat_interval_sec=heartbeat_interval_sec,
        )
        self._last_server_time_ms = 0.0

    def build_subscription_payload(self) -> Dict[str, Any]:
        """Binance streams in query URL are automatically subscribed; return ping payload."""
        return {"method": "LIST_SUBSCRIPTIONS", "id": 1}

    def parse_message(self, raw_msg: str, local_ts_ms: float) -> Optional[Tuple[str, List, List, float]]:
        data = json.loads(raw_msg)

        if "stream" in data and "data" in data:
            stream_name = data["stream"]
            payload = data["data"]
            raw_symbol = stream_name.split("@")[0].lower()

            # Handle bookTicker stream (contains real exchange timestamp E/T and top-of-book)
            if "@bookTicker" in stream_name:
                exch_ts = float(payload.get("E") or payload.get("T") or (local_ts_ms - (clock_sync.rtt_ms / 2.0)))
                self._last_server_time_ms = exch_ts
                bids = [(float(payload["b"]), float(payload["B"]))]
                asks = [(float(payload["a"]), float(payload["A"]))]
                return raw_symbol, bids, asks, exch_ts

            # Handle depth10 stream
            elif "@depth" in stream_name:
                bids_raw = payload.get("bids", [])
                asks_raw = payload.get("asks", [])
                bids = [(float(p), float(a)) for p, a in bids_raw]
                asks = [(float(p), float(a)) for p, a in asks_raw]

                # Use event time if present, or recent bookTicker server time + elapsed
                if "E" in payload:
                    exchange_ts_ms = float(payload["E"])
                elif self._last_server_time_ms > 0:
                    exchange_ts_ms = self._last_server_time_ms
                else:
                    exchange_ts_ms = local_ts_ms - (clock_sync.rtt_ms / 2.0)

                return raw_symbol, bids, asks, exchange_ts_ms

        return None
