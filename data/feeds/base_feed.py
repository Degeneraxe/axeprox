"""Abstract base WebSocket feed client for cryptocurrency exchanges.

Provides:
- Resilient connection lifecycle management
- Exponential backoff reconnection with randomized jitter
- Heartbeat / ping-pong liveness monitoring
- Per-message dual timestamping (exchange_ts and local_ts)
- Transit latency tracking
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import websockets

from data.ntp_sync import clock_sync

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"


class BaseExchangeFeed(ABC):
    """Abstract resilient WebSocket feed client."""

    def __init__(
        self,
        exchange_name: str,
        ws_url: str,
        symbols: List[str],
        on_book_update: Optional[Callable[[str, str, List, List, float, float], None]] = None,
        heartbeat_interval_sec: float = 15.0,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        backoff_multiplier: float = 2.0,
    ):
        self.exchange_name = exchange_name.lower()
        self.ws_url = ws_url
        self.symbols = symbols
        self.on_book_update = on_book_update
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.initial_backoff_sec = initial_backoff_sec
        self.max_backoff_sec = max_backoff_sec
        self.backoff_multiplier = backoff_multiplier

        self.state = ConnectionState.DISCONNECTED
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_msg_local_ts = 0.0

        # Transit latency metrics (ms)
        self.transit_latencies: List[float] = []
        self.max_latency_history = 100

    @abstractmethod
    def build_subscription_payload(self) -> Dict[str, Any] | List[Dict[str, Any]] | str:
        """Returns the payload(s) needed to subscribe to market depth/ticker channels."""
        pass

    @abstractmethod
    def parse_message(self, raw_msg: str, local_ts_ms: float) -> Optional[Tuple[str, List, List, float]]:
        """Parses raw WS message.
        
        Returns:
            Optional tuple: (raw_symbol, bids_list, asks_list, exchange_ts_ms)
        """
        pass

    async def start(self) -> None:
        """Starts the feed client background listener loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name=f"{self.exchange_name}_ws_loop")
        logger.info("[%s] Feed loop initiated.", self.exchange_name)

    async def stop(self) -> None:
        """Stops the feed cleanly."""
        self._running = False
        self.state = ConnectionState.STOPPED
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[%s] Feed stopped.", self.exchange_name)

    async def _run_loop(self) -> None:
        backoff = self.initial_backoff_sec

        while self._running:
            try:
                self.state = ConnectionState.CONNECTING
                logger.info("[%s] Connecting to %s ...", self.exchange_name, self.ws_url)

                async with websockets.connect(
                    self.ws_url,
                    ping_interval=None,  # We manage application-level heartbeats
                    max_size=10_000_000,
                    close_timeout=5.0,
                ) as ws:
                    self._ws = ws
                    self.state = ConnectionState.CONNECTED
                    backoff = self.initial_backoff_sec
                    self._last_msg_local_ts = time.time()
                    logger.info("[%s] Connected successfully.", self.exchange_name)

                    # Send subscription
                    sub = self.build_subscription_payload()
                    if isinstance(sub, list):
                        for p in sub:
                            await ws.send(json.dumps(p) if isinstance(p, dict) else str(p))
                    elif isinstance(sub, dict):
                        await ws.send(json.dumps(sub))
                    elif isinstance(sub, str):
                        await ws.send(sub)

                    # Start heartbeat monitor
                    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(ws))

                    # Message loop
                    async for raw_msg in ws:
                        if not self._running:
                            break
                        local_ts_ms = clock_sync.get_corrected_time_ms()
                        self._last_msg_local_ts = time.time()

                        try:
                            parsed = self.parse_message(raw_msg, local_ts_ms)
                            if parsed:
                                raw_symbol, bids, asks, exchange_ts_ms = parsed
                                latency = max(0.0, local_ts_ms - exchange_ts_ms)
                                self._record_latency(latency)

                                if self.on_book_update:
                                    self.on_book_update(
                                        self.exchange_name,
                                        raw_symbol,
                                        bids,
                                        asks,
                                        exchange_ts_ms,
                                        local_ts_ms,
                                    )
                        except Exception as parse_err:
                            logger.error("[%s] Error parsing message: %s", self.exchange_name, parse_err, exc_info=True)

            except (websockets.ConnectionClosed, asyncio.CancelledError, Exception) as e:
                if not self._running:
                    break
                self.state = ConnectionState.RECONNECTING
                jitter = random.uniform(0.8, 1.2)
                sleep_sec = min(backoff * jitter, self.max_backoff_sec)
                logger.warning("[%s] Disconnected (%s). Reconnecting in %.2fs...", self.exchange_name, e, sleep_sec)
                await asyncio.sleep(sleep_sec)
                backoff = min(backoff * self.backoff_multiplier, self.max_backoff_sec)
            finally:
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()

    async def _heartbeat_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        """Periodic heartbeat monitor to detect half-open connections."""
        while self._running:
            await asyncio.sleep(self.heartbeat_interval_sec)
            idle_sec = time.time() - self._last_msg_local_ts
            if idle_sec > (self.heartbeat_interval_sec * 2):
                logger.warning("[%s] No messages received for %.1fs. Forcing reconnect.", self.exchange_name, idle_sec)
                await ws.close()
                break
            try:
                # Send ping
                pong_waiter = await ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=5.0)
            except Exception as ping_err:
                logger.warning("[%s] Ping failed (%s). Triggering reconnect.", self.exchange_name, ping_err)
                await ws.close()
                break

    def _record_latency(self, latency_ms: float) -> None:
        self.transit_latencies.append(latency_ms)
        if len(self.transit_latencies) > self.max_latency_history:
            self.transit_latencies.pop(0)

    @property
    def avg_transit_latency_ms(self) -> float:
        if not self.transit_latencies:
            return 0.0
        return sum(self.transit_latencies) / len(self.transit_latencies)
