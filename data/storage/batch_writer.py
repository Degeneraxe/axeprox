"""High-throughput asynchronous batched storage writer.

Implements:
- Asynchronous queue with non-blocking enqueue
- Batch flushes based on size threshold or timeout interval
- Depth snapshot throttling (Amendment 6)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from data.normalizer import OrderBookSnapshot
from data.storage.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class BatchWriter:
    """Buffers order book ticks in memory and writes them in high-speed batches."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        batch_size: int = 50,
        flush_interval_sec: float = 0.5,
        depth_throttle_ms: int = 100,
    ):
        self.db = db_manager
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.depth_throttle_ms = depth_throttle_ms

        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=10000)
        self._staged: List[Dict[str, Any]] = []
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._last_depth_write: Dict[Tuple[str, str], float] = {}

    def enqueue_snapshot(self, snap: OrderBookSnapshot) -> None:
        """Serializes snapshot and enqueues tick for writing."""
        if not self._running:
            return

        now_ms = time.monotonic() * 1000.0
        pair_key = (snap.exchange, snap.symbol)

        # Depth throttling (Amendment 6): include full depth levels only every depth_throttle_ms
        include_full_depth = False
        last_t = self._last_depth_write.get(pair_key, 0.0)
        if (now_ms - last_t) >= self.depth_throttle_ms:
            include_full_depth = True
            self._last_depth_write[pair_key] = now_ms

        bids_serialized = [{"p": lvl.price, "a": lvl.amount} for lvl in snap.bids[:10]] if include_full_depth else []
        asks_serialized = [{"p": lvl.price, "a": lvl.amount} for lvl in snap.asks[:10]] if include_full_depth else []

        tick_record = {
            "exchange": snap.exchange,
            "symbol": snap.symbol,
            "quote_currency": snap.quote_currency,
            "best_bid": snap.best_bid.price if snap.best_bid else 0.0,
            "best_ask": snap.best_ask.price if snap.best_ask else 0.0,
            "mid_price": snap.mid_price,
            "top_spread": snap.top_spread,
            "imbalance": snap.compute_imbalance(5),
            "bid_depth": bids_serialized,
            "ask_depth": asks_serialized,
            "exchange_ts_ms": snap.exchange_ts_ms,
            "local_ts_ms": snap.local_ts_ms,
            "transit_latency_ms": snap.transit_latency_ms,
            "is_stale": snap.is_stale,
            "ts_source": getattr(snap, "ts_source", "exchange"),
        }

        if self.queue.full():
            # Keep arrival order across both the worker batch and queued snapshots.
            self._drain_and_write()
        self.queue.put_nowait(tick_record)

    def _drain_and_write(self) -> None:
        while not self.queue.empty():
            self._staged.append(self.queue.get_nowait())
            self.queue.task_done()
        if self._staged:
            self.db.insert_ticks_batch(self._staged)
            self._staged.clear()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_worker(), name="batch_writer_worker")
        logger.info("[Storage] Batch writer worker started.")

    async def stop(self) -> None:
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush_remaining()
        logger.info("[Storage] Batch writer stopped.")

    async def _flush_worker(self) -> None:
        batch = self._staged
        last_flush_time = time.monotonic()

        while self._running:
            try:
                # Calculate remaining wait time before forced interval flush
                timeout = max(0.01, self.flush_interval_sec - (time.monotonic() - last_flush_time))
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                    self.queue.task_done()
                except asyncio.TimeoutError:
                    pass

                # Check flush condition
                if len(batch) >= self.batch_size or ((time.monotonic() - last_flush_time) >= self.flush_interval_sec and batch):
                    self.db.insert_ticks_batch(batch)
                    batch.clear()
                    last_flush_time = time.monotonic()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Storage] Error during batch write: %s", e, exc_info=True)
                await asyncio.sleep(0.1)

    async def _flush_remaining(self) -> None:
        self._drain_and_write()
