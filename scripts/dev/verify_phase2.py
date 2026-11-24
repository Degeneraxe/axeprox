"""Phase 2 Storage Layer Verification Script.

Verifies:
1. TimescaleDB path probing & diagnostic reporting (Amendment 8)
2. Database initialization and table migration creation
3. Live tick buffering and batched insertion through BatchWriter
4. Data integrity verification: queries recent rows and prints schema layout
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from data.storage import BatchWriter, DatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase2Verifier")


async def run_phase2_verification():
    print("=" * 80)
    print("ArbSense Phase 2 Verification: Storage Engine & Batched Ingestion")
    print("=" * 80)

    # 1. Initialize Database Manager & Probe TimescaleDB Path (Amendment 8)
    print("\n[Step 1] Probing TimescaleDB connection & initializing storage...")
    db = DatabaseManager(
        sqlite_path="data/arbsense.db",
        migrations_dir="data/storage/migrations",
    )
    backend = db.initialize()
    print(f"    - Storage Backend Active: {backend.upper()}")
    print(f"    - Diagnostic: {db.connection_diagnostic}")

    # 2. Start Batch Writer
    print("\n[Step 2] Initializing asynchronous BatchWriter...")
    writer = BatchWriter(
        db_manager=db,
        batch_size=10,
        flush_interval_sec=0.2,
        depth_throttle_ms=50,
    )
    await writer.start()

    # 3. Insert Sample Live Ticks through BatchWriter
    print("\n[Step 3] Enqueueing 25 test ticks across Binance, Coinbase, and Kraken...")
    now_ms = time.time() * 1000.0
    for i in range(25):
        exch = ["binance", "coinbase", "kraken"][i % 3]
        tick = {
            "exchange": exch,
            "symbol": "BTC/USDT",
            "quote_currency": "USDT",
            "best_bid": 81600.0 + (i * 0.5),
            "best_ask": 81601.0 + (i * 0.5),
            "mid_price": 81600.5 + (i * 0.5),
            "top_spread": 1.0,
            "imbalance": 0.15 * (1 if i % 2 == 0 else -1),
            "bid_depth": [{"p": 81600.0 + i, "a": 1.2}],
            "ask_depth": [{"p": 81601.0 + i, "a": 0.8}],
            "exchange_ts_ms": now_ms + (i * 10),
            "local_ts_ms": now_ms + (i * 10) + 25.0,
            "transit_latency_ms": 25.0,
            "is_stale": False,
        }
        await writer.queue.put(tick)

    # Allow buffer to flush
    await asyncio.sleep(0.6)
    await writer.stop()

    # 4. Verify Database Row Count & Query Sample Data
    total_ticks = db.get_tick_count()
    print(f"\n[Step 4] Querying verified ticks from database (Total Stored: {total_ticks})...")

    recent_rows = db.get_recent_ticks(limit=4)
    print("\n" + "=" * 80)
    print("EVIDENCE: Recent Order Book Ticks Query Output")
    print("=" * 80)
    for r in recent_rows:
        print(f"ID: {r['id']:<4d} | Time: {r['time'][:19]} | Exchange: {r['exchange'].upper():8s} | Symbol: {r['symbol']}")
        print(f"     Bid: ${r['best_bid']:.2f} | Ask: ${r['best_ask']:.2f} | Mid: ${r['mid_price']:.2f} | Spread: ${r['top_spread']:.2f}")
        print(f"     Imbalance: {r['imbalance']:+.3f} | Transit Latency: {r['transit_latency_ms']:.1f}ms")
        depth = json.loads(r["bid_depth"])
        print(f"     Bid Depth (Level 1): {depth}")
        print("-" * 80)

    db.close()
    print("\nPhase 2 Storage Layer verification complete.")


if __name__ == "__main__":
    asyncio.run(run_phase2_verification())
