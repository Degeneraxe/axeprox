"""Phase 1 Verification Script.

Executes live concurrent ingestion from Binance, Coinbase, and Kraken:
1. Performs clock synchronization (NTP / exchange REST) and measures system offset.
2. Connects to live public WebSockets for Binance, Coinbase, and Kraken.
3. Ingests real-time L2 order books for BTC and ETH.
4. Enforces the stale-book filter (max quote age).
5. Computes transit latency (local_ts - exchange_ts) across all venues.
6. Displays live order book tables and latency statistics.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from data.feeds import BinanceFeed, CoinbaseFeed, KrakenFeed
from data.normalizer import OrderBookNormalizer
from data.ntp_sync import clock_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase1Verifier")


async def run_live_verification(duration_sec: int = 15):
    print("=" * 80)
    print("ArbSense Phase 1 Verification: Live Multi-Exchange WebSocket Ingestion")
    print("=" * 80)

    # 1. Load config
    config = AppConfig.load()
    print(f"[*] Configuration loaded from config.yaml")
    print(f"    - Supported pairs: {config.markets.supported_pairs}")
    print(f"    - Stale quote threshold: {config.time_sync.max_quote_age_ms} ms")

    # 2. NTP / REST Clock Sync
    print("\n[Step 1] Synchronizing system clock...")
    offset_ms = clock_sync.sync()
    print(f"    - Sync Method: {clock_sync.method_used}")
    print(f"    - Measured Clock Offset: {offset_ms:+.2f} ms")
    print(f"    - Round-Trip Latency (RTT): {clock_sync.rtt_ms:.2f} ms")

    # 3. Initialize Normalizer
    normalizer = OrderBookNormalizer(
        max_quote_age_ms=config.time_sync.max_quote_age_ms,
        default_quote=config.markets.primary_quote,
    )

    update_counts = {"binance": 0, "coinbase": 0, "kraken": 0}
    latencies = {"binance": [], "coinbase": [], "kraken": []}

    def on_book_update(exchange: str, raw_symbol: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        snap = normalizer.update_book(exchange, raw_symbol, bids, asks, exch_ts, loc_ts)
        update_counts[exchange] = update_counts.get(exchange, 0) + 1
        lat = snap.transit_latency_ms
        if len(latencies[exchange]) < 200:
            latencies[exchange].append(lat)

    # 4. Instantiate Live Feeds
    print("\n[Step 2] Initializing live WebSocket clients...")
    binance = BinanceFeed(
        symbols=["btcusdt", "ethusdt"],
        on_book_update=on_book_update,
    )
    coinbase = CoinbaseFeed(
        symbols=["BTC-USDT", "ETH-USDT", "BTC-USD", "ETH-USD"],
        on_book_update=on_book_update,
    )
    kraken = KrakenFeed(
        symbols=["XBT/USDT", "ETH/USDT", "XBT/USD", "ETH/USD"],
        on_book_update=on_book_update,
    )

    # 5. Start Feeds Concurrently
    feeds = [binance, coinbase, kraken]
    for feed in feeds:
        await feed.start()

    print(f"\n[Step 3] Streaming live ticks for {duration_sec} seconds...")
    start_time = time.time()

    # Progress loop
    while time.time() - start_time < duration_sec:
        await asyncio.sleep(3.0)
        now_ms = clock_sync.get_corrected_time_ms()
        elapsed = int(time.time() - start_time)
        print(f"    [Elapsed {elapsed:02d}s] Ticks received -> Binance: {update_counts['binance']} | Coinbase: {update_counts['coinbase']} | Kraken: {update_counts['kraken']}")

    # 6. Capture Live Snapshot WHILE feeds are actively streaming (proves ACTIVE status)
    now_ms = clock_sync.get_corrected_time_ms()
    print("\n" + "=" * 80)
    print("EVIDENCE: Live Level 2 Normalized Order Books Snapshot (Active Streaming)")
    print("=" * 80)

    for symbol in ["BTC/USDT", "ETH/USDT"]:
        print(f"\n--- Symbol: {symbol} ---")
        for exch in ["binance", "coinbase", "kraken"]:
            snap = normalizer.get_snapshot(exch, symbol, now_ms)
            if not snap:
                snap = normalizer.get_snapshot(exch, symbol.replace("USDT", "USD"), now_ms)

            if snap and snap.best_bid and snap.best_ask:
                stale_tag = "STALE" if snap.is_stale else "ACTIVE"
                print(f"[{exch.upper():8s}] Status: {stale_tag:6s} | Age: {snap.quote_age_ms:5.1f}ms | Transit: {snap.transit_latency_ms:5.1f}ms")
                print(f"             Best Bid: ${snap.best_bid.price:>10.2f} (Vol: {snap.best_bid.amount:6.4f})")
                print(f"             Best Ask: ${snap.best_ask.price:>10.2f} (Vol: {snap.best_ask.amount:6.4f})")
                print(f"             Spread:   ${snap.top_spread:>10.2f} | OBI(top5): {snap.compute_imbalance(5):+0.3f}")
            else:
                print(f"[{exch.upper():8s}] No snapshot captured yet.")

    # 7. Stop Feeds Cleanly
    print("\n[Step 4] Stopping feeds cleanly...")
    for feed in feeds:
        await feed.stop()

    # 8. Print Latency & Performance Statistics
    print("\n" + "=" * 80)
    print("EVIDENCE: Transit Latency & Throughput Metrics (local_ts - exchange_ts)")
    print("=" * 80)
    for exch, lat_list in latencies.items():
        if lat_list:
            avg_lat = sum(lat_list) / len(lat_list)
            min_lat = min(lat_list)
            max_lat = max(lat_list)
            p50 = sorted(lat_list)[len(lat_list) // 2]
            p95 = sorted(lat_list)[int(len(lat_list) * 0.95)]
            print(f"[{exch.upper():8s}] Ticks: {update_counts[exch]:>4d} | Avg: {avg_lat:6.1f}ms | Median: {p50:6.1f}ms | p95: {p95:6.1f}ms | Min/Max: {min_lat:.1f}/{max_lat:.1f}ms")
        else:
            print(f"[{exch.upper():8s}] No latency samples recorded.")

    print("\nPhase 1 verification complete.")


if __name__ == "__main__":
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    asyncio.run(run_live_verification(duration_sec=duration))
