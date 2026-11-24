"""Verification of WebSocket Order Books against Exchange REST Tickers.

Task 1:
- For BTC/USDT and ETH/USDT, every 5 seconds for 5 minutes (or configurable duration),
  compares our normalizer's best bid/ask per exchange with the exchange's official
  REST ticker at the exact same moment:
  * Binance: https://api.binance.com/api/v3/ticker/bookTicker
  * Coinbase: https://api.exchange.coinbase.com/products/{id}/ticker
  * Kraken: https://api.kraken.com/0/public/Ticker
- Computes differences in basis points (bps) per exchange and flags differences > 3 bps.
- Logs whether any book is crossed (best_bid >= best_ask).
- Logs top-of-book update frequency (updates/sec) per feed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.feeds import BinanceFeed, CoinbaseFeed, KrakenFeed
from data.normalizer import OrderBookNormalizer

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


def fetch_rest_ticker(exchange: str, symbol: str) -> Optional[Tuple[float, float]]:
    """Fetches real-time best bid and ask directly from exchange REST ticker API."""
    try:
        headers = {"User-Agent": "ArbSenseVerifier/1.0"}
        if exchange == "binance":
            pair = symbol.replace("/", "").replace("-", "").upper()
            url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={pair}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                return float(data["bidPrice"]), float(data["askPrice"])

        elif exchange == "coinbase":
            prod = "BTC-USD" if "BTC" in symbol else "ETH-USD"
            url = f"https://api.exchange.coinbase.com/products/{prod}/ticker"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                return float(data["bid"]), float(data["ask"])

        elif exchange == "kraken":
            pair = "XBTUSDT" if "BTC" in symbol else "ETHUSDT"
            url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                res = data.get("result", {})
                pair_key = list(res.keys())[0] if res else None
                if pair_key:
                    bids = res[pair_key]["b"]
                    asks = res[pair_key]["a"]
                    return float(bids[0]), float(asks[0])
    except Exception:
        pass
    return None


async def run_verifier(duration_sec: int = 300, interval_sec: float = 5.0):
    print("=" * 105)
    print("ArbSense Book Integrity & REST Ticker Verification (Task 1)")
    print(f"Monitoring BTC and ETH across Binance, Coinbase, Kraken every {interval_sec}s for {duration_sec}s")
    print("=" * 105)

    norm = OrderBookNormalizer()
    feed_updates = {"binance": 0, "coinbase": 0, "kraken": 0}
    crossed_detections = {"binance": 0, "coinbase": 0, "kraken": 0}

    def on_update(exch: str, raw_sym: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        feed_updates[exch] = feed_updates.get(exch, 0) + 1
        snap = norm.update_book(exch, raw_sym, bids, asks, exch_ts, loc_ts)
        if snap.is_crossed:
            crossed_detections[exch] = crossed_detections.get(exch, 0) + 1

    feeds = [
        BinanceFeed(symbols=["btcusdt", "ethusdt"], on_book_update=on_update),
        CoinbaseFeed(symbols=["BTC-USD", "ETH-USD"], on_book_update=on_update),
        KrakenFeed(symbols=["XBT/USDT", "ETH/USDT"], on_book_update=on_update),
    ]

    for f in feeds:
        await f.start()

    print("[*] WebSocket feeds connected. Warming up books for 5 seconds...")
    await asyncio.sleep(5.0)

    start_t = time.time()
    last_t = start_t
    all_diffs: List[Dict[str, Any]] = []

    header = f"{'Time':<8} {'Exch':<9} {'Symbol':<9} {'WS Bid':<10} {'REST Bid':<10} {'Bid Diff':<10} {'WS Ask':<10} {'REST Ask':<10} {'Ask Diff':<10} {'Crossed?':<9} {'Rate (up/s)'}"
    print("\n" + header)
    print("-" * len(header))

    check_num = 0
    while time.time() - start_t < duration_sec:
        await asyncio.sleep(interval_sec)
        now = time.time()
        elapsed = now - start_t
        dt = now - last_t
        last_t = now
        check_num += 1

        t_str = f"{int(elapsed)}s"
        rates = {ex: feed_updates[ex] / max(1.0, elapsed) for ex in feed_updates}

        for symbol in ["BTC/USDT", "ETH/USDT"]:
            for exch in ["binance", "coinbase", "kraken"]:
                snap = norm.books.get((exch, symbol))
                if not snap:
                    snap = norm.books.get((exch, symbol.replace("USDT", "USD")))
                
                rest_quote = fetch_rest_ticker(exch, symbol)
                if not snap or not rest_quote:
                    continue

                ws_bid = snap.best_bid.price if snap.best_bid else 0.0
                ws_ask = snap.best_ask.price if snap.best_ask else 0.0
                r_bid, r_ask = rest_quote

                bid_diff_bps = abs(ws_bid - r_bid) / r_bid * 10000.0 if r_bid > 0 else 0.0
                ask_diff_bps = abs(ws_ask - r_ask) / r_ask * 10000.0 if r_ask > 0 else 0.0

                is_cr = "YES (!)" if snap.is_crossed else "NO"
                bid_flag = " *" if bid_diff_bps > 3.0 else ""
                ask_flag = " *" if ask_diff_bps > 3.0 else ""

                row_info = {
                    "time": t_str,
                    "exchange": exch,
                    "symbol": symbol,
                    "ws_bid": ws_bid,
                    "rest_bid": r_bid,
                    "bid_diff_bps": bid_diff_bps,
                    "ws_ask": ws_ask,
                    "rest_ask": r_ask,
                    "ask_diff_bps": ask_diff_bps,
                    "is_crossed": snap.is_crossed,
                }
                all_diffs.append(row_info)

                print(
                    f"{t_str:<8} {exch:<9} {symbol:<9} {ws_bid:<10.2f} {r_bid:<10.2f} "
                    f"{bid_diff_bps:<7.2f}{bid_flag:<3} {ws_ask:<10.2f} {r_ask:<10.2f} "
                    f"{ask_diff_bps:<7.2f}{ask_flag:<3} {is_cr:<9} {rates[exch]:<6.1f}"
                )

    for f in feeds:
        await f.stop()

    print("\n" + "=" * 105)
    print("REST VERIFICATION SUMMARY (5-Minute Audit)")
    print("=" * 105)
    print(f"Total Comparison Checks:    {len(all_diffs)}")
    print(f"Total Crossed Books Logged: Binance: {crossed_detections['binance']}, Coinbase: {crossed_detections['coinbase']}, Kraken: {crossed_detections['kraken']}")
    
    for exch in ["binance", "coinbase", "kraken"]:
        ex_diffs = [d for d in all_diffs if d["exchange"] == exch]
        if ex_diffs:
            avg_bid_d = sum(d["bid_diff_bps"] for d in ex_diffs) / len(ex_diffs)
            avg_ask_d = sum(d["ask_diff_bps"] for d in ex_diffs) / len(ex_diffs)
            max_bid_d = max(d["bid_diff_bps"] for d in ex_diffs)
            max_ask_d = max(d["ask_diff_bps"] for d in ex_diffs)
            over_3 = sum(1 for d in ex_diffs if d["bid_diff_bps"] > 3.0 or d["ask_diff_bps"] > 3.0)
            pct_over_3 = (over_3 / len(ex_diffs)) * 100.0
            print(f"[{exch.upper():<8}] Avg Bid Diff: {avg_bid_d:.2f} bps | Avg Ask Diff: {avg_ask_d:.2f} bps | Max Diff: {max(max_bid_d, max_ask_d):.2f} bps | >3bps: {over_3}/{len(ex_diffs)} ({pct_over_3:.1f}%) | Update Rate: {rates[exch]:.1f} updates/s")


def main():
    duration = 300  # 5 minutes as required
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    asyncio.run(run_verifier(duration_sec=duration))


if __name__ == "__main__":
    main()
