"""Comprehensive 5-minute REST verification audit for SOL, ADA, and XRP.

Evaluates WebSocket order book accuracy against official REST tickers across:
- Binance: SOLUSDT, ADAUSDT, XRPUSDT
- Coinbase USD: SOL-USD, ADA-USD, XRP-USD
- Coinbase USDT: SOL-USDT, ADA-USDT, XRP-USDT
- Kraken: SOL/USDT, ADA/USDT, XRP/USDT

Pass criteria:
- Average difference < 2.0 bps
- Fewer than 3.0% of comparison checks above 3.0 bps
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import urllib.request
from collections import defaultdict
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
    headers = {"User-Agent": "ArbSenseVerifier/1.0"}
    try:
        if exchange == "binance":
            pair = symbol.upper()
            url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={pair}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                return float(data["bidPrice"]), float(data["askPrice"])

        elif exchange.startswith("coinbase"):
            prod = symbol.upper()
            url = f"https://api.exchange.coinbase.com/products/{prod}/ticker"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                bid = float(data.get("bid", 0.0))
                ask = float(data.get("ask", 0.0))
                if bid > 0 and ask > 0:
                    return bid, ask

        elif exchange == "kraken":
            pair = symbol.upper()
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


async def run_audit(duration_sec: int = 300, interval_sec: float = 5.0):
    print("=" * 115)
    print("ArbSense 5-Minute REST Order Book Audit for SOL, ADA, XRP")
    print(f"Sampling every {interval_sec}s for {duration_sec}s across Binance, Coinbase (USD & USDT), and Kraken")
    print("=" * 115)

    norm = OrderBookNormalizer()
    feed_updates = defaultdict(int)
    crossed_detections = defaultdict(int)

    def on_update(exch: str, raw_sym: str, bids: list, asks: list, exch_ts: float, loc_ts: float):
        feed_updates[exch] += 1
        snap = norm.update_book(exch, raw_sym, bids, asks, exch_ts, loc_ts)
        if snap.is_crossed:
            crossed_detections[exch] += 1

    binance_symbols = ["solusdt", "adausdt", "xrpusdt"]
    coinbase_symbols = ["SOL-USD", "SOL-USDT", "ADA-USD", "ADA-USDT", "XRP-USD", "XRP-USDT"]
    kraken_symbols = ["SOL/USDT", "ADA/USDT", "XRP/USDT"]

    feeds = [
        BinanceFeed(symbols=binance_symbols, on_book_update=on_update),
        CoinbaseFeed(symbols=coinbase_symbols, on_book_update=on_update),
        KrakenFeed(symbols=kraken_symbols, on_book_update=on_update),
    ]

    for f in feeds:
        await f.start()

    print("[*] WebSocket feeds connected. Warming up books for 6 seconds...")
    await asyncio.sleep(6.0)

    start_t = time.time()
    all_diffs: List[Dict[str, Any]] = []

    # Map of audit targets: (audit_label, venue_id, lookup_sym, rest_sym, base_asset)
    targets = [
        # Binance
        ("binance", "binance", "SOL/USDT", "SOLUSDT", "SOL"),
        ("binance", "binance", "ADA/USDT", "ADAUSDT", "ADA"),
        ("binance", "binance", "XRP/USDT", "XRPUSDT", "XRP"),
        # Coinbase USD
        ("coinbase_usd", "coinbase", "SOL/USD", "SOL-USD", "SOL"),
        ("coinbase_usd", "coinbase", "ADA/USD", "ADA-USD", "ADA"),
        ("coinbase_usd", "coinbase", "XRP/USD", "XRP-USD", "XRP"),
        # Coinbase USDT
        ("coinbase_usdt", "coinbase", "SOL/USDT", "SOL-USDT", "SOL"),
        ("coinbase_usdt", "coinbase", "ADA/USDT", "ADA-USDT", "ADA"),
        ("coinbase_usdt", "coinbase", "XRP/USDT", "XRP-USDT", "XRP"),
        # Kraken
        ("kraken", "kraken", "SOL/USDT", "SOLUSDT", "SOL"),
        ("kraken", "kraken", "ADA/USDT", "ADAUSDT", "ADA"),
        ("kraken", "kraken", "XRP/USDT", "XRPUSDT", "XRP"),
    ]

    header = f"{'Time':<7} {'Venue':<14} {'Asset':<6} {'Pair':<11} {'WS Bid':<10} {'REST Bid':<10} {'Bid Diff':<9} {'WS Ask':<10} {'REST Ask':<10} {'Ask Diff':<9} {'Crossed?'}"
    print("\n" + header)
    print("-" * len(header))

    iteration = 0
    while (time.time() - start_t) < duration_sec:
        await asyncio.sleep(interval_sec)
        iteration += 1
        elapsed = time.time() - start_t
        t_str = f"{int(elapsed)}s"

        for label, venue, ws_sym, rest_sym, asset in targets:
            snap = norm.books.get((venue, ws_sym))
            rest_quote = fetch_rest_ticker("binance" if venue == "binance" else ("kraken" if venue == "kraken" else "coinbase"), rest_sym)
            if not snap or not rest_quote:
                continue

            ws_bid = snap.best_bid.price if snap.best_bid else 0.0
            ws_ask = snap.best_ask.price if snap.best_ask else 0.0
            r_bid, r_ask = rest_quote

            bid_diff_bps = (abs(ws_bid - r_bid) / r_bid) * 10000.0 if r_bid > 0 else 0.0
            ask_diff_bps = (abs(ws_ask - r_ask) / r_ask) * 10000.0 if r_ask > 0 else 0.0
            max_diff = max(bid_diff_bps, ask_diff_bps)

            is_cr = "YES" if snap.is_crossed else "NO"
            flag = " *" if max_diff > 3.0 else ""

            row_info = {
                "venue_label": label,
                "asset": asset,
                "pair": rest_sym,
                "bid_diff_bps": bid_diff_bps,
                "ask_diff_bps": ask_diff_bps,
                "max_diff_bps": max_diff,
                "is_crossed": snap.is_crossed,
            }
            all_diffs.append(row_info)

            # Periodically log sampling row (every 30 seconds to keep stdout uncluttered)
            if iteration % 6 == 0:
                print(
                    f"{t_str:<7} {label:<14} {asset:<6} {rest_sym:<11} {ws_bid:<10.4f} {r_bid:<10.4f} "
                    f"{bid_diff_bps:<7.2f}  {ws_ask:<10.4f} {r_ask:<10.4f} {ask_diff_bps:<7.2f}{flag:<3} {is_cr}"
                )

    for f in feeds:
        await f.stop()

    print("\n" + "=" * 115)
    print("5-MINUTE REST AUDIT FINAL SUMMARY & PAIR ADMISSIBILITY REPORT")
    print("Criteria for inclusion: Average Difference < 2.0 bps AND Checks > 3.0 bps < 3.0%")
    print("=" * 115)

    summary_header = f"{'Venue':<15} {'Asset':<7} {'Pair':<12} {'Checks':<8} {'Avg Diff (bps)':<16} {'Max Diff (bps)':<16} {'>3.0 bps Checks':<20} {'Status'}"
    print(summary_header)
    print("-" * len(summary_header))

    passed_assets = set()
    failed_reasons = []

    for label, venue, ws_sym, rest_sym, asset in targets:
        match_diffs = [d for d in all_diffs if d["venue_label"] == label and d["pair"] == rest_sym]
        if not match_diffs:
            print(f"{label:<15} {asset:<7} {rest_sym:<12} {'0':<8} {'N/A':<16} {'N/A':<16} {'N/A':<20} FAILED (No Data)")
            failed_reasons.append(f"{label} {rest_sym}: No REST comparison checks recorded")
            continue

        n_checks = len(match_diffs)
        avg_diff = sum((d["bid_diff_bps"] + d["ask_diff_bps"]) / 2.0 for d in match_diffs) / n_checks
        max_diff = max(d["max_diff_bps"] for d in match_diffs)
        over_3 = sum(1 for d in match_diffs if d["max_diff_bps"] > 3.0)
        pct_over_3 = (over_3 / n_checks) * 100.0

        passes = (avg_diff < 2.0) and (pct_over_3 < 3.0)
        status_str = "PASS" if passes else "FAIL"

        print(
            f"{label:<15} {asset:<7} {rest_sym:<12} {n_checks:<8} "
            f"{avg_diff:<16.2f} {max_diff:<16.2f} {f'{over_3}/{n_checks} ({pct_over_3:.1f}%)':<20} {status_str}"
        )

        if not passes:
            failed_reasons.append(
                f"{label} {rest_sym}: avg_diff={avg_diff:.2f} bps (threshold <2.0 bps), >3bps={pct_over_3:.1f}% (threshold <3.0%)"
            )

    print("\n" + "=" * 115)
    print("EVALUATION & ADMISSIBILITY SUMMARY:")
    print("=" * 115)
    for r in failed_reasons:
        print(f" [DROP REASON] {r}")


def main():
    duration = 300
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    asyncio.run(run_audit(duration_sec=duration))


if __name__ == "__main__":
    main()
