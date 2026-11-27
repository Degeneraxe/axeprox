"""Timed SOL WebSocket/REST comparison; no collector or database writes."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.dev.verify_remaining_pairs import fetch_rest_ticker
from data.feeds import BinanceFeed, CoinbaseFeed, KrakenFeed


async def main(resume_log=None):
    targets = [(BinanceFeed, "binance", "solusdt", "SOLUSDT"),
               (KrakenFeed, "kraken", "SOL/USDT", "SOLUSDT"),
               (CoinbaseFeed, "coinbase", "SOL-USD", "SOL-USD"),
               (CoinbaseFeed, "coinbase", "SOL-USDT", "SOL-USDT")]
    latest = {}
    samples = {symbol: [] for _, _, symbol, _ in targets}

    # Import only the uninterrupted prefix of an earlier audit. Time spent
    # offline contributes zero seconds to the cumulative sampling span.
    if resume_log:
        for line in Path(resume_log).read_text(encoding="utf-8-sig").splitlines():
            if "Disconnected" in line or "No messages received" in line:
                break
            if line.startswith("check "):
                _, venue, sym, elapsed, ws_bid, ws_ask, rest_bid, rest_ask = line.split()
                b, a = abs(float(ws_bid)/float(rest_bid)-1)*10000, abs(float(ws_ask)/float(rest_ask)-1)*10000
                samples[sym].append((float(elapsed), b, a))
                print(line, flush=True)
    offset = min((rows[-1][0] for rows in samples.values() if rows), default=0)

    def update(venue, symbol, bids, asks, exchange_ms, receipt_ms):
        if bids and asks:
            latest[symbol.lower()] = (bids[0][0], asks[0][0], time.monotonic(), receipt_ms)

    feeds = [cls(symbols=[symbol], on_book_update=update) for cls, _, symbol, _ in targets]

    async def compare(venue, symbol, rest):
        quote = await asyncio.to_thread(fetch_rest_ticker, venue, rest)
        book = latest.get(symbol.lower())
        now = time.monotonic()
        if not quote or not book or min(*quote, *book[:2]) <= 0:
            return
        bid, ask = [abs(x/y-1)*10000 for x,y in zip(book[:2], quote)]
        elapsed = offset + now-start
        feed = next(f for f in feeds if f.symbols == [symbol])
        if feed.connected_since_ms is None or feed.state.name != "CONNECTED" or book[3] < feed.connected_since_ms:
            return
        samples[symbol].append((elapsed, bid, ask))
        print("check", venue, symbol, round(elapsed,2), *book[:2], *quote, flush=True)

    try:
        for feed in feeds:
            await feed.start()
        await asyncio.sleep(6)
        start = time.monotonic()
        while time.monotonic()-start < 2400:
            if all(len(rows)>=100 and rows[-1][0]-rows[0][0]>=1800 for rows in samples.values()):
                break
            await asyncio.sleep(12)
            await asyncio.gather(*(compare(v,s,r) for _,v,s,r in targets))
    finally:
        await asyncio.gather(*(f.stop() for f in feeds))
    for _, venue, symbol, _ in targets:
        rows = samples[symbol]
        span = rows[-1][0]-rows[0][0] if rows else 0
        avg = mean((b+a)/2 for _,b,a in rows) if rows else None
        over = sum(max(b,a)>3 for _,b,a in rows)
        passed = len(rows)>=100 and span>=1800 and avg<2 and over/len(rows)<.03
        print("RESULT", json.dumps(dict(venue=venue,pair=symbol,checks=len(rows),
              span_s=round(span,2),mean_bps=avg,over3=over,passed=passed)),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-log", help="Reuse valid checks before the first disconnect")
    asyncio.run(main(parser.parse_args().resume_log))
