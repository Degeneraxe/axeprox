"""Read-only fee screen and exact-depth coverage check for the eight-hour backup."""
import argparse
import csv
import json
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal as D
from pathlib import Path

from replay_size_delay import walk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(args.backup.resolve().as_uri()+"?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    fees = json.loads(db.execute("SELECT config_json FROM run_metadata LIMIT 1").fetchone()[0])["fee_scenarios"]["scenarios"]
    limits = json.loads((Path(__file__).resolve().parents[1]/"docs/smoke-market-limits.json").read_text())["pairs"]
    books, stamps = defaultdict(list), defaultdict(list)
    for b in db.execute("""SELECT * FROM orderbook_ticks WHERE exchange IN ('binance','kraken')
            AND symbol IN ('BTC/USDT','ETH/USDT') AND quote_currency='USDT' ORDER BY local_ts_ms,id"""):
        key = b["exchange"], b["symbol"]
        books[key].append(b)
        stamps[key].append(b["local_ts_ms"])

    def pair_at(venues, symbol, t):
        result = []
        for venue in venues:
            key = venue, symbol
            ts = stamps[key]
            if not ts or not ts[0] <= t <= ts[-1]:
                return None
            result.append(books[key][bisect_right(ts,t)-1])
        return result

    parser_ids = [r[0] for r in db.execute("""SELECT event_id FROM spread_events
        WHERE symbol IN ('BTC/USDT','ETH/USDT') AND
        ((buy_exchange='binance' AND sell_exchange='kraken') OR
         (buy_exchange='kraken' AND sell_exchange='binance'))
        ORDER BY CAST(open_time AS REAL),event_id""")]
    sample_ids = {parser_ids[i*(len(parser_ids)-1)//199] for i in range(200)} if len(parser_ids)>=200 else set(parser_ids)
    sample = Counter()
    n = Counter()
    totals = defaultdict(Counter)
    for e in db.execute("SELECT * FROM spread_events ORDER BY CAST(open_time AS REAL),event_id"):
        n["events"] += 1
        venues = e["buy_exchange"], e["sell_exchange"]
        if set(venues) != {"binance","kraken"} or e["symbol"] not in limits:
            continue
        n["native_venue_candidates"] += 1
        ticks = sorted(json.loads(e["lifetime_ticks_json"]), key=lambda x:x["time_ms"])
        first = ticks[0]
        start = first["time_ms"]
        ts = [x["time_ms"] for x in ticks]
        symbol = e["symbol"]
        rules = [limits[symbol][v] for v in venues]
        at_open = pair_at(venues,symbol,start)
        matched = at_open and abs(at_open[0]["best_ask"]-first["buy_ask"])<1e-8 and abs(at_open[1]["best_bid"]-first["sell_bid"])<1e-8
        n["matched_native_open"] += bool(matched)
        for delay in (0,25,50,85,100,250,500):
            t = start+delay
            quote = ticks[bisect_right(ts,t)-1] if t <= ts[-1] else None
            legs = pair_at(venues,symbol,t) if matched else None
            fresh = legs and all(t-b["local_ts_ms"]<=250 and 0<b["best_bid"]<b["best_ask"] for b in legs) and abs(legs[0]["local_ts_ms"]-legs[1]["local_ts_ms"])<=100
            for tier,schedule in fees.items():
                bf,sf = [D(str(schedule[v]["taker_bps"]))/10000 for v in venues]
                for size in (50,100,250,500,1000):
                    s = totals[delay,tier,size]
                    if quote:
                        s["quotes"] += 1
                        s["fee_clears"] += D(str(quote["sell_bid"]))*(1-sf)>D(str(quote["buy_ask"]))*(1+bf)
                    if delay==85 and tier=="vip3" and size==100 and e["event_id"] in sample_ids and quote:
                        sample["observable"] += 1
                        sample["clear"] += D(str(quote["sell_bid"]))*(1-sf)>D(str(quote["buy_ask"]))*(1+bf)
                    if not fresh:
                        continue
                    s["fresh_books"] += 1
                    step = max(D(v["step"]) for v in rules)
                    qty = (D(size)/D(str(first["buy_ask"]))//step)*step
                    if any(qty < D(v["min_qty"]) or qty > D(v.get("max_qty","Infinity")) or
                           qty*D(str(price)) < D(v["min_cost"]) for v,price in zip(rules,(legs[0]["best_ask"],legs[1]["best_bid"]))):
                        continue
                    amounts = []
                    for book,field,v in zip(legs,("ask_depth","bid_depth"),rules):
                        levels = json.loads(book[field] or "[]")
                        amounts.append(walk([(x["p"],x["a"]) for x in levels],qty,D(v["tick"])))
                    if None not in amounts:
                        pnl = amounts[1]*(1-sf)-amounts[0]*(1+bf)
                        s["priced"] += 1
                        s["wins"] += pnl>0
                        s["sum_usdt"] += pnl
    db.close()
    print(dict(n))
    print("Systematic sample", len(sample_ids), "VIP3 85 ms observable", sample["observable"], "fee clears", sample["clear"])
    fields = ["quotes","fee_clears","fresh_books","priced","wins","sum_usdt"]
    with args.output.open("w",newline="") as out:
        writer = csv.writer(out,lineterminator="\n")
        writer.writerow(["delay_ms","tier","size_usdt",*fields])
        for (delay,tier,size),s in sorted(totals.items()):
            writer.writerow([delay,tier,size,*[s[k] if k!="sum_usdt" or s["priced"] else "" for k in fields]])
            if tier=="vip3" and size==100:
                print(delay, "ms", {k:s[k] for k in fields if k!="sum_usdt" or s["priced"]})


if __name__ == "__main__":
    main()
