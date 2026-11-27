"""Replay native BTC/ETH smoke events with fixed decision sizes and as-of arrivals."""
import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal as D
from pathlib import Path


def walk(levels, quantity, tick):
    remaining = quantity
    total = D(0)
    for price, amount in levels:
        price, amount = D(str(price)), D(str(amount))
        if price <= 0 or amount < 0 or price % tick:
            raise ValueError("Invalid archived price or quantity for the recorded pair limits")
        take = min(remaining, amount)
        total += price * take
        remaining -= take
        if remaining == 0:
            return total
    return None


def replay(db, limits):
    fees = json.loads(db.execute("SELECT config_json FROM run_metadata").fetchone()[0])["fee_scenarios"]["scenarios"]
    end = db.execute("SELECT MAX(local_ts_ms) FROM orderbook_ticks").fetchone()[0]
    stats = defaultdict(lambda: dict(events=0, fresh=0, top_wins=0, priced=0, wins=0, pnl=D(0)))
    for row in db.execute("SELECT * FROM spread_events"):
        symbol, buy, sell = row["symbol"], row["buy_exchange"], row["sell_exchange"]
        evidence = json.loads(row["execution_evidence_json"])
        if symbol not in limits or {buy,sell} != {"kraken","binance"} or evidence["status"] == "conversion_required":
            continue
        first = json.loads(row["lifetime_ticks_json"])[0]
        venues = [limits[symbol][v] for v in (buy,sell)]
        step = max(D(v["step"]) for v in venues)
        for delay in (25,50,85,100,250,500):
            arrival = first["time_ms"] + delay
            books = [db.execute("""SELECT * FROM orderbook_ticks WHERE exchange=? AND symbol=?
                    AND local_ts_ms<=? ORDER BY local_ts_ms DESC,id DESC LIMIT 1""",
                    (v,symbol,arrival)).fetchone() for v in (buy,sell)]
            valid = arrival <= end and all(b and arrival-b["local_ts_ms"] <= 250
                    and 0 < b["best_bid"] < b["best_ask"] for b in books)
            if valid:
                valid = abs(books[0]["local_ts_ms"]-books[1]["local_ts_ms"]) <= 100
            for size in (50,100,250,500,1000):
                qty = (D(size)/D(str(first["buy_ask"]))//step)*step
                for tier,schedule in fees.items():
                    s = stats[delay,size,tier]
                    s["events"] += 1
                    if not valid:
                        continue
                    ask,bid = D(str(books[0]["best_ask"])), D(str(books[1]["best_bid"]))
                    if any(qty % D(v["step"]) or qty < D(v["min_qty"]) or qty > D(v.get("max_qty","Infinity"))
                           or qty*price < D(v["min_cost"]) for v,price in zip(venues,(ask,bid))):
                        continue
                    bf,sf = [D(str(schedule[v]["taker_bps"]))/10000 for v in (buy,sell)]
                    s["fresh"] += 1
                    s["top_wins"] += int(bid*(1-sf)>ask*(1+bf))
                    sides = []
                    for b,leg,key in zip(books,("buy","sell"),("asks","bids")):
                        if b["local_ts_ms"] == evidence.get(leg+"_book_ms") and leg+"_"+key in evidence:
                            sides.append(evidence[leg+"_"+key])
                        else:
                            sides.append([(x["p"],x["a"]) for x in json.loads(b["ask_depth" if leg=="buy" else "bid_depth"] or "[]")])
                    cost, proceeds = [walk(levels,qty,D(v["tick"])) for levels,v in zip(sides,venues)]
                    if cost is None or proceeds is None:
                        continue
                    pnl = proceeds*(1-sf)-cost*(1+bf)
                    s["priced"] += 1
                    s["wins"] += int(pnl>0)
                    s["pnl"] += pnl
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    rules = Path(__file__).resolve().parents[1] / "docs/smoke-market-limits.json"
    with sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro",uri=True) as db:
        db.row_factory = sqlite3.Row
        stats = replay(db,json.loads(rules.read_text())["pairs"])
    with args.output.open("w",newline="",encoding="utf8") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(["delay_ms","target_USDT","fee_scenario","events","fresh_valid","positive_top_bounds","priced","profitable","sum_pnl_USDT"])
        for (delay,size,tier),s in sorted(stats.items()):
            writer.writerow([delay,size,tier,s["events"],s["fresh"],s["top_wins"],s["priced"],s["wins"],round(s["pnl"],8) if s["priced"] else ""])
            if tier=="vip3" and size==100:
                print(f'{delay} ms, 100 USDT: {s["fresh"]}/{s["events"]} fresh, {s["top_wins"]} positive best-price bounds, {s["wins"]}/{s["priced"]} priced wins, sum {s["pnl"]:.4f} USDT')


if __name__ == "__main__":
    main()
