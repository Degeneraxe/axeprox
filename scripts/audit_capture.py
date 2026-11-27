"""Audit a finished SQLite capture. Open the source read-only; use decimal disk units."""
import argparse
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def storage(db, path, seconds, hours):
    count = db.execute("SELECT count(*) FROM orderbook_ticks").fetchone()[0]
    size = path.stat().st_size
    # Measure permanent tables and their indexes on a private compacted copy.
    with sqlite3.connect(":memory:") as copy:
        db.backup(copy)
        copy.execute("DROP TABLE orderbook_ticks")
        copy.execute("VACUUM")
        permanent = (copy.execute("PRAGMA page_count").fetchone()[0]
                     * copy.execute("PRAGMA page_size").fetchone()[0])
    day_factor = 86400 / seconds
    daily = size * day_factor / 1e9
    permanent_daily = permanent * day_factor / 1e9
    retained = (daily - permanent_daily) * hours / 24 + permanent_daily * 5
    print(f"raw_rows={count} seconds={seconds:g} rows/s={count/seconds:.2f} "
          f"DB_bytes={size} MB/min={size*60/seconds/1e6:.3f} gross_GB/day={daily:.3f}")
    print(f"permanent_upper_GB/day={permanent_daily:.3f} "
          f"projected_5day_DB_GB={retained:.3f} replay_hours={hours:g}")


def legacy_books(db, row, evidence, exact=False):
    """Recover top-of-book and timestamp provenance from retained raw ticks."""
    legs = []
    for side in ("buy", "sell"):
        ts = evidence.get(side + "_book_ms") if exact else evidence["arrival_ms"]
        op = "=" if exact else "<="
        leg = db.execute(f"""SELECT best_bid,best_ask,local_ts_ms,ts_source,transit_latency_ms
            FROM orderbook_ticks WHERE exchange=? AND symbol=? AND local_ts_ms {op} ?
            ORDER BY local_ts_ms DESC, id DESC LIMIT 1""",
            (row[side + "_exchange"], row["symbol"], ts)).fetchone()
        legs.append(leg)
    return legs


def diagnose(db, rows):
    causes = ("no_book", "older_than_250ms", "more_than_100ms_apart", "crossed_or_empty",
              "connection_warmup", "unknown")
    grouped = {}
    dropped = priced = unknown = 0
    first = db.execute("SELECT MIN(local_ts_ms) FROM orderbook_ticks").fetchone()[0]
    # An unpruned legacy smoke DB retains every receipt (including unsampled top-of-book).
    count, max_id = db.execute("SELECT COUNT(*),MAX(id) FROM orderbook_ticks").fetchone()
    complete = count == max_id and first is not None
    for row in rows:
        ev = json.loads(row["execution_evidence_json"] or "{}")
        if ev.get("status") == "missing_books":
            route = row["symbol"], row["buy_exchange"], row["sell_exchange"]
            cause = ev.get("missing_cause")
            if cause is None and complete:
                legs = legacy_books(db, row, ev)
                if any(b is None for b in legs):
                    cause = "no_book"
                elif any(ev["arrival_ms"] - b["local_ts_ms"] > 250 for b in legs):
                    cause = "older_than_250ms"
                elif any(b["best_bid"] <= 0 or b["best_ask"] <= b["best_bid"] for b in legs):
                    cause = "crossed_or_empty"
                elif abs(legs[0]["local_ts_ms"] - legs[1]["local_ts_ms"]) > 100:
                    cause = "more_than_100ms_apart"
            grouped.setdefault(route, Counter())[cause or "unknown"] += 1
        elif ev.get("status") == "priced":
            priced += 1
            if "buy" in ev and "sell" in ev:
                dropped += any(ev[s].get("exchange_age_over_500ms") for s in ("buy", "sell"))
            else:
                legs = legacy_books(db, row, ev, exact=True)
                if any(b is None for b in legs):
                    unknown += 1
                else:
                    dropped += any(b["ts_source"] == "exchange" and
                                   ev["arrival_ms"]-b["local_ts_ms"]+b["transit_latency_ms"] > 500
                                   for b in legs)
    print("missing_books primary cause (only affected routes)")
    print("symbol buy sell", *causes)
    for route, counts in sorted(grouped.items()):
        print(*route, *(counts[c] for c in causes))
    print(f"exchange age >500ms: would drop {dropped} of {priced} priced events; unknown={unknown}")


def outcomes(db, cfg):
    states = ["priced", "missing_books", "insufficient_depth", "conversion_required", "capture_ended"]
    counts = Counter()
    rows = list(db.execute("SELECT * FROM spread_events ORDER BY open_time"))
    diagnose(db, rows)
    priced = []
    for row in rows:
        evidence = json.loads(row["execution_evidence_json"] or "{}")
        status = evidence.get("status", "unknown")
        if status not in states:
            raise ValueError(f"Unexpected status {status} for {row['event_id']}")
        counts[row["symbol"], row["buy_exchange"], row["sell_exchange"], status] += 1
        if status == "priced":
            priced.append((row, evidence))
    print("symbol buy sell", *states)
    for symbol in cfg["markets"]["supported_pairs"]:
        venues = [name for name, pairs in cfg["markets"]["exchange_symbols"].items() if symbol in pairs]
        for buy, sell in itertools.permutations(venues, 2):
            print(symbol, buy, sell, *(counts[symbol, buy, sell, s] for s in states))
    print("fee n min_USDT p10 median p90 max mean positive_share boot95_low boot95_high")
    for tier in cfg["fee_scenarios"]["scenarios"]:
        pnl = [row[f"net_peak_{tier}"] * ev["buy_vwap"] * ev["quantity"] / 1e4 for row, ev in priced]
        if not pnl:
            print(tier, "0 of 0 priced events; exact CI unavailable")
            continue
        values = np.asarray(pnl)
        wins = int(np.count_nonzero(values > 0))
        exact = binomtest(wins, len(values)).proportion_ci(method="exact")
        print(f"{tier}: {wins} of {len(values)} priced events; exact 95% CI [{exact.low:.5f}, {exact.high:.5f}]")
        share = wins / len(values)
        # Resample one-minute clusters to preserve within-burst dependence.
        by_minute = {}
        for (row, ev), v in zip(priced, values):
            group = by_minute.setdefault(int(ev["arrival_ms"] // 60000), [0, 0])
            group[0] += int(v > 0)
            group[1] += 1
        clusters = np.asarray(list(by_minute.values()))
        rng = np.random.default_rng(42)
        draws = rng.integers(len(clusters), size=(10000, len(clusters)))
        sums = clusters[draws].sum(axis=1)
        ci = np.quantile(sums[:, 0] / sums[:, 1], [.025, .975])
        print(tier, len(values), *(f"{v:.5f}" for v in
              [*np.quantile(values, [0, .1, .5, .9, 1]), values.mean(), share, *ci]))
    print("priced_minute_clusters", len({int(ev["arrival_ms"]//60000) for _, ev in priced}))
    print("exchange n p50_ms p90_ms p99_ms max_ms")
    for (venue,) in db.execute("SELECT DISTINCT exchange FROM orderbook_ticks ORDER BY exchange"):
        values = [r[0] for r in db.execute(
            "SELECT transit_latency_ms FROM orderbook_ticks WHERE exchange=? AND ts_source='exchange'", (venue,))]
        stats = [f"{x:.2f}" for x in np.quantile(values, [.5, .9, .99, 1])] if values else ["n/a"]
        print(venue, len(values), *stats)
    for row in db.execute("SELECT time, offset_ms, rtt_ms, method, error_bound_ms FROM clock_offsets"):
        print("clock", *row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--seconds", type=float, required=True, help="Timed streaming interval")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("seconds must be positive")
    with sqlite3.connect(args.database.resolve().as_uri()+"?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        cfg = json.loads(db.execute("SELECT config_json FROM run_metadata ORDER BY run_id DESC LIMIT 1").fetchone()[0])
        hours = cfg["storage"].get("replay_window_hours", 120)
        storage(db, args.database, args.seconds, hours)
        if "replay_window_hours" in cfg["storage"]:
            outcomes(db, cfg)


if __name__ == "__main__":
    main()

