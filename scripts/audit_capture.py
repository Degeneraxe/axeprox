"""Audit a finished SQLite capture. Open the source read-only; use decimal disk units."""
import argparse
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np


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


def outcomes(db, cfg):
    states = ["priced", "missing_books", "insufficient_depth", "conversion_required", "capture_ended"]
    counts = Counter()
    rows = list(db.execute("SELECT * FROM spread_events ORDER BY open_time"))
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
            print(tier, "0 (no priced events)")
            continue
        values = np.asarray(pnl)
        share = np.mean(values > 0)
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

