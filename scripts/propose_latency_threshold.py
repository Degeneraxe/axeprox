"""Summarize post-warm-up exchange-timestamped receipt delays from a SQLite capture."""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

import numpy as np


def latency_stats(a):
    p50, p75, p90, p95, p99 = np.percentile(a, [50, 75, 90, 95, 99])
    return dict(count=len(a), min=a.min(), p50=p50, p75=p75, p90=p90,
                p95=p95, p99=p99, max=a.max(), mean=a.mean(), std=a.std())


def analyze_latencies(db_path, min_hours=None, legacy_single_session=False):
    path = Path(db_path).resolve()
    if min_hours is not None and not 0 < min_hours < float("inf"):
        raise ValueError("--hours must be positive and finite")
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        counts = dict(db.execute("SELECT ts_source, COUNT(*) FROM orderbook_ticks GROUP BY ts_source"))
        columns = {r[1] for r in db.execute("PRAGMA table_info(orderbook_ticks)")}
        if legacy_single_session:
            if "warmup_complete" in columns and db.execute(
                "SELECT 1 FROM orderbook_ticks WHERE warmup_complete IS NOT NULL LIMIT 1"
            ).fetchone():
                raise ValueError("Legacy override requires entirely unmarked ticks")
            runs = db.execute("SELECT COUNT(*) FROM run_metadata").fetchone()[0]
            count, last = db.execute("SELECT COUNT(*), MAX(id) FROM orderbook_ticks").fetchone()
            if runs != 1 or count != last:
                raise ValueError("Legacy override requires one complete unpruned capture")
            warm = """local_ts_ms >= (SELECT MAX(first_ms) + 60000 FROM
                      (SELECT MIN(local_ts_ms) AS first_ms FROM orderbook_ticks GROUP BY exchange))"""
        elif "warmup_complete" in columns:
            warm = "warmup_complete = 1"
        else:
            raise ValueError("No warm-up markers. For a verified uninterrupted capture use --legacy-single-session")
        query = f"""SELECT exchange, transit_latency_ms FROM orderbook_ticks
                    WHERE ts_source='exchange' AND transit_latency_ms IS NOT NULL AND {warm}"""
        params = []
        if min_hours is not None:
            query += " AND local_ts_ms >= (SELECT MAX(local_ts_ms) FROM orderbook_ticks) - ?"
            params.append(min_hours * 3600000)
        by_exchange = defaultdict(lambda: array("d"))
        for venue, delay in db.execute(query, params):
            by_exchange[venue].append(delay)
        fields = {r[1] for r in db.execute("PRAGMA table_info(clock_offsets)")}
        bounds = []
        if "error_bound_ms" in fields:
            bounds = list(db.execute("SELECT time,method,error_bound_ms FROM clock_offsets ORDER BY id"))
    arrays = {venue: np.frombuffer(values, dtype=np.float64) for venue, values in by_exchange.items()}
    return dict(counts_by_source=counts, bounds=bounds,
                per_exchange={venue: latency_stats(values) for venue, values in arrays.items()},
                overall=latency_stats(np.concatenate(list(arrays.values()))) if arrays else None)


def main():
    parser = argparse.ArgumentParser(description="Propose viability latency threshold from measured exchange latencies.")
    parser.add_argument("--db", type=str, default="data/arbsense.db", help="Path to SQLite database")
    parser.add_argument("--hours", type=float, default=None, help="Filter for last N hours of data")
    parser.add_argument("--legacy-single-session", action="store_true", help="Use latest venue first tick plus 60 s for an uninterrupted unpruned legacy capture")
    args = parser.parse_args()

    print("=" * 80)
    print("ArbSense Empirical Latency Threshold Analysis Tool")
    print(f"Database Target: {args.db}")
    print("=" * 80)

    try:
        results = analyze_latencies(args.db, min_hours=args.hours, legacy_single_session=args.legacy_single_session)
    except Exception as e:
        print(f"Error reading database: {e}")
        sys.exit(1)

    print("Warm-up filter:", "legacy latest first receipt + 60 s" if args.legacy_single_session else "recorded all-feed readiness")
    for stamp, method, bound in results["bounds"]:
        print("clock_error_bound_ms", stamp, method, bound)
    if not results["bounds"]:
        print("clock_error_bound_ms unavailable")
    counts = results["counts_by_source"]
    print("\n[Tick Source Breakdown]")
    total_ticks = sum(counts.values())
    print(f"Total Ticks: {total_ticks:,}")
    for src, cnt in counts.items():
        pct = (cnt / total_ticks * 100.0) if total_ticks > 0 else 0.0
        print(f"  - {src}: {cnt:,} ({pct:.1f}%)")

    overall = results["overall"]
    if not overall or overall["count"] == 0:
        print("\nNo exchange-timestamped latency ticks found in database.")
        print("Note: Binance depth snapshots are marked 'estimated'.")
        print("Kraken and Coinbase ticks with exchange timestamps are used for latency measurement.")
        sys.exit(0)

    print("\n[Empirical Latency Statistics (ts_source='exchange')]")
    print(f"{'Scope':<15} {'Count':>8} {'Min':>8} {'p50':>8} {'p75':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'Mean':>8} {'Std':>8}")
    print("-" * 92)

    for exch, st in results["per_exchange"].items():
        print(f"{exch:<15} {st['count']:>8d} {st['min']:>8.1f} {st['p50']:>8.1f} {st['p75']:>8.1f} {st['p90']:>8.1f} {st['p95']:>8.1f} {st['p99']:>8.1f} {st['mean']:>8.1f} {st['std']:>8.1f}")

    print("-" * 92)
    print(f"{'OVERALL':<15} {overall['count']:>8d} {overall['min']:>8.1f} {overall['p50']:>8.1f} {overall['p75']:>8.1f} {overall['p90']:>8.1f} {overall['p95']:>8.1f} {overall['p99']:>8.1f} {overall['mean']:>8.1f} {overall['std']:>8.1f}")
    print("-" * 92)

    print("These are inbound market-data delays. The VM execution threshold is set separately.")


if __name__ == "__main__":
    main()
