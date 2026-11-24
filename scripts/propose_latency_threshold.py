"""Propose Viability Latency Threshold from Measured Exchange Latencies (Item 5).

Reads orderbook_ticks with ts_source='exchange' from the database,
computes empirical percentiles (p50, p90, p95, p99, mean, std) overall and per exchange,
and proposes a frozen viability_latency_threshold_ms to set in config.yaml.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import AppConfig


def analyze_latencies(db_path: str, min_hours: Optional[float] = None) -> Dict[str, any]:
    """Analyzes transit latency from orderbook_ticks for exchange-timestamped ticks."""
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    uri = f"file:{p.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    cur = conn.cursor()

    # 1. Tick count by ts_source
    cur.execute("SELECT ts_source, COUNT(*) FROM orderbook_ticks GROUP BY ts_source")
    counts_by_source = {row[0]: row[1] for row in cur.fetchall()}

    # 2. Query latencies
    query = """
        SELECT exchange, transit_latency_ms, time
        FROM orderbook_ticks
        WHERE ts_source = 'exchange'
          AND transit_latency_ms IS NOT NULL
          AND transit_latency_ms > 0
    """
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return {
            "counts_by_source": counts_by_source,
            "overall": None,
            "per_exchange": {},
        }

    # Group latencies per exchange and overall
    latencies_all = [r[1] for r in rows]
    by_exchange: Dict[str, List[float]] = {}
    for exch, lat, _ in rows:
        by_exchange.setdefault(exch.lower(), []).append(lat)

    def _calc_stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr, dtype=np.float64)
        return {
            "count": len(a),
            "min": float(np.min(a)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
        }

    overall_stats = _calc_stats(latencies_all)
    per_exchange_stats = {exch: _calc_stats(lats) for exch, lats in by_exchange.items()}

    return {
        "counts_by_source": counts_by_source,
        "overall": overall_stats,
        "per_exchange": per_exchange_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Propose viability latency threshold from measured exchange latencies.")
    parser.add_argument("--db", type=str, default="data/arbsense.db", help="Path to SQLite database")
    parser.add_argument("--hours", type=float, default=None, help="Filter for last N hours of data")
    args = parser.parse_args()

    print("=" * 80)
    print("ArbSense Empirical Latency Threshold Analysis Tool")
    print(f"Database Target: {args.db}")
    print("=" * 80)

    try:
        results = analyze_latencies(args.db, min_hours=args.hours)
    except Exception as e:
        print(f"Error reading database: {e}")
        sys.exit(1)

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

    p50 = overall["p50"]
    p90 = overall["p90"]
    print("\n[Threshold Recommendation]")
    print(f"  Aggressive (p50): {p50:.1f} ms")
    print(f"  Conservative (p90): {p90:.1f} ms")
    print("\nTo freeze this threshold in config.yaml, set:")
    print("dataset:")
    print(f"  viability_latency_threshold_ms: {p90:.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
