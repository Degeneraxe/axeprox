"""Break-Even Analysis for Stored Arbitrage Spread Events.

Requirement 6:
From stored events in the validated database:
1. Computes effective gross spread = peak_spread_bps - depth_walked_slippage.
2. Derives the empirical distribution of required round-trip fee hurdle (bps) for profitability.
3. Reports the fraction of events profitable at 0, 5, 10, and 20 bps round-trip cost hurdles.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config.settings import AppConfig


def run_break_even_analysis(
    db_path: str = "data/arbsense.db",
    version: str | None = None,
    exclude_censored: bool = True,
) -> Dict[str, Any]:
    version = version or AppConfig.load().raw["dataset"]["version"]
    full_path = Path(db_path)
    if not full_path.is_absolute():
        full_path = PROJECT_ROOT / full_path

    if not full_path.exists():
        raise FileNotFoundError(f"Database not found at {full_path}")

    conn = sqlite3.connect(str(full_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
        SELECT *
        FROM spread_events
        WHERE dataset_version = ?
    """
    params: List[Any] = [version]
    if exclude_censored:
        query += " AND is_right_censored = 0"

    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"No events found for dataset_version='{version}' (exclude_censored={exclude_censored}).")
        return {}

    events = [dict(r) for r in rows]
    print(f"Selected {len(events)} event rows.")

    # Compute break-even round-trip fee hurdle per event: peak_spread_bps - slippage_bps
    # (i.e. the maximum round-trip fee an execution can bear before becoming net negative)
    break_evens: List[float] = []
    slippages: List[float] = []
    peaks: List[float] = []
    per_symbol: Dict[str, List[float]] = {}

    for ev in events:
        peak = float(ev["peak_spread_bps"] if ev["peak_spread_bps"] is not None else 0.0)
        if ev["slippage_bps"] is None:
            continue
        slip = float(ev["slippage_bps"])
        if version == "v2.2":
            evidence = json.loads(ev.get("execution_evidence_json") or "{}")
            if evidence.get("status") != "priced":
                continue
            be = 10000 * (evidence["sell_vwap"] / evidence["buy_vwap"] - 1)
        else:
            be = peak - slip
        break_evens.append(be)
        slippages.append(slip)
        peaks.append(peak)

        sym = ev["symbol"]
        if sym not in per_symbol:
            per_symbol[sym] = []
        per_symbol[sym].append(be)

    total_n = len(break_evens)
    if not total_n:
        print("No priced observations for this analysis.")
        return {}
    if version != "v2.2":
        print("Historical peak-based screening estimates.")
    arr = np.array(break_evens, dtype=np.float64)
    arr_peaks = np.array(peaks, dtype=np.float64)
    arr_slips = np.array(slippages, dtype=np.float64)

    # Percentiles
    p10 = float(np.percentile(arr, 10))
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p90 = float(np.percentile(arr, 90))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Fee hurdles
    hurdles = [0.0, 5.0, 10.0, 20.0]
    fractions: Dict[float, Tuple[int, float]] = {}
    for h in hurdles:
        cnt = int(np.sum(arr > h))
        pct = (cnt / total_n) * 100.0
        fractions[h] = (cnt, pct)

    # Print comprehensive report
    print("=" * 90)
    print(f"ARBSENSE BREAK-EVEN & FEE HURDLE ANALYSIS (Dataset: {version})")
    print(f"Source DB: {full_path.name} | Total Events Analyzed: {total_n} (Non-Censored: {exclude_censored})")
    print("=" * 90)

    print("\n1. SPREAD & SLIPPAGE COMPONENTS (bps):")
    print(f"  - Peak Spread:      Mean = {np.mean(arr_peaks):5.2f} bps | Median = {np.median(arr_peaks):5.2f} bps | Max = {np.max(arr_peaks):5.2f} bps")
    print(f"  - Slippage (VWAP):  Mean = {np.mean(arr_slips):5.2f} bps | Median = {np.median(arr_slips):5.2f} bps | Max = {np.max(arr_slips):5.2f} bps")
    print(f"  - Effective Spread: Mean = {mean_val:5.2f} bps | Median = {p50:5.2f} bps | Max = {max_val:5.2f} bps")

    print("\n2. DISTRIBUTION OF BREAK-EVEN ROUND-TRIP FEE HURDLES:")
    print(f"  {'Metric':<20} {'Value (bps)':>12}")
    print("  " + "-" * 32)
    print(f"  {'Min':<20} {min_val:>12.2f}")
    print(f"  {'p10':<20} {p10:>12.2f}")
    print(f"  {'p25':<20} {p25:>12.2f}")
    print(f"  {'p50 (Median)':<20} {p50:>12.2f}")
    print(f"  {'Mean':<20} {mean_val:>12.2f}")
    print(f"  {'p75':<20} {p75:>12.2f}")
    print(f"  {'p90':<20} {p90:>12.2f}")
    print(f"  {'p95':<20} {p95:>12.2f}")
    print(f"  {'p99':<20} {p99:>12.2f}")
    print(f"  {'Max':<20} {max_val:>12.2f}")
    print(f"  {'Std Dev':<20} {std_val:>12.2f}")

    print("\n3. FRACTION PROFITABLE AT BENCHMARK ROUND-TRIP FEE HURDLES:")
    print(f"  {'Hurdle Tier':<25} {'Threshold':<15} {'Profitable Events':<18} {'Fraction Profitable':<20}")
    print("  " + "-" * 78)
    for h in hurdles:
        cnt, pct = fractions[h]
        desc = (
            "Zero Cost (Gross > Slip)" if h == 0.0 else
            "Ultra-VIP Maker (5 bps)" if h == 5.0 else
            "Tier-1 VIP / Neg Maker (10 bps)" if h == 10.0 else
            "Standard Institutional (20 bps)"
        )
        print(f"  {desc:<25} {f'> {h:.0f} bps':<15} {f'{cnt} / {total_n}':<18} {f'{pct:6.2f}%':<20}")

    print("\n4. BREAK-EVEN DISTRIBUTION BY PAIR:")
    for sym, vals in sorted(per_symbol.items()):
        s_arr = np.array(vals)
        s_cnt_0 = int(np.sum(s_arr > 0.0))
        s_cnt_5 = int(np.sum(s_arr > 5.0))
        print(f"  - {sym:<10} (N={len(vals):>3}): Mean = {np.mean(s_arr):4.2f} bps | Median = {np.median(s_arr):4.2f} bps | "
              f"P(>0bps) = {s_cnt_0/len(vals)*100:5.1f}% | P(>5bps) = {s_cnt_5/len(vals)*100:5.1f}%")

    print("=" * 90)

    return {
        "total_events": total_n,
        "mean_bps": mean_val,
        "median_bps": p50,
        "p10_bps": p10,
        "p90_bps": p90,
        "fractions": fractions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArbSense Break-Even Spread Analysis")
    parser.add_argument("--db", default="data/arbsense.db", help="Path to SQLite database")
    parser.add_argument("--version", default=None, help="Dataset version (defaults to active config)")
    parser.add_argument("--include-censored", action="store_true", help="Include right-censored events")
    args = parser.parse_args()

    run_break_even_analysis(
        db_path=args.db,
        version=args.version,
        exclude_censored=not args.include_censored,
    )
