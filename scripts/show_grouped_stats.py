"""Calculates and displays comprehensive grouped statistics on the fresh database (Step 4)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import numpy as np
from collections import defaultdict


def main(db_path: str = "data/arbsense.db"):
    from data.storage.db_manager import DatabaseManager
    # Ensure migrations are applied if connecting to existing DB
    try:
        db_mgr = DatabaseManager(sqlite_path=db_path)
        db_mgr.initialize()
        db_mgr.close()
    except Exception:
        pass

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("=" * 125)
    print("FRESH RUN: DATABASE INTEGRITY & SUMMARY STATISTICS (Step 4)")
    print("=" * 125)

    c.execute("""
        SELECT 
            COUNT(*) as tot_ev,
            SUM(feature_window_json IS NOT NULL) as with_win,
            SUM(is_viable = 1) as viable,
            SUM(is_right_censored = 1) as censored,
            SUM(is_suspect = 1) as suspect,
            SUM(is_executable = 1) as executable,
            SUM(is_profitable_retail = 1) as prof_retail,
            SUM(is_profitable_vip1 = 1) as prof_vip1,
            SUM(is_profitable_vip3 = 1) as prof_vip3,
            SUM(is_profitable_zero_maker = 1) as prof_zero_maker,
            MIN(open_time) as min_t,
            MAX(open_time) as max_t
        FROM spread_events
        WHERE dataset_version != 'invalid_v0' OR dataset_version IS NULL
    """)
    row = dict(c.fetchone())
    tot_ev = row["tot_ev"] or 0
    with_win = row["with_win"] or 0
    viable = row["viable"] or 0
    censored = row["censored"] or 0
    suspect = row["suspect"] or 0
    executable = row["executable"] or 0
    min_t = row["min_t"]
    max_t = row["max_t"]

    tot_ticks = c.execute("SELECT COUNT(*) FROM orderbook_ticks").fetchone()[0]
    try:
        cur_seq = c.execute("SELECT seq FROM sqlite_sequence WHERE name = 'orderbook_ticks'").fetchone()
        cumul_ticks = cur_seq[0] if cur_seq else tot_ticks
    except Exception:
        cumul_ticks = tot_ticks

    print(f"Total Ticks in Active Buffer:    {tot_ticks}")
    print(f"Total Cumulative Ingested Ticks: {cumul_ticks}")
    print(f"Total Spread Events Opened:      {tot_ev}")
    print(f"Events with 20-Step Windows:     {with_win} ({(with_win/max(1,tot_ev))*100:.1f}%)")
    print(f"Viable Events:                   {viable} ({(viable/max(1,tot_ev))*100:.2f}%)")
    print(f"Right-Censored Events:           {censored} ({(censored/max(1,tot_ev))*100:.2f}%)")
    print(f"Suspect Flagged Events (>80bps): {suspect} ({(suspect/max(1,tot_ev))*100:.2f}%)")
    print(f"Earliest Event Time (Epoch ms):  {min_t}")
    print(f"Latest Event Time (Epoch ms):    {max_t}")

    # Compute run duration
    elapsed_hours = 0.5  # default minimum 30 min
    if min_t is not None and max_t is not None:
        try:
            diff_ms = float(max_t) - float(min_t)
            if diff_ms > 0:
                elapsed_hours = max(0.01, diff_ms / 3600000.0)
        except Exception:
            pass

    print("\n" + "=" * 125)
    print("STEP 3 & 4: LABEL BALANCE ACROSS ALL TARGETS & FEE SCENARIOS")
    print("=" * 125)
    labels_to_show = [
        ("is_executable (Primary)", executable),
        ("is_profitable_retail", row["prof_retail"] or 0),
        ("is_profitable_vip1", row["prof_vip1"] or 0),
        ("is_profitable_vip3", row["prof_vip3"] or 0),
        ("is_profitable_zero_maker", row["prof_zero_maker"] or 0),
        ("is_viable (Legacy)", viable),
    ]
    for lbl_desc, pos_count in labels_to_show:
        neg_count = tot_ev - pos_count
        pos_pct = (pos_count / max(1, tot_ev)) * 100.0
        print(f"  {lbl_desc:<32}: {pos_count:>5} positive / {neg_count:>5} negative ({pos_pct:>5.1f}%)")

    print("\n" + "=" * 125)
    print("GROUPED METRICS PER (SYMBOL, BUY, SELL) WITH AVG / P50 / P90 AND PROJECTIONS")
    print("=" * 125)

    c.execute("""
        SELECT 
            symbol,
            buy_exchange,
            sell_exchange,
            open_spread_bps,
            peak_spread_bps,
            duration_ms,
            cost_hurdle_bps,
            feature_window_json,
            is_suspect,
            is_executable
        FROM spread_events
        WHERE dataset_version != 'invalid_v0' OR dataset_version IS NULL
        ORDER BY symbol, buy_exchange, sell_exchange
    """)
    all_events = [dict(r) for r in c.fetchall()]

    # Group by (symbol, buy, sell)
    groups = defaultdict(list)
    for ev in all_events:
        key = (ev["symbol"], ev["buy_exchange"], ev["sell_exchange"])
        groups[key].append(ev)

    header = (
        f"{'Symbol':<9} {'Buy':<8} {'Sell':<8} {'N':<5} "
        f"{'Open bps (avg/p50/p90)':<24} "
        f"{'Peak bps (avg/p50/p90)':<24} "
        f"{'Duration ms (avg/p50/p90)':<26} "
        f"{'Win':<4} {'Susp':<4} {'Ev/hr':<7} {'Est/day'}"
    )
    print(header)
    print("-" * len(header))

    # Sort groups by count descending
    sorted_keys = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)

    for k in sorted_keys:
        evs = groups[k]
        n = len(evs)
        sym, buy_ex, sell_ex = k

        opens = [float(e["open_spread_bps"]) for e in evs if e["open_spread_bps"] is not None]
        peaks = [float(e["peak_spread_bps"]) for e in evs if e["peak_spread_bps"] is not None]
        durs = [float(e["duration_ms"]) for e in evs if e["duration_ms"] is not None]
        n_win = sum(1 for e in evs if e["feature_window_json"] is not None)
        n_susp = sum(1 for e in evs if e.get("is_suspect", 0) == 1)

        avg_o, p50_o, p90_o = np.mean(opens) if opens else 0.0, np.percentile(opens, 50) if opens else 0.0, np.percentile(opens, 90) if opens else 0.0
        avg_p, p50_p, p90_p = np.mean(peaks) if peaks else 0.0, np.percentile(peaks, 50) if peaks else 0.0, np.percentile(peaks, 90) if peaks else 0.0
        avg_d, p50_d, p90_d = np.mean(durs) if durs else 0.0, np.percentile(durs, 50) if durs else 0.0, np.percentile(durs, 90) if durs else 0.0

        o_str = f"{avg_o:.1f}/{p50_o:.1f}/{p90_o:.1f}"
        p_str = f"{avg_p:.1f}/{p50_p:.1f}/{p90_p:.1f}"
        d_str = f"{avg_d:.0f}/{p50_d:.0f}/{p90_d:.0f}"

        ev_per_hr = n / elapsed_hours
        est_per_day = ev_per_hr * 24.0

        print(
            f"{sym:<9} {buy_ex:<8} {sell_ex:<8} {n:<5} "
            f"{o_str:<24} "
            f"{p_str:<24} "
            f"{d_str:<26} "
            f"{n_win:<4} {n_susp:<4} {ev_per_hr:<7.1f} {est_per_day:<7.0f}"
        )

    # Per-pair hourly and daily projections
    print("\n" + "=" * 125)
    print("PAIR-LEVEL EVENT FREQUENCY & DAILY PROJECTIONS")
    print("=" * 125)
    pair_counts = defaultdict(int)
    for (sym, _, _), evs in groups.items():
        pair_counts[sym] += len(evs)

    for sym, cnt in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
        hr_rate = cnt / elapsed_hours
        day_rate = hr_rate * 24.0
        print(f"  {sym:<10}: {cnt:>5} events | {hr_rate:>6.1f} events/hour | {day_rate:>7.0f} estimated events/day")

    conn.close()


if __name__ == "__main__":
    main()
