"""Morning Diagnostic & Analytics Report for ArbSense Long Run.

Connects to data/arbsense.db in read-only mode, computes:
- Run uptime and event generation rate (events/hour)
- Total events and feature window coverage
- Persistence horizon label balances (500ms, 2s, 5s)
- Right-censoring share
- Per-(symbol, direction) event counts and average metrics
- Latest 60s status line from logs/run_live.log
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_morning_report(db_path: str = "data/arbsense.db", log_path: str = "logs/run_live.log"):
    full_db = PROJECT_ROOT / db_path
    full_log = PROJECT_ROOT / log_path

    if not full_db.exists():
        print(f"[Error] Database file not found at: {full_db}")
        return

    conn = sqlite3.connect(f"file:{full_db.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Run Metadata & Start Time
    cur.execute("SELECT run_id, dataset_version, config_hash, git_commit, latency_threshold_ms, created_at FROM run_metadata ORDER BY run_id DESC LIMIT 1")
    meta_row = cur.fetchone()
    meta_dict = dict(meta_row) if meta_row else {}

    # 2. Event Aggregate Metrics
    cur.execute("""
        SELECT 
            COUNT(*) as tot_ev,
            SUM(feature_window_json IS NOT NULL) as with_win,
            SUM(is_viable = 1) as viable_cnt,
            SUM(is_right_censored = 1) as censored_cnt,
            SUM(is_executable = 1) as exec_cnt,
            SUM(is_persistent_500ms = 1) as p500_cnt,
            SUM(is_persistent_2000ms = 1) as p2000_cnt,
            SUM(is_persistent_5000ms = 1) as p5000_cnt,
            MIN(open_time) as min_t,
            MAX(open_time) as max_t
        FROM spread_events
        WHERE dataset_version = 'v2.0'
    """)
    agg = dict(cur.fetchone())

    # 3. Tick Aggregates
    try:
        cur.execute("SELECT COUNT(*) FROM orderbook_ticks")
        tot_ticks = cur.fetchone()[0]
    except Exception:
        tot_ticks = 0

    # 4. Grouped per (symbol, direction)
    cur.execute("""
        SELECT 
            symbol,
            buy_exchange,
            sell_exchange,
            COUNT(*) as n,
            AVG(open_spread_bps) as avg_open,
            AVG(peak_spread_bps) as avg_peak,
            AVG(duration_ms) as avg_dur
        FROM spread_events
        WHERE dataset_version = 'v2.0'
        GROUP BY symbol, buy_exchange, sell_exchange
        ORDER BY n DESC
    """)
    grouped_rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Calculate Uptime
    now_ts = time.time()
    min_t = agg.get("min_t")
    max_t = agg.get("max_t")

    uptime_hours = 0.01
    uptime_str = "0h 01m"
    if meta_dict.get("created_at"):
        try:
            # Format: '2026-11-22 19:30:58'
            dt_start = datetime.strptime(meta_dict["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            uptime_sec = max(1.0, datetime.now(timezone.utc).timestamp() - dt_start.timestamp())
            uptime_hours = max(0.01, uptime_sec / 3600.0)
            h = int(uptime_sec // 3600)
            m = int((uptime_sec % 3600) // 60)
            uptime_str = f"{h}h {m:02d}m ({uptime_sec:.0f}s)"
        except Exception:
            pass
    elif min_t and max_t:
        try:
            span_ms = float(max_t) - float(min_t)
            uptime_hours = max(0.01, span_ms / 3600000.0)
            h = int(uptime_hours)
            m = int((uptime_hours - h) * 60)
            uptime_str = f"{h}h {m:02d}m"
        except Exception:
            pass

    tot_ev = agg["tot_ev"] or 0
    with_win = agg["with_win"] or 0
    censored = agg["censored_cnt"] or 0
    ev_per_hr = tot_ev / uptime_hours
    win_pct = (with_win / max(1, tot_ev)) * 100.0
    cens_pct = (censored / max(1, tot_ev)) * 100.0

    print("=" * 95)
    print("ARBSENSE MORNING DIAGNOSTIC REPORT (v2.0 Long Run)")
    print("=" * 95)
    print(f"Timestamp:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database:                {full_db.name} (Read-Only Mode)")
    print(f"Active Dataset Version:  {meta_dict.get('dataset_version', 'v2.0')}")
    print(f"Active Git Commit:       {meta_dict.get('git_commit', 'unknown')[:10]}")
    print(f"Config Hash:             {meta_dict.get('config_hash', 'unknown')[:16]}...")
    print(f"Latency Threshold:       {meta_dict.get('latency_threshold_ms', 85.0)} ms")
    print(f"Run Uptime:              {uptime_str}")
    print(f"Total Ticks Ingested:    {tot_ticks:,}")
    print(f"Total Spread Events:     {tot_ev:,}")
    print(f"Event Generation Rate:   {ev_per_hr:.1f} events/hour ({ev_per_hr * 24:.0f} estimated/day)")
    print(f"Events with 20-Step Win: {with_win} ({win_pct:.1f}%) [Requirement: 100%]")
    print(f"Right-Censored Share:    {censored} ({cens_pct:.2f}%) [Requirement: < 5%]")

    print("\n" + "-" * 95)
    print("PERSISTENCE HORIZON LABEL BALANCES (Non-Censored Samples)")
    print("-" * 95)
    uncensored_n = max(1, tot_ev - censored)
    for lbl, cnt in [
        ("is_persistent_500ms (Primary Target)", agg["p500_cnt"] or 0),
        ("is_persistent_2000ms", agg["p2000_cnt"] or 0),
        ("is_persistent_5000ms", agg["p5000_cnt"] or 0),
        ("is_executable (Fee-Independent)", agg["exec_cnt"] or 0),
    ]:
        pos = cnt
        neg = tot_ev - pos
        pct = (pos / max(1, tot_ev)) * 100.0
        print(f"  {lbl:<38}: {pos:>5} pos / {neg:>5} neg ({pct:>5.1f}%)")

    print("\n" + "-" * 95)
    print("PER-(SYMBOL, DIRECTION) SPREAD DISTRIBUTION")
    print("-" * 95)
    print(f"{'Symbol':<10} {'Direction':<24} {'Count':<7} {'Avg Open bps':<15} {'Avg Peak bps':<15} {'Avg Dur (ms)':<15}")
    print("-" * 95)
    if grouped_rows:
        for r in grouped_rows:
            direction = f"{r['buy_exchange']} -> {r['sell_exchange']}"
            print(f"{r['symbol']:<10} {direction:<24} {r['n']:<7} {r['avg_open']:<15.2f} {r['avg_peak']:<15.2f} {r['avg_dur']:<15.1f}")
    else:
        print("  (No events recorded yet)")

    print("\n" + "-" * 95)
    print("LATEST COLLECTOR STATUS LINE (from logs/run_live.log)")
    print("-" * 95)
    last_status = "No status lines logged yet."
    if full_log.exists():
        with open(full_log, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if "[STATUS 60s]" in line]
            if lines:
                last_status = lines[-1]
    print(f"  {last_status}")
    print("=" * 95)


if __name__ == "__main__":
    generate_morning_report()
