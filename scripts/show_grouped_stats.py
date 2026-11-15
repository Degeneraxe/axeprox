"""Calculates and displays comprehensive grouped statistics on the fresh database."""
import sqlite3

def main():
    db_path = "data/arbsense.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    print("=" * 115)
    print("FRESH RUN: DATABASE INTEGRITY & SUMMARY STATISTICS")
    print("=" * 115)

    c.execute("""
        SELECT 
            COUNT(*),
            SUM(feature_window_json IS NOT NULL),
            SUM(is_viable = 1),
            SUM(is_right_censored = 1),
            SUM(is_suspect = 1),
            MIN(open_time),
            MAX(open_time)
        FROM spread_events
    """)
    tot_ev, with_win, viable, censored, suspect, min_t, max_t = c.fetchone()
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

    print("\n" + "=" * 115)
    print("GROUPED STATISTICS PER (SYMBOL, BUY VENUE, SELL VENUE)")
    print("=" * 115)
    
    header = f"{'Symbol':<10} {'Buy Venue':<10} {'Sell Venue':<10} {'Count':<7} {'Avg Open':<10} {'Avg Peak':<10} {'Avg Hurdle':<12} {'Min Dur':<10} {'Avg Dur':<10} {'Max Dur':<10} {'With Win':<9} {'Viable'}"
    print(header)
    print("-" * len(header))

    c.execute("""
        SELECT 
            symbol,
            buy_exchange,
            sell_exchange,
            COUNT(*),
            AVG(open_spread_bps),
            AVG(peak_spread_bps),
            AVG(cost_hurdle_bps),
            MIN(duration_ms),
            AVG(duration_ms),
            MAX(duration_ms),
            SUM(feature_window_json IS NOT NULL),
            SUM(is_viable = 1)
        FROM spread_events
        GROUP BY symbol, buy_exchange, sell_exchange
        ORDER BY COUNT(*) DESC
    """)
    rows = c.fetchall()

    for r in rows:
        sym, b_ex, s_ex, cnt, avg_o, avg_p, avg_h, min_d, avg_d, max_d, w_win, viab = r
        min_d_str = f"{min_d:.1f}ms" if min_d is not None else "N/A"
        avg_d_str = f"{avg_d:.1f}ms" if avg_d is not None else "N/A"
        max_d_str = f"{max_d:.1f}ms" if max_d is not None else "N/A"
        print(
            f"{sym:<10} {b_ex:<10} {s_ex:<10} {cnt:<7} "
            f"{avg_o:<10.2f} {avg_p:<10.2f} {avg_h:<12.2f} "
            f"{min_d_str:<10} {avg_d_str:<10} {max_d_str:<10} "
            f"{w_win:<9} {viab}"
        )

if __name__ == "__main__":
    main()
