"""Helper script to inspect and print 20 raw spread events with lifetime ticks."""
import json
import sqlite3
import sys

def main():
    db_path = "data/arbsense.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT 
            event_id, symbol, buy_exchange, sell_exchange,
            open_time, close_time, open_spread_bps, peak_spread_bps,
            cost_hurdle_bps, duration_ms, is_viable, is_suspect,
            lifetime_ticks_json
        FROM spread_events
        WHERE duration_ms IS NOT NULL
        ORDER BY open_time ASC
        LIMIT 20
    """)
    rows = c.fetchall()
    print("=" * 100)
    print(f"RAW SPREAD EVENTS AUDIT ({len(rows)} Events with Lifetime Ticks)")
    print("=" * 100)
    
    for i, r in enumerate(rows, 1):
        evt_id, sym, buy_ex, sell_ex, o_t, c_t, o_s, p_s, h_s, dur, via, susp, ticks_json = r
        ticks = json.loads(ticks_json) if ticks_json else []
        print(f"\n--- [Event #{i}] ID: {evt_id} | {sym} ({buy_ex} -> {sell_ex}) ---")
        print(f"    Open Spread:    {o_s:.2f} bps | Peak Spread: {p_s:.2f} bps | Cost Hurdle: {h_s:.2f} bps")
        print(f"    Real Duration:  {dur:.1f} ms ({len(ticks)} ticks observed during lifetime)")
        print(f"    Viable:         {'YES' if via else 'NO'} | Suspect: {'YES' if susp else 'NO'}")
        print("    Observed Lifetime Ticks:")
        for j, t in enumerate(ticks, 1):
            t_ms = t.get("time_ms", 0.0)
            gross = t.get("gross_spread_bps", 0.0)
            ask = t.get("buy_ask", 0.0)
            bid = t.get("sell_bid", 0.0)
            print(f"       Tick {j:02d} @ {t_ms:.0f} ms: Gross={gross:6.2f} bps | Buy Ask={ask:10.2f} | Sell Bid={bid:10.2f}")

if __name__ == "__main__":
    main()
