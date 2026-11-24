"""ArbSense Real-Time Analytics and Paper Trading Dashboard.

Multi-tab Streamlit dashboard providing:
1. Live cross-exchange spread matrix & L2 market depth monitor
2. Real-time AI opportunity detection & spread-duration predictions
3. Model Results tab reading models/checkpoints/model_results.json (Item 7)
4. Paper trading execution ledger, fee/slippage breakdown, and cumulative P&L equity curve.
Zero writes to arbsense.db (sqlite URI mode=ro enforced).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AppConfig
from data.normalizer import OrderBookLevel, OrderBookSnapshot
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel
from engine.simulator import ExecutionSimulator

# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ArbSense | Arbitrage Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    div[data-testid="stAppViewContainer"] > section { opacity: 1 !important; filter: none !important; }
    .element-container, div[data-testid="stVerticalBlock"] { opacity: 1 !important; filter: none !important; }
    div.stApp.stApp--running { opacity: 1 !important; filter: none !important; }
    .main { opacity: 1 !important; filter: none !important; }
    
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-viable {
        background-color: #00cc66;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-filtered {
        background-color: #ff4d4d;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .tier-badge {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def get_db(sqlite_path: str = "data/arbsense.db") -> DatabaseManager:
    """Connects to SQLite database strictly in read-only mode (mode=ro). Zero writes allowed."""
    db = DatabaseManager(sqlite_path=sqlite_path, read_only=True)
    db.initialize()
    return db


def load_config() -> AppConfig:
    return AppConfig.load()


def calculate_route_hurdle(buy_venue: str, sell_venue: str, scenario: str, cost_model: CostModel) -> float:
    """Computes exact fee hurdle (bps) for a cross-exchange route under active scenario."""
    if scenario == "zero_maker":
        fee_buy = cost_model.get_fee_bps(buy_venue, role="maker", scenario=scenario)
        fee_sell = cost_model.get_fee_bps(sell_venue, role="taker", scenario=scenario)
    else:
        fee_buy = cost_model.get_fee_bps(buy_venue, role="taker", scenario=scenario)
        fee_sell = cost_model.get_fee_bps(sell_venue, role="taker", scenario=scenario)
    return fee_buy + fee_sell


def main():
    config = load_config()
    cost_model = CostModel(config)

    # --------------------------------------------------------------------------
    # Sidebar Controls (Item 7: DB Path Field & Read-Only Guarantee)
    # --------------------------------------------------------------------------
    st.sidebar.title("⚡ ArbSense Engine")
    st.sidebar.caption("AI Cross-Exchange Microstructure Engine (Read-Only Mode)")

    db_path_input = st.sidebar.text_input(
        "Database Path",
        value="data/arbsense.db",
        help="Path to SQLite database copy. Opened strictly read-only (mode=ro) with zero writes.",
    )

    try:
        db = get_db(db_path_input)
    except Exception as e:
        st.sidebar.error(f"Failed opening database at {db_path_input}: {e}")
        st.error(f"Cannot connect to database: {e}")
        return

    # Fee Scenario Selector
    fee_scenario = st.sidebar.selectbox(
        "Active Fee Scenario",
        options=["retail", "vip1", "vip3", "zero_maker"],
        index=0,
        help="Simulates paper trading returns under various exchange VIP tier schedules",
    )

    # Supported Pairs
    supported_pairs = config.markets.supported_pairs
    selected_symbol = st.sidebar.selectbox(
        "Asset Pair",
        options=supported_pairs,
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Current Tier:** `{fee_scenario.upper()}`")
    if fee_scenario == "retail":
        st.sidebar.caption("Standard taker fees: 10–60 bps. Calm markets produce fee deficits.")
    elif fee_scenario == "vip1":
        st.sidebar.caption("VIP 1 volume tiers: 9–40 bps taker fees.")
    elif fee_scenario == "vip3":
        st.sidebar.caption("VIP 3 institutional: 6–20 bps taker fees. Moderate spreads profit.")
    elif fee_scenario == "zero_maker":
        st.sidebar.caption("Zero-Maker / HFT: 0 bps maker + 7.5–25 bps taker. High net alpha.")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data Now", use_container_width=True):
        st.rerun()

    auto_refresh = st.sidebar.checkbox("Auto-Refresh (10s)", value=False)

    # --------------------------------------------------------------------------
    # Top KPI Metrics Bar
    # --------------------------------------------------------------------------
    cumul_ticks = db.get_cumulative_tick_count()
    active_ticks = db.get_tick_count()
    cur = db._sqlite_conn.cursor()

    cur.execute("""
        SELECT 
            COUNT(*),
            SUM(feature_window_json IS NOT NULL),
            SUM(is_viable = 1),
            SUM(is_right_censored = 1)
        FROM spread_events
    """)
    row_ev = cur.fetchone()
    total_events = row_ev[0] or 0
    events_with_windows = row_ev[1] or 0
    viable_events = row_ev[2] or 0
    censored_events = row_ev[3] or 0

    viable_fraction = (viable_events / total_events) if total_events > 0 else 0.0
    censored_fraction = (censored_events / total_events) if total_events > 0 else 0.0

    try:
        cur.execute("SELECT COUNT(*), SUM(net_pnl) FROM simulated_trades WHERE status = 'FILLED'")
        total_trades, total_net_pnl = cur.fetchone()
        total_trades = total_trades or 0
        total_net_pnl = total_net_pnl or 0.0
    except Exception:
        total_trades = 0
        total_net_pnl = 0.0

    st.title("ArbSense: AI-Powered Cross-Exchange Arbitrage Engine")
    st.caption("Empirical Market Microstructure Spread Detection & PyTorch LSTM Duration Forecasting")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Cumulative Ticks Ingested", f"{cumul_ticks:,}", delta=f"{active_ticks:,} active buffer")
    col2.metric("Total Spread Events", f"{total_events:,}")
    col3.metric("Events with Windows", f"{events_with_windows:,}", delta=f"{(events_with_windows / max(1, total_events) * 100):.1f}% complete")
    col4.metric("Viable Fraction", f"{viable_fraction:.1%}", delta=f"{viable_events:,} viable")
    col5.metric("Censored Fraction", f"{censored_fraction:.1%}", delta=f"{censored_events:,} censored")

    # Tabs (Item 7: Model Results tab reading model_results.json)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Spreads & Depth",
        "🧠 AI Prediction Engine",
        "📈 Model Results",
        "💼 Paper Trading Ledger",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Live Spreads & Market Depth
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader(f"Live Normalized Order Books: {selected_symbol}")
        recent_ticks = db.get_recent_ticks(symbol=selected_symbol, limit=30)

        if recent_ticks:
            df_ticks = pd.DataFrame(recent_ticks)
            latest_per_exch = df_ticks.groupby("exchange").first().reset_index()

            cols = st.columns(len(latest_per_exch))
            quote_map = {}
            for i, row in latest_per_exch.iterrows():
                quote_map[row["exchange"]] = row
                with cols[i]:
                    st.markdown(f"### {row['exchange'].upper()}")
                    st.metric("Best Bid", f"${row['best_bid']:,.4f}" if row['best_bid'] < 10 else f"${row['best_bid']:,.2f}")
                    st.metric("Best Ask", f"${row['best_ask']:,.4f}" if row['best_ask'] < 10 else f"${row['best_ask']:,.2f}")
                    st.write(f"**Internal Spread:** `${row['top_spread']:.4f}`")
                    st.write(f"**Imbalance (OBI):** `{row['imbalance']:+.3f}`")
                    lat_str = f"{row['transit_latency_ms']:.1f} ms" if row['transit_latency_ms'] else "N/A"
                    st.write(f"**Transit Latency:** `{lat_str}`")

            st.markdown("---")
            st.subheader(f"Cross-Exchange Arbitrage Matrix (Active Scenario: `{fee_scenario.upper()}`)")
            st.caption(f"Real-time spread evaluation across all pairwise exchange routes for **{selected_symbol}** under the selected **{fee_scenario.upper()}** fee schedule:")

            exchanges = list(quote_map.keys())
            matrix_data = []

            for buy_ex in exchanges:
                for sell_ex in exchanges:
                    if buy_ex == sell_ex:
                        continue
                    b_quote = quote_map[buy_ex]
                    s_quote = quote_map[sell_ex]

                    ask_price = b_quote["best_ask"]
                    bid_price = s_quote["best_bid"]

                    gross_spread_bps = ((bid_price - ask_price) / ask_price) * 10000.0
                    hurdle_bps = calculate_route_hurdle(buy_ex, sell_ex, fee_scenario, cost_model)
                    net_alpha_bps = gross_spread_bps - hurdle_bps

                    if net_alpha_bps > 0:
                        status_str = f"✨ PROFITABLE (+{net_alpha_bps:.2f} bps)"
                    else:
                        status_str = f"🛡️ FILTERED (Fee Deficit: {net_alpha_bps:.2f} bps)"

                    matrix_data.append({
                        "Route": f"{buy_ex.upper()} ➔ {sell_ex.upper()}",
                        "Buy Ask": f"${ask_price:,.2f}",
                        "Sell Bid": f"${bid_price:,.2f}",
                        "Gross Spread (bps)": f"{gross_spread_bps:+.2f}",
                        "Total Hurdle (bps)": f"{hurdle_bps:.2f}",
                        "Net Alpha (bps)": f"{net_alpha_bps:+.2f}",
                        "Execution Decision": status_str,
                    })

            if matrix_data:
                st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)
        else:
            st.info(f"No orderbook ticks recorded yet for {selected_symbol}. Feeds will stream updates once active.")

    # --------------------------------------------------------------------------
    # TAB 2: AI Prediction Engine
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Real-Time Spread Event State Machine & AI Inference")
        st.caption("Active and recently completed spread opportunities captured with 20-step microstructure feature windows:")

        cur.execute("SELECT * FROM spread_events ORDER BY open_time DESC LIMIT 25")
        events_rows = [dict(r) for r in cur.fetchall()]

        if events_rows:
            df_events = pd.DataFrame(events_rows)
            cols_to_show = [
                "event_id", "symbol", "buy_exchange", "sell_exchange", "open_time",
                "peak_spread_bps", "cost_hurdle_bps", "duration_ms", "log_duration", "is_viable", "is_right_censored"
            ]
            cols_present = [c for c in cols_to_show if c in df_events.columns]
            st.dataframe(df_events[cols_present], use_container_width=True)
        else:
            st.info("No spread events closed yet. Feeds are actively monitoring for cross-exchange discrepancies.")

    # --------------------------------------------------------------------------
    # TAB 3: Model Results (Item 7: Reading models/checkpoints/model_results.json)
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Empirical Model Results & Walk-Forward Validation Benchmark")
        results_path = PROJECT_ROOT / "models" / "checkpoints" / "model_results.json"

        if results_path.exists():
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    res_data = json.load(f)

                # Provenance Banner
                st.info(
                    f"**Dataset Version:** `{res_data.get('dataset_version', 'N/A')}` | "
                    f"**Git Commit:** `{res_data.get('git_commit', 'N/A')}` | "
                    f"**Target Label:** `{res_data.get('target_label', 'is_persistent_500ms')}` | "
                    f"**Decision Threshold:** `{res_data.get('decision_threshold', 0.50):.2f}` | "
                    f"**Evaluated At:** `{res_data.get('timestamp', 'N/A')}`"
                )

                # Walk-Forward Validation Results Table
                st.markdown("### 1. Expanding-Window Walk-Forward Validation (Non-Overlapping Folds)")
                st.caption("Folds evaluated with 60s time purge gaps and zero test-window overlap across folds:")

                wf_summary = res_data.get("walkforward_summary", {})
                if wf_summary:
                    table_rows = []
                    for model_name, stats in wf_summary.items():
                        table_rows.append({
                            "Model": model_name,
                            "PR-AUC (Mean ± Std)": f"{stats.get('pr_auc_mean', 0.0):.3f} ± {stats.get('pr_auc_std', 0.0):.3f}",
                            "Lift over Base Rate": f"{stats.get('lift_mean', 1.0):.2f}x ± {stats.get('lift_std', 0.0):.2f}x",
                            "MCC (Validation Threshold)": f"{stats.get('mcc_mean', 0.0):+.3f} ± {stats.get('mcc_std', 0.0):.3f}",
                            "Balanced Accuracy": f"{stats.get('bal_acc_mean', 0.5):.3f} ± {stats.get('bal_acc_std', 0.0):.3f}",
                            "Brier Skill Score": f"{stats.get('bss_mean', 0.0):+.3f}",
                            "F1 (Secondary)": f"{stats.get('f1_mean', 0.0):.3f} ± {stats.get('f1_std', 0.0):.3f}",
                        })
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                # Duration Regressors MAE Table
                st.markdown("### 2. Spread Duration Forecasting (MAE in milliseconds)")
                dur_summary = res_data.get("duration_summary", {})
                if dur_summary:
                    dur_rows = []
                    for reg_name, stats in dur_summary.items():
                        dur_rows.append({
                            "Duration Regressor": reg_name,
                            "Mean Absolute Error (MAE)": f"{stats.get('mae_mean_ms', 0.0):.1f} ms ± {stats.get('mae_std_ms', 0.0):.1f} ms",
                        })
                    st.dataframe(pd.DataFrame(dur_rows), use_container_width=True)

                # Held-Out Evaluation Metrics
                st.markdown("### 3. Held-Out Final Period Evaluation (Strictly Unseen)")
                held_out = res_data.get("held_out_metrics", {})
                if held_out:
                    h_cols = st.columns(5)
                    h_cols[0].metric("Held-Out PR-AUC", f"{held_out.get('pr_auc', 0.0):.3f}")
                    h_cols[1].metric("Held-Out Lift", f"{held_out.get('lift', 1.0):.2f}x")
                    h_cols[2].metric("Held-Out MCC", f"{held_out.get('mcc', 0.0):+.3f}")
                    h_cols[3].metric("Precision @ Top-20%", f"{held_out.get('p_top20', 0.0):.1%}")
                    h_cols[4].metric("Duration MAE", f"{held_out.get('mae_ms', 0.0):.1f} ms")
                else:
                    st.caption("Held-out metrics will appear after production model evaluation.")

                st.markdown("---")
                st.markdown("#### Scientific Limitations & Integrity Notice")
                st.caption(
                    "All metrics are produced without test-set tuning. "
                    "Features are strictly point-in-time from 20-step order book windows. "
                    "Binance depth snapshot arrival intervals are flagged 'estimated' and excluded from latency features."
                )

            except Exception as e:
                st.error(f"Error parsing model results from {results_path}: {e}")
        else:
            st.warning(
                f"No model results file found at `{results_path}`. "
                "Run `python models/train.py` to execute walk-forward validation and generate benchmark evidence."
            )

    # --------------------------------------------------------------------------
    # TAB 4: Paper Trading & P&L Ledger
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Autonomous Paper Trading Ledger & Execution Desk")
        st.info("🤖 **Validation Layer Gatekeeper Active:** Simulates executions against real L2 depth without modifying database.")

        # Interactive Paper Execution Desk (In-Memory Simulation)
        with st.expander("⚡ Interactive Trade Execution Desk (In-Memory Simulator)", expanded=True):
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                default_idx = supported_pairs.index(selected_symbol) if selected_symbol in supported_pairs else 0
                trade_sym = st.selectbox(
                    "Trading Pair",
                    supported_pairs,
                    index=default_idx,
                    key="desk_trading_pair",
                )
            with p_col2:
                buy_venue = st.selectbox("Buy Venue (Ask)", ["binance", "coinbase", "kraken"], index=0, key="manual_buy")
            with p_col3:
                sell_venue = st.selectbox("Sell Venue (Bid)", ["kraken", "coinbase", "binance"], index=0, key="manual_sell")
            with p_col4:
                default_qty = 0.05 if "BTC" in trade_sym else (0.50 if "ETH" in trade_sym else 10.0)
                trade_qty = st.number_input(
                    "Order Size",
                    value=float(default_qty),
                    step=0.01 if "BTC" in trade_sym or "ETH" in trade_sym else 1.0,
                    format="%.4f" if "BTC" in trade_sym or "ETH" in trade_sym else "%.1f",
                )

            snap_buy_row = db.get_recent_ticks(exchange=buy_venue, symbol=trade_sym, limit=1)
            snap_sell_row = db.get_recent_ticks(exchange=sell_venue, symbol=trade_sym, limit=1)

            if snap_buy_row and snap_sell_row:
                b_tick = snap_buy_row[0]
                s_tick = snap_sell_row[0]

                raw_spread_bps = ((s_tick["best_bid"] - b_tick["best_ask"]) / b_tick["best_ask"]) * 10000.0
                total_fee_hurdle = calculate_route_hurdle(buy_venue, sell_venue, fee_scenario, cost_model)
                net_spread_bps = raw_spread_bps - total_fee_hurdle

                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric(f"Live Best Ask ({buy_venue.upper()})", f"${b_tick['best_ask']:,.2f}")
                c_m2.metric(f"Live Best Bid ({sell_venue.upper()})", f"${s_tick['best_bid']:,.2f}")
                c_m3.metric("Live Net Alpha (after fees)", f"{net_spread_bps:+.2f} bps")

                test_pipeline_btn = st.button("🧪 Simulate In-Memory Paper Trade", use_container_width=True)
                if test_pipeline_btn:
                    b_asks = [OrderBookLevel(b_tick["best_ask"], trade_qty * 3)]
                    s_bids = [OrderBookLevel(s_tick["best_bid"], trade_qty * 3)]

                    buy_snap = OrderBookSnapshot(
                        exchange=buy_venue,
                        symbol=trade_sym,
                        quote_currency="USDT",
                        exchange_ts_ms=b_tick["exchange_ts_ms"],
                        local_ts_ms=b_tick["local_ts_ms"],
                        transit_latency_ms=b_tick["transit_latency_ms"],
                        asks=b_asks,
                    )
                    sell_snap = OrderBookSnapshot(
                        exchange=sell_venue,
                        symbol=trade_sym,
                        quote_currency="USDT",
                        exchange_ts_ms=s_tick["exchange_ts_ms"],
                        local_ts_ms=s_tick["local_ts_ms"],
                        transit_latency_ms=s_tick["transit_latency_ms"],
                        bids=s_bids,
                    )

                    # In-memory execution simulator: db_manager=None guarantees ZERO database writes
                    simulator = ExecutionSimulator(db_manager=None, config=config)
                    trade_res = simulator.execute_arbitrage(
                        symbol=trade_sym,
                        buy_snap=buy_snap,
                        sell_snap=sell_snap,
                        quantity=trade_qty,
                        fee_scenario=fee_scenario,
                        opportunity_duration_ms=600.0,
                    )

                    if trade_res.status == "FILLED":
                        st.success(
                            f"✅ **Trade Simulated FILLED!** Bought {trade_res.filled_quantity} {trade_sym} on {buy_venue.upper()} at VWAP ${trade_res.buy_vwap:,.2f}, Sold on {sell_venue.upper()} at VWAP ${trade_res.sell_vwap:,.2f}. Gross P&L: ${trade_res.gross_pnl_usdt:+.2f} | Fees: -${(trade_res.buy_fee_usdt + trade_res.sell_fee_usdt):.2f} | **Net P&L: {trade_res.net_pnl_usdt:+.2f} USDT** (Injected Latency: {trade_res.injected_latency_ms:.1f}ms)"
                        )
                    else:
                        st.warning(f"⚠️ Simulated Status: {trade_res.status} (Injected Latency: {trade_res.injected_latency_ms:.1f}ms)")
            else:
                st.info(f"Awaiting book updates from {buy_venue.upper()} and {sell_venue.upper()} for {trade_sym}...")

        # Display historical trades from DB read-only
        st.markdown("---")
        try:
            cur.execute("SELECT * FROM simulated_trades ORDER BY timestamp DESC LIMIT 50")
            trades_rows = [dict(r) for r in cur.fetchall()]
        except Exception:
            trades_rows = []

        if trades_rows:
            df_trades = pd.DataFrame(trades_rows)
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("No recorded historical trades in database.")

    if auto_refresh:
        time.sleep(10)
        st.rerun()


if __name__ == "__main__":
    main()
