-- ==============================================================================
-- ArbSense TimescaleDB Schema Migration 001
-- Hypertables for High-Frequency Crypto Tick Storage and Arbitrage Analytics
-- ==============================================================================

-- Enable TimescaleDB extension if running in PostgreSQL
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 1. High-Frequency Order Book Ticks Hypertable
CREATE TABLE IF NOT EXISTS orderbook_ticks (
    time TIMESTAMPTZ NOT NULL,
    exchange VARCHAR(32) NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    quote_currency VARCHAR(16) NOT NULL DEFAULT 'USDT',
    best_bid NUMERIC(18, 8) NOT NULL,
    best_ask NUMERIC(18, 8) NOT NULL,
    mid_price NUMERIC(18, 8) NOT NULL,
    top_spread NUMERIC(18, 8) NOT NULL,
    imbalance NUMERIC(8, 4) NOT NULL,
    bid_depth JSONB,
    ask_depth JSONB,
    exchange_ts_ms NUMERIC(18, 3) NOT NULL,
    local_ts_ms NUMERIC(18, 3) NOT NULL,
    transit_latency_ms NUMERIC(12, 3) NOT NULL,
    is_stale BOOLEAN NOT NULL DEFAULT FALSE
);

-- Convert orderbook_ticks into TimescaleDB Hypertable partitioned by time (1-hour chunks)
SELECT create_hypertable('orderbook_ticks', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 hour');

-- Indices for rapid querying
CREATE INDEX IF NOT EXISTS idx_ticks_exch_sym_time ON orderbook_ticks (exchange, symbol, time DESC);

-- 2. Spread Events Table (Arbitrage Lifecycle Windows)
CREATE TABLE IF NOT EXISTS spread_events (
    event_id VARCHAR(64) PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    buy_exchange VARCHAR(32) NOT NULL,
    sell_exchange VARCHAR(32) NOT NULL,
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ,
    peak_spread_bps NUMERIC(10, 4) NOT NULL,
    open_spread_bps NUMERIC(10, 4) NOT NULL,
    cost_hurdle_bps NUMERIC(10, 4) NOT NULL,
    duration_ms NUMERIC(12, 3),
    log_duration NUMERIC(10, 4),
    is_viable BOOLEAN NOT NULL DEFAULT FALSE,
    is_right_censored BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_spread_events_time ON spread_events (open_time DESC);

-- 3. Simulated Trades Table (Paper Execution Ledger)
CREATE TABLE IF NOT EXISTS simulated_trades (
    trade_id VARCHAR(64) PRIMARY KEY,
    event_id VARCHAR(64) REFERENCES spread_events(event_id),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    buy_exchange VARCHAR(32) NOT NULL,
    sell_exchange VARCHAR(32) NOT NULL,
    quantity NUMERIC(18, 8) NOT NULL,
    buy_vwap NUMERIC(18, 8) NOT NULL,
    sell_vwap NUMERIC(18, 8) NOT NULL,
    buy_fee NUMERIC(18, 8) NOT NULL,
    sell_fee NUMERIC(18, 8) NOT NULL,
    gross_pnl NUMERIC(18, 8) NOT NULL,
    net_pnl NUMERIC(18, 8) NOT NULL,
    latency_ms NUMERIC(10, 2) NOT NULL,
    status VARCHAR(32) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_time ON simulated_trades (timestamp DESC);
