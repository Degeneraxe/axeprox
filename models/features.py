"""Microstructure feature engineering module for ArbSense.

Extracts time-series features from order book depth snapshots:
1. Order Book Imbalance (OBI) across top 1, 5, and 10 levels
2. Inter-exchange spread velocity and acceleration (dSpread/dt, d^2Spread/dt^2)
3. Multi-scale rolling price volatility
4. Network transit latency differentials (Buy latency - Sell latency)
5. Micro-price deviations and internal venue spreads
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


FEATURE_NAMES = [
    "gross_spread_bps",
    "spread_velocity",
    "spread_acceleration",
    "obi_buy_top5",
    "obi_sell_top5",
    "obi_differential",
    "rolling_vol_buy_bps",
    "rolling_vol_sell_bps",
    "transit_lat_buy_ms",
    "transit_lat_sell_ms",
    "transit_lat_diff_ms",
    "internal_spread_buy",
    "internal_spread_sell",
    "microprice_diff_bps",
]


def compute_microprice(snap: Any) -> float:
    """Calculates volume-weighted micro-price: (P_ask * V_bid + P_bid * V_ask) / (V_bid + V_ask)."""
    if not snap.best_bid or not snap.best_ask:
        return 0.0
    p_bid, v_bid = snap.best_bid.price, snap.best_bid.amount
    p_ask, v_ask = snap.best_ask.price, snap.best_ask.amount
    tot_vol = v_bid + v_ask
    if tot_vol <= 0:
        return (p_bid + p_ask) / 2.0
    return (p_ask * v_bid + p_bid * v_ask) / tot_vol


def extract_step_features(
    snap_buy: Any,
    snap_sell: Any,
    gross_spread_bps: float,
    prev_spread_bps: Optional[float] = None,
    prev_velocity: Optional[float] = None,
    vol_buy_bps: float = 1.0,
    vol_sell_bps: float = 1.0,
) -> List[float]:
    """Extracts a 14-dimensional feature vector for a single timestep."""
    # 1. Spread dynamics
    velocity = (gross_spread_bps - prev_spread_bps) if prev_spread_bps is not None else 0.0
    accel = (velocity - prev_velocity) if prev_velocity is not None else 0.0

    # 2. Order book imbalance
    obi_buy = snap_buy.compute_imbalance(5)
    obi_sell = snap_sell.compute_imbalance(5)
    obi_diff = obi_buy - obi_sell

    # 3. Transit latency dynamics
    lat_buy = float(snap_buy.transit_latency_ms)
    lat_sell = float(snap_sell.transit_latency_ms)
    lat_diff = lat_buy - lat_sell

    # 4. Internal venue spreads
    top_spread_buy = float(snap_buy.top_spread)
    top_spread_sell = float(snap_sell.top_spread)

    # 5. Micro-price differential
    mp_buy = compute_microprice(snap_buy)
    mp_sell = compute_microprice(snap_sell)
    mp_diff_bps = ((mp_sell - mp_buy) / mp_buy * 10000.0) if mp_buy > 0 else 0.0

    return [
        float(gross_spread_bps),
        float(velocity),
        float(accel),
        float(obi_buy),
        float(obi_sell),
        float(obi_diff),
        float(vol_buy_bps),
        float(vol_sell_bps),
        float(lat_buy),
        float(lat_sell),
        float(lat_diff),
        float(top_spread_buy),
        float(top_spread_sell),
        float(mp_diff_bps),
    ]
