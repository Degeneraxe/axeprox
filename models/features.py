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


def compute_rolling_volatility_bps(prices: Any, min_observations: int = 5) -> float:
    """Computes sample standard deviation of relative price log-returns in basis points.
    
    Unified shared feature function used across both live streaming paths (CostModel,
    Detector, LivePredictor) and offline training dataset generation (train.py).
    """
    p_list = list(prices) if not isinstance(prices, (list, np.ndarray)) else prices
    if len(p_list) < min_observations:
        return 1.0  # Default 1 bp baseline volatility when history is insufficient

    p_arr = np.asarray(p_list, dtype=np.float64)
    if np.any(p_arr <= 0):
        return 1.0

    returns = np.diff(np.log(p_arr))
    if len(returns) == 0:
        return 1.0

    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else float(np.std(returns))
    return float(std_ret * 10000.0)



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


def _extract_ts(tick: Any) -> float:
    """Helper to extract millisecond timestamp from snapshot or dict."""
    if hasattr(tick, "local_ts_ms"):
        return float(tick.local_ts_ms)
    if isinstance(tick, dict):
        if "local_ts_ms" in tick and tick["local_ts_ms"] is not None:
            return float(tick["local_ts_ms"])
        if "time" in tick and tick["time"] is not None:
            from datetime import datetime
            try:
                dt_str = str(tick["time"]).replace("Z", "+00:00")
                return datetime.fromisoformat(dt_str).timestamp() * 1000.0
            except Exception:
                pass
    return 0.0


def asof_join_ticks(
    buy_ticks: List[Any],
    sell_ticks: List[Any],
    max_staleness_ms: float = 1500.0,
    seq_len: int = 20,
) -> List[Tuple[Any, Any]]:
    """Aligns buy and sell ticks by timestamp using an as-of join with max staleness (Requirement 2).
    
    Never aligns by arbitrary array index. For every distinct event observation timestamp,
    matches the most recent tick on both venues that arrived at or before T, strictly enforcing
    that neither venue's quote age exceeds max_staleness_ms.
    
    Returns:
        List of (buy_tick, sell_tick) pairs, truncated to the most recent seq_len steps.
    """
    if not buy_ticks or not sell_ticks:
        return []

    import bisect

    b_sorted = sorted(buy_ticks, key=_extract_ts)
    s_sorted = sorted(sell_ticks, key=_extract_ts)

    b_ts = [_extract_ts(t) for t in b_sorted]
    s_ts = [_extract_ts(t) for t in s_sorted]

    # All observation timestamps across both streams
    all_ts = sorted(list(set(b_ts + s_ts)))

    aligned_pairs: List[Tuple[Any, Any]] = []
    seen_keys = set()

    for t in all_ts:
        # Find latest buy tick <= t
        b_idx = bisect.bisect_right(b_ts, t) - 1
        # Find latest sell tick <= t
        s_idx = bisect.bisect_right(s_ts, t) - 1

        if b_idx < 0 or s_idx < 0:
            continue

        b_tick = b_sorted[b_idx]
        s_tick = s_sorted[s_idx]

        b_age = t - b_ts[b_idx]
        s_age = t - s_ts[s_idx]

        if b_age <= max_staleness_ms and s_age <= max_staleness_ms:
            # Avoid duplicate consecutive pairs if neither side changed
            pair_key = (id(b_tick), id(s_tick))
            if pair_key not in seen_keys:
                seen_keys.add(pair_key)
                aligned_pairs.append((b_tick, s_tick))

    return aligned_pairs[-seq_len:]

