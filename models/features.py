"""Microstructure feature engineering module for ArbSense.

Extracts time-series features from order book depth snapshots:
1. Order Book Imbalance (OBI) across top 1, 5, and 10 levels
2. Inter-exchange spread velocity and acceleration (dSpread/dt, d^2Spread/dt^2)
3. Multi-scale rolling price volatility
4. Network transit latency differentials (Buy latency - Sell latency)
5. Micro-price deviations and internal venue spreads
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class FeatureStep(dict):
    """Dictionary representing a single timestep in a feature window.

    Persists raw metadata (real transit latencies, ts_source strings, is_measured flags)
    and point-in-time microstructure fields. Implements sequence, slicing, and numpy
    array protocols so np.array(feature_window) and legacy indexing continue to work seamlessly.
    """
    def __init__(self, data: Dict[str, Any], vector: List[float]):
        super().__init__(data)
        self.vector = [float(v) for v in vector]

    def __iter__(self):
        return iter(self.vector)

    def __len__(self):
        return len(self.vector)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (int, slice)):
            return self.vector[item]
        return super().__getitem__(item)

    def __array__(self, dtype=None):
        return np.array(self.vector, dtype=dtype)


def get_feature_names(use_latency: bool = False) -> List[str]:
    """Returns sequence feature column names based on features.use_latency switch."""
    if use_latency:
        return [
            "gross_spread_bps",
            "spread_velocity",
            "spread_acceleration",
            "obi_buy_top5",
            "obi_sell_top5",
            "obi_differential",
            "rolling_vol_buy_bps",
            "rolling_vol_sell_bps",
            "transit_lat_buy_ms",
            "is_measured_buy",
            "transit_lat_sell_ms",
            "is_measured_sell",
            "transit_lat_diff_ms",
            "internal_spread_buy",
            "internal_spread_sell",
            "microprice_diff_bps",
        ]
    return [
        "gross_spread_bps",
        "spread_velocity",
        "spread_acceleration",
        "obi_buy_top5",
        "obi_sell_top5",
        "obi_differential",
        "rolling_vol_buy_bps",
        "rolling_vol_sell_bps",
        "internal_spread_buy",
        "internal_spread_sell",
        "microprice_diff_bps",
    ]


FEATURE_NAMES = get_feature_names(use_latency=False)


def get_tabular_feature_names(use_latency: bool = False) -> List[str]:
    """Returns tabular feature column names based on features.use_latency switch."""
    base = [
        "open_spread_bps",
        "spread_vs_hurdle",
        "spread_velocity_3",
        "spread_velocity_5",
        "spread_velocity_10",
        "spread_accel_3",
        "spread_accel_5",
        "spread_accel_10",
        "volatility_3",
        "volatility_5",
        "volatility_10",
        "obi_buy_top5",
        "obi_sell_top5",
        "obi_diff",
        "obi_buy_mean_5",
        "obi_sell_mean_5",
        "vol_diff_5",
        "is_buy_leg_mover",
    ]
    if use_latency:
        lat = [
            "transit_lat_buy_ms",
            "is_measured_buy",
            "transit_lat_sell_ms",
            "is_measured_sell",
            "transit_lat_diff_ms",
        ]
    else:
        lat = []
    tail = [
        "time_since_prev_event_s",
        "event_count_last_60s",
        "hour_of_day",
        "hour_sin",
        "hour_cos",
    ]
    return base + lat + tail


TABULAR_FEATURE_NAMES = get_tabular_feature_names(use_latency=False)


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


def build_feature_step(
    gross_bps: float,
    vel: float,
    acc: float,
    obi_b: float,
    obi_s: float,
    vol_b: float,
    vol_s: float,
    top_b: float,
    top_s: float,
    mp_diff: float,
    raw_lat_b: float,
    raw_lat_s: float,
    ts_src_b: str,
    ts_src_s: str,
    use_latency: bool = False,
    imputed_latency_ms: float = 85.0,
) -> FeatureStep:
    """Builds a FeatureStep dict with raw latency persistence and numerical vector."""
    is_meas_b = 1.0 if ts_src_b == "exchange" else 0.0
    is_meas_s = 1.0 if ts_src_s == "exchange" else 0.0

    # Raw dictionary holding honest latency values and flags
    step_dict = {
        "gross_spread_bps": float(gross_bps),
        "spread_velocity": float(vel),
        "spread_acceleration": float(acc),
        "obi_buy_top5": float(obi_b),
        "obi_sell_top5": float(obi_s),
        "obi_differential": float(obi_b - obi_s),
        "rolling_vol_buy_bps": float(vol_b),
        "rolling_vol_sell_bps": float(vol_s),
        "lat_buy_ms": float(raw_lat_b),
        "lat_sell_ms": float(raw_lat_s),
        "ts_src_buy": str(ts_src_b),
        "ts_src_sell": str(ts_src_s),
        "is_measured_buy": float(is_meas_b),
        "is_measured_sell": float(is_meas_s),
        "internal_spread_buy": float(top_b),
        "internal_spread_sell": float(top_s),
        "microprice_diff_bps": float(mp_diff),
    }

    if not use_latency:
        vector = [
            float(gross_bps),
            float(vel),
            float(acc),
            float(obi_b),
            float(obi_s),
            float(obi_b - obi_s),
            float(vol_b),
            float(vol_s),
            float(top_b),
            float(top_s),
            float(mp_diff),
        ]
    else:
        # NaN-safe representation: value + is_measured indicator, with imputed value for estimated ticks
        safe_lat_b = float(raw_lat_b) if (is_meas_b == 1.0 and not math.isnan(raw_lat_b)) else float(imputed_latency_ms)
        safe_lat_s = float(raw_lat_s) if (is_meas_s == 1.0 and not math.isnan(raw_lat_s)) else float(imputed_latency_ms)
        safe_lat_diff = safe_lat_b - safe_lat_s
        vector = [
            float(gross_bps),
            float(vel),
            float(acc),
            float(obi_b),
            float(obi_s),
            float(obi_b - obi_s),
            float(vol_b),
            float(vol_s),
            float(safe_lat_b),
            float(is_meas_b),
            float(safe_lat_s),
            float(is_meas_s),
            float(safe_lat_diff),
            float(top_b),
            float(top_s),
            float(mp_diff),
        ]

    return FeatureStep(step_dict, vector)


def extract_step_features(
    snap_buy: Any,
    snap_sell: Any,
    gross_spread_bps: float,
    prev_spread_bps: Optional[float] = None,
    prev_velocity: Optional[float] = None,
    vol_buy_bps: float = 1.0,
    vol_sell_bps: float = 1.0,
    use_latency: bool = False,
    imputed_latency_ms: float = 85.0,
) -> List[float]:
    """Extracts a feature vector for a single timestep with use_latency support."""
    # 1. Spread dynamics
    velocity = (gross_spread_bps - prev_spread_bps) if prev_spread_bps is not None else 0.0
    accel = (velocity - prev_velocity) if prev_velocity is not None else 0.0

    # 2. Order book imbalance
    obi_buy = snap_buy.compute_imbalance(5) if hasattr(snap_buy, "compute_imbalance") else snap_buy.get("imbalance", 0.0)
    obi_sell = snap_sell.compute_imbalance(5) if hasattr(snap_sell, "compute_imbalance") else snap_sell.get("imbalance", 0.0)

    # 3. Transit latency dynamics
    b_ts_src = getattr(snap_buy, "ts_source", snap_buy.get("ts_source", "exchange") if isinstance(snap_buy, dict) else "exchange")
    s_ts_src = getattr(snap_sell, "ts_source", snap_sell.get("ts_source", "exchange") if isinstance(snap_sell, dict) else "exchange")
    raw_lat_b = float(getattr(snap_buy, "transit_latency_ms", snap_buy.get("transit_latency_ms", 0.0) if isinstance(snap_buy, dict) else 0.0))
    raw_lat_s = float(getattr(snap_sell, "transit_latency_ms", snap_sell.get("transit_latency_ms", 0.0) if isinstance(snap_sell, dict) else 0.0))

    # 4. Internal venue spreads
    top_spread_buy = float(getattr(snap_buy, "top_spread", snap_buy.get("top_spread", 0.0) if isinstance(snap_buy, dict) else 0.0))
    top_spread_sell = float(getattr(snap_sell, "top_spread", snap_sell.get("top_spread", 0.0) if isinstance(snap_sell, dict) else 0.0))

    # 5. Micro-price differential
    mp_buy = compute_microprice(snap_buy) if not isinstance(snap_buy, dict) else float(snap_buy.get("mid_price", 0.0))
    mp_sell = compute_microprice(snap_sell) if not isinstance(snap_sell, dict) else float(snap_sell.get("mid_price", 0.0))
    mp_diff_bps = ((mp_sell - mp_buy) / mp_buy * 10000.0) if mp_buy > 0 else 0.0

    step = build_feature_step(
        gross_bps=gross_spread_bps,
        vel=velocity,
        acc=accel,
        obi_b=obi_buy,
        obi_s=obi_sell,
        vol_b=vol_buy_bps,
        vol_s=vol_sell_bps,
        top_b=top_spread_buy,
        top_s=top_spread_sell,
        mp_diff=mp_diff_bps,
        raw_lat_b=raw_lat_b,
        raw_lat_s=raw_lat_s,
        ts_src_b=b_ts_src,
        ts_src_s=s_ts_src,
        use_latency=use_latency,
        imputed_latency_ms=imputed_latency_ms,
    )
    return step.vector


def parse_feature_window(
    win_data: Any,
    use_latency: bool = False,
    imputed_latency_ms: float = 85.0,
) -> np.ndarray:
    """Parses raw feature window (JSON string, list of dicts, or legacy list) into a 2D float32 ndarray."""
    if win_data is None:
        dim = 11 if not use_latency else 16
        return np.zeros((0, dim), dtype=np.float32)

    if isinstance(win_data, str):
        try:
            win_data = json.loads(win_data)
        except Exception:
            dim = 11 if not use_latency else 16
            return np.zeros((0, dim), dtype=np.float32)

    rows: List[List[float]] = []
    for step in win_data:
        if isinstance(step, FeatureStep):
            rows.append(step.vector)
        elif isinstance(step, dict):
            gross_bps = float(step.get("gross_spread_bps", 0.0))
            vel = float(step.get("spread_velocity", 0.0))
            acc = float(step.get("spread_acceleration", 0.0))
            obi_b = float(step.get("obi_buy_top5", 0.0))
            obi_s = float(step.get("obi_sell_top5", 0.0))
            obi_diff = float(step.get("obi_differential", obi_b - obi_s))
            vol_b = float(step.get("rolling_vol_buy_bps", 1.0))
            vol_s = float(step.get("rolling_vol_sell_bps", 1.0))
            top_b = float(step.get("internal_spread_buy", 0.0))
            top_s = float(step.get("internal_spread_sell", 0.0))
            mp_diff = float(step.get("microprice_diff_bps", 0.0))

            if not use_latency:
                rows.append([
                    gross_bps, vel, acc, obi_b, obi_s, obi_diff,
                    vol_b, vol_s, top_b, top_s, mp_diff,
                ])
            else:
                raw_lat_b = float(step.get("lat_buy_ms", imputed_latency_ms))
                raw_lat_s = float(step.get("lat_sell_ms", imputed_latency_ms))
                ts_src_b = str(step.get("ts_src_buy", "exchange"))
                ts_src_s = str(step.get("ts_src_sell", "exchange"))
                is_meas_b = float(step.get("is_measured_buy", 1.0 if ts_src_b == "exchange" else 0.0))
                is_meas_s = float(step.get("is_measured_sell", 1.0 if ts_src_s == "exchange" else 0.0))

                safe_lat_b = raw_lat_b if (is_meas_b == 1.0 and not math.isnan(raw_lat_b)) else float(imputed_latency_ms)
                safe_lat_s = raw_lat_s if (is_meas_s == 1.0 and not math.isnan(raw_lat_s)) else float(imputed_latency_ms)
                safe_lat_diff = safe_lat_b - safe_lat_s

                rows.append([
                    gross_bps, vel, acc, obi_b, obi_s, obi_diff,
                    vol_b, vol_s,
                    safe_lat_b, is_meas_b, safe_lat_s, is_meas_s, safe_lat_diff,
                    top_b, top_s, mp_diff,
                ])
        elif isinstance(step, (list, tuple)):
            if len(step) >= 14:
                # Legacy 14-element list:
                # 0: gross, 1: vel, 2: acc, 3: obi_b, 4: obi_s, 5: obi_diff, 6: vol_b, 7: vol_s,
                # 8: lat_b, 9: lat_s, 10: lat_diff, 11: top_b, 12: top_s, 13: mp_diff
                if not use_latency:
                    rows.append([
                        float(step[0]), float(step[1]), float(step[2]), float(step[3]),
                        float(step[4]), float(step[5]), float(step[6]), float(step[7]),
                        float(step[11]), float(step[12]), float(step[13])
                    ])
                else:
                    lat_b = float(step[8])
                    lat_s = float(step[9])
                    is_b = 1.0 if lat_b != 0.0 else 0.0
                    is_s = 1.0 if lat_s != 0.0 else 0.0
                    safe_b = lat_b if is_b == 1.0 else float(imputed_latency_ms)
                    safe_s = lat_s if is_s == 1.0 else float(imputed_latency_ms)
                    rows.append([
                        float(step[0]), float(step[1]), float(step[2]), float(step[3]),
                        float(step[4]), float(step[5]), float(step[6]), float(step[7]),
                        safe_b, is_b, safe_s, is_s, safe_b - safe_s,
                        float(step[11]), float(step[12]), float(step[13])
                    ])
            elif len(step) == 11 and not use_latency:
                rows.append([float(x) for x in step])
            elif len(step) == 16 and use_latency:
                rows.append([float(x) for x in step])
            else:
                rows.append([float(x) for x in step])
        else:
            rows.append(step)

    return np.array(rows, dtype=np.float32)


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


def engineer_tabular_features(
    seq_arr: np.ndarray,
    event_dict: Dict[str, Any],
    prev_event_open_ts: Optional[float] = None,
    recent_event_times: Optional[List[float]] = None,
    use_latency: bool = False,
    imputed_latency_ms: float = 85.0,
) -> np.ndarray:
    """Computes all point-in-time tabular microstructure features from 20-step window and event fields.
    
    Unified feature engineering module in models.features shared by training and inference.
    """
    from datetime import datetime, timezone

    open_ts = float(event_dict.get("open_time", 0.0))
    time_since_prev_s = min(600.0, (open_ts - prev_event_open_ts) / 1000.0) if prev_event_open_ts else 600.0
    
    if recent_event_times is not None:
        event_count_60s = sum(1 for t in recent_event_times if t >= open_ts - 60000.0)
    else:
        event_count_60s = 1

    dt_utc = datetime.fromtimestamp(open_ts / 1000.0, tz=timezone.utc)
    hour = dt_utc.hour + dt_utc.minute / 60.0
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)

    s = seq_arr[:, 0]  # gross spread bps across window
    open_bps = float(event_dict.get("open_spread_bps", s[-1] if len(s) > 0 else 0.0))
    hurdle_bps = float(event_dict.get("cost_hurdle_bps", 0.0))
    spread_vs_hurdle = open_bps - hurdle_bps

    # Velocities and accelerations over 3, 5, 10 steps
    vel_3 = float(s[-1] - s[-3]) if len(s) >= 3 else 0.0
    vel_5 = float(s[-1] - s[-5]) if len(s) >= 5 else 0.0
    vel_10 = float(s[-1] - s[-10]) if len(s) >= 10 else 0.0
    acc_3 = float((s[-1] - s[-3]) - (s[-3] - s[-5])) if len(s) >= 5 else 0.0
    acc_5 = float((s[-1] - s[-5]) - (s[-5] - s[-9])) if len(s) >= 9 else 0.0
    acc_10 = float((s[-1] - s[-10]) - (s[-10] - s[-19])) if len(s) >= 19 else 0.0

    # Multi-scale volatility
    vol_3 = float(np.std(s[-3:])) if len(s) >= 3 else 1.0
    vol_5 = float(np.std(s[-5:])) if len(s) >= 5 else 1.0
    vol_10 = float(np.std(s[-10:])) if len(s) >= 10 else 1.0

    # Order book imbalances across scales
    obi_b_top5 = float(seq_arr[-1, 3])
    obi_s_top5 = float(seq_arr[-1, 4])
    obi_diff = float(seq_arr[-1, 5])
    obi_b_mean5 = float(np.mean(seq_arr[-5:, 3])) if len(seq_arr) >= 5 else obi_b_top5
    obi_s_mean5 = float(np.mean(seq_arr[-5:, 4])) if len(seq_arr) >= 5 else obi_s_top5

    # Which leg moved most in last 5 steps
    vol_b_5 = float(np.mean(seq_arr[-5:, 6])) if len(seq_arr) >= 5 else 1.0
    vol_s_5 = float(np.mean(seq_arr[-5:, 7])) if len(seq_arr) >= 5 else 1.0
    vol_diff_5 = vol_b_5 - vol_s_5
    is_buy_mover = 1.0 if vol_b_5 >= vol_s_5 else 0.0

    base_vec = [
        open_bps,
        spread_vs_hurdle,
        vel_3,
        vel_5,
        vel_10,
        acc_3,
        acc_5,
        acc_10,
        vol_3,
        vol_5,
        vol_10,
        obi_b_top5,
        obi_s_top5,
        obi_diff,
        obi_b_mean5,
        obi_s_mean5,
        vol_diff_5,
        is_buy_mover,
    ]

    if use_latency:
        if seq_arr.shape[1] >= 16:
            lat_b = float(seq_arr[-1, 8])
            is_meas_b = float(seq_arr[-1, 9])
            lat_s = float(seq_arr[-1, 10])
            is_meas_s = float(seq_arr[-1, 11])
            lat_diff = float(seq_arr[-1, 12])
        else:
            lat_b = float(imputed_latency_ms)
            is_meas_b = 0.0
            lat_s = float(imputed_latency_ms)
            is_meas_s = 0.0
            lat_diff = 0.0
        lat_vec = [lat_b, is_meas_b, lat_s, is_meas_s, lat_diff]
    else:
        lat_vec = []

    tail_vec = [
        time_since_prev_s,
        float(event_count_60s),
        float(hour),
        hour_sin,
        hour_cos,
    ]

    return np.array(base_vec + lat_vec + tail_vec, dtype=np.float32)
