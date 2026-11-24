"""Spread Event Tracker and Ground-Truth Duration/Viability Labeler.

Implements Amendments 1, 5, and 6:
- Tracks spread opening, persistence, and closure lifecycle
- Calculates ground-truth duration_ms, log_duration, and is_viable
- Flags right-censored events where spreads do not close before window expiration
- Strict anti-leakage guarantee: features recorded at open, labels derived at closure
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Deque

from config.settings import AppConfig
from data.storage.db_manager import DatabaseManager
from engine.cost_model import CostModel
from engine.detector import SpreadCandidate
from models.features import asof_join_ticks, compute_rolling_volatility_bps, extract_step_features, build_feature_step

logger = logging.getLogger(__name__)


def _extract_ts(tick: Any) -> float:
    """Helper to extract epoch ms timestamp from snapshot or dict."""
    if hasattr(tick, "local_ts_ms"):
        return float(tick.local_ts_ms)
    if isinstance(tick, dict):
        if "local_ts_ms" in tick and tick["local_ts_ms"] is not None:
            return float(tick["local_ts_ms"])
        if "time" in tick and tick["time"] is not None:
            val = tick["time"]
            if isinstance(val, (int, float)):
                return float(val)
            try:
                dt_str = str(val).replace("Z", "+00:00")
                return datetime.fromisoformat(dt_str).timestamp() * 1000.0
            except Exception:
                pass
    return 0.0


@dataclass
class ActiveSpreadEvent:
    """Tracks an ongoing arbitrage opportunity window with epoch ms timestamps."""
    event_id: str
    symbol: str
    buy_exchange: str
    sell_exchange: str
    open_time_ms: float
    open_spread_bps: float
    cost_hurdle_bps: float
    peak_spread_bps: float
    last_seen_ms: float
    tick_count: int = 1
    features_at_open: Dict[str, float] = field(default_factory=dict)
    feature_window: List[Any] = field(default_factory=list)
    vol_buy_bps: float = 1.0
    vol_sell_bps: float = 1.0
    latency_threshold_ms: float = 85.0
    is_closed: bool = False
    is_right_censored: bool = False
    is_suspect: bool = False
    close_time_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    log_duration: Optional[float] = None
    is_viable: bool = False
    lifetime_ticks: List[Dict[str, Any]] = field(default_factory=list)
    dataset_version: str = "v1.0"
    slippage_bps: float = 0.0
    is_executable: bool = False
    net_peak_retail: float = 0.0
    is_profitable_retail: bool = False
    net_peak_vip1: float = 0.0
    is_profitable_vip1: bool = False
    net_peak_vip3: float = 0.0
    is_profitable_vip3: bool = False
    net_peak_zero_maker: float = 0.0
    is_profitable_zero_maker: bool = False
    is_persistent_500ms: bool = False
    is_persistent_2000ms: bool = False
    is_persistent_5000ms: bool = False


class SpreadTracker:
    """Manages active spread windows and generates ground-truth training records."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        config: Optional[AppConfig] = None,
        on_event_closed: Optional[Callable[[ActiveSpreadEvent], None]] = None,
        normalizer: Optional[Any] = None,
    ):
        self.db = db_manager
        self.config = config or AppConfig.load()
        self.on_event_closed = on_event_closed
        self.normalizer = normalizer
        self.cost_model = CostModel(self.config)

        # In-memory per-(exchange, symbol) ring buffer used by both live inference and tracker (Requirement 3)
        if normalizer is not None and hasattr(normalizer, "ring_buffers"):
            self.ring_buffers = normalizer.ring_buffers
        else:
            self.ring_buffers: Dict[Tuple[str, str], Deque[Any]] = defaultdict(lambda: deque(maxlen=100))

        # Pair key: (symbol, buy_exchange, sell_exchange) -> ActiveSpreadEvent
        self.active_events: Dict[Tuple[str, str, str], ActiveSpreadEvent] = {}

        # Fixed viability latency threshold per dataset version (Item 5: ONE frozen threshold)
        ds_cfg = self.config.raw.get("dataset", {})
        lbl_cfg = self.config.raw.get("labeling", {})
        trk_cfg = self.config.raw.get("tracking", {})
        self.dataset_version = ds_cfg.get("version", "v1.0")
        self.latency_hurdle_ms = float(
            ds_cfg.get("viability_latency_threshold_ms", 85.0)
        )
        self.max_staleness_ms = float(
            trk_cfg.get("max_staleness_ms", getattr(self.config.time_sync, "max_quote_age_ms", 1500.0))
        )
        self.inactivity_timeout_ms = float(
            trk_cfg.get("inactivity_timeout_ms", 1500.0)
        )

        self.max_window_ms = float(lbl_cfg.get("right_censored_max_window_ms", 30000.0))
        self.use_log_duration = bool(lbl_cfg.get("use_log_duration", True))

        self.completed_events: List[ActiveSpreadEvent] = []

    def record_tick(self, snapshot: Any) -> None:
        """Appends tick to in-memory ring buffer (Requirement 3)."""
        key = (snapshot.exchange.lower(), snapshot.symbol)
        self.ring_buffers[key].append(snapshot)

    def _build_point_in_time_feature_window(
        self,
        symbol: str,
        buy_exch: str,
        sell_exch: str,
        open_time_ms: float,
        seq_len: int = 20,
        use_latency: Optional[bool] = None,
        imputed_latency_ms: Optional[float] = None,
    ) -> Tuple[List[Any], float, float]:
        """Builds 20-step feature window directly from in-memory ring buffer using as-of timestamp join (Requirement 1, 2, 3)."""
        if use_latency is None:
            use_latency = bool(self.config.raw.get("features", {}).get("use_latency", False))
        if imputed_latency_ms is None:
            imputed_latency_ms = self.latency_hurdle_ms

        b_key = (buy_exch.lower(), symbol)
        s_key = (sell_exch.lower(), symbol)

        b_ticks_raw = self.ring_buffers.get(b_key, [])
        if not b_ticks_raw and "USDT" in symbol:
            b_ticks_raw = self.ring_buffers.get((buy_exch.lower(), symbol.replace("USDT", "USD")), [])

        s_ticks_raw = self.ring_buffers.get(s_key, [])
        if not s_ticks_raw and "USDT" in symbol:
            s_ticks_raw = self.ring_buffers.get((sell_exch.lower(), symbol.replace("USDT", "USD")), [])

        b_ticks = [t for t in b_ticks_raw if _extract_ts(t) <= open_time_ms]
        s_ticks = [t for t in s_ticks_raw if _extract_ts(t) <= open_time_ms]

        # 50-tick rolling volatility calculation per exchange
        b_prices = [(t.mid_price if hasattr(t, "mid_price") else t["mid_price"]) for t in b_ticks[-50:]] if b_ticks else []
        s_prices = [(t.mid_price if hasattr(t, "mid_price") else t["mid_price"]) for t in s_ticks[-50:]] if s_ticks else []
        vol_b = compute_rolling_volatility_bps(b_prices) if b_prices else 1.0
        vol_s = compute_rolling_volatility_bps(s_prices) if s_prices else 1.0

        # As-of join by timestamp with max staleness (Requirement 2 & 3)
        aligned_pairs = asof_join_ticks(b_ticks, s_ticks, max_staleness_ms=self.max_staleness_ms, seq_len=seq_len)
        if len(aligned_pairs) < seq_len:
            return [], vol_b, vol_s

        seq: List[Any] = []
        prev_spread = None
        prev_vel = None
        basis_rate = getattr(self.normalizer, "usdt_usd_basis_rate", 1.0000) if self.normalizer else 1.0000

        for b_tick, s_tick in aligned_pairs:
            ask_buy = float(b_tick.best_ask.price if hasattr(b_tick, "best_ask") and b_tick.best_ask else (b_tick.get("best_ask", 0.0) if isinstance(b_tick, dict) else 0.0))
            bid_sell = float(s_tick.best_bid.price if hasattr(s_tick, "best_bid") and s_tick.best_bid else (s_tick.get("best_bid", 0.0) if isinstance(s_tick, dict) else 0.0))
            b_quote = getattr(b_tick, "quote_currency", "USDT") if not isinstance(b_tick, dict) else b_tick.get("quote_currency", "USDT")
            s_quote = getattr(s_tick, "quote_currency", "USDT") if not isinstance(s_tick, dict) else s_tick.get("quote_currency", "USDT")
            if b_quote == "USD" and hasattr(b_tick, "get_adjusted_price"):
                ask_buy = b_tick.get_adjusted_price(ask_buy, target_quote="USDT", basis_rate=basis_rate)
            elif b_quote == "USD" and basis_rate > 0:
                ask_buy = ask_buy / basis_rate
            if s_quote == "USD" and hasattr(s_tick, "get_adjusted_price"):
                bid_sell = s_tick.get_adjusted_price(bid_sell, target_quote="USDT", basis_rate=basis_rate)
            elif s_quote == "USD" and basis_rate > 0:
                bid_sell = bid_sell / basis_rate

            gross_bps = ((bid_sell - ask_buy) / ask_buy * 10000.0) if ask_buy > 0 else 0.0

            vel = (gross_bps - prev_spread) if prev_spread is not None else 0.0
            acc = (vel - prev_vel) if prev_vel is not None else 0.0
            prev_spread = gross_bps
            prev_vel = vel

            obi_b = float(b_tick.compute_imbalance(5) if hasattr(b_tick, "compute_imbalance") else b_tick.get("imbalance", 0.0))
            obi_s = float(s_tick.compute_imbalance(5) if hasattr(s_tick, "compute_imbalance") else s_tick.get("imbalance", 0.0))
            b_ts_src = str(getattr(b_tick, "ts_source", b_tick.get("ts_source", "exchange") if isinstance(b_tick, dict) else "exchange"))
            s_ts_src = str(getattr(s_tick, "ts_source", s_tick.get("ts_source", "exchange") if isinstance(s_tick, dict) else "exchange"))
            lat_b = float(b_tick.transit_latency_ms if hasattr(b_tick, "transit_latency_ms") else b_tick.get("transit_latency_ms", 0.0))
            lat_s = float(s_tick.transit_latency_ms if hasattr(s_tick, "transit_latency_ms") else s_tick.get("transit_latency_ms", 0.0))
            top_b = float(b_tick.top_spread if hasattr(b_tick, "top_spread") else b_tick.get("top_spread", 0.0))
            top_s = float(s_tick.top_spread if hasattr(s_tick, "top_spread") else s_tick.get("top_spread", 0.0))
            mp_b = float(b_tick.mid_price if hasattr(b_tick, "mid_price") else b_tick.get("mid_price", 0.0))
            mp_s = float(s_tick.mid_price if hasattr(s_tick, "mid_price") else s_tick.get("mid_price", 0.0))
            mp_diff = ((mp_s - mp_b) / mp_b * 10000.0) if mp_b > 0 else 0.0

            step = build_feature_step(
                gross_bps=gross_bps,
                vel=vel,
                acc=acc,
                obi_b=obi_b,
                obi_s=obi_s,
                vol_b=vol_b,
                vol_s=vol_s,
                top_b=top_b,
                top_s=top_s,
                mp_diff=mp_diff,
                raw_lat_b=lat_b,
                raw_lat_s=lat_s,
                ts_src_b=b_ts_src,
                ts_src_s=s_ts_src,
                use_latency=use_latency,
                imputed_latency_ms=imputed_latency_ms,
            )
            seq.append(step)

        return seq, vol_b, vol_s

    def process_candidate(
        self,
        candidate: SpreadCandidate,
        features: Optional[Dict[str, float]] = None,
        feature_window: Optional[List[List[float]]] = None,
        vol_buy_bps: Optional[float] = None,
        vol_sell_bps: Optional[float] = None,
    ) -> Optional[ActiveSpreadEvent]:
        """Updates active spread state machine with newly scanned candidate."""
        key = (candidate.symbol, candidate.buy_exchange, candidate.sell_exchange)
        now_ms = float(candidate.timestamp_ms)  # Stored as epoch ms (Requirement 4)

        # 1. Existing ongoing event for this pair
        if key in self.active_events:
            event = self.active_events[key]

            # Record tick observed during event's lifetime (Task 3)
            tick_record = {
                "time_ms": now_ms,
                "gross_spread_bps": round(float(candidate.gross_spread_bps), 4),
                "buy_ask": float(candidate.buy_ask),
                "sell_bid": float(candidate.sell_bid),
                "hurdle_bps": round(float(candidate.total_hurdle_bps), 4),
            }
            event.lifetime_ticks.append(tick_record)
            event.tick_count += 1
            event.last_seen_ms = now_ms

            # Update peak spread on every tick (Task 3)
            if candidate.gross_spread_bps > event.peak_spread_bps:
                event.peak_spread_bps = candidate.gross_spread_bps

            # Check for right-censoring timeout (Amendment 5)
            if (now_ms - event.open_time_ms) >= self.max_window_ms:
                return self._close_event(key, now_ms, right_censored=True)

            # Events must close when the spread actually falls below the hurdle (Task 3)
            # Fix D: Hysteresis closure buffer (hurdle - hysteresis_bps)
            det_cfg = self.config.raw.get("detection", {})
            hysteresis_bps = float(det_cfg.get("hysteresis_bps", 1.0))
            close_hurdle = (event.cost_hurdle_bps - hysteresis_bps) if event.open_spread_bps >= event.cost_hurdle_bps else float(self.config.detection.gross_spread_threshold_bps)
            if candidate.gross_spread_bps < close_hurdle:
                return self._close_event(key, now_ms, right_censored=False)

            return event

        # 2. No ongoing event -> open a new one if candidate meets detection criteria
        elif candidate.is_gross_candidate:
            seq_len = int(self.config.raw.get("dataset", {}).get("sequence_length", 20))

            # Ring buffer check: wait until ring buffers hold at least seq_len ticks per leg
            b_key = (candidate.buy_exchange.lower(), candidate.symbol)
            s_key = (candidate.sell_exchange.lower(), candidate.symbol)
            b_buf = self.ring_buffers.get(b_key, [])
            if not b_buf and "USDT" in candidate.symbol:
                b_buf = self.ring_buffers.get((candidate.buy_exchange.lower(), candidate.symbol.replace("USDT", "USD")), [])
            s_buf = self.ring_buffers.get(s_key, [])
            if not s_buf and "USDT" in candidate.symbol:
                s_buf = self.ring_buffers.get((candidate.sell_exchange.lower(), candidate.symbol.replace("USDT", "USD")), [])

            if len(b_buf) < seq_len or len(s_buf) < seq_len:
                return None

            event_id = f"evt_{uuid.uuid4().hex[:12]}"

            # Build 20-step feature window directly from in-memory ring buffer (Requirement 1, 2, 3)
            if feature_window is not None:
                feat_win = feature_window
                v_buy = vol_buy_bps if vol_buy_bps is not None else 1.0
                v_sell = vol_sell_bps if vol_sell_bps is not None else 1.0
            else:
                feat_win, v_buy, v_sell = self._build_point_in_time_feature_window(
                    candidate.symbol, candidate.buy_exchange, candidate.sell_exchange, now_ms, seq_len=seq_len
                )

            # Never persist an event with an empty/incomplete window (Task 2)
            if not feat_win or len(feat_win) < seq_len:
                return None

            tick_record = {
                "time_ms": now_ms,
                "gross_spread_bps": round(float(candidate.gross_spread_bps), 4),
                "buy_ask": float(candidate.buy_ask),
                "sell_bid": float(candidate.sell_bid),
                "hurdle_bps": round(float(candidate.total_hurdle_bps), 4),
            }
            max_plausible = float(self.config.raw.get("detection", {}).get("max_plausible_spread_bps", 80.0))
            is_suspect = (candidate.gross_spread_bps > max_plausible)

            new_event = ActiveSpreadEvent(
                event_id=event_id,
                symbol=candidate.symbol,
                buy_exchange=candidate.buy_exchange,
                sell_exchange=candidate.sell_exchange,
                open_time_ms=now_ms,
                open_spread_bps=candidate.gross_spread_bps,
                cost_hurdle_bps=candidate.total_hurdle_bps,
                peak_spread_bps=candidate.gross_spread_bps,
                last_seen_ms=now_ms,
                tick_count=1,
                features_at_open=features or {},
                feature_window=feat_win,
                vol_buy_bps=v_buy,
                vol_sell_bps=v_sell,
                latency_threshold_ms=self.latency_hurdle_ms,
                is_suspect=is_suspect,
                lifetime_ticks=[tick_record],
                dataset_version=self.dataset_version,
            )
            self.active_events[key] = new_event

            # Persist event at open with complete 20-step feature window, suspect flag, and ticks (Requirement 1 & 3 & 4)
            if self.db:
                self.db.insert_spread_event({
                    "event_id": new_event.event_id,
                    "symbol": new_event.symbol,
                    "buy_exchange": new_event.buy_exchange,
                    "sell_exchange": new_event.sell_exchange,
                    "open_time": new_event.open_time_ms,  # Epoch ms (Requirement 4)
                    "open_spread_bps": new_event.open_spread_bps,
                    "cost_hurdle_bps": new_event.cost_hurdle_bps,
                    "peak_spread_bps": new_event.peak_spread_bps,
                    "feature_window_json": json.dumps(new_event.feature_window) if new_event.feature_window else None,
                    "vol_buy_bps": new_event.vol_buy_bps,
                    "vol_sell_bps": new_event.vol_sell_bps,
                    "latency_threshold_ms": new_event.latency_threshold_ms,
                    "is_suspect": new_event.is_suspect,
                    "lifetime_ticks_json": json.dumps(new_event.lifetime_ticks),
                    "dataset_version": new_event.dataset_version,
                })
            return new_event

        return None

    def check_timeouts(self, current_time_ms: float) -> List[ActiveSpreadEvent]:
        """Closes events that have stopped receiving updates or exceeded max duration."""
        closed: List[ActiveSpreadEvent] = []
        for key, event in list(self.active_events.items()):
            time_since_last = current_time_ms - event.last_seen_ms
            total_duration = current_time_ms - event.open_time_ms

            # Inactivity closure (no ticks for > inactivity_timeout_ms means spread closed)
            if time_since_last > self.inactivity_timeout_ms:
                closed_evt = self._close_event(key, event.last_seen_ms, right_censored=False)
                if closed_evt:
                    closed.append(closed_evt)
            # Max window timeout (Amendment 5)
            elif total_duration >= self.max_window_ms:
                closed_evt = self._close_event(key, current_time_ms, right_censored=True)
                if closed_evt:
                    closed.append(closed_evt)
        return closed

    def _close_event(
        self, key: Tuple[str, str, str], close_time_ms: float, right_censored: bool
    ) -> Optional[ActiveSpreadEvent]:
        """Finalizes spread event, computes labels, and persists to DB (timestamps as epoch ms)."""
        event = self.active_events.pop(key, None)
        if not event:
            return None

        event.is_closed = True
        event.close_time_ms = float(close_time_ms)
        event.is_right_censored = right_censored

        # Sanity gate: flag suspect spreads (Task 4)
        max_plausible = float(self.config.raw.get("detection", {}).get("max_plausible_spread_bps", 80.0))
        if event.open_spread_bps > max_plausible or event.peak_spread_bps > max_plausible:
            event.is_suspect = True

        # Calculate duration = real close time minus open time (Task 3)
        duration = max(1.0, event.close_time_ms - event.open_time_ms)
        event.duration_ms = duration
        event.log_duration = math.log(1.0 + duration) if self.use_log_duration else duration

        # Ground truth viability label (Requirement 3):
        # Viable = Stays open longer than the event's fixed dataset threshold AND peak spread exceeded total hurdle
        event.is_viable = (duration >= event.latency_threshold_ms) and (event.peak_spread_bps >= event.cost_hurdle_bps)

        # Depth-walked slippage calculation at configured trade size (Step 3)
        trade_sizes = self.config.raw.get("simulation", {}).get("trade_sizes", {})
        qty = float(trade_sizes.get(event.symbol, 0.0))
        if self.normalizer and qty > 0:
            b_sym = event.symbol if (event.buy_exchange.lower(), event.symbol) in self.normalizer.books else event.symbol.replace("USDT", "USD")
            s_sym = event.symbol if (event.sell_exchange.lower(), event.symbol) in self.normalizer.books else event.symbol.replace("USDT", "USD")
            snap_buy = self.normalizer.get_snapshot(event.buy_exchange, b_sym, event.close_time_ms) or self.normalizer.books.get((event.buy_exchange.lower(), b_sym))
            snap_sell = self.normalizer.get_snapshot(event.sell_exchange, s_sym, event.close_time_ms) or self.normalizer.books.get((event.sell_exchange.lower(), s_sym))
            if snap_buy and snap_sell and snap_buy.best_ask and snap_sell.best_bid:
                p_ask = snap_buy.best_ask.price
                p_bid = snap_sell.best_bid.price
                buy_res = snap_buy.compute_vwap("buy", qty)
                buy_vwap = buy_res[0] if isinstance(buy_res, tuple) else buy_res
                sell_res = snap_sell.compute_vwap("sell", qty)
                sell_vwap = sell_res[0] if isinstance(sell_res, tuple) else sell_res
                slip_buy = ((buy_vwap - p_ask) / p_ask * 10000.0) if p_ask > 0 and buy_vwap > 0 else 0.0
                slip_sell = ((p_bid - sell_vwap) / p_bid * 10000.0) if p_bid > 0 and sell_vwap > 0 else 0.0
                event.slippage_bps = max(0.0, round(slip_buy + slip_sell, 2))
            else:
                event.slippage_bps = 0.0
        else:
            event.slippage_bps = 0.0

        # Step 3 Label Definition: is_executable (fee-independent primary classification target)
        event.is_executable = (duration >= event.latency_threshold_ms) and (event.peak_spread_bps >= event.slippage_bps)

        # Step 3 Label Definition: Persistence targets (duration >= h)
        event.is_persistent_500ms = bool(duration >= 500.0)
        event.is_persistent_2000ms = bool(duration >= 2000.0)
        event.is_persistent_5000ms = bool(duration >= 5000.0)

        # Step 3 Label Definition: is_profitable_<scenario> for EVERY scenario in config
        scenarios = ["retail", "vip1", "vip3", "zero_maker"]
        for scen in scenarios:
            hurdle, _, _ = self.cost_model.get_hurdle_bps(event.buy_exchange, event.sell_exchange, event.symbol, scenario=scen)
            net_peak = round(event.peak_spread_bps - hurdle, 2)
            is_prof = bool(net_peak > 0)
            setattr(event, f"net_peak_{scen}", net_peak)
            setattr(event, f"is_profitable_{scen}", is_prof)

        self.completed_events.append(event)

        # Update in database with epoch ms timestamp, suspect flag, and redesigned labels (Step 1 & 3)
        if self.db:
            self.db.update_spread_event(event.event_id, {
                "close_time": event.close_time_ms,  # Epoch ms
                "peak_spread_bps": event.peak_spread_bps,
                "duration_ms": event.duration_ms,
                "log_duration": event.log_duration,
                "is_viable": event.is_viable,
                "is_right_censored": event.is_right_censored,
                "latency_threshold_ms": event.latency_threshold_ms,
                "is_suspect": event.is_suspect,
                "lifetime_ticks_json": json.dumps(event.lifetime_ticks),
                "dataset_version": event.dataset_version,
                "slippage_bps": event.slippage_bps,
                "is_executable": event.is_executable,
                "is_persistent_500ms": int(event.is_persistent_500ms),
                "is_persistent_2000ms": int(event.is_persistent_2000ms),
                "is_persistent_5000ms": int(event.is_persistent_5000ms),
                "net_peak_retail": event.net_peak_retail,
                "is_profitable_retail": event.is_profitable_retail,
                "net_peak_vip1": event.net_peak_vip1,
                "is_profitable_vip1": event.is_profitable_vip1,
                "net_peak_vip3": event.net_peak_vip3,
                "is_profitable_vip3": event.is_profitable_vip3,
                "net_peak_zero_maker": event.net_peak_zero_maker,
                "is_profitable_zero_maker": event.is_profitable_zero_maker,
            })

        if self.on_event_closed:
            self.on_event_closed(event)

        return event

