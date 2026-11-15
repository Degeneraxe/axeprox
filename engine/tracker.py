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
from engine.detector import SpreadCandidate
from models.features import asof_join_ticks, compute_rolling_volatility_bps, extract_step_features

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
    feature_window: List[List[float]] = field(default_factory=list)
    vol_buy_bps: float = 1.0
    vol_sell_bps: float = 1.0
    latency_threshold_ms: float = 85.0
    is_closed: bool = False
    is_right_censored: bool = False
    close_time_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    log_duration: Optional[float] = None
    is_viable: bool = False


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

        # In-memory per-(exchange, symbol) ring buffer used by both live inference and tracker (Requirement 3)
        if normalizer is not None and hasattr(normalizer, "ring_buffers"):
            self.ring_buffers = normalizer.ring_buffers
        else:
            self.ring_buffers: Dict[Tuple[str, str], Deque[Any]] = defaultdict(lambda: deque(maxlen=100))

        # Pair key: (symbol, buy_exchange, sell_exchange) -> ActiveSpreadEvent
        self.active_events: Dict[Tuple[str, str, str], ActiveSpreadEvent] = {}

        # Fixed viability latency threshold per dataset version (Requirement 3: never recompute at startup)
        ds_cfg = self.config.raw.get("dataset", {})
        lbl_cfg = self.config.raw.get("labeling", {})
        self.dataset_version = ds_cfg.get("version", "v1.0")
        self.latency_hurdle_ms = float(
            ds_cfg.get("viability_latency_threshold_ms", lbl_cfg.get("execution_latency_assumption_ms", 85.0))
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
    ) -> Tuple[List[List[float]], float, float]:
        """Builds 20-step feature window directly from in-memory ring buffer using as-of timestamp join (Requirement 1, 2, 3)."""
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
        aligned_pairs = asof_join_ticks(b_ticks, s_ticks, max_staleness_ms=1500.0, seq_len=seq_len)
        if len(aligned_pairs) < seq_len:
            return [], vol_b, vol_s

        seq: List[List[float]] = []
        prev_spread = None
        prev_vel = None

        for b_tick, s_tick in aligned_pairs:
            ask_buy = float(b_tick.best_ask.price if hasattr(b_tick, "best_ask") and b_tick.best_ask else (b_tick.get("best_ask", 0.0) if isinstance(b_tick, dict) else 0.0))
            bid_sell = float(s_tick.best_bid.price if hasattr(s_tick, "best_bid") and s_tick.best_bid else (s_tick.get("best_bid", 0.0) if isinstance(s_tick, dict) else 0.0))
            gross_bps = ((bid_sell - ask_buy) / ask_buy * 10000.0) if ask_buy > 0 else 0.0

            vel = (gross_bps - prev_spread) if prev_spread is not None else 0.0
            acc = (vel - prev_vel) if prev_vel is not None else 0.0
            prev_spread = gross_bps
            prev_vel = vel

            obi_b = float(b_tick.compute_imbalance(5) if hasattr(b_tick, "compute_imbalance") else b_tick.get("imbalance", 0.0))
            obi_s = float(s_tick.compute_imbalance(5) if hasattr(s_tick, "compute_imbalance") else s_tick.get("imbalance", 0.0))
            lat_b = float(b_tick.transit_latency_ms if hasattr(b_tick, "transit_latency_ms") else b_tick.get("transit_latency_ms", 0.0))
            lat_s = float(s_tick.transit_latency_ms if hasattr(s_tick, "transit_latency_ms") else s_tick.get("transit_latency_ms", 0.0))
            top_b = float(b_tick.top_spread if hasattr(b_tick, "top_spread") else b_tick.get("top_spread", 0.0))
            top_s = float(s_tick.top_spread if hasattr(s_tick, "top_spread") else s_tick.get("top_spread", 0.0))
            mp_b = float(b_tick.mid_price if hasattr(b_tick, "mid_price") else b_tick.get("mid_price", 0.0))
            mp_s = float(s_tick.mid_price if hasattr(s_tick, "mid_price") else s_tick.get("mid_price", 0.0))
            mp_diff = ((mp_s - mp_b) / mp_b * 10000.0) if mp_b > 0 else 0.0

            seq.append([
                float(gross_bps),
                float(vel),
                float(acc),
                float(obi_b),
                float(obi_s),
                float(obi_b - obi_s),
                float(vol_b),
                float(vol_s),
                float(lat_b),
                float(lat_s),
                float(lat_b - lat_s),
                float(top_b),
                float(top_s),
                float(mp_diff),
            ])

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
            # Check for right-censoring timeout (Amendment 5)
            if (now_ms - event.open_time_ms) >= self.max_window_ms:
                self._close_event(key, now_ms, right_censored=True)
                return None

            # Still active? (Candidate is still positive gross spread)
            if candidate.gross_spread_bps > 0:
                event.last_seen_ms = now_ms
                event.tick_count += 1
                if candidate.gross_spread_bps > event.peak_spread_bps:
                    event.peak_spread_bps = candidate.gross_spread_bps
                return event
            else:
                # Spread closed / collapsed
                return self._close_event(key, now_ms, right_censored=False)

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
            )
            self.active_events[key] = new_event

            # Persist event at open with complete 20-step feature window and threshold (Requirement 1 & 3 & 4)
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
                })
            return new_event

        return None

    def check_timeouts(self, current_time_ms: float) -> List[ActiveSpreadEvent]:
        """Closes events that have stopped receiving updates or exceeded max duration."""
        closed: List[ActiveSpreadEvent] = []
        for key, event in list(self.active_events.items()):
            time_since_last = current_time_ms - event.last_seen_ms
            total_duration = current_time_ms - event.open_time_ms

            # Inactivity closure (no ticks for > 1500ms means spread closed)
            if time_since_last > 1500.0:
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

        # Calculate duration
        duration = max(1.0, event.close_time_ms - event.open_time_ms)
        event.duration_ms = duration
        event.log_duration = math.log(1.0 + duration) if self.use_log_duration else duration

        # Ground truth viability label (Requirement 3):
        # Viable = Stays open longer than the event's fixed dataset threshold AND peak spread exceeded total hurdle
        event.is_viable = (duration >= event.latency_threshold_ms) and (event.peak_spread_bps >= event.cost_hurdle_bps)

        self.completed_events.append(event)

        # Update in database with epoch ms timestamp (Requirement 4)
        if self.db:
            self.db.update_spread_event(event.event_id, {
                "close_time": event.close_time_ms,  # Epoch ms
                "peak_spread_bps": event.peak_spread_bps,
                "duration_ms": event.duration_ms,
                "log_duration": event.log_duration,
                "is_viable": event.is_viable,
                "is_right_censored": event.is_right_censored,
                "latency_threshold_ms": event.latency_threshold_ms,
            })

        if self.on_event_closed:
            self.on_event_closed(event)

        return event

