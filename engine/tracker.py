"""Spread Event Tracker and Ground-Truth Duration/Viability Labeler.

Implements Amendments 1, 5, and 6:
- Tracks spread opening, persistence, and closure lifecycle
- Calculates ground-truth duration_ms, log_duration, and is_viable
- Flags right-censored events where spreads do not close before window expiration
- Strict anti-leakage guarantee: features recorded at open, labels derived at closure
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.settings import AppConfig
from data.storage.db_manager import DatabaseManager
from engine.detector import SpreadCandidate

logger = logging.getLogger(__name__)


@dataclass
class ActiveSpreadEvent:
    """Tracks an ongoing arbitrage opportunity window."""
    event_id: str
    symbol: str
    buy_exchange: str
    sell_exchange: str
    open_time_ms: float
    open_time_iso: str
    open_spread_bps: float
    cost_hurdle_bps: float
    peak_spread_bps: float
    last_seen_ms: float
    tick_count: int = 1
    features_at_open: Dict[str, float] = field(default_factory=dict)
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
    ):
        self.db = db_manager
        self.config = config or AppConfig.load()
        self.on_event_closed = on_event_closed

        # Pair key: (symbol, buy_exchange, sell_exchange) -> ActiveSpreadEvent
        self.active_events: Dict[Tuple[str, str, str], ActiveSpreadEvent] = {}

        # Labeling parameters: derive viability latency threshold from measured transit_latency_ms in DB
        lat_stats = self.db.get_empirical_latency_stats() if self.db else {}
        lbl_cfg = self.config.raw.get("labeling", {})
        if lat_stats and lat_stats.get("count", 0) >= 10:
            self.latency_hurdle_ms = float(lat_stats["viability_threshold_ms"])
        else:
            self.latency_hurdle_ms = float(lbl_cfg.get("execution_latency_assumption_ms", 50.0))

        self.max_window_ms = float(lbl_cfg.get("right_censored_max_window_ms", 30000.0))
        self.use_log_duration = bool(lbl_cfg.get("use_log_duration", True))

        self.completed_events: List[ActiveSpreadEvent] = []

    def process_candidate(
        self, candidate: SpreadCandidate, features: Optional[Dict[str, float]] = None
    ) -> Optional[ActiveSpreadEvent]:
        """Updates active spread state machine with newly scanned candidate."""
        key = (candidate.symbol, candidate.buy_exchange, candidate.sell_exchange)
        now_ms = candidate.timestamp_ms

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
            event_id = f"evt_{uuid.uuid4().hex[:12]}"
            open_iso = datetime.fromtimestamp(now_ms / 1000.0, timezone.utc).isoformat()
            new_event = ActiveSpreadEvent(
                event_id=event_id,
                symbol=candidate.symbol,
                buy_exchange=candidate.buy_exchange,
                sell_exchange=candidate.sell_exchange,
                open_time_ms=now_ms,
                open_time_iso=open_iso,
                open_spread_bps=candidate.gross_spread_bps,
                cost_hurdle_bps=candidate.total_hurdle_bps,
                peak_spread_bps=candidate.gross_spread_bps,
                last_seen_ms=now_ms,
                tick_count=1,
                features_at_open=features or {},
            )
            self.active_events[key] = new_event

            # Record open in database
            if self.db:
                self.db.insert_spread_event({
                    "event_id": new_event.event_id,
                    "symbol": new_event.symbol,
                    "buy_exchange": new_event.buy_exchange,
                    "sell_exchange": new_event.sell_exchange,
                    "open_time": new_event.open_time_iso,
                    "open_spread_bps": new_event.open_spread_bps,
                    "cost_hurdle_bps": new_event.cost_hurdle_bps,
                    "peak_spread_bps": new_event.peak_spread_bps,
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
        """Finalizes spread event, computes labels, and persists to DB."""
        event = self.active_events.pop(key, None)
        if not event:
            return None

        event.is_closed = True
        event.close_time_ms = close_time_ms
        event.is_right_censored = right_censored

        # Calculate duration
        duration = max(1.0, close_time_ms - event.open_time_ms)
        event.duration_ms = duration
        event.log_duration = math.log(1.0 + duration) if self.use_log_duration else duration

        # Ground truth viability label:
        # Viable = Stays open longer than empirical execution latency AND peak spread exceeded total hurdle
        event.is_viable = (duration >= self.latency_hurdle_ms) and (event.peak_spread_bps >= event.cost_hurdle_bps)

        self.completed_events.append(event)

        # Update in database
        if self.db:
            close_iso = datetime.fromtimestamp(close_time_ms / 1000.0, timezone.utc).isoformat()
            self.db.update_spread_event(event.event_id, {
                "close_time": close_iso,
                "peak_spread_bps": event.peak_spread_bps,
                "duration_ms": event.duration_ms,
                "log_duration": event.log_duration,
                "is_viable": event.is_viable,
                "is_right_censored": event.is_right_censored,
            })

        if self.on_event_closed:
            self.on_event_closed(event)

        return event
