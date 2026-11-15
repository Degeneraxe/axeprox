"""Dynamic cost hurdle and fee modeling engine.

Implements Amendments 1 & 4:
- Configurable fee scenarios (retail, vip1, vip3, zero_maker)
- Dynamic volatility hurdle: Hurdle = Total_Taker_Fees + k * Rolling_Volatility
- Net spread and viability computation separate from gross spread detection
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from config.settings import AppConfig
from models.features import compute_rolling_volatility_bps


class CostModel:
    """Calculates venue fee schedules, rolling market volatility, and dynamic hurdles."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()
        # Rolling price windows for volatility estimation: symbol -> Deque[mid_price]
        self.rolling_ticks = self.config.raw.get("dynamic_hurdle", {}).get("rolling_window_ticks", 50)
        self.vol_k = float(self.config.raw.get("dynamic_hurdle", {}).get("volatility_multiplier_k", 1.5))
        self._price_history: Dict[str, Deque[float]] = {}

    def update_price(self, symbol: str, mid_price: float) -> None:
        """Appends mid-price to rolling window for volatility calculation."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.rolling_ticks)
        self._price_history[symbol].append(mid_price)

    def get_rolling_volatility_bps(self, symbol: str) -> float:
        """Computes sample standard deviation of relative price returns in basis points."""
        prices = self._price_history.get(symbol)
        if not prices:
            return 1.0  # Default 1 bp volatility baseline
        return compute_rolling_volatility_bps(prices)

    def get_fee_bps(self, exchange: str, role: str = "taker", scenario: Optional[str] = None) -> float:
        """Returns fee in basis points for an exchange."""
        return self.config.get_fee_bps(exchange, role=role, scenario=scenario)

    def get_hurdle_bps(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        scenario: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Calculates dynamic cost hurdle.
        
        Returns:
            Tuple of (total_hurdle_bps, total_fee_bps, vol_buffer_bps)
        """
        fee_buy = self.get_fee_bps(buy_exchange, role="taker", scenario=scenario)
        fee_sell = self.get_fee_bps(sell_exchange, role="taker", scenario=scenario)
        total_fees = fee_buy + fee_sell

        vol_bps = self.get_rolling_volatility_bps(symbol)
        vol_buffer = self.vol_k * vol_bps

        total_hurdle = total_fees + vol_buffer
        return total_hurdle, total_fees, vol_buffer

    def compute_net_spread_bps(
        self,
        gross_spread_bps: float,
        buy_exchange: str,
        sell_exchange: str,
        scenario: Optional[str] = None,
    ) -> float:
        """Computes net spread after venue fees."""
        fee_buy = self.get_fee_bps(buy_exchange, role="taker", scenario=scenario)
        fee_sell = self.get_fee_bps(sell_exchange, role="taker", scenario=scenario)
        return gross_spread_bps - (fee_buy + fee_sell)
