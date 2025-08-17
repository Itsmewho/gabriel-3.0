import pandas as pd
from typing import Dict, Any
from .base_strat import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 10))
        self.tp_pips = float(config.get("tp_pips", 50))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", False))
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 100))
        self.use_be = bool(config.get("USE_BREAK_EVEN_STOP", False))
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 50))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 10))
        self.require_atr_guard = bool(config.get("require_atr_guard", True))
        self.require_slope = bool(config.get("require_slope", False))

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < 2 or not self.backtester:
            return

        prev, curr = data.iloc[-2], data.iloc[-1]

        if self.require_atr_guard and curr.get("atr_guard_ok", 1) != 1:
            return

        if self.require_slope:
            slope = curr["sma_fast"] - prev["sma_fast"]
            if slope < 0:
                return

        if prev["sma_fast"] < prev["sma_slow"] and curr["sma_fast"] > curr["sma_slow"]:
            self.backtester.open_trade(
                "buy",
                curr["close"],
                curr["close"] - (self.sl_pips * self.pip_size),
                curr["close"] + (self.tp_pips * self.pip_size),
                curr["time"],
            )
        elif (
            prev["sma_fast"] > prev["sma_slow"] and curr["sma_fast"] < curr["sma_slow"]
        ):
            self.backtester.open_trade(
                "sell",
                curr["close"],
                curr["close"] + (self.sl_pips * self.pip_size),
                curr["close"] - (self.tp_pips * self.pip_size),
                curr["time"],
            )
