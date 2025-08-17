import pandas as pd
from typing import Dict, Any

# Assuming base_strat.py contains the BaseStrategy class
from .base_strat import BaseStrategy


class IchimokuStrategy(BaseStrategy):
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

    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester or len(data) < 2:
            return

        prev, curr = data.iloc[-2], data.iloc[-1]

        if self.require_atr_guard and curr.get("atr_guard_ok", 1) != 1:
            return

        bullish = (
            prev["tenkan_sen"] < prev["kijun_sen"]
            and curr["tenkan_sen"] > curr["kijun_sen"]
            and curr["close"] > curr["senkou_span_a_now"]
            and curr["close"] > curr["senkou_span_b_now"]
            and curr["close"] > curr["chikou_span"]
        )

        bearish = (
            prev["tenkan_sen"] > prev["kijun_sen"]
            and curr["tenkan_sen"] < curr["kijun_sen"]
            and curr["close"] < curr["senkou_span_a_now"]
            and curr["close"] < curr["senkou_span_b_now"]
            and curr["chikou_span"] < curr["close"]
        )

        price = curr["close"]
        if bullish:
            self.backtester.open_trade(
                "buy",
                price,
                price - self.sl_pips * self.pip_size,
                price + self.tp_pips * self.pip_size,
                curr["time"],
            )
        elif bearish:
            self.backtester.open_trade(
                "sell",
                price,
                price + self.sl_pips * self.pip_size,
                price - self.tp_pips * self.pip_size,
                curr["time"],
            )
