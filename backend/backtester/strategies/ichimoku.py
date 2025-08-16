import pandas as pd
from typing import Dict, Any

# Assuming base_strat.py contains the BaseStrategy class
from .base_strat import BaseStrategy


class IchimokuStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 10))
        self.tp_pips = float(config.get("tp_pips", 60))

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < 2 or not self.backtester:
            return
        prev, curr = data.iloc[-2], data.iloc[-1]

        is_bullish_cross = (
            prev["tenkan_sen"] < prev["kijun_sen"]
            and curr["tenkan_sen"] > curr["kijun_sen"]
        )

        price_above_cloud = (
            curr["close"] > curr["senkou_span_a_now"]
            and curr["close"] > curr["senkou_span_b_now"]
        )

        chikou_confirm_bullish = curr["close"] > curr["chikou_span"]

        if is_bullish_cross and price_above_cloud and chikou_confirm_bullish:
            self.backtester.open_trade(
                "buy",
                curr["close"],
                curr["close"] - (self.sl_pips * self.pip_size),
                curr["close"] + (self.tp_pips * self.pip_size),
                curr["time"],
            )

        is_bearish_cross = (
            prev["tenkan_sen"] > prev["kijun_sen"]
            and curr["tenkan_sen"] < curr["kijun_sen"]
        )

        price_below_cloud = (
            curr["close"] < curr["senkou_span_a_now"]
            and curr["close"] < curr["senkou_span_b_now"]
        )

        # This is the standard bearish confirmation.
        chikou_confirm_bearish = curr["chikou_span"] < curr["close"]

        if is_bearish_cross and price_below_cloud and chikou_confirm_bearish:
            self.backtester.open_trade(
                "sell",
                curr["close"],
                curr["close"] + (self.sl_pips * self.pip_size),
                curr["close"] - (self.tp_pips * self.pip_size),
                curr["time"],
            )
