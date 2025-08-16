import pandas as pd
from typing import Dict, Any
from .base_strat import BaseStrategy


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 10))
        self.tp_pips = float(config.get("tp_pips", 50))

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < 2 or not self.backtester:
            return
        prev, curr = data.iloc[-2], data.iloc[-1]

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
