import pandas as pd
from typing import Dict, Any
from .base_strat import BaseStrategy


class MidnightOpenStrategy(BaseStrategy):
    """
    Opens a long trade at configurable time and closes it after a fixed holding period.
    Includes optional trend and volatility filters.
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 0))
        self.tp_pips = float(config.get("tp_pips", 0))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", False))
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 100))
        self.use_be = bool(config.get("USE_BREAK_EVEN_STOP", False))
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 50))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 10))

        self.entry_hour = int(config.get("entry_hour", 0))
        self.entry_minute = int(config.get("entry_minute", 10))
        self.hold_minutes = int(config.get("hold_minutes", 120))
        self.require_atr_guard = bool(config.get("require_atr_guard", True))
        self.require_trend = bool(config.get("require_trend", False))

        self.open_times = set()  # Track already opened dates

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < 1 or not self.backtester:
            return

        curr = data.iloc[-1]
        t = pd.to_datetime(curr["time"])

        # --- Entry Conditions ---
        if t.hour == self.entry_hour and t.minute == self.entry_minute:
            trade_date = t.date()
            if trade_date in self.open_times:
                return

            # --- ATR Volatility Filter ---
            if self.require_atr_guard and curr.get("atr_guard_ok", 1) != 1:
                return

            # --- Trend Filter ---
            if (
                self.require_trend
                and curr.get("ema_fast")
                and curr["close"] < curr["ema_fast"]
            ):
                return

            price = curr["close"]
            sl = price - (self.sl_pips * self.pip_size) if self.sl_pips > 0 else None
            tp = price + (self.tp_pips * self.pip_size) if self.tp_pips > 0 else None
            self.backtester.open_trade("buy", price, sl, tp, curr["time"])
            self.open_times.add(trade_date)

        # --- Exit Condition (Time-Based Close) ---
        t_minus_hold = t - pd.Timedelta(minutes=self.hold_minutes)
        close_time_str = t_minus_hold.strftime("%Y-%m-%d %H:%M")

        for trade in list(self.backtester.open_trades):
            if trade.entry_time.strftime("%Y-%m-%d %H:%M") == close_time_str:
                self.backtester.close_trade(
                    trade, "Time Close", curr["close"], curr["time"]
                )
