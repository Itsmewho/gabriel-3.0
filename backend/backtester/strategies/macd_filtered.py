import pandas as pd
from typing import Dict, Any, Optional
from .base_strat import BaseStrategy


class FilteredMACDStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 70))
        self.tp_pips = float(config.get("tp_pips", 180))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", False))
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 95))
        self.use_be = bool(config.get("USE_BREAK_EVEN_STOP", False))
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 38))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 6))
        self.require_atr_guard = bool(config.get("require_atr_guard", True))
        self.require_hist_2bar = bool(config.get("require_hist_2bar", True))
        self.hist_fallback_1bar = bool(config.get("hist_fallback_1bar", True))
        self.require_trend_confirmation = bool(
            config.get("require_trend_confirmation", True)
        )
        self.only_subzero_cross = bool(config.get("only_subzero_cross", True))
        self.avoid_kc_mid = bool(config.get("avoid_kc_mid", True))
        self.kc_mid_band_mult = float(config.get("kc_mid_band_mult", 0.25))  # * ATR
        self.atr_min_for_kc = float(config.get("atr_min_for_kc", 0.0003))

    def _away_from_kc_mid(self, curr: pd.Series) -> bool:
        if not self.avoid_kc_mid:
            return True
        kc_mid = curr.get("kc_mid")
        atr = float(curr.get("atr", 0) or 0)
        if atr < self.atr_min_for_kc:
            return True  # disable clamp in ultra-low vol
        if kc_mid is None or pd.isna(kc_mid) or atr == 0:
            return True
        band = self.kc_mid_band_mult * atr
        return abs(float(curr["close"]) - float(kc_mid)) > band

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < 3 or not self.backtester:
            return
        curr = data.iloc[-1]
        prev = data.iloc[-2]
        prev2 = data.iloc[-3]

        if self.require_atr_guard and int(curr.get("atr_guard_ok", 0)) != 1:
            return
        if self.require_trend_confirmation and float(curr.get("ema_fast", 0)) <= float(
            curr.get("ema_slow", 0)
        ):
            return

        # Histogram momentum requirement with fallback
        h0 = float(pd.to_numeric(curr.get("macd_hist", 0), errors="coerce") or 0)
        h1 = float(pd.to_numeric(prev.get("macd_hist", 0), errors="coerce") or 0)
        h2 = float(pd.to_numeric(prev2.get("macd_hist", 0), errors="coerce") or 0)
        if self.require_hist_2bar:
            hist_ok = h0 > h1 > h2
        else:
            hist_ok = h0 > h1
        if not hist_ok and self.hist_fallback_1bar:
            hist_ok = h0 > h1
        if not hist_ok:
            return

        if not self._away_from_kc_mid(curr):
            return

        # Crosses
        buy_cross = (prev["macd_line"] <= prev["macd_signal"]) and (
            curr["macd_line"] > curr["macd_signal"]
        )
        sell_cross = (prev["macd_line"] >= prev["macd_signal"]) and (
            curr["macd_line"] < curr["macd_signal"]
        )

        price = float(curr["close"])
        if buy_cross:
            if self.only_subzero_cross and float(curr.get("macd_line", 0)) >= 0:
                return
            self.backtester.open_trade(
                "buy",
                price,
                price - self.sl_pips * self.pip_size,
                price + self.tp_pips * self.pip_size,
                curr["time"],
            )
            return
        if sell_cross:
            if self.only_subzero_cross and float(curr.get("macd_line", 0)) <= 0:
                return
            self.backtester.open_trade(
                "sell",
                price,
                price + self.sl_pips * self.pip_size,
                price - self.tp_pips * self.pip_size,
                curr["time"],
            )
