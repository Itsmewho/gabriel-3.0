import pandas as pd
from typing import Dict, Any, Optional, Set
from .base_strat import BaseStrategy


# ---------------------------------
# Helpers
# ---------------------------------
def _in_allowed_session(row: pd.Series, allowed: Optional[Set[int]]) -> bool:
    if not allowed:
        return True
    sid = row.get("session_id")
    if pd.isna(sid):
        return False
    try:
        return int(sid) in allowed
    except Exception:
        return False


def _atr_ok(row: pd.Series, require: bool) -> bool:
    return True if not require else int(row.get("atr_guard_ok", 0)) == 1


class MACDStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 70))
        self.tp_pips = float(config.get("tp_pips", 200))
        self.use_trailing = True
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 85))
        self.use_be = True
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 32))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 6))
        self.require_atr_guard = True
        self.delay_bars = int(config.get("delay_bars", 1))

    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester:
            return
        needed = max(2, self.delay_bars + 1)
        if len(data) < needed:
            return
        curr = data.iloc[-1]
        if not _atr_ok(curr, self.require_atr_guard):
            return

        d = self.delay_bars
        idx_prev = -(d + 1)
        idx_cross = -d

        prev_bar = data.iloc[idx_prev]
        cross_bar = data.iloc[idx_cross]

        cross_up_then = (prev_bar["macd_line"] <= prev_bar["macd_signal"]) and (
            cross_bar["macd_line"] > cross_bar["macd_signal"]
        )
        cross_dn_then = (prev_bar["macd_line"] >= prev_bar["macd_signal"]) and (
            cross_bar["macd_line"] < cross_bar["macd_signal"]
        )

        still_long = curr["macd_line"] > curr["macd_signal"]
        still_short = curr["macd_line"] < curr["macd_signal"]

        price = float(curr["close"])
        if cross_up_then and still_long:
            self.backtester.open_trade(
                "buy",
                price,
                price - self.sl_pips * self.pip_size,
                price + self.tp_pips * self.pip_size,
                curr["time"],
            )
        elif cross_dn_then and still_short:
            self.backtester.open_trade(
                "sell",
                price,
                price + self.sl_pips * self.pip_size,
                price - self.tp_pips * self.pip_size,
                curr["time"],
            )
