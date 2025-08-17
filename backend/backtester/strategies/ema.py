import pandas as pd
from typing import Dict, Any, Optional, Set
from .base_strat import BaseStrategy


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


class EMACrossoverStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 55))
        self.tp_pips = float(config.get("tp_pips", 160))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", False))
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 65))
        self.use_be = bool(config.get("USE_BREAK_EVEN_STOP", False))
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 22))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 4))
        self.require_atr_guard = bool(config.get("require_atr_guard", True))
        self.require_slope = bool(config.get("require_slope", True))
        self.require_persistence_bars = int(config.get("require_persistence_bars", 2))
        self.min_ma_sep_atr_mult = float(config.get("min_ma_sep_atr_mult", 0.5))
        self.allowed_sessions = set(config.get("session_ids", []))

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < max(2, self.require_persistence_bars) or not self.backtester:
            return
        curr = data.iloc[-1]
        prev = data.iloc[-2]

        if not _atr_ok(curr, self.require_atr_guard):
            return
        if not _in_allowed_session(curr, self.allowed_sessions):
            return

        # Slope filter on fast EMA
        if self.require_slope and (curr["ema_fast"] - prev["ema_fast"]) <= 0:
            # only take longs when fast rising; shorts handled below
            pass

        # Persistence: last N bars must be in the new regime
        if self.require_persistence_bars > 1:
            lastN = data.tail(self.require_persistence_bars)
            long_ok_persist = (lastN["ema_fast"] > lastN["ema_slow"]).all()
            short_ok_persist = (lastN["ema_fast"] < lastN["ema_slow"]).all()
        else:
            long_ok_persist = curr["ema_fast"] > curr["ema_slow"]
            short_ok_persist = curr["ema_fast"] < curr["ema_slow"]

        # Minimum separation to avoid micro crosses (ATR is in price units)
        atr = float(curr.get("atr", 0) or 0)
        sep = abs(float(curr["ema_fast"]) - float(curr["ema_slow"]))
        if atr > 0 and sep < self.min_ma_sep_atr_mult * atr:
            return

        # Classic cross confirmation using prev bar
        cross_up = (
            prev["ema_fast"] <= prev["ema_slow"] and curr["ema_fast"] > curr["ema_slow"]
        )
        cross_dn = (
            prev["ema_fast"] >= prev["ema_slow"] and curr["ema_fast"] < curr["ema_slow"]
        )

        price = float(curr["close"])
        if cross_up and long_ok_persist:
            sl = price - (self.sl_pips * self.pip_size)
            tp = price + (self.tp_pips * self.pip_size)
            self.backtester.open_trade("buy", price, sl, tp, curr["time"])
            return
        if cross_dn and short_ok_persist:
            sl = price + (self.sl_pips * self.pip_size)
            tp = price - (self.tp_pips * self.pip_size)
            self.backtester.open_trade("sell", price, sl, tp, curr["time"])
            return
