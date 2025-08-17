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


class RSIOscillatorStrategy(BaseStrategy):
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 65))
        self.tp_pips = float(config.get("tp_pips", 180))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", False))
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 75))
        self.use_be = bool(config.get("USE_BREAK_EVEN_STOP", False))
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 28))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 5))
        self.require_atr_guard = bool(config.get("require_atr_guard", True))
        self.use_cross_back = bool(config.get("use_cross_back", True))
        self.countertrend_only = bool(config.get("countertrend_only", True))

    def generate_signals(self, data: pd.DataFrame):
        if len(data) < 2 or not self.backtester:
            return
        prev, curr = data.iloc[-2], data.iloc[-1]
        if not _atr_ok(curr, self.require_atr_guard):
            return

        rsi_prev = float(prev.get("rsi", 50))
        rsi_curr = float(curr.get("rsi", 50))
        price = float(curr["close"])

        # Long: oversold then cross back above 30
        long_sig = (
            (rsi_prev < 30 <= rsi_curr) if self.use_cross_back else (rsi_curr < 30)
        )
        if long_sig:
            if self.countertrend_only and float(curr.get("ema_fast", price)) < price:
                pass  # countertrend: price above ema_fast? then skip long
            sl = price - self.sl_pips * self.pip_size
            tp = price + self.tp_pips * self.pip_size
            self.backtester.open_trade("buy", price, sl, tp, curr["time"])
            return

        # Short: overbought then cross back below 70
        short_sig = (
            (rsi_prev > 70 >= rsi_curr) if self.use_cross_back else (rsi_curr > 70)
        )
        if short_sig:
            if self.countertrend_only and float(curr.get("ema_fast", price)) > price:
                pass  # countertrend: price below ema_fast? then skip short
            sl = price + self.sl_pips * self.pip_size
            tp = price - self.tp_pips * self.pip_size
            self.backtester.open_trade("sell", price, sl, tp, curr["time"])
            return
