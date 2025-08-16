import pandas as pd
from typing import Dict, Any, List
from .base_strat import BaseStrategy

REQ: List[str] = [
    "kc_upper",
    "kc_lower",
    "kc_mid",
    "kc_slope",
    "renko_close",
    "renko_dir",
    "renko_flip",
]


class KeltnerRenkoStrategy(BaseStrategy):
    """Breakout strategy combining Keltner Channels with Renko state.

    Long:
      - Renko regime up (renko_dir==1) or fresh flip up (renko_flip==1)
      - AND renko_close > kc_upper
      - Optional: kc_slope > 0

    Short:
      - Renko regime down or fresh flip down
      - AND renko_close < kc_lower
      - Optional: kc_slope < 0

    Filters:
      - allowed_sessions via session_id
      - event_block_minutes via event_minutes_to_event
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 10))
        self.tp_pips = float(config.get("tp_pips", 30))
        self.require_slope = bool(config.get("require_slope", True))
        self.allowed_sessions = set(config.get("allowed_sessions", []))
        self.event_block_minutes = int(config.get("event_block_minutes", 0))

    def _has_cols(self, df: pd.DataFrame) -> bool:
        return all(c in df.columns for c in REQ)

    def _session_ok(self, row: pd.Series) -> bool:
        if not self.allowed_sessions:
            return True
        sid = row.get("session_id")
        if pd.isna(sid):
            return False
        return int(sid) in self.allowed_sessions

    def _event_ok(self, row: pd.Series) -> bool:
        if self.event_block_minutes <= 0:
            return True
        m = row.get("event_minutes_to_event")
        if m is None or pd.isna(m):
            return True
        return abs(float(m)) > self.event_block_minutes

    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester or len(data) < 2:
            return
        if not self._has_cols(data):
            return

        prev, curr = data.iloc[-2], data.iloc[-1]
        if not self._session_ok(curr) or not self._event_ok(curr):
            return

        # Long
        renko_up = (
            int(curr.get("renko_dir", 0)) == 1 or int(curr.get("renko_flip", 0)) == 1
        )
        kc_break_up = pd.notna(curr.get("kc_upper")) and float(
            curr["renko_close"]
        ) > float(curr["kc_upper"])
        slope_ok_up = (not self.require_slope) or int(curr.get("kc_slope", 0)) > 0
        if renko_up and kc_break_up and slope_ok_up:
            price = float(curr["close"])
            sl = price - (self.sl_pips * self.pip_size)
            tp = price + (self.tp_pips * self.pip_size)
            self.backtester.open_trade("buy", price, sl, tp, curr["time"])
            return

        # Short
        renko_dn = int(curr.get("renko_dir", 0)) == -1 or (
            int(curr.get("renko_flip", 0)) == 1 and int(prev.get("renko_dir", 0)) == 1
        )
        # If flip flagged but direction not yet -1 in same bar, infer from close vs kc bands using prev dir
        kc_break_dn = pd.notna(curr.get("kc_lower")) and float(
            curr["renko_close"]
        ) < float(curr["kc_lower"])
        slope_ok_dn = (not self.require_slope) or int(curr.get("kc_slope", 0)) < 0
        if renko_dn and kc_break_dn and slope_ok_dn:
            price = float(curr["close"])
            sl = price + (self.sl_pips * self.pip_size)
            tp = price - (self.tp_pips * self.pip_size)
            self.backtester.open_trade("sell", price, sl, tp, curr["time"])
            return
