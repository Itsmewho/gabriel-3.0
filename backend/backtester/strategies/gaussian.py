import pandas as pd
from typing import Dict, Any, List
from .base_strat import BaseStrategy

REQUIRED_COLS: List[str] = [
    "gauss_mid",
    "gauss_upper",
    "gauss_lower",
    "gauss_slope",
]


class GaussianChannelStrategy(BaseStrategy):
    """Breakout strategy on a causal Gaussian channel.

    Entries:
      - Long: close > gauss_upper AND gauss_slope > 0
      - Short: close < gauss_lower AND gauss_slope < 0
    Exits handled by backtester via SL/TP.

    Optional gates if columns exist:
      - Session: only trade when session_id in allowed_sessions
      - Event filter: skip when |minutes_to_event| <= event_block_minutes
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        # Risk params
        self.sl_pips = float(config.get("sl_pips", 10))
        self.tp_pips = float(config.get("tp_pips", 30))
        # Filters
        self.use_slope = bool(config.get("use_slope", True))
        self.allowed_sessions = set(config.get("allowed_sessions", []))  # e.g., {1,2,3}
        self.event_block_minutes = int(
            config.get("event_block_minutes", 0)
        )  # 0 disables

    # --- helpers ---
    def _has_required(self, data: pd.DataFrame) -> bool:
        return all(col in data.columns for col in REQUIRED_COLS)

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
        minutes_to_event = row.get("event_minutes_to_event")
        if minutes_to_event is None or pd.isna(minutes_to_event):
            return True
        # Block inside the symmetric window around the event
        return abs(float(minutes_to_event)) > self.event_block_minutes

    # --- core ---
    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester or len(data) < 2:
            return
        if not self._has_required(data):
            return

        prev, curr = data.iloc[-2], data.iloc[-1]

        # Optional gates
        if not self._session_ok(curr):
            return
        if not self._event_ok(curr):
            return

        # Long breakout
        if pd.notna(curr["gauss_upper"]) and curr["close"] > curr["gauss_upper"]:
            if not self.use_slope or curr["gauss_slope"] > 0:
                self.backtester.open_trade(
                    "buy",
                    float(curr["close"]),
                    float(curr["close"]) - (self.sl_pips * self.pip_size),
                    float(curr["close"]) + (self.tp_pips * self.pip_size),
                    curr["time"],
                )
                return

        # Short breakout
        if pd.notna(curr["gauss_lower"]) and curr["close"] < curr["gauss_lower"]:
            if not self.use_slope or curr["gauss_slope"] < 0:
                self.backtester.open_trade(
                    "sell",
                    float(curr["close"]),
                    float(curr["close"]) + (self.sl_pips * self.pip_size),
                    float(curr["close"]) - (self.tp_pips * self.pip_size),
                    curr["time"],
                )
                return
