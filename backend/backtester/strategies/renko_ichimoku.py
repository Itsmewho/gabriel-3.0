import pandas as pd
from typing import Dict, Any, List
from .base_strat import BaseStrategy

REQ: List[str] = [
    "renko_close",
    "renko_dir",
    "renko_flip",
    "senkou_span_a_now",
    "senkou_span_b_now",
    "ichi_price_above_cloud",
    "ichi_price_below_cloud",
    "ichi_tenkan_above_kijun",
    "ichi_cloud_color",
]


class RenkoIchimokuStrategy(BaseStrategy):
    """Continuation strategy using Renko direction with Ichimoku cloud state.

    Long:
      - Price above cloud (ichi_price_above_cloud==1) AND tenkan above kijun
      - Renko regime up or fresh flip up

    Short:
      - Price below cloud AND tenkan below kijun
      - Renko regime down or fresh flip down

    Filters:
      - allowed_sessions via session_id
      - event_block_minutes via event_minutes_to_event
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 12))
        self.tp_pips = float(config.get("tp_pips", 36))
        self.require_cloud_color = bool(
            config.get("require_cloud_color", False)
        )  # cloud color in trend direction
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

        cloud_top = max(
            float(curr["senkou_span_a_now"]), float(curr["senkou_span_b_now"])
        )
        cloud_bot = min(
            float(curr["senkou_span_a_now"]), float(curr["senkou_span_b_now"])
        )

        # Long conditions
        above_cloud = (
            int(curr.get("ichi_price_above_cloud", 0)) == 1
            or float(curr["close"]) > cloud_top
        )
        tenkan_kijun_bull = int(curr.get("ichi_tenkan_above_kijun", 0)) == 1
        cloud_color_bull = (
            int(curr.get("ichi_cloud_color", 0)) > 0
            if self.require_cloud_color
            else True
        )
        renko_up = (
            int(curr.get("renko_dir", 0)) == 1 or int(curr.get("renko_flip", 0)) == 1
        )

        if above_cloud and tenkan_kijun_bull and cloud_color_bull and renko_up:
            price = float(curr["close"])
            sl = price - (self.sl_pips * self.pip_size)
            tp = price + (self.tp_pips * self.pip_size)
            self.backtester.open_trade("buy", price, sl, tp, curr["time"])
            return

        # Short conditions
        below_cloud = (
            int(curr.get("ichi_price_below_cloud", 0)) == 1
            or float(curr["close"]) < cloud_bot
        )
        tenkan_kijun_bear = int(curr.get("ichi_tenkan_above_kijun", 0)) == 0
        cloud_color_bear = (
            int(curr.get("ichi_cloud_color", 0)) < 0
            if self.require_cloud_color
            else True
        )
        renko_dn = int(curr.get("renko_dir", 0)) == -1 or (
            int(curr.get("renko_flip", 0)) == 1 and int(prev.get("renko_dir", 0)) == 1
        )

        if below_cloud and tenkan_kijun_bear and cloud_color_bear and renko_dn:
            price = float(curr["close"])
            sl = price + (self.sl_pips * self.pip_size)
            tp = price - (self.tp_pips * self.pip_size)
            self.backtester.open_trade("sell", price, sl, tp, curr["time"])
            return
