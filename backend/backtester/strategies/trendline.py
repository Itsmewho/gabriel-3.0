import pandas as pd
from typing import Dict, Any, List, Optional
from .base_strat import BaseStrategy

# Expected features from add_trendline_from_pivots(...)
# tl_highs_value, tl_highs_slope, dist_close_to_tl_highs
# tl_lows_value,  tl_lows_slope,  dist_close_to_tl_lows


class TrendlineStrategy(BaseStrategy):
    """Trendline breakout/retest strategy using pivot-derived dynamic lines.

    Modes:
      - breakout: enter on close through the active line
      - retest:   enter on proximity to line followed by acceptance (close back with line as support/resistance)

    Config:
      sl_pips, tp_pips
      mode: "breakout" | "retest"
      side: "both" | "long" | "short"
      use_highs_for_resistance: bool  # for long breakouts through highs line
      tolerance_pips: float           # proximity for retest mode
      require_slope_sign: bool        # enforce expected slope direction
      allowed_sessions: list[int]
      event_block_minutes: int
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 12))
        self.tp_pips = float(config.get("tp_pips", 36))
        self.mode = str(config.get("mode", "breakout"))  # breakout|retest
        self.side = str(config.get("side", "both"))  # both|long|short
        self.use_highs_for_resistance = bool(
            config.get("use_highs_for_resistance", True)
        )
        self.tolerance_pips = float(config.get("tolerance_pips", 3.0))
        self.require_slope_sign = bool(config.get("require_slope_sign", True))
        self.allowed_sessions = set(config.get("allowed_sessions", []))
        self.event_block_minutes = int(config.get("event_block_minutes", 0))

    # --- helpers ---
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

    def _line_ok(self, row: pd.Series, *, for_long: bool) -> Optional[float]:
        """Return active line value for the side or None if not available."""
        if for_long:
            # Option A: breaking above resistance from highs
            if self.use_highs_for_resistance and pd.notna(row.get("tl_highs_value")):
                return float(row["tl_highs_value"])
            # Option B: bouncing off support from lows
            if pd.notna(row.get("tl_lows_value")):
                return float(row["tl_lows_value"])
            return None
        else:
            # For shorts, prefer highs line as resistance or lows line as breakdown
            if pd.notna(row.get("tl_highs_value")):
                return float(row["tl_highs_value"])
            if pd.notna(row.get("tl_lows_value")):
                return float(row["tl_lows_value"])
            return None

    def _slope_ok(self, row: pd.Series, *, for_long: bool) -> bool:
        if not self.require_slope_sign:
            return True
        hs = row.get("tl_highs_slope")
        ls = row.get("tl_lows_slope")
        if for_long:
            # Expect down-sloping highs line for breakout or up-sloping lows line for bounce
            if self.use_highs_for_resistance and pd.notna(hs):
                return float(hs) < 0
            if pd.notna(ls):
                return float(ls) > 0
            return True
        else:
            # Expect up-sloping highs line for rejection or down-sloping lows line for breakdown
            if pd.notna(hs):
                return float(hs) > 0
            if pd.notna(ls):
                return float(ls) < 0
            return True

    # --- core ---
    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester or len(data) < 2:
            return
        curr = data.iloc[-1]

        if not self._session_ok(curr) or not self._event_ok(curr):
            return

        price = float(curr["close"]) if pd.notna(curr.get("close")) else None
        if price is None:
            return

        # Long logic
        if self.side in ("both", "long"):
            line_val = self._line_ok(curr, for_long=True)
            if line_val is not None and self._slope_ok(curr, for_long=True):
                if self.mode == "breakout":
                    if price > line_val:
                        sl = price - (self.sl_pips * self.pip_size)
                        tp = price + (self.tp_pips * self.pip_size)
                        self.backtester.open_trade("buy", price, sl, tp, curr["time"])
                        return
                else:  # retest
                    dist = None
                    if pd.notna(curr.get("dist_close_to_tl_lows")):
                        dist = abs(float(curr.get("dist_close_to_tl_lows")))  # type: ignore
                    if pd.notna(curr.get("dist_close_to_tl_highs")):
                        d2 = abs(float(curr.get("dist_close_to_tl_highs")))  # type: ignore
                        dist = min(dist, d2) if dist is not None else d2
                    tol = self.tolerance_pips * self.pip_size
                    if dist is not None and dist <= tol and price >= line_val:
                        sl = price - (self.sl_pips * self.pip_size)
                        tp = price + (self.tp_pips * self.pip_size)
                        self.backtester.open_trade("buy", price, sl, tp, curr["time"])
                        return

        # Short logic
        if self.side in ("both", "short"):
            line_val = self._line_ok(curr, for_long=False)
            if line_val is not None and self._slope_ok(curr, for_long=False):
                if self.mode == "breakout":
                    if price < line_val:
                        sl = price + (self.sl_pips * self.pip_size)
                        tp = price - (self.tp_pips * self.pip_size)
                        self.backtester.open_trade("sell", price, sl, tp, curr["time"])
                        return
                else:  # retest
                    dist = None
                    if pd.notna(curr.get("dist_close_to_tl_highs")):
                        dist = abs(float(curr.get("dist_close_to_tl_highs")))  # type: ignore
                    if pd.notna(curr.get("dist_close_to_tl_lows")):
                        d2 = abs(float(curr.get("dist_close_to_tl_lows")))  # type: ignore
                        dist = min(dist, d2) if dist is not None else d2
                    tol = self.tolerance_pips * self.pip_size
                    if dist is not None and dist <= tol and price <= line_val:
                        sl = price + (self.sl_pips * self.pip_size)
                        tp = price - (self.tp_pips * self.pip_size)
                        self.backtester.open_trade("sell", price, sl, tp, curr["time"])
                        return
