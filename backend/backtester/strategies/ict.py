import pandas as pd
from typing import Dict, Any, List
from .base_strat import BaseStrategy

REQUIRED_ANY_LONG: List[str] = [
    # At least one of these should exist for long logic to work robustly
    "sweep_dn",
    "fvg_up",
    "fvg_up_active",
    "fvg_up_mid_tag",
    "ob_bull_active",
    "ob_bull_mitigated",
]
REQUIRED_ANY_SHORT: List[str] = [
    "sweep_up",
    "fvg_dn",
    "fvg_dn_active",
    "fvg_dn_mid_tag",
    "ob_bear_active",
    "ob_bear_mitigated",
]


class ICTStrategy(BaseStrategy):
    """Rule-based ICT scaffold. No look-ahead.

    Long idea (precise but simple):
      - Liquidity sweep of lows (sweep_dn == 1)
      - AND either bullish FVG active with mid tag OR bullish OB mitigated
      - Optional premium/discount: prefer discount (in_discount == 1) if available

    Short idea:
      - Liquidity sweep of highs (sweep_up == 1)
      - AND either bearish FVG active with mid tag OR bearish OB mitigated
      - Optional premium/discount: prefer premium (in_premium == 1) if available

    Filters:
      - allowed_sessions via session_id
      - event_block_minutes via event_minutes_to_event
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        # Risk params
        self.sl_pips = float(config.get("sl_pips", 12))
        self.tp_pips = float(config.get("tp_pips", 36))
        # Filters
        self.allowed_sessions = set(config.get("allowed_sessions", []))
        self.event_block_minutes = int(config.get("event_block_minutes", 0))
        self.require_discount_premium = bool(
            config.get("require_discount_premium", False)
        )
        # Use mid-tag or OB mitigation as confirmation
        self.require_mid_or_ob = bool(config.get("require_mid_or_ob", True))

    # --- helpers ---
    def _cols_exist(self, data: pd.DataFrame, cols: List[str]) -> bool:
        return all(c in data.columns for c in cols)

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

    def _discount_ok(self, row: pd.Series, long: bool) -> bool:
        if not self.require_discount_premium:
            return True
        flag = row.get("in_discount") if long else row.get("in_premium")
        if flag is None or pd.isna(flag):
            return True  # do not block if feature absent
        return bool(int(flag) == 1)

    # --- core ---
    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester or len(data) < 2:
            return
        curr = data.iloc[-1]

        if not self._session_ok(curr) or not self._event_ok(curr):
            return

        # Long setup
        can_long = "sweep_dn" in data.columns and int(curr.get("sweep_dn", 0)) == 1
        if can_long:
            confirm_long = True
            if self.require_mid_or_ob:
                mid_ok = (
                    int(curr.get("fvg_up_active", 0)) == 1
                    and int(curr.get("fvg_up_mid_tag", 0)) == 1
                )
                ob_ok = (
                    int(curr.get("ob_bull_active", 0)) == 1
                    and int(curr.get("ob_bull_mitigated", 0)) == 1
                )
                confirm_long = mid_ok or ob_ok
            if confirm_long and self._discount_ok(curr, True):
                price = float(curr["close"])
                sl = price - (self.sl_pips * self.pip_size)
                tp = price + (self.tp_pips * self.pip_size)
                self.backtester.open_trade("buy", price, sl, tp, curr["time"])
                return

        # Short setup
        can_short = "sweep_up" in data.columns and int(curr.get("sweep_up", 0)) == 1
        if can_short:
            confirm_short = True
            if self.require_mid_or_ob:
                mid_ok = (
                    int(curr.get("fvg_dn_active", 0)) == 1
                    and int(curr.get("fvg_dn_mid_tag", 0)) == 1
                )
                ob_ok = (
                    int(curr.get("ob_bear_active", 0)) == 1
                    and int(curr.get("ob_bear_mitigated", 0)) == 1
                )
                confirm_short = mid_ok or ob_ok
            if confirm_short and self._discount_ok(curr, False):
                price = float(curr["close"])
                sl = price + (self.sl_pips * self.pip_size)
                tp = price - (self.tp_pips * self.pip_size)
                self.backtester.open_trade("sell", price, sl, tp, curr["time"])
                return
