import pandas as pd
from typing import Dict, Any, List, Optional
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
    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.sl_pips = float(config.get("sl_pips", 95))
        self.tp_pips = float(config.get("tp_pips", 320))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", False))
        self.trail_dist_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 110))
        self.use_be = bool(config.get("USE_BREAK_EVEN_STOP", False))
        self.be_trigger_pips = float(config.get("BE_TRIGGER_PIPS", 42))
        self.be_offset_pips = float(config.get("BE_OFFSET_PIPS", 10))
        self.require_atr_guard = bool(config.get("require_atr_guard", True))
        self.require_mid_or_ob = bool(config.get("require_mid_or_ob", True))
        self.require_discount_premium = bool(
            config.get("require_discount_premium", True)
        )
        self.block_near_pdh_pdl = bool(config.get("block_near_pdh_pdl", True))
        self.fresh_confirm_bars = int(config.get("fresh_confirm_bars", 1))
        self.one_trade_per_swing = bool(config.get("one_trade_per_swing", True))
        # Track state
        self._last_sweep = 0  # 1=up, -1=down, 0=none
        self._prev_day = None
        self._pdh = None  # previous day's high
        self._pdl = None  # previous day's low
        self._day_high = None
        self._day_low = None

    def _update_daily_levels(self, t: pd.Timestamp, high: float, low: float):
        d = t.date()
        if self._prev_day is None:
            self._prev_day = d
            self._day_high = high
            self._day_low = low
            return
        if d != self._prev_day:
            # rollover: yesterday's H/L become PDH/PDL
            self._pdh, self._pdl = self._day_high, self._day_low
            self._day_high, self._day_low = high, low
            self._prev_day = d
        else:
            self._day_high = max(self._day_high or high, high)
            self._day_low = min(self._day_low or low, low)

    def _near_pdh_pdl(self, price: float, atr: float) -> bool:
        if not self.block_near_pdh_pdl:
            return False
        if self._pdh is None or self._pdl is None or atr is None or atr == 0:
            return False
        tol = 0.25 * float(atr)
        return (abs(price - float(self._pdh)) <= tol) or (
            abs(price - float(self._pdl)) <= tol
        )

    def _fresh_confirm(self, curr: pd.Series, prev: Optional[pd.Series]):
        if not self.require_mid_or_ob:
            return True
        mid_ok = (
            int(curr.get("fvg_up_active", 0)) == 1
            and int(curr.get("fvg_up_mid_tag", 0)) == 1
        )
        ob_ok = (
            int(curr.get("ob_bull_active", 0)) == 1
            and int(curr.get("ob_bull_mitigated", 0)) == 1
        )
        if mid_ok or ob_ok:
            return True
        if self.fresh_confirm_bars >= 1 and prev is not None:
            mid_ok_p = (
                int(prev.get("fvg_up_active", 0)) == 1
                and int(prev.get("fvg_up_mid_tag", 0)) == 1
            )
            ob_ok_p = (
                int(prev.get("ob_bull_active", 0)) == 1
                and int(prev.get("ob_bull_mitigated", 0)) == 1
            )
            return mid_ok_p or ob_ok_p
        return False

    def _fresh_confirm_short(self, curr: pd.Series, prev: Optional[pd.Series]):
        if not self.require_mid_or_ob:
            return True
        mid_ok = (
            int(curr.get("fvg_dn_active", 0)) == 1
            and int(curr.get("fvg_dn_mid_tag", 0)) == 1
        )
        ob_ok = (
            int(curr.get("ob_bear_active", 0)) == 1
            and int(curr.get("ob_bear_mitigated", 0)) == 1
        )
        if mid_ok or ob_ok:
            return True
        if self.fresh_confirm_bars >= 1 and prev is not None:
            mid_ok_p = (
                int(prev.get("fvg_dn_active", 0)) == 1
                and int(prev.get("fvg_dn_mid_tag", 0)) == 1
            )
            ob_ok_p = (
                int(prev.get("ob_bear_active", 0)) == 1
                and int(prev.get("ob_bear_mitigated", 0)) == 1
            )
            return mid_ok_p or ob_ok_p
        return False

    def generate_signals(self, data: pd.DataFrame):
        if not self.backtester or len(data) < 2:
            return
        prev, curr = data.iloc[-2], data.iloc[-1]

        # Update daily levels for PDH/PDL tracking
        t = pd.to_datetime(curr["time"]) if "time" in curr else pd.Timestamp.utcnow()
        self._update_daily_levels(
            t,
            float(curr.get("high", curr.get("close", 0))),
            float(curr.get("low", curr.get("close", 0))),
        )

        if self.require_atr_guard and int(curr.get("atr_guard_ok", 0)) != 1:
            return

        atr = float(curr.get("atr", 0) or 0)
        price = float(curr["close"])

        # Long setup
        if int(curr.get("sweep_dn", 0)) == 1:
            if self.one_trade_per_swing and self._last_sweep == -1:
                return
            if not self._fresh_confirm(curr, prev):
                return
            if self.require_discount_premium and int(curr.get("in_discount", 0)) != 1:
                return
            if self._near_pdh_pdl(price, atr):
                return
            self._last_sweep = -1
            self.backtester.open_trade(
                "buy",
                price,
                price - self.sl_pips * self.pip_size,
                price + self.tp_pips * self.pip_size,
                curr["time"],
            )
            return

        # Short setup
        if int(curr.get("sweep_up", 0)) == 1:
            if self.one_trade_per_swing and self._last_sweep == 1:
                return
            if not self._fresh_confirm_short(curr, prev):
                return
            if self.require_discount_premium and int(curr.get("in_premium", 0)) != 1:
                return
            if self._near_pdh_pdl(price, atr):
                return
            self._last_sweep = 1
            self.backtester.open_trade(
                "sell",
                price,
                price + self.sl_pips * self.pip_size,
                price - self.tp_pips * self.pip_size,
                curr["time"],
            )
