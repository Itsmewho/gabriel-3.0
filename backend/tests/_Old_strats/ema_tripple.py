from __future__ import annotations
from typing import Any, Dict, Optional
import math
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE, Trade


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class EmaTripleConfirm(BaseStrategy):
    """
    Entry when FAST crosses SLOW with MID confirming and a long-term TREND_EMA filter.

    Anti-chop (no ATR/Vol/Events):
      - EMA spread threshold
      - EMA slope direction
      - Cooldown bars after entry
      - Optional *confirmation bars* (cross must persist N bars)

    Buy
      - prev_fast <= prev_slow and curr_fast > curr_slow  (bull cross)
      - curr_fast >= curr_mid >= curr_slow                 (3-line alignment)
      - close > trend_ema                                  (trend filter)
      - spread/slope pass
      - cross persists for CONFIRM_BARS bars if > 0

    Sell is symmetric.

    Management
      - Break-even after SL_PIPS + BE_TRIGGER_EXTRA_PIPS in favor
      - Optional trailing once >= SL_PIPS in favor
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 3301,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # Core EMA periods
        self.fast = int(config.get("FAST_EMA", 12))
        self.mid = int(config.get("MID_EMA", 30))
        self.slow = int(config.get("SLOW_EMA", 26))
        self.trend = int(config.get("TREND_EMA", 150))

        # Risk params
        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 40))

        # BE / trailing
        self.be_extra = float(config.get("BE_TRIGGER_EXTRA_PIPS", 1))
        self.be_offset = float(config.get("BE_OFFSET_PIPS", 2))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", True))
        self.trail_dist = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 10))

        # Whipsaw controls (no ATR):
        self.min_ema_spread_pips = float(config.get("MIN_EMA_SPREAD_PIPS", 0))
        self.require_slope = bool(config.get("REQUIRE_SLOPE", False))
        self.min_slope = float(
            config.get("MIN_SLOPE", 0.0)
        )  # abs(ema_t - ema_t-1) in price units
        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 0))

        # Temporal confirmation
        self.confirm_bars = int(config.get("CONFIRM_BARS", 0))  # 0 disables
        self._confirm_side: Optional[str] = None
        self._confirm_count: int = 0

        self.prev_row: Optional[pd.Series] = None
        self.magic = magic
        self.cooldown = 0

    # ---------- utils ----------
    def _open_count_for_me(self, broker) -> int:
        return sum(
            getattr(tr, "strategy_id", None) == self.name for tr in broker.open_trades
        )

    @staticmethod
    def _fav_move_pips(tr: Trade, row: pd.Series) -> float:
        hi = float(row.get("high", row.get("High", float("nan"))))
        lo = float(row.get("low", row.get("Low", float("nan"))))
        return (
            (hi - tr.entry_price) / PIP_SIZE
            if tr.side == "buy"
            else (tr.entry_price - lo) / PIP_SIZE
        )

    def _ema_vals(self, s: pd.Series) -> tuple[float, float, float, float]:
        f = float(s.get(f"ema_{self.fast}"))
        m = float(s.get(f"ema_{self.mid}"))
        sl = float(s.get(f"ema_{self.slow}"))
        tr = float(s.get(f"ema_{self.trend}"))
        return f, m, sl, tr

    def _spread_ok(self, f: float, m: float, sl: float) -> bool:
        if self.min_ema_spread_pips <= 0:
            return True
        pip = PIP_SIZE
        return (abs(f - m) / pip >= self.min_ema_spread_pips) and (
            abs(m - sl) / pip >= self.min_ema_spread_pips
        )

    def _slope_ok(
        self,
        prev_f: float,
        prev_m: float,
        prev_sl: float,
        f: float,
        m: float,
        sl: float,
        side: str,
    ) -> bool:
        if not self.require_slope:
            return True
        df, dm, dsl = f - prev_f, m - prev_m, sl - prev_sl
        if side == "buy":
            return (df >= self.min_slope) and (dm >= 0) and (dsl >= 0)
        else:
            return (df <= -self.min_slope) and (dm <= 0) and (dsl <= 0)

    # ---------- trade management ----------
    def _manage(self, broker, row: pd.Series):
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) != self.name:
                continue
            fav = self._fav_move_pips(tr, row)
            be_thr = self.sl_pips + self.be_extra
            if not getattr(tr, "be_applied", False) and fav >= be_thr:
                broker.set_break_even(
                    trade_id=tr.id, be_pips=be_thr, offset_pips=self.be_offset
                )
                tr.be_applied = True
            if (
                self.use_trailing
                and getattr(tr, "trailing_sl_distance", None) is None
                and fav >= self.sl_pips
            ):
                tr.trailing_sl_distance = self.trail_dist

    # ---------- main hook ----------
    def on_bar(self, broker, t, row: pd.Series):
        self._manage(broker, row)

        if self.cooldown > 0:
            self.cooldown -= 1

        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None
        if self.prev_row is None:
            self.prev_row = row
            return None

        try:
            prev_f, prev_m, prev_sl, prev_tr = self._ema_vals(self.prev_row)
            f, m, sl, tr = self._ema_vals(row)
            close = float(row["close"])  # assume OHLC columns exist
        except Exception:
            self.prev_row = row
            return None

        # Determine raw cross side for this bar
        cross_side = None
        if prev_f <= prev_sl and f > sl:
            cross_side = "buy"
        elif prev_f >= prev_sl and f < sl:
            cross_side = "sell"

        # Maintain confirmation counters
        if self.confirm_bars > 0:
            if cross_side is None:
                # If no fresh cross, but we were counting, ensure side still valid via current separation
                if self._confirm_side == "buy" and f > sl:
                    self._confirm_count += 1
                elif self._confirm_side == "sell" and f < sl:
                    self._confirm_count += 1
                else:
                    self._confirm_side, self._confirm_count = None, 0
            else:
                # Fresh cross: start or continue counting
                if self._confirm_side == cross_side:
                    self._confirm_count += 1
                else:
                    self._confirm_side, self._confirm_count = cross_side, 1
        else:
            # Confirmation disabled -> tie side directly to cross
            self._confirm_side = cross_side
            self._confirm_count = 1 if cross_side else 0

        # Candidate entry only if confirmation satisfied
        ready = (
            self._confirm_side is not None
            and self._confirm_count >= max(1, self.confirm_bars)
            and self.cooldown == 0
        )

        side = self._confirm_side if ready else None

        if side:
            alignment_ok = (f >= m >= sl) if side == "buy" else (f <= m <= sl)
            trend_ok = (close > tr) if side == "buy" else (close < tr)
            spread_ok = self._spread_ok(f, m, sl)
            slope_ok = self._slope_ok(prev_f, prev_m, prev_sl, f, m, sl, side)

            if alignment_ok and trend_ok and spread_ok and slope_ok:
                req = SizeRequest(
                    balance=broker.balance,
                    sl_pips=self.sl_pips,
                    value_per_pip=self._value_per_pip_1lot(broker),
                )
                lots = _round_to_step(
                    self.sizer.size(req).lots,
                    broker.cfg.VOLUME_STEP,
                    broker.cfg.VOLUME_MIN,
                    9999.0,
                )
                if lots > 0:
                    trd = broker.open_trade(
                        side=side,
                        price=close,
                        wanted_lots=lots,
                        sl_pips=self.sl_pips,
                        tp_pips=self.tp_pips,
                        t=t,
                        strategy_id=self.name,
                        magic=self.magic,
                    )
                    self.prev_row = row
                    self.cooldown = self.cooldown_bars
                    # reset confirmation state after entry
                    self._confirm_side, self._confirm_count = None, 0
                    return trd

        self.prev_row = row
        return None


"""
# Feature requirements
# Ensure feature pipeline builds: ema_FAST, ema_MID, ema_SLOW, ema_TREND

# Example config
# strategies = [
#   EmaTripleConfirm(
#     symbol=SYMBOL,
#     config={
#       "name": "EMA_Triple",
#       "FAST_EMA": 14,
#       "MID_EMA": 30,
#       "SLOW_EMA": 50,
#       "TREND_EMA": 150,
#       "SL_PIPS": 10,
#       "TP_PIPS": 40,
#       "USE_TRAILING_STOP": True,
#       "TRAILING_STOP_DISTANCE_PIPS": 10,
#       "BE_TRIGGER_EXTRA_PIPS": 1,
#       "BE_OFFSET_PIPS": 2,
#       "MIN_EMA_SPREAD_PIPS": 0.2,
#       "REQUIRE_SLOPE": True,
#       "MIN_SLOPE": 0.00002,
#       "COOLDOWN_BARS": 2,
#       "CONFIRM_BARS": 1,   # wait 1 extra bar with fast on correct side
#     },
#     strat_cfg=cfg_map["EMA_Triple"],
#     governor=governor,
#   )
# ]
"""
