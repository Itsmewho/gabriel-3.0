## USELESSS for this


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


class AtrChannelBreakoutStrategy(BaseStrategy):
    """
    Entry:
      - LONG  if close crosses above SMA(n) + entry_mult * ATR(n)
      - SHORT if close crosses below SMA(n) - entry_mult * ATR(n)

    Risk:
      - SL = sl_mult * ATR(n)
      - TP = tp_mult * ATR(n)
      - Break-even & trailing only after >= +1R (in pips, using intrabar extremes)
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 3001,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)
        self.n = int(config.get("ATR_PERIOD", 40))
        self.ma_n = int(config.get("MA_PERIOD", 40))
        self.entry_mult = float(config.get("ENTRY_ATR_MULT", 1.0))
        self.sl_mult = float(config.get("SL_ATR_MULT", 1.0))
        self.tp_mult = float(config.get("TP_ATR_MULT", 2.0))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", True))
        self.trailing_mult = float(config.get("TRAIL_ATR_MULT", 1.0))
        self.be_extra = float(config.get("BE_TRIGGER_EXTRA_PIPS", 0.0))
        self.be_offset = float(config.get("BE_OFFSET_PIPS", 2.0))
        self.magic = magic
        self.prev_row: Optional[pd.Series] = None

    # ---------- helpers ----------
    def _open_count_for_me(self, broker) -> int:
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    @staticmethod
    def _fav_move_pips(tr: Trade, row: pd.Series) -> float:
        high = float(row.get("high", row.get("High")))  # type: ignore
        low = float(row.get("low", row.get("Low")))  # type: ignore
        return (
            (high - tr.entry_price) / PIP_SIZE
            if tr.side == "buy"
            else (tr.entry_price - low) / PIP_SIZE
        )

    def _manage_positions(self, broker, row: pd.Series, sl_pips_current: float) -> None:
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) != self.name:
                continue
            fav = self._fav_move_pips(tr, row)
            be_threshold = sl_pips_current + self.be_extra
            if not getattr(tr, "be_applied", False) and fav >= be_threshold:
                broker.set_break_even(
                    trade_id=tr.id, be_pips=be_threshold, offset_pips=self.be_offset
                )
                tr.be_applied = True
            if (
                self.use_trailing
                and tr.trailing_sl_distance is None
                and fav >= sl_pips_current
            ):
                tr.trailing_sl_distance = max(
                    1.0, self.trailing_mult * sl_pips_current
                )  # trail distance in pips

    # ---------- main ----------
    def on_bar(self, broker, t, row: pd.Series):
        # need features
        try:
            atr_price = float(row[f"atr_{self.n}"])  # ATR in price units
            sma_price = float(row[f"sma_{self.ma_n}"])
            close = float(row["close"])
        except Exception:
            self.prev_row = row
            return None

        # convert ATR to pips for sizing/management
        atr_pips = max(0.1, atr_price / PIP_SIZE)
        sl_pips = self.sl_mult * atr_pips
        tp_pips = self.tp_mult * atr_pips

        # always manage existing positions first
        self._manage_positions(broker, row, sl_pips_current=sl_pips)

        # risk gates
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None

        # need prior bar to detect "cross"
        if self.prev_row is None:
            self.prev_row = row
            return None

        try:
            prev_close = float(self.prev_row["close"])
            prev_sma = float(self.prev_row[f"sma_{self.ma_n}"])
            prev_atr = float(self.prev_row[f"atr_{self.n}"])
        except Exception:
            self.prev_row = row
            return None

        prev_upper = prev_sma + self.entry_mult * prev_atr
        prev_lower = prev_sma - self.entry_mult * prev_atr
        curr_upper = sma_price + self.entry_mult * atr_price
        curr_lower = sma_price - self.entry_mult * atr_price

        side: Optional[str] = None
        # cross above upper channel → long
        if prev_close <= prev_upper and close > curr_upper:
            side = "buy"
        # cross below lower channel → short
        elif prev_close >= prev_lower and close < curr_lower:
            side = "sell"

        if side:
            req = SizeRequest(
                balance=broker.balance,
                sl_pips=sl_pips,
                value_per_pip=self._value_per_pip_1lot(broker),
            )
            sized = self.sizer.size(req)
            lots = _round_to_step(
                sized.lots, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
            )
            if lots > 0:
                tr = broker.open_trade(
                    side=side,
                    price=close,
                    wanted_lots=lots,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    t=t,
                    strategy_id=self.name,
                    magic=self.magic,
                )
                # do NOT arm trailing/BE here; we gate them post +1R in _manage_positions()
                self.prev_row = row
                return tr

        self.prev_row = row
        return None
