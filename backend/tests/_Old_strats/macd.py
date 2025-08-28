# strategies/macd_crossover.py
from __future__ import annotations
from typing import Any, Dict, Optional
import math, pandas as pd
from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import Trade, PIP_SIZE


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class MacdCrossover(BaseStrategy):
    """
    Buy when MACD crosses ABOVE signal below the zero-line.
    Sell when MACD crosses BELOW signal above the zero-line.
    Optional BE/trailing are gated to trigger only after >= 1R.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 3101,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)
        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 60))
        self.be_extra = float(config.get("BE_TRIGGER_EXTRA_PIPS", 1))
        self.be_offset = float(config.get("BE_OFFSET_PIPS", 2))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", True))
        self.trail_dist = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 10))
        # column names (override if your DF uses different names)
        self.col_macd = config.get("MACD_COL", "macd")
        self.col_signal = config.get("MACD_SIGNAL_COL", "macd_signal")
        self.prev_row: Optional[pd.Series] = None
        self.magic = magic

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

    def on_bar(self, broker, t, row: pd.Series):
        # always manage existing
        self._manage(broker, row)

        # risk gates
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
            pm, ps = float(self.prev_row[self.col_macd]), float(  # type: ignore
                self.prev_row[self.col_signal]  # type: ignore
            )
            cm, cs = float(row[self.col_macd]), float(row[self.col_signal])  # type: ignore
            close = float(row["close"])
        except Exception:
            self.prev_row = row
            return None

        side = None
        # Bullish cross below zero
        if pm < ps and cm > cs and cm < 0:
            side = "buy"
        # Bearish cross above zero
        elif pm > ps and cm < cs and cm > 0:
            side = "sell"

        if side:
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
                tr = broker.open_trade(
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
                return tr

        self.prev_row = row
        return None
