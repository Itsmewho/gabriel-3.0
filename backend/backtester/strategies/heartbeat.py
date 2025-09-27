from __future__ import annotations
from typing import Any, Dict, Optional, Deque, List, Sequence
import math
from collections import deque

import numpy as np
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE


# --- helpers ---
EPS = 1e-9


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


def _full_body_above(row: pd.Series, level: float) -> bool:
    return float(row["open"]) > level and float(row["close"]) > level


def _full_body_below(row: pd.Series, level: float) -> bool:
    return float(row["open"]) < level and float(row["close"]) < level


def _slope_norm_reg(y: Sequence[float], atr_last: float) -> float:
    """Linear-regression slope per bar, normalized by last ATR, mapped to (0,1)."""
    y_arr = np.asarray(y, dtype=float)
    if y_arr.size < 2:
        return 0.5
    x = np.arange(y_arr.size, dtype=float)
    beta1 = np.polyfit(x, y_arr, 1)[0]  # slope per bar
    d = beta1 / max(abs(atr_last), EPS)
    return 0.5 + math.atan(d) / math.pi


class PtlAdvancedFilteredReversal(BaseStrategy):
    """Stop-and-reverse on PTL flips with Keltner and EMA-pack confirmations.

    Entries require: PTL flip + candle body relative to KC mid + KC width cap.
    Exits may be ignored using ema_400 position vs KC and EMA-pack regression slope.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 2029,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # columns
        self.ptl_trena_col = self.config.get("PTL_TRENA_COL", "ptl_trena")
        self.kc_upper_col = self.config.get("KC_UPPER_COL", "kc_30_30_2.5_upper")
        self.kc_middle_col = self.config.get("KC_MIDDLE_COL", "kc_30_30_2.5_mid")
        self.kc_lower_col = self.config.get("KC_LOWER_COL", "kc_30_30_2.5_lower")
        self.ema_pack_cols: List[str] = self.config.get(
            "EMA_PACK_COLS", ["ema_80", "ema_100", "ema_200"]
        )
        self.ema400_col = self.config.get("EMA_400_COL", "ema_400")
        self.atr_col = self.config.get("ATR_COL", "atr_24")

        # params
        self.sl_pips = float(self.config.get("SL_PIPS", 18))
        self.tp_pips = float(self.config.get("TP_PIPS", 27))
        self.max_kc_pips = float(self.config.get("MAX_KC_PIPS", 18.0))
        self.sell_slope_thresh = float(self.config.get("SELL_SLOPE_THRESH", 0.48))
        self.buy_slope_thresh = float(self.config.get("BUY_SLOPE_THRESH", 0.52))
        self.pack_window = int(
            self.config.get("PACK_WINDOW", 10)
        )  # bars for regression
        self.debug = bool(self.config.get("DEBUG_REASON", False))

        self._required_cols = [
            self.ptl_trena_col,
            self.kc_upper_col,
            self.kc_middle_col,
            self.kc_lower_col,
            *self.ema_pack_cols,
            self.ema400_col,
            self.atr_col,
            "open",
            "close",
        ]

        # state
        self.magic = magic
        self.prev_row: Optional[pd.Series] = None
        self._validated = False
        self._current_side: Optional[str] = None
        self._buy_trade_ema400_override = False
        self._recent_bars: Deque[pd.Series] = deque(maxlen=self.pack_window)

    # --- slope ---
    def _pack_slope(self) -> float:
        """Average normalized LR slope over EMA pack using last pack_window bars."""
        if len(self._recent_bars) < 2:
            return 0.5
        atr_last = float(self._recent_bars[-1].get(self.atr_col, 1.0) or 1.0)
        sl_sum = 0.0
        n = 0
        for col in self.ema_pack_cols:
            try:
                y = [b[col] for b in self._recent_bars]
                sl = _slope_norm_reg(y, atr_last)
                sl_sum += sl
                n += 1
            except Exception:
                continue
        return (sl_sum / max(n, 1)) if n else 0.5

    # --- open ---
    def _open_new_trade(self, broker, t, row: pd.Series, side: str):
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            return
        if any(tr.strategy_id == self.name for tr in broker.open_trades):
            return
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=self.sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        lots_raw = self.sizer.size(req).lots
        lots = _round_to_step(
            lots_raw, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        if lots <= 0:
            return
        tr = broker.open_trade(
            side=side,
            price=float(row["close"]),
            wanted_lots=lots,
            sl_pips=self.sl_pips,
            tp_pips=self.tp_pips,
            t=t,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)

    # --- main ---
    def on_bar(self, broker, t, row: pd.Series):
        self._recent_bars.append(row)
        if not self._validated:
            missing = [c for c in self._required_cols if c not in row.index]
            if missing:
                if self.debug:
                    print(f"{t} | {self.name} :: missing columns: {missing}")
                return
            self._validated = True

        if self.prev_row is None:
            self.prev_row = row
            return

        prev_ptl, cur_ptl = self.prev_row[self.ptl_trena_col], row[self.ptl_trena_col]
        flip_side = (
            "sell"
            if prev_ptl == 0 and cur_ptl == 1
            else ("buy" if prev_ptl == 1 and cur_ptl == 0 else None)
        )
        if not flip_side:
            self.prev_row = row
            return

        kc_mid = float(row[self.kc_middle_col])
        kc_low = float(row[self.kc_lower_col])
        kc_up = float(row[self.kc_upper_col])
        ema400 = float(row[self.ema400_col])
        ema_slope = self._pack_slope()

        # --- EXIT FILTERS with AND logic ---
        if self._current_side and flip_side != self._current_side:
            if self._current_side == "sell" and flip_side == "buy":
                # keep SELL only if long-term AND short-term both bearish
                if (ema400 > kc_mid) and (ema_slope <= self.sell_slope_thresh):
                    if self.debug:
                        print(
                            f"{t} | {self.name} :: hold SELL (ema400>mid and slope={ema_slope:.3f}<=th)"
                        )
                    self.prev_row = row
                    return
            if self._current_side == "buy" and flip_side == "sell":
                # maintain BUY override only while ema400 < lower AND slope is bullish
                if ema400 < kc_low:
                    self._buy_trade_ema400_override = True
                elif ema400 > kc_mid:
                    self._buy_trade_ema400_override = False
                if self._buy_trade_ema400_override and (
                    ema_slope >= self.buy_slope_thresh
                ):
                    if self.debug:
                        print(
                            f"{t} | {self.name} :: hold BUY (ema400<lower and slope={ema_slope:.3f}>=th)"
                        )
                    self.prev_row = row
                    return

        # --- ENTRY FILTERS ---
        kc_width = kc_up - kc_low
        if kc_width > (self.max_kc_pips * PIP_SIZE):
            if self.debug:
                print(
                    f"{t} | {self.name} :: skip entry: kc_width {kc_width:.6f} > {self.max_kc_pips} pips"
                )
            self.prev_row = row
            return
        if ((flip_side == "sell") and not _full_body_above(row, kc_mid)) or (
            (flip_side == "buy") and not _full_body_below(row, kc_mid)
        ):
            if self.debug:
                print(
                    f"{t} | {self.name} :: skip entry: body not on correct side of kc_mid {kc_mid:.6f}"
                )
            self.prev_row = row
            return

        # --- stop-and-reverse ---
        side_to_close = "buy" if flip_side == "sell" else "sell"
        for tr in list(broker.open_trades):
            if (
                getattr(tr, "strategy_id", None) == self.name
                and tr.side == side_to_close
            ):
                broker.close_trade(tr, float(row["close"]), "PTL Advanced Reversal", t)

        print(f"{t} | *** PTL FLIP to {flip_side.upper()} CONFIRMED *** by {self.name}")
        self._open_new_trade(broker, t, row, flip_side)
        self._current_side = flip_side

        self.prev_row = row
