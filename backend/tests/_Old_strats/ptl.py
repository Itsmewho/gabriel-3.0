from __future__ import annotations
from typing import Any, Dict, Optional, Deque
import math
import pandas as pd
from collections import deque

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


# --- Helper Functions ---
def _slope_norm(cur: float, prev: float, denom: float) -> float:
    eps = 1e-9
    d = (cur - prev) / max(abs(denom), eps)
    return 0.5 + math.atan(d) / math.pi


def _full_body_above(row: pd.Series, level: float) -> bool:
    return float(row["open"]) > level and float(row["close"]) > level


def _full_body_below(row: pd.Series, level: float) -> bool:
    return float(row["open"]) < level and float(row["close"]) < level


class PtlKeltnerStrategy(BaseStrategy):
    """
    Enters on PTL trend flips with several layers of confirmation:
    1.  KC Middle Cross: If a flip occurs on the "wrong" side of the KC middle,
        it waits for a full candle to cross to the correct side before entering.
    2.  Momentum/Exhaustion Filter: When reversing a position, it checks for
        persisting EMA momentum AND recent price exhaustion outside the KC bands.
        If these conditions are met, the reversal is executed and the *next*
        signal in the opposite direction is ignored to prevent whipsaws.
    3.  Standard Reversal: If the momentum/exhaustion conditions are not met,
        it performs a standard stop-and-reverse.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 2025,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # --- Configurable Parameters ---
        self.ptl_trena_col = self.config.get("PTL_TRENA_COL", "ptl_trena")
        self.kc_upper_col = self.config.get("KC_UPPER_COL", "kc_18_18_2.0_upper")
        self.kc_lower_col = self.config.get("KC_LOWER_COL", "kc_18_18_2.0_lower")
        self.kc_middle_col = self.config.get("KC_MIDDLE_COL", "kc_18_18_2.0_middle")
        self.ema60_col = self.config.get("EMA_60_COL", "ema_60")
        self.ema90_col = self.config.get("EMA_90_COL", "ema_90")
        self.atr_col = self.config.get("ATR_COL", "atr_18")

        self.sl_pips = float(self.config.get("SL_PIPS", 18))
        self.tp_pips = float(self.config.get("TP_PIPS", 27))

        self.ema_slope_win_min = int(self.config.get("EMA_SLOPE_WINDOW_MIN", 5))
        self.exhaustion_win_min = int(self.config.get("EXHAUSTION_WINDOW_MIN", 10))
        self.exhaustion_candle_count = int(
            self.config.get("EXHAUSTION_CANDLE_COUNT", 2)
        )

        # --- Internal State ---
        self.magic = magic
        self.prev_row: Optional[pd.Series] = None
        self._recent_bars: Deque[pd.Series] = deque(maxlen=self.exhaustion_win_min + 5)
        self._await_confirm_side: Optional[str] = None
        self._ignore_next_buy: bool = False
        self._ignore_next_sell: bool = False

    def _get_recent_bars(self, minutes: int) -> list[pd.Series]:
        """Gets all bars from the buffer within the last N minutes."""
        if not self._recent_bars:
            return []

        cutoff = self._recent_bars[-1].name - pd.Timedelta(minutes=minutes)
        # Iterate from right to left for efficiency
        for i in range(len(self._recent_bars) - 1, -1, -1):
            if self._recent_bars[i].name < cutoff:
                return list(self._recent_bars)[i + 1 :]
        return list(self._recent_bars)

    def _calculate_ema_slope(self) -> float:
        """Calculates the average normalized slope of EMAs over the defined window."""
        slope_bars = self._get_recent_bars(self.ema_slope_win_min)
        if len(slope_bars) < 2:
            return 0.5  # Neutral slope

        first_bar, last_bar = slope_bars[0], slope_bars[-1]
        denom = float(last_bar.get(self.atr_col, 1.0) or 1e-9)

        s60 = _slope_norm(last_bar[self.ema60_col], first_bar[self.ema60_col], denom)
        s90 = _slope_norm(last_bar[self.ema90_col], first_bar[self.ema90_col], denom)

        return (s60 + s90) / 2.0

    def _open_new_trade(self, broker, t, row: pd.Series, side: str):
        """Centralized logic for opening a trade."""
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            return

        open_trades = [
            tr
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        ]
        if len(open_trades) >= self.strat_cfg.max_concurrent_trades:
            return

        req = SizeRequest(
            balance=broker.balance,
            sl_pips=self.sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        lots = self.sizer.size(req).lots
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

    def on_bar(self, broker, t, row: pd.Series):
        self._recent_bars.append(row)
        if self.prev_row is None:
            self.prev_row = row
            return

        # --- 1. PTL Flip Detection ---
        prev_trena = self.prev_row[self.ptl_trena_col]
        current_trena = row[self.ptl_trena_col]

        flip_side: Optional[str] = None
        if prev_trena == 1 and current_trena == 0:
            flip_side = "buy"
        elif prev_trena == 0 and current_trena == 1:
            flip_side = "sell"

        # --- 2. Handle KC Middle Cross Confirmation (if waiting) ---
        if self._await_confirm_side:
            side_needed = self._await_confirm_side
            kc_mid = row[self.kc_middle_col]

            confirmed = False
            if side_needed == "buy" and _full_body_above(row, kc_mid):
                confirmed = True
            elif side_needed == "sell" and _full_body_below(row, kc_mid):
                confirmed = True

            if confirmed:
                self._open_new_trade(broker, t, row, side_needed)
                self._await_confirm_side = None

            # If a new, opposite flip happens while waiting, cancel the wait.
            if flip_side and flip_side != self._await_confirm_side:
                self._await_confirm_side = None
            else:  # Otherwise, continue waiting
                self.prev_row = row
                return

        # --- 3. Process a New Flip Signal ---
        if not flip_side:
            self.prev_row = row
            return

        # If we need to ignore this signal, do so and reset the flag.
        if (flip_side == "buy" and self._ignore_next_buy) or (
            flip_side == "sell" and self._ignore_next_sell
        ):
            self._ignore_next_buy = self._ignore_next_sell = False
            self.prev_row = row
            return

        # --- 4. Main Entry Logic on Flip ---
        # Close all opposite trades first.
        side_to_close = "buy" if flip_side == "sell" else "sell"
        for tr in list(broker.open_trades):
            if (
                getattr(tr, "strategy_id", None) == self.name
                and tr.side == side_to_close
            ):
                broker.close_trade(tr, float(row["close"]), "PTL Reversal", t)

        # --- 4a. Check for Momentum/Exhaustion Condition ---
        exhaustion_bars = self._get_recent_bars(self.exhaustion_win_min)
        ema_slope = self._calculate_ema_slope()

        if flip_side == "buy":  # Buy signal occurred
            bodies_below_lower = sum(
                1
                for bar in exhaustion_bars
                if _full_body_below(bar, bar[self.kc_lower_col])
            )

            # If EMA slope is still down AND we had recent exhaustion, open the buy and ignore the next sell
            if ema_slope < 0.5 and bodies_below_lower >= self.exhaustion_candle_count:
                self._open_new_trade(broker, t, row, "buy")
                self._ignore_next_sell = True  # Ignore the next sell signal
            else:  # --- 4b. Standard Entry ---
                self._open_new_trade(broker, t, row, "buy")

        elif flip_side == "sell":  # Sell signal occurred
            bodies_above_upper = sum(
                1
                for bar in exhaustion_bars
                if _full_body_above(bar, bar[self.kc_upper_col])
            )

            # If EMA slope is still up AND we had recent exhaustion, open the sell and ignore the next buy
            if ema_slope > 0.5 and bodies_above_upper >= self.exhaustion_candle_count:
                self._open_new_trade(broker, t, row, "sell")
                self._ignore_next_buy = True  # Ignore the next buy signal
            else:  # --- 4b. Standard Entry ---
                self._open_new_trade(broker, t, row, "sell")

        self.prev_row = row

    # governor = RiskGovernor(cfg_map)
    # strategies = [
    #     PtlKeltnerStrategy(
    #         symbol=symbol,
    #         config={
    #             "name": STRATEGY_NAME,
    #             # --- Risk ---
    #             "SL_PIPS": 18,
    #             "TP_PIPS": 27,
    #             # --- Confirmation & Filters ---
    #             "EMA_SLOPE_WINDOW_MIN": 5,  # Check slope over last 5 mins
    #             "EXHAUSTION_WINDOW_MIN": 10,  # Check for exhaustion candles in last 10 mins
    #             "EXHAUSTION_CANDLE_COUNT": 2,  # Require 2 candles outside band for exhaustion signal
    #             # --- Column Names (ensure these match your feature_spec) ---
    #             "PTL_TRENA_COL": "ptl_trena",
    #             "KC_UPPER_COL": "kc_18_18_2.0_upper",
    #             "KC_LOWER_COL": "kc_18_18_2.0_lower",
    #             "KC_MIDDLE_COL": "kc_18_18_2.0_middle",
    #             "EMA_60_COL": "ema_60",
    #             "EMA_90_COL": "ema_90",
    #             "ATR_COL": "atr_18",
    #         },
    #         strat_cfg=cfg_map["PTL_KC_Conditional"],
    #         governor=governor,
    #     )
    # ]
