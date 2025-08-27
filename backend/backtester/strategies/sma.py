from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = math.floor(x / step) * step
    lots = max(min_lot, lots)
    lots = min(max_lot, lots)
    return round(lots, 8)


@dataclass
class _State:
    last_fast: Optional[float] = None
    last_slow: Optional[float] = None
    # pending confirmation after a detected cross
    pending: Optional[str] = None  # "bull" | "bear"
    pending_count: int = 0


class SMACross(BaseStrategy):
    """
    Simple SMA crossover strategy (no look-ahead).

    Config:
      name: str = "SMA_X"
      FAST: int = 20
      SLOW: int = 50
      CONFIRM_BARS: int = 0   # require this many bars after cross before entry
      SL_PIPS: float = 15
      TP_PIPS: float = 30
      ALLOW_REVERSE: bool = True  # close and flip on opposite cross
      MAX_HOLD_MIN: int = 0       # hard exit minutes after entry (0 disables)
      FIXED_LOTS: Optional[float] = None
      USE_BREAK_EVEN_STOP, BE_TRIGGER_PIPS, BE_OFFSET_PIPS,
      USE_TRAILING_STOP, TRAILING_STOP_DISTANCE_PIPS,
      USE_TP_EXTENSION, NEAR_TP_BUFFER_PIPS, TP_EXTENSION_PIPS: optional pass-throughs

    Notes:
      - Maintains rolling SMA using O(1) updates.
      - Enters only when warmed up (>= SLOW samples) and confirmation satisfied.
      - One position per strategy. Opposite cross closes or reverses depending on ALLOW_REVERSE.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 4101,
    ) -> None:
        super().__init__(symbol, config or {"name": "SMA_X"}, strat_cfg, governor)
        self.magic = magic

        self.fast_n = int(self.config.get("FAST", 20))
        self.slow_n = int(self.config.get("SLOW", 50))
        if self.fast_n <= 0 or self.slow_n <= 1 or self.fast_n >= self.slow_n:
            raise ValueError("FAST must be >0 and < SLOW; SLOW > 1")

        self.confirm_bars = int(self.config.get("CONFIRM_BARS", 0))
        self.sl_pips = float(self.config.get("SL_PIPS", 15))
        self.tp_pips = float(self.config.get("TP_PIPS", 30))
        self.allow_reverse = bool(self.config.get("ALLOW_REVERSE", True))
        self.max_hold_min = int(self.config.get("MAX_HOLD_MIN", 0))
        self.fixed_lots = self.config.get("FIXED_LOTS", None)

        # rolling windows
        self._q_fast: deque[float] = deque(maxlen=self.fast_n)
        self._q_slow: deque[float] = deque(maxlen=self.slow_n)
        self._sum_fast: float = 0.0
        self._sum_slow: float = 0.0

        self._st = _State()

    # --- sizing ---
    def _size_lots(self, broker, sl_pips: float) -> float:
        if self.fixed_lots is not None:
            return _round_to_step(
                float(self.fixed_lots),
                broker.cfg.VOLUME_STEP,
                broker.cfg.VOLUME_MIN,
                self.strat_cfg.lot_max,
            )
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        sized = self.sizer.size(req)
        return _round_to_step(
            sized.lots,
            broker.cfg.VOLUME_STEP,
            broker.cfg.VOLUME_MIN,
            self.strat_cfg.lot_max,
        )

    # --- helpers ---
    def _warm(self) -> bool:
        return len(self._q_slow) == self.slow_n

    def _sma_fast(self) -> float:
        return self._sum_fast / max(1, len(self._q_fast))

    def _sma_slow(self) -> float:
        return self._sum_slow / max(1, len(self._q_slow))

    def _update_rolling(self, close_px: float) -> None:
        if len(self._q_fast) == self.fast_n:
            self._sum_fast -= self._q_fast[0]
        if len(self._q_slow) == self.slow_n:
            self._sum_slow -= self._q_slow[0]
        self._q_fast.append(close_px)
        self._q_slow.append(close_px)
        self._sum_fast += close_px
        self._sum_slow += close_px

    def _mine_open(self, broker) -> list:
        return [
            tr
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        ]

    def _close_all_mine(self, broker, t, price, reason="signal_exit") -> None:
        for tr in list(self._mine_open(broker)):
            try:
                broker.close_trade(tr, price, reason, t)
            except TypeError:
                broker.close_trade(tr, price, reason, t)

    # --- main ---
    def on_bar(self, broker, t: pd.Timestamp, row):
        close_px = float(row["close"])

        # time-based exit
        if self.max_hold_min > 0:
            for tr in list(self._mine_open(broker)):
                if (t - tr.entry_time) >= pd.Timedelta(minutes=self.max_hold_min):
                    try:
                        broker.close_trade(tr, close_px, "time_exit", t)
                    except TypeError:
                        broker.close_trade(tr, close_px, "time_exit", t)

        # update rolling SMA with CURRENT close, then compute current/previous SMAs
        prev_fast, prev_slow = self._st.last_fast, self._st.last_slow
        self._update_rolling(close_px)
        if not self._warm():
            # store last for next step but do nothing until warm
            self._st.last_fast = self._sma_fast()
            self._st.last_slow = self._sma_slow()
            return None

        cur_fast = self._sma_fast()
        cur_slow = self._sma_slow()

        crossed_up = False
        crossed_dn = False
        if prev_fast is not None and prev_slow is not None:
            crossed_up = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
            crossed_dn = (prev_fast >= prev_slow) and (cur_fast < cur_slow)

        # update last now for next bar comparisons
        self._st.last_fast = cur_fast
        self._st.last_slow = cur_slow

        # confirmation handling
        def _confirm(side: str) -> bool:
            if self.confirm_bars <= 0:
                return True
            if self._st.pending is None:
                self._st.pending = side
                self._st.pending_count = 1
                return False
            if self._st.pending == side:
                self._st.pending_count += 1
                return self._st.pending_count >= self.confirm_bars
            # different side arrived; reset
            self._st.pending = side
            self._st.pending_count = 1
            return False

        mine = self._mine_open(broker)

        # Opposite signal management
        if mine:
            tr = mine[0]
            if tr.side == "buy" and crossed_dn:
                self._close_all_mine(broker, t, close_px, "xdown_exit")
                if not self.allow_reverse:
                    return None
                # fall-through to potential sell entry with confirmation
                if not _confirm("bear"):
                    return None
                side = "sell"
            elif tr.side == "sell" and crossed_up:
                self._close_all_mine(broker, t, close_px, "xup_exit")
                if not self.allow_reverse:
                    return None
                if not _confirm("bull"):
                    return None
                side = "buy"
            else:
                return None
        else:
            # No open position: look for fresh entries
            side = None
            if crossed_up and _confirm("bull"):
                side = "buy"
            elif crossed_dn and _confirm("bear"):
                side = "sell"
            if side is None:
                return None

        # governor
        if self.governor:
            chk = self.governor.allow_new_trade(self.name)
            if not chk.ok:
                return None

        # local concurrency
        if len(self._mine_open(broker)) >= max(1, self.strat_cfg.max_concurrent_trades):
            return None

        # size and open
        lots = self._size_lots(broker, self.sl_pips)
        if lots <= 0:
            return None

        # simple fallbacks
        fb1 = _round_to_step(
            lots * 0.5,
            broker.cfg.VOLUME_STEP,
            broker.cfg.VOLUME_MIN,
            self.strat_cfg.lot_max,
        )
        fb2 = _round_to_step(
            lots * 0.25,
            broker.cfg.VOLUME_STEP,
            broker.cfg.VOLUME_MIN,
            self.strat_cfg.lot_max,
        )
        fallbacks = [x for x in [fb1, fb2] if broker.cfg.VOLUME_MIN <= x < lots]

        tr = broker.open_trade(
            side=side,  # type: ignore[arg-type]
            price=close_px,
            wanted_lots=lots,
            sl_pips=self.sl_pips,
            tp_pips=self.tp_pips,
            t=t,
            fallbacks=fallbacks,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)
            # reset pending after action
            self._st.pending = None
            self._st.pending_count = 0
            return tr
        return None


# --- Example wiring  ---
"""
from backtester.strategies.sma_crossover import SMACross

cfg_map = {
    "SMA_X_KELLY": StrategyConfig(
        risk_mode=RiskMode.HALF_KELLY,
        risk_pct=0.01,
        kelly_p=0.53,
        kelly_rr=1.6,
        kelly_cap_pct=0.02,
        lot_min=cfg.VOLUME_MIN,
        lot_step=cfg.VOLUME_STEP,
        lot_max=100.0,
        max_risk_pct_per_trade=0.02,
        max_concurrent_trades=1,
    ),
    "SMA_X_FIXED": StrategyConfig(
        risk_mode=RiskMode.FIXED,
        risk_pct=0.01,
        lot_min=cfg.VOLUME_MIN,
        lot_step=cfg.VOLUME_STEP,
        lot_max=100.0,
        max_risk_pct_per_trade=0.01,
        max_concurrent_trades=1,
    ),
}

governor = RiskGovernor(cfg_map)

strategies = [
    SMACross(
        symbol=symbol,
        config={
            "name": "SMA_X_KELLY",
            "FAST": 20,
            "SLOW": 50,
            "CONFIRM_BARS": 1,
            "SL_PIPS": 15,
            "TP_PIPS": 30,
            "ALLOW_REVERSE": True,
            "MAX_HOLD_MIN": 0,
            "USE_BREAK_EVEN_STOP": True,
            "BE_TRIGGER_PIPS": 8,
            "BE_OFFSET_PIPS": 1,
        },
        strat_cfg=cfg_map["SMA_X_KELLY"],
        governor=governor,
    ),
    SMACross(
        symbol=symbol,
        config={
            "name": "SMA_X_FIXED",
            "FAST": 20,
            "SLOW": 50,
            "CONFIRM_BARS": 1,
            "SL_PIPS": 15,
            "TP_PIPS": 30,
            "ALLOW_REVERSE": True,
            "MAX_HOLD_MIN": 0,
        },
        strat_cfg=cfg_map["SMA_X_FIXED"],
        governor=governor,
    ),
]
"""
