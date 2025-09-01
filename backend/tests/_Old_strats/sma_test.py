# strategies/sma_crossover_simple.py
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


class SmaCrossoverSimple(BaseStrategy):
    """
    SMA fast/slow crossover with *delayed* management:
    - Break-even only after +1R (>= SL_PIPS in favor)
    - Start trailing only after +1R
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 2001,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)
        self.fast = int(config.get("FAST_MA", 12))
        self.slow = int(config.get("SLOW_MA", 24))
        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 50))
        self.be_trigger_extra = float(
            config.get("BE_TRIGGER_EXTRA_PIPS", 1)
        )  # added: BE beyond 1R by N pips
        self.be_offset = float(config.get("BE_OFFSET_PIPS", 2))
        self.trail_distance_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 10))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", True))
        self.magic = magic
        self.prev_row: Optional[pd.Series] = None

    # ------------------ helpers ------------------
    def _open_count_for_me(self, broker) -> int:
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    @staticmethod
    def _favorable_move_pips(tr: Trade, row: pd.Series) -> float:
        """Use intrabar extremes for true trigger behavior."""
        high = float(row.get("high", row.get("High", float("nan"))))
        low = float(row.get("low", row.get("Low", float("nan"))))
        if tr.side == "buy":
            return (high - tr.entry_price) / PIP_SIZE
        else:
            return (tr.entry_price - low) / PIP_SIZE

    def _manage_open_positions(self, broker, row: pd.Series) -> None:
        """Apply BE/trailing only after +1R progression."""
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) != self.name:
                continue

            fav = self._favorable_move_pips(tr, row)

            # Break-even after >= 1R (+ optional extra)
            be_threshold = self.sl_pips + self.be_trigger_extra
            if not getattr(tr, "be_applied", False) and fav >= be_threshold:
                broker.set_break_even(
                    trade_id=tr.id,
                    be_pips=self.sl_pips + self.be_trigger_extra,
                    offset_pips=self.be_offset,
                )
                # mark locally as applied to avoid repeated calls (broker also tracks)
                tr.be_applied = True

            # Start trailing only after >= 1R
            if (
                self.use_trailing
                and getattr(tr, "trailing_sl_distance", None) is None
                and fav >= self.sl_pips
            ):
                tr.trailing_sl_distance = (
                    self.trail_distance_pips
                )  # broker.on_bar() reads this to trail via highs/lows

    # ------------------ main hook ------------------
    def on_bar(self, broker, t, row: pd.Series):
        # Risk gate
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            # Still manage existing positions
            self._manage_open_positions(broker, row)
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            self._manage_open_positions(broker, row)
            return None

        # Manage existing positions before looking for new ones
        self._manage_open_positions(broker, row)

        # Need prior bar for crossover
        if self.prev_row is None:
            self.prev_row = row
            return None

        # Read features
        try:
            prev_fast = float(self.prev_row[f"sma_{self.fast}"])
            prev_slow = float(self.prev_row[f"sma_{self.slow}"])
            curr_fast = float(row[f"sma_{self.fast}"])
            curr_slow = float(row[f"sma_{self.slow}"])
            close = float(row["close"])
        except Exception:
            self.prev_row = row
            return None

        # CROSSOVER: invert mapping if desired
        side: Optional[str] = None
        if prev_fast < prev_slow and curr_fast > curr_slow:
            side = "buy"
        elif prev_fast > prev_slow and curr_fast < curr_slow:
            side = "sell"

        if side:
            # Size
            req = SizeRequest(
                balance=broker.balance,
                sl_pips=self.sl_pips,
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
                    sl_pips=self.sl_pips,
                    tp_pips=self.tp_pips,
                    t=t,
                    strategy_id=self.name,
                    magic=self.magic,
                )
                if tr:
                    # Do NOT pre-arm trailing or BE here; we gate them in _manage_open_positions()
                    pass
                self.prev_row = row
                return tr

        self.prev_row = row
        return None


# cfg_map = {
#     "SMA_CROSS": StrategyConfig(
#         risk_mode=RiskMode.FIXED,
#         risk_pct=0.1,
#         lot_min=cfg.VOLUME_MIN,
#         lot_step=cfg.VOLUME_STEP,
#         lot_max=100.0,
#         max_risk_pct_per_trade=0.1,
#         max_drawdown_pct=0.3,
#         max_concurrent_trades=5,
#     ),
# }
# governor = RiskGovernor(cfg_map)

# strategies = [
#     SmaCrossoverSimple(
#         symbol=SYMBOL,
#         config={
#             "name": "SMA_CROSS",
#             "FAST_MA": 12,
#             "SLOW_MA": 24,
#             "SL_PIPS": 10,
#             "TP_PIPS": 50,
#             # optional: BaseStrategy honors these if present
#             "USE_BREAK_EVEN_STOP": True,
#             "BE_TRIGGER_PIPS": 8,
#             "BE_OFFSET_PIPS": 2,
#             "USE_TP_EXTENSION": False,
#         },
#         strat_cfg=cfg_map["SMA_CROSS"],
#         governor=governor,
#     )
# ]
