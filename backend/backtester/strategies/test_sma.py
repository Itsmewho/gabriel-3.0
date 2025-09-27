from __future__ import annotations
from typing import Any, Optional, Sequence, Dict, Set  # noqa: F401
import math
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class MultiStageConfirmationCross(BaseStrategy):
    """
    A stateful strategy that waits for a multi-stage sequence of crossovers.
    This version supports up to three confirmation stages and includes a state
    check to correctly identify conditions that are already met.
    """

    def __init__(
        self,
        symbol: str,
        config: dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 7705,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        self.fast_pack: Set[str] = set(
            config.get("FAST_PACK", ["sma_high_30", "sma_low_30", "ema_50", "ema_14"])
        )

        # --- Stage Definitions (fully configurable) ---
        # --- Stage Definitions  ---
        self.stage1_signal: Dict[str, str] = config.get("STAGE1_SIGNAL")
        self.stage2_signal: Dict[str, str] = config.get("STAGE2_SIGNAL")
        # --- NEW: Optional Third Stage ---
        self.stage3_signal: Dict[str, str] = config.get("STAGE3_SIGNAL")

        self.window_bars = int(config.get("CONFIRM_WINDOW_BARS", 30))
        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 50))
        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 5))
        self.eps = float(config.get("EPS", 0.0) or 0.0)

        self.prev_row: Optional[pd.Series] = None
        self.cooldown = 0
        self.magic = magic
        self.pending: Optional[Dict[str, Any]] = None

    # --- Utilities ---
    def _open_count_for_me(self, broker) -> int:
        return sum(
            getattr(tr, "strategy_id", None) == self.name for tr in broker.open_trades
        )

    def _check_cross(self, fast_val, prev_fast, sig_val, prev_sig) -> Optional[str]:
        if prev_fast <= prev_sig - self.eps and fast_val > sig_val + self.eps:
            return "buy"
        if prev_fast >= prev_sig + self.eps and fast_val < sig_val - self.eps:
            return "sell"
        return None

    def _is_on_correct_side(self, fast_val: float, sig_val: float, side: str) -> bool:
        if side == "buy":
            return fast_val > sig_val + self.eps
        else:
            return fast_val < sig_val - self.eps

    def _fire_trade(self, broker, t, row: pd.Series, side: str):
        price = float(row["close"])
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
                price=price,
                wanted_lots=lots,
                sl_pips=self.sl_pips,
                tp_pips=self.tp_pips,
                t=t,
                strategy_id=self.name,
                magic=self.magic,
            )
            if tr:
                self.setup_trade(broker, tr)
                self.cooldown = self.cooldown_bars
                print(f"{t} | *** {side.upper()} FIRED *** by {self.name}")
                return tr
        return None

    # --- Main Hook ---
    def on_bar(self, broker, t, row: pd.Series):
        if self.prev_row is None:
            self.prev_row = row
            return None

        if self.cooldown > 0:
            self.cooldown -= 1

        if (
            self.cooldown > 0
            or self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades
            or (self.governor and not self.governor.allow_new_trade(self.name).ok)
        ):
            self.prev_row = row
            return None

        if self.pending and (row.name - self.pending["t0"]) > pd.Timedelta(
            minutes=self.window_bars
        ):
            print(f"{t} | Pending {self.pending['side']} signal timed out.")
            self.pending = None

        if self.pending is None:
            for side in ("buy", "sell"):
                sig_line = self.stage1_signal[side]
                crosses = {
                    ln
                    for ln in self.fast_pack
                    if self._check_cross(
                        row[ln],
                        self.prev_row[ln],
                        row[sig_line],
                        self.prev_row[sig_line],
                    )
                    == side
                }
                if crosses:
                    print(
                        f"{t} | New {side} signal started by {crosses} crossing {sig_line}. Window started."
                    )
                    self.pending = {
                        "side": side,
                        "stage": 1,
                        "t0": row.name,
                        "stage1_crossed": crosses,
                        "stage2_crossed": set(),
                        "stage3_crossed": set(),
                    }
                    break

        if self.pending:
            side = self.pending["side"]

            # --- STAGE 1 LOGIC ---
            if self.pending["stage"] == 1:
                sig_line = self.stage1_signal[side]
                for fast_line in self.fast_pack - self.pending["stage1_crossed"]:
                    if (
                        self._check_cross(
                            row[fast_line],
                            self.prev_row[fast_line],
                            row[sig_line],
                            self.prev_row[sig_line],
                        )
                        == side
                    ):
                        self.pending["stage1_crossed"].add(fast_line)
                if self.pending["stage1_crossed"] == self.fast_pack:
                    print(
                        f"{t} | >>> Stage 1 COMPLETE for {side}. Awaiting Stage 2. <<<"
                    )
                    self.pending["stage"] = 2

            # --- STAGE 2 LOGIC ---
            if self.pending["stage"] == 2:
                sig_line = self.stage2_signal[side]
                for fast_line in self.fast_pack - self.pending["stage2_crossed"]:
                    if self._check_cross(
                        row[fast_line],
                        self.prev_row[fast_line],
                        row[sig_line],
                        self.prev_row[sig_line],
                    ) == side or self._is_on_correct_side(
                        row[fast_line], row[sig_line], side
                    ):
                        self.pending["stage2_crossed"].add(fast_line)
                if self.pending["stage2_crossed"] == self.fast_pack:
                    print(
                        f"{t} | >>> Stage 2 COMPLETE for {side}. Awaiting Stage 3. <<<"
                    )
                    self.pending["stage"] = 3

            # --- STAGE 3 LOGIC ---
            if self.pending["stage"] == 3:
                sig_line = self.stage3_signal[side]
                for fast_line in self.fast_pack - self.pending["stage3_crossed"]:
                    if self._check_cross(
                        row[fast_line],
                        self.prev_row[fast_line],
                        row[sig_line],
                        self.prev_row[sig_line],
                    ) == side or self._is_on_correct_side(
                        row[fast_line], row[sig_line], side
                    ):
                        self.pending["stage3_crossed"].add(fast_line)
                if self.pending["stage3_crossed"] == self.fast_pack:
                    tr = self._fire_trade(broker, t, row, side)
                    self.pending = None
                    self.prev_row = row
                    return tr

        self.prev_row = row
        return None
