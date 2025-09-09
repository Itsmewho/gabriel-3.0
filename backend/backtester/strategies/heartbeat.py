from __future__ import annotations
from typing import Any, Optional, Set, Dict
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import Trade


class HeartbeatCrossover(BaseStrategy):
    """
    An asymmetrical crossover strategy with a dynamic, trend-following exit.

    -   **Entry:** Standard 'fast_pack' alignment crossover.
    -   **Exit:** Uses a standard exit by default. However, if the trade is profitable
        AND the short-term trend (defined by 'TREND_CHECK_MA') is confirming,
        it switches to a much looser 'loosened_exit_line', allowing profits to run.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 8801,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # --- Indicator Configuration ---
        self.fast_pack: Set[str] = set(config.get("FAST_PACK"))
        self.signal_line: str = config.get("SIGNAL_LINE")
        self.exit_trigger_line: str = config.get("EXIT_TRIGGER_LINE")
        self.exit_against_line: str = config.get("EXIT_AGAINST_LINE")

        # --- NEW: Trend-Following Exit Configuration ---
        self.use_trend_exit = config.get("USE_TREND_EXIT", False)
        if self.use_trend_exit:
            self.trend_check_ma = config.get("TREND_CHECK_MA")
            self.loosened_exit_line = config.get("LOOSENED_EXIT_AGAINST_LINE")

        # --- Risk and Standard Parameters ---
        self.sl_pips = float(config.get("SL_PIPS", 0))
        self.tp_pips = float(config.get("TP_PIPS", 0))
        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 1))
        self.eps = float(config.get("EPS", 0.0) or 0.0)
        self.prev_row: Optional[pd.Series] = None
        self.cooldown = 0
        self.magic = magic

    # --- Utilities ---
    def _get_my_trade(self, broker) -> Optional[Trade]:
        for trade in broker.open_trades:
            if getattr(trade, "strategy_id", None) == self.name:
                return trade
        return None

    def _is_in_profit(self, trade: Trade, current_price: float) -> bool:
        """Checks if a trade is currently profitable, excluding costs."""
        if trade.side == "buy":
            return current_price > trade.entry_price
        else:
            return current_price < trade.entry_price

    def _is_trend_confirming(self, row, side):
        if (
            not self.trend_check_ma
            or pd.isna(row.get(self.trend_check_ma, None))
            or self.prev_row is None
            or pd.isna(self.prev_row.get(self.trend_check_ma, None))
        ):
            return False
        cur = float(row[self.trend_check_ma])
        prev = float(self.prev_row[self.trend_check_ma])
        return (cur > prev) if side == "buy" else (cur < prev)

    def _are_all_on_side(self, row: pd.Series, side: str) -> bool:
        signal_value = row[self.signal_line]
        if pd.isna(signal_value):
            return False
        vals = [row[l] for l in self.fast_pack]  # noqa: E741
        if any(pd.isna(v) for v in vals):
            return False
        return (
            all(v > signal_value + self.eps for v in vals)
            if side == "buy"
            else all(v < signal_value - self.eps for v in vals)
        )

    def _check_cross(self, fast_val, prev_fast, sig_val, prev_sig) -> Optional[str]:
        # CROSS UP: was below (prev_fast <= prev_sig) → now above (fast_val > sig_val)
        if prev_fast <= prev_sig - self.eps and fast_val > sig_val + self.eps:
            return "buy"
        # CROSS DOWN: was above → now below
        if prev_fast >= prev_sig + self.eps and fast_val < sig_val - self.eps:
            return "sell"
        return None

    def _fire_trade(self, broker, t, row: pd.Series, side: str):
        price = float(row["close"])
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=self.sl_pips if self.sl_pips > 0 else 1000,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        lots = self.sizer.size(req).lots
        if lots <= 0:
            return None
        tr = broker.open_trade(
            side=side,
            price=price,
            wanted_lots=lots,
            sl_pips=self.sl_pips if self.sl_pips > 0 else None,
            tp_pips=self.tp_pips if self.tp_pips > 0 else None,
            t=t,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)
            self.cooldown = self.cooldown_bars
            return tr
        return None

    # --- Main Hook ---
    def on_bar(self, broker, t, row: pd.Series):
        if self.prev_row is None:
            self.prev_row = row
            return None

        if self.cooldown > 0:
            self.cooldown -= 1

        open_trade = self._get_my_trade(broker)

        # --- State 1: In a trade, manage exit ---
        if open_trade:
            active_exit_against_line = self.exit_against_line

            # --- DYNAMIC EXIT LOGIC ---
            if self.use_trend_exit:
                is_profitable = self._is_in_profit(open_trade, row["close"])
                is_trending = self._is_trend_confirming(row, open_trade.side)

                if is_profitable and is_trending:
                    active_exit_against_line = self.loosened_exit_line
                    if not getattr(open_trade, "is_exit_loosened", False):
                        setattr(open_trade, "is_exit_loosened", True)
                        print(
                            f"{t} | Trade {open_trade.id} is profitable and trending. Exit loosened."
                        )
                else:
                    if getattr(open_trade, "is_exit_loosened", False):
                        setattr(open_trade, "is_exit_loosened", False)
                        print(
                            f"{t} | Trade {open_trade.id} condition ended. Exit tightened."
                        )

            # --- Check for the exit signal using the determined line ---
            exit_side = "sell" if open_trade.side == "buy" else "buy"
            exit_cross_event = self._check_cross(
                row[self.exit_trigger_line],
                self.prev_row[self.exit_trigger_line],
                row[active_exit_against_line],
                self.prev_row[active_exit_against_line],
            )

            if exit_cross_event == exit_side:
                reason = f"Invalidation: {self.exit_trigger_line} recrossed {active_exit_against_line}"
                print(f"{t} | EXITING trade {open_trade.id} due to: {reason}")
                broker.close_trade(open_trade, row["close"], reason, t)
                self.cooldown = self.cooldown_bars

        # --- State 2: Flat, looking for an entry ---
        else:
            if self.cooldown > 0 or (
                self.governor and not self.governor.allow_new_trade(self.name).ok
            ):
                self.prev_row = row
                return None

            is_buy_signal = self._are_all_on_side(row, "buy")
            was_buy_signal = self._are_all_on_side(self.prev_row, "buy")
            is_sell_signal = self._are_all_on_side(row, "sell")
            was_sell_signal = self._are_all_on_side(self.prev_row, "sell")

            if is_buy_signal and not was_buy_signal:
                self._fire_trade(broker, t, row, "buy")
            elif is_sell_signal and not was_sell_signal:
                self._fire_trade(broker, t, row, "sell")

        self.prev_row = row
        return None
