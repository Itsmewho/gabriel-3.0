# strategies/sma.py

from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
import math  # ADDED: Required for rounding lots
from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


# ADDED: Standalone helper function for rounding trade volume to the broker's specifications.
def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    """Rounds a value down to the nearest step size, respecting min/max lots."""
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class SmaConfluenceStrategy(BaseStrategy):
    """
    Trades based on a confluence of factors:
    1. Strong Trend: Defined by the alignment of 4 SMAs (12, 20, 50, 150).
    2. Pullback Entry: Enters on a pullback to the medium-term SMA (50).
    3. Volume Confirmation: Entry must be confirmed by a volume spike.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 1001,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        self.fast_ma = int(self.config.get("FAST_MA", 12))
        self.short_ma = int(self.config.get("SHORT_MA", 20))
        self.medium_ma = int(self.config.get("MEDIUM_MA", 50))
        self.slow_ma = int(self.config.get("SLOW_MA", 150))

        self.sl_pips = float(self.config.get("SL_PIPS", 20))
        self.tp_pips = float(self.config.get("TP_PIPS", 40))

        self.volume_ma_period = int(self.config.get("VOLUME_MA", 20))
        self.volume_factor = float(self.config.get("VOLUME_FACTOR", 1.5))
        self.prev_row: Optional[pd.Series] = None
        self.magic = magic

    # ADDED: Helper method to check for open trades managed by this specific strategy instance.
    def _open_count_for_me(self, broker) -> int:
        """Counts open trades associated with this strategy's name."""
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    def _get_required_values(self, row: pd.Series) -> Optional[dict]:
        """Safely extracts all required SMA and volume values."""
        try:
            return {
                "fast_sma": float(row[f"sma_{self.fast_ma}"]),
                "short_sma": float(row[f"sma_{self.short_ma}"]),
                "medium_sma": float(row[f"sma_{self.medium_ma}"]),
                "slow_sma": float(row[f"sma_{self.slow_ma}"]),
                "volume": float(row["tick_volume"]),
                "avg_volume": float(row[f"volume_sma_{self.volume_ma_period}"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def on_bar(self, broker, t, row: pd.Series):
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            return None

        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None

        if self.prev_row is None:
            self.prev_row = row
            return None

        current_vals = self._get_required_values(row)
        prev_vals = self._get_required_values(self.prev_row)
        if not current_vals or not prev_vals:
            self.prev_row = row
            return None

        close = float(row["close"])
        prev_low = float(self.prev_row["low"])
        prev_high = float(self.prev_row["high"])
        side = None

        is_bullish_trend = (
            current_vals["fast_sma"] > current_vals["short_sma"]
            and current_vals["short_sma"] > current_vals["medium_sma"]
            and current_vals["medium_sma"] > current_vals["slow_sma"]
        )

        is_pullback_buy = (
            prev_low < prev_vals["medium_sma"] and close > current_vals["medium_sma"]
        )

        if is_bullish_trend and is_pullback_buy:
            side = "buy"

        is_bearish_trend = (
            current_vals["fast_sma"] < current_vals["short_sma"]
            and current_vals["short_sma"] < current_vals["medium_sma"]
            and current_vals["medium_sma"] < current_vals["slow_sma"]
        )

        is_pullback_sell = (
            prev_high > prev_vals["medium_sma"] and close < current_vals["medium_sma"]
        )

        if is_bearish_trend and is_pullback_sell:
            side = "sell"

        if side:
            has_volume_confirmation = current_vals["volume"] > (
                current_vals["avg_volume"] * self.volume_factor
            )
            if not has_volume_confirmation:
                self.prev_row = row
                return None

            req = SizeRequest(
                balance=broker.balance,
                sl_pips=self.sl_pips,
                value_per_pip=self._value_per_pip_1lot(broker),
            )
            sized = self.sizer.size(req)

            # CORRECTED: Call the standalone helper function, not a class method.
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
                    self.setup_trade(broker, tr)
                self.prev_row = row
                return tr

        self.prev_row = row
        return None
