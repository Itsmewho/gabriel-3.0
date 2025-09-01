from __future__ import annotations
from typing import Any, Optional, Sequence
import pandas as pd
import math

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class EmaSignalLineMultiCross(BaseStrategy):
    """K-of-N crosses within a rolling window, tracked with a last-cross ledger.

    Each bar:
      1) Detect per-line cross vs signal with tolerance `eps`.
      2) Record `last_i[line]` and `last_dir[line]`.
      3) For side in {buy,sell}: count lines where `last_dir==side` and
         `bar_i - last_i <= timeframe_bars`. Optionally require that those
         lines are currently all on the same side of the signal.
      4) Fire once per bar, then cooldown.
    """

    def __init__(
        self,
        symbol: str,
        config: dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 3377,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)
        self.signal_ema_period = int(config.get("SIGNAL_EMA", 150))
        self.signal_col = f"ema_{self.signal_ema_period}"
        self.other_lines: Sequence[str] = list(config.get("OTHER_LINES", []))
        self.timeframe_bars = int(config.get("TIMEFRAME_BARS", 60))
        self.required_count = int(
            config.get("REQUIRE_AT_LEAST", max(1, len(self.other_lines)))
        )
        self.eps = float(config.get("EPS", 0.0))
        self.require_current_alignment = bool(
            config.get("REQUIRE_CURRENT_ALIGNMENT", True)
        )

        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 50))
        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 5))
        self.magic = magic

        self.prev_row: Optional[pd.Series] = None
        self.cooldown = 0
        self.bar_i = 0
        # last-cross ledger
        self.last_i: dict[str, int] = {}
        self.last_dir: dict[str, Optional[str]] = {}
        for c in self.other_lines:
            self.last_i[c] = -(10**9)
            self.last_dir[c] = None

    # utils
    def _open_count_for_me(self, broker) -> int:
        return sum(
            getattr(tr, "strategy_id", None) == self.name for tr in broker.open_trades
        )

    def _values_ok(self, row: pd.Series) -> bool:
        need = [self.signal_col, *self.other_lines]
        for c in need:
            if c not in row.index or pd.isna(row[c]):
                return False
        return True

    @staticmethod
    def _cross_dir(
        prev_line: float, line: float, prev_sig: float, sig: float, eps: float
    ) -> Optional[str]:
        if prev_line <= prev_sig - eps and line > sig + eps:
            return "buy"
        if prev_line >= prev_sig + eps and line < sig - eps:
            return "sell"
        return None

    def on_bar(self, broker, t, row: pd.Series):
        self.bar_i += 1
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.prev_row is None:
            self.prev_row = row
            return None
        if not self._values_ok(row) or not self._values_ok(self.prev_row):
            self.prev_row = row
            return None
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None

        sig = float(row[self.signal_col])
        psig = float(self.prev_row[self.signal_col])

        # 1) update ledger
        for c in self.other_lines:
            line = float(row[c])
            pline = float(self.prev_row[c])
            d = self._cross_dir(pline, line, psig, sig, self.eps)
            if d:
                self.last_i[c] = self.bar_i
                self.last_dir[c] = d

        # 2) evaluate cluster(s)
        for side in ("buy", "sell"):
            recent = [
                c
                for c in self.other_lines
                if self.last_dir[c] == side
                and (self.bar_i - self.last_i[c]) <= self.timeframe_bars
            ]
            if len(recent) < self.required_count or self.cooldown > 0:
                continue
            if self.require_current_alignment:
                if side == "buy" and not all(
                    float(row[c]) > sig + self.eps for c in recent
                ):
                    continue
                if side == "sell" and not all(
                    float(row[c]) < sig - self.eps for c in recent
                ):
                    continue

            price = float(row.get("close", row.get("Close")))
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
                    # prevent same cluster re-firing immediately
                    for c in recent:
                        self.last_i[c] = -(10**9)
                    break

        self.prev_row = row
        return None


"""
STRATEGY_NAME = "EMA_BURST"


# Request only what you plot or use
feature_spec = {
"ema": [14, 30, 50, 150, 200, 400, 600, 650],
"sma_high": [50],
"sma_low": [50],
}


cfg_map = {
STRATEGY_NAME: StrategyConfig(
risk_mode=RiskMode.FIXED,
risk_pct=0.10,
lot_min=cfg.VOLUME_MIN,
lot_step=cfg.VOLUME_STEP,
lot_max=100.0,
max_risk_pct_per_trade=0.10,
max_drawdown_pct=0.30,
max_concurrent_trades=5,
),
}


governor = RiskGovernor(cfg_map)


# Ledger-based version does NOT need .bind_parent_df()
strat = EmaSignalLineMultiCross(
symbol=SYMBOL,
config={
"name": STRATEGY_NAME,
"SIGNAL_EMA": 150, # 150 or 200
"OTHER_LINES": [
"ema_14","ema_30","ema_50","ema_400","ema_600","ema_650",
"sma_high_50","sma_low_50",
],
# Window and confirmation
"TIMEFRAME_BARS": 30, # bars since each line's last cross
"REQUIRE_AT_LEAST": 8, # K-of-N; use len(OTHER_LINES) to require ALL
"REQUIRE_CURRENT_ALIGNMENT": True, # all contributing lines must be on the same side NOW
# Cross tolerance
"EPS": 0.00002, # ~0.2 pip on EURUSD; set 0.0 if you want any touch
# Risk and management
"SL_PIPS": 10,
"TP_PIPS": 50,
"COOLDOWN_BARS": 2,
"USE_BREAK_EVEN_STOP": False,
"USE_TRAILING_STOP": False,
"USE_TP_EXTENSION": False,
},
strat_cfg=cfg_map[STRATEGY_NAME],
governor=governor,
)


strategies = [strat]


# Notes:
# - TIMEFRAME_BARS replaces CONFIRM_WINDOW_MIN.
# - EPS avoids micro “touch” crosses; tune per symbol/timeframe.
# - If signals are too rare, lower REQUIRE_AT_LEAST or increase TIMEFRAME_BARS.
# - If you see noise, raise EPS or enforce REQUIRE_CURRENT_ALIGNMENT=True.
"""
