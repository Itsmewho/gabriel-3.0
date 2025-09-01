# backtester/strategies/test_sma.py
from __future__ import annotations
from typing import Any, Optional, Sequence
import math
import pandas as pd
from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class SmaHLFastPackSignalCross(BaseStrategy):
    """
    Trade when ALL fast lines cross EMA450 in the SAME direction within a bar window.
      fast_pack default: [sma_high_30, sma_low_30, ema_50, ema_14]
    BUY  = all cross UP above EMA450 within window
    SELL = all cross DOWN below EMA450 within window
    """

    def __init__(
        self,
        symbol: str,
        config: dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 7703,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        n_hi = int(config.get("SMA_HIGH_N", 30))
        n_lo = int(config.get("SMA_LOW_N", 30))
        self.fast_pack: Sequence[str] = config.get(
            "FAST_PACK",
            [f"sma_high_{n_hi}", f"sma_low_{n_lo}", "ema_50", "ema_14"],
        )
        self.ema_sig = f"ema_{int(config.get('EMA_SIGNAL_N', 450))}"

        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 50))
        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 0))
        self.window_bars = int(config.get("CONFIRM_WINDOW_BARS", 30))
        self.eps = float(config.get("EPS", 0.0) or 0.0)

        self.prev_row: Optional[pd.Series] = None
        self.cooldown = 0
        # pending: {"side": "buy"/"sell", "crossed": set(str), "t0": Timestamp}
        self.pending: Optional[dict] = None

        self.parent_df: Optional[pd.DataFrame] = None
        self.magic = magic

    def bind_parent_df(self, df: pd.DataFrame):
        self.parent_df = df
        return self

    def _open_count_for_me(self, broker) -> int:
        return sum(
            getattr(tr, "strategy_id", None) == self.name for tr in broker.open_trades
        )

    def _xup(self, pa, a, pb, b) -> bool:
        e = self.eps
        return (pa <= pb + e) and (a > b + e)

    def _xdn(self, pa, a, pb, b) -> bool:
        e = self.eps
        return (pa >= pb - e) and (a < b - e)

    def _fire(self, broker, t, row, side: str):
        if self.sl_pips <= 0 or self.tp_pips <= 0:
            return None
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
        if lots <= 0:
            return None
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

    def on_bar(self, broker, t, row: pd.Series):
        # required columns
        need = set(self.fast_pack) | {self.ema_sig, "close"}
        if any(c not in row.index for c in need):
            self.prev_row = row
            return None

        if self.prev_row is None:
            self.prev_row = row
            return None

        if (
            self.cooldown > 0
            or self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades
        ):
            self.cooldown = max(0, self.cooldown - 1)
            self.prev_row = row
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None

        # expire window
        if self.pending is not None and (row.name - self.pending["t0"]) > pd.Timedelta(
            minutes=self.window_bars
        ):
            self.pending = None

        prev_sig, sig = self.prev_row[self.ema_sig], row[self.ema_sig]

        # detect crosses this bar for all fast lines
        crosses: dict[str, str] = {}
        for ln in self.fast_pack:
            pa, a = self.prev_row[ln], row[ln]
            if self._xup(pa, a, prev_sig, sig):
                crosses[ln] = "buy"
                print(f"{t} | {ln} CROSS UP over {self.ema_sig}")
            elif self._xdn(pa, a, prev_sig, sig):
                crosses[ln] = "sell"
                print(f"{t} | {ln} CROSS DOWN under {self.ema_sig}")

        # update pending window
        if crosses:
            # determine majority direction on this bar among crossed lines
            dirs = set(crosses.values())
            # if both directions happened on same bar, split preference: reset and start fresh per first seen
            side_this_bar = next(iter(dirs))
            if self.pending is None or self.pending["side"] != side_this_bar:
                self.pending = {"side": side_this_bar, "crossed": set(), "t0": row.name}
            # add only lines that match the pending side
            for ln, sd in crosses.items():
                if sd == self.pending["side"]:
                    self.pending["crossed"].add(ln)

        # confirm when all fast lines crossed same way within window
        if self.pending and self.pending["crossed"] >= set(self.fast_pack):
            side = self.pending["side"]
            tr = self._fire(broker, t, row, side)
            self.pending = None
            self.prev_row = row
            return tr

        self.prev_row = row
        return None
