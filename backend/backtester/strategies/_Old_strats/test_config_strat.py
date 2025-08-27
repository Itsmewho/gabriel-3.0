# Test config strat

import numpy as np
from typing import Any, Dict, Optional
from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


def _round_to_step(x, step, min_lot, max_lot):
    import math

    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class RandomEntryStrategyConfig(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        magic: Optional[int] = 1001,
        governor=None,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)
        self.every_n_minutes = int(self.config.get("EVERY_N_MINUTES", 30))
        self.magic = magic

    def _should_fire(self, t) -> bool:
        return (t.minute % self.every_n_minutes == 0) and (getattr(t, "second", 0) == 0)

    def _open_count_for_me(self, broker) -> int:
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    def on_bar(self, broker, t, row):
        if not self._should_fire(t):
            return None

        # Governor: drawdown pause check
        if self.governor:
            chk = self.governor.allow_new_trade(self.name)
            if not chk.ok:
                return None

        # Concurrency limit (local check against broker state)
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            return None

        sl_pps = float(self.config.get("SL_PIPS", 20))
        tp_pps = float(self.config.get("TP_PIPS", 30))
        if sl_pps <= 0:
            return None

        req = SizeRequest(
            balance=broker.balance,
            sl_pips=sl_pps,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        sized = self.sizer.size(req)
        lots = _round_to_step(
            sized.lots, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        if lots <= 0:
            return None

        px = float(row["close"])
        side = "buy" if np.random.rand() > 0.5 else "sell"
        fb1 = _round_to_step(
            lots * 0.5, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        fb2 = _round_to_step(
            lots * 0.25, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        fallbacks = [x for x in [fb1, fb2] if x < lots and x >= broker.cfg.VOLUME_MIN]

        tr = broker.open_trade(
            side=side,
            price=px,
            wanted_lots=lots,
            sl_pips=sl_pps,
            tp_pips=tp_pps,
            t=t,
            fallbacks=fallbacks,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)
        return tr


class RandomEntryStrategyFixed(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any] | None = None,
        magic: Optional[int] = 2002,
        strat_cfg=None,
        governor=None,
    ):
        super().__init__(symbol, config or {}, governor=governor)
        self.every_n_minutes = 45
        self.magic = magic

    def _should_fire(self, t) -> bool:
        return t.minute % self.every_n_minutes == 0

    def _open_count_for_me(self, broker) -> int:
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    def on_bar(self, broker, t, row):
        if not self._should_fire(t):
            return None
        if self.governor:
            chk = self.governor.allow_new_trade(self.name)
            if not chk.ok:
                return None
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            return None

        px = float(row["close"])
        side = "buy" if np.random.rand() > 0.5 else "sell"

        tr = broker.open_trade(
            side=side,
            price=px,
            wanted_lots=0.20,
            sl_pips=15,
            tp_pips=25,
            t=t,
            fallbacks=[0.10, 0.05],
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            tr.trailing_sl_distance = 10
            tr.near_tp_buffer_pips = 2
            tr.tp_extension_pips = 3
        return tr
