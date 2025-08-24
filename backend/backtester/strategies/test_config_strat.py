import numpy as np
from typing import Any, Dict
from backtester.strategies.base_strat import BaseStrategy


class RandomEntryStrategyConfig(BaseStrategy):
    """
    Config-driven: uses self.config for SL/TP/lot sizing + risk helpers.
    Fires every N minutes if no open trades.
    """

    def __init__(self, symbol: str, config: Dict[str, Any]):
        super().__init__(symbol, config)
        self.every_n_minutes = int(self.config.get("EVERY_N_MINUTES", 30))

    def _should_fire(self, t) -> bool:
        return (t.minute % self.every_n_minutes == 0) and (
            t.second == 0 if hasattr(t, "second") else True
        )

    def on_bar(self, broker, t, row):
        if broker.open_trades:
            return
        if not self._should_fire(t):
            return

        px = float(row["close"])
        side = "buy" if np.random.rand() > 0.5 else "sell"
        lots = float(self.config.get("LOTS", 0.10))
        sl_pps = float(self.config.get("SL_PIPS", 20))
        tp_pps = float(self.config.get("TP_PIPS", 30))
        fallbacks = self.config.get("FALLBACK_LOTS") or [lots * 0.5, lots * 0.25]

        tr = broker.open_trade(
            side=side,
            price=px,
            wanted_lots=lots,
            sl_pips=sl_pps,
            tp_pips=tp_pps,
            t=t,
            fallbacks=fallbacks,
        )
        if tr:
            self.setup_trade(broker, tr)


class RandomEntryStrategyFixed(BaseStrategy):
    """
    Hardcoded: ignores config (simple smoke test).
    Fires every 45 minutes if no open trades.
    """

    def __init__(self, symbol: str, config: Dict[str, Any] = None):  # type: ignore
        super().__init__(symbol, config or {})
        self.every_n_minutes = 45

    def _should_fire(self, t) -> bool:
        return t.minute % self.every_n_minutes == 0

    def on_bar(self, broker, t, row):
        if broker.open_trades:
            return
        if not self._should_fire(t):
            return

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
        )
        if tr:
            # Example per-trade overrides (hardcoded)
            tr.trailing_sl_distance = 10  # pips
            tr.near_tp_buffer_pips = 2
            tr.tp_extension_pips = 3
            # No break-even in fixed variant
