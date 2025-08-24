from typing import Any, Dict
from backtester.broker.main_broker import Broker, Trade


class BaseStrategy:
    def __init__(self, symbol: str, config: Dict[str, Any]):
        self.symbol = symbol
        self.config = config or {}

    def setup_trade(self, broker: Broker, tr: Trade):
        # Break-even
        if self.config.get("USE_BREAK_EVEN_STOP", False):
            broker.set_break_even(
                tr.id,
                be_pips=self.config.get("BE_TRIGGER_PIPS", 0),
                offset_pips=self.config.get("BE_OFFSET_PIPS", 0),
            )
        # Trailing SL (per-trade override)
        if self.config.get("USE_TRAILING_STOP", False):
            tr.trailing_sl_distance = self.config.get(
                "TRAILING_STOP_DISTANCE_PIPS", None
            )
        # TP extension (per-trade override)
        if self.config.get("USE_TP_EXTENSION", False):
            tr.near_tp_buffer_pips = self.config.get("NEAR_TP_BUFFER_PIPS", None)
            tr.tp_extension_pips = self.config.get("TP_EXTENSION_PIPS", None)

    def on_bar(self, broker: Broker, t, row):
        raise NotImplementedError
