# Base strat

from typing import Any, Dict, Optional
from backtester.broker.main_broker import Broker, Trade
from backtester.account_management.sizer import AccountSizer
from backtester.account_management.types import StrategyConfig
from backtester.broker import PIP_SIZE  # EURUSD pip size


class BaseStrategy:
    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg: Optional[StrategyConfig] = None,
        governor=None,  # <-- inject the shared RiskGovernor
    ):
        self.symbol = symbol
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.strat_cfg = strat_cfg or StrategyConfig()
        self.sizer = AccountSizer(self.strat_cfg)
        self.governor = governor  # <-- store it (may be None)

    def _value_per_pip_1lot(self, broker: Broker) -> float:
        # vpp (per 1.0 lot) = CONTRACT_SIZE * PIP_SIZE
        return broker.cfg.CONTRACT_SIZE * PIP_SIZE

    def setup_trade(self, broker: Broker, tr: Trade):
        if self.config.get("USE_BREAK_EVEN_STOP", False):
            broker.set_break_even(
                tr.id,
                be_pips=self.config.get("BE_TRIGGER_PIPS", 0),
                offset_pips=self.config.get("BE_OFFSET_PIPS", 0),
            )
        if self.config.get("USE_TRAILING_STOP", False):
            tr.trailing_sl_distance = self.config.get(
                "TRAILING_STOP_DISTANCE_PIPS", None
            )
        if self.config.get("USE_TP_EXTENSION", False):
            tr.near_tp_buffer_pips = self.config.get("NEAR_TP_BUFFER_PIPS", None)
            tr.tp_extension_pips = self.config.get("TP_EXTENSION_PIPS", None)

    def on_bar(self, broker: Broker, t, row):
        raise NotImplementedError
