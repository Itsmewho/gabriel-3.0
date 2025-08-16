import pandas as pd
from typing import Optional, Dict, Any


class BaseStrategy:
    def __init__(self, symbol: str, config: Dict[str, Any]):
        self.symbol = symbol
        self.config = config
        self.backtester: Optional[Any] = None
        self.pip_size = 0.0001

    def set_backtester(self, backtester: Any):
        self.backtester = backtester
        if hasattr(backtester, "pip_size"):
            self.pip_size = backtester.pip_size

    def generate_signals(self, data: pd.DataFrame):
        raise NotImplementedError

    def get_name(self) -> str:
        return self.config.get("name", self.__class__.__name__)
