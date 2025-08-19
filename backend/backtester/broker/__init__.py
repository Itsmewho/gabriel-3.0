# __init__.py

from dataclasses import dataclass
import pandas as pd
from typing import Optional

PIP_SIZE = 0.0001  # EURUSD


@dataclass
class BrokerConfig:
    INITIAL_BALANCE: float
    SYMBOL: str
    SPREAD_PIPS: float
    SWAP_LONG_POINTS: float
    SWAP_SHORT_POINTS: float
    CONTRACT_SIZE: int
    MARGIN_PER_LOT: float
    STOP_OUT_LEVEL_PCT: float
    VOLUME_STEP: float
    VOLUME_MIN: float
    MIN_SLIPPAGE_PIPS: int = 0
    MAX_SLIPPAGE_PIPS: int = 0
    COMMISSION_PER_LOT_PER_SIDE: float | None = None


@dataclass
class Trade:
    id: int
    side: str  # buy/sell
    entry_price: float
    entry_time: pd.Timestamp
    lot_size: float
    sl: Optional[float]
    tp: Optional[float]
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None

    # running stats
    pnl: float = 0.0
    highest_price_during_trade: float = float("-inf")
    lowest_price_during_trade: float = float("inf")

    # SL/TP audit
    sl_first: Optional[float] = None
    sl_last: Optional[float] = None
    sl_mod_count: int = 0
    tp_first: Optional[float] = None
    tp_last: Optional[float] = None
    tp_mod_count: int = 0

    # costs
    commission_paid: float = 0.0  # sum of open+close commissions
    swap_paid: float = 0.0  # cumulative swaps


__all__ = ["PIP_SIZE", "BrokerConfig", "Trade"]
