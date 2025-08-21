# __init__.py

from dataclasses import dataclass
import pandas as pd
from typing import Optional, List

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
    RESUME_MARGIN_LEVEL_PCT: float = 120.0  # auto-resume threshold
    AUTO_PAUSE_ON_REJECTION: bool = True  # pause after all fallbacks fail
    FALLBACK_LOTS: Optional[List[float]] = None  # absolute fallback lot sizes
    FALLBACK_FRACTIONS: List[float] = (0.5, 0.25, 0.10)  # type: ignore # if no absolute lots
    NEAR_TP_BUFFER_PIPS: float = 2.0  # trigger zone
    TP_EXTENSION_PIPS: float = 3.0  # bump size

    # Break-even (optional)
    BREAK_EVEN_ENABLE: bool = False
    BREAK_EVEN_TRIGGER_PIPS: float = (
        0.0  # move SL to BE once price moves this many pips in favor
    )
    BREAK_EVEN_OFFSET_PIPS: float = (
        0.0  # add/subtract this many pips beyond BE (locking in +X pips)
    )


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
    sl_reason: str | None = None  # "break_even" | "trailing_sl" | None
    tp_reason: str | None = None  # "tp_extend"  | None

    # costs
    commission_paid: float = 0.0  # open + close
    swap_paid: float = 0.0  # cumulative rollover debits

    # balances (for audit)
    balance_at_open: Optional[float] = None  # BEFORE open fee
    balance_at_close: Optional[float] = None  # AFTER close is booked

    # Break-even
    be_applied: bool = False
    be_price: Optional[float] = None
    be_trigger_pips: float | None = None
    be_offset_pips: float = 0.0

    # Trailing overrides
    trailing_sl_distance: float | None = None  # strategy-specific trailing stop
    near_tp_buffer_pips: float | None = None  # strategy override
    tp_extension_pips: float | None = None  # strategy override


__all__ = ["PIP_SIZE", "BrokerConfig", "Trade"]
