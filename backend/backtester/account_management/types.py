# Types

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class RiskMode(str, Enum):
    FIXED = "fixed"
    HALF_KELLY = "half_kelly"


@dataclass
class StrategyConfig:
    # sizing
    risk_mode: RiskMode = RiskMode.FIXED
    risk_pct: float = 0.03  # for FIXED (e.g. 1%)
    kelly_p: float = 0.55  # win probability
    kelly_rr: float = 1.5  # reward/risk ratio
    kelly_cap_pct: float = 0.02  # cap fraction of balance risked (safety)

    # broker lot constraints (pass from your BrokerConfig)
    lot_min: float = 0.01
    lot_step: float = 0.01
    lot_max: float = 100.0

    # risk limits
    max_risk_pct_per_trade: float = 0.03
    max_concurrent_trades: int = 5
    max_drawdown_pct: float = 0.3  # pause if equity falls 30% from peak


@dataclass
class Allocation:
    strategy_id: str
    weight: float  # 0..1
    equity_budget: float = 0.0


@dataclass
class SizeRequest:
    balance: float
    sl_pips: float
    value_per_pip: float  # lots * contract * pip_size (you’ll pass per 1.0 lot)
    wanted_rr: Optional[float] = None  # optional


@dataclass
class SizeResult:
    lots: float
    risk_amount: float  # $ risked (approx at SL distance)
    reason: Optional[str] = None


@dataclass
class GovCheck:
    ok: bool
    reason: Optional[str] = None


@dataclass
class LedgerEntry:
    time: pd.Timestamp
    strategy_id: str
    kind: str  # "open" | "close" | "deposit" | "withdraw" | "transfer"
    amount: float  # pnl or cash flow (+/-)
    equity_before: float
    equity_after: float
    trade_id: Optional[int] = None
