# cost_engine.py

import pandas as pd
from . import PIP_SIZE, BrokerConfig, Trade

DEFAULT_COMMISSION_PER_LOT_PER_SIDE = 3.50  # USD per lot per side


def _per_side_commission(cfg: BrokerConfig) -> float:
    return cfg.COMMISSION_PER_LOT_PER_SIDE or DEFAULT_COMMISSION_PER_LOT_PER_SIDE


def apply_spread(cfg: BrokerConfig, side: str, raw_price: float) -> float:
    adj = cfg.SPREAD_PIPS * PIP_SIZE
    return raw_price + adj if side == "buy" else raw_price - adj


def value_per_pip(cfg: BrokerConfig, lots: float) -> float:
    return lots * cfg.CONTRACT_SIZE * PIP_SIZE


def commission_open(cfg: BrokerConfig, lots: float) -> float:
    return _per_side_commission(cfg) * lots


def commission_close(cfg: BrokerConfig, lots: float) -> float:
    return _per_side_commission(cfg) * lots


def swap_cost(cfg: BrokerConfig, trade: Trade, t: pd.Timestamp) -> float:
    if not isinstance(t, pd.Timestamp):
        t = pd.to_datetime(t)
    swap_points = cfg.SWAP_LONG_POINTS if trade.side == "buy" else cfg.SWAP_SHORT_POINTS
    # 1 point = 1/10 pip on MT5
    points_to_price = PIP_SIZE / 10.0
    swap_in_price = swap_points * points_to_price
    fee = cfg.CONTRACT_SIZE * swap_in_price * trade.lot_size
    if t.weekday() == 2:  # Wednesday triple
        fee *= 3
    return fee
