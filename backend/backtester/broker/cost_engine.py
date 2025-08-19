# cost_engine.py

import pandas as pd
from . import PIP_SIZE, BrokerConfig, Trade

DEFAULT_COMMISSION_PER_LOT_PER_SIDE = 3.50  # USD per lot per side (MT5-Broker)


def _per_side_commission(cfg: BrokerConfig) -> float:
    return (
        cfg.COMMISSION_PER_LOT_PER_SIDE
        if cfg.COMMISSION_PER_LOT_PER_SIDE is not None
        else DEFAULT_COMMISSION_PER_LOT_PER_SIDE
    )


def apply_spread(cfg: BrokerConfig, side: str, raw_price: float) -> float:
    # use module-level PIP_SIZE, not cfg.PIP_SIZE
    adj = cfg.SPREAD_PIPS * PIP_SIZE
    return raw_price + adj if side == "buy" else raw_price - adj


def value_per_pip(cfg: BrokerConfig, lots: float) -> float:
    # use module-level PIP_SIZE, not cfg.PIP_SIZE
    return lots * cfg.CONTRACT_SIZE * PIP_SIZE


def commission_open(cfg: BrokerConfig, lots: float) -> float:
    return _per_side_commission(cfg) * lots


def commission_close(cfg: BrokerConfig, lots: float) -> float:
    return _per_side_commission(cfg) * lots


def swap_cost(cfg: BrokerConfig, trade: Trade, t: pd.Timestamp) -> float:
    """
    Calculate overnight swap cost for a single trade.
    MT5-style: swap points × contract size × lots × pip value.
    Wednesday rollover = 3x.
    """
    if not isinstance(t, pd.Timestamp):
        t = pd.to_datetime(t)

    # side-based swap points
    swap_points = cfg.SWAP_LONG_POINTS if trade.side == "buy" else cfg.SWAP_SHORT_POINTS

    # convert points → price units
    points_to_price = PIP_SIZE / 10.0  # 1 point = 1/10 pip
    swap_in_price = swap_points * points_to_price

    # value in account currency
    value_per_point = cfg.CONTRACT_SIZE * swap_in_price
    fee = value_per_point * trade.lot_size

    # Wednesday triple rollover
    if t.weekday() == 2:
        fee *= 3

    return fee
