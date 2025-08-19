# close_orders.py

import pandas as pd
from . import Trade, BrokerConfig, PIP_SIZE
from .cost_engine import value_per_pip, commission_close


def close_trade(
    cfg: BrokerConfig, trade: Trade, exit_price: float, reason: str, t: pd.Timestamp
) -> float:
    if trade.side == "buy":
        pips = (exit_price - trade.entry_price) / PIP_SIZE
    else:
        pips = (trade.entry_price - exit_price) / PIP_SIZE
    # broker/close_orders.py
    gross = pips * value_per_pip(cfg, trade.lot_size)  # no costs
    fee_close = commission_close(cfg, trade.lot_size)
    trade.commission_paid += fee_close
    trade.exit_price = exit_price
    trade.exit_time = t
    trade.exit_reason = reason
    trade.pnl = gross - trade.commission_paid - trade.swap_paid
    return trade.pnl
