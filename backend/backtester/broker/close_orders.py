# close_orders.py

import pandas as pd
from . import Trade, BrokerConfig, PIP_SIZE
from .cost_engine import value_per_pip, commission_close


def close_trade(
    cfg: BrokerConfig, trade: Trade, exit_price: float, reason: str, t: pd.Timestamp
) -> tuple[float, float]:
    # gross (price-only PnL)
    if trade.side == "buy":
        pips = (exit_price - trade.entry_price) / PIP_SIZE
    else:
        pips = (trade.entry_price - exit_price) / PIP_SIZE
    gross = pips * value_per_pip(cfg, trade.lot_size)

    # commission at close (hits account and trade now)
    fee_close = commission_close(cfg, trade.lot_size)
    trade.commission_paid += fee_close

    # stamp
    trade.exit_price = exit_price
    trade.exit_time = t
    if reason == "Take Profit" and getattr(trade, "tp_mod_count", 0) > 0:
        reason = "Take Profit (extended)"
    trade.exit_reason = reason

    # full net for audit
    full_net = gross - trade.commission_paid - trade.swap_paid
    trade.pnl = full_net

    # balance delta to apply now (open fee & swaps already applied):
    balance_delta_now = gross - fee_close

    return balance_delta_now, full_net
