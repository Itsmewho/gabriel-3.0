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

    # normalize exit reason
    r = (reason or "").strip()

    # TP: if any TP extension occurred, label as extended
    if r in ("TP", "Take Profit"):
        r = (
            "Take Profit (extended)"
            if getattr(trade, "tp_mod_count", 0) > 0
            else "Take Profit"
        )

    # SL: if SL was moved to break-even, label accordingly
    elif r in ("SL", "Stop Loss"):
        r = (
            "Break Even (SL)"
            if getattr(trade, "sl_reason", None) == "break_even"
            else "Stop Loss"
        )

    trade.exit_reason = r
    # full net for audit
    full_net = gross - trade.commission_paid - trade.swap_paid
    trade.pnl = full_net

    # balance delta to apply now (open fee & swaps already applied):
    balance_delta_now = gross - fee_close

    return balance_delta_now, full_net
