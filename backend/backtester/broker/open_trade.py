# open_trades.py

import pandas as pd
from . import Trade, BrokerConfig, PIP_SIZE
from .cost_engine import apply_spread, commission_open


def calc_sl_tp(
    entry: float, side: str, sl_pips: float, tp_pips: float
) -> tuple[float, float]:
    sl_adj = sl_pips * PIP_SIZE
    tp_adj = tp_pips * PIP_SIZE
    if side == "buy":
        return entry - sl_adj, entry + tp_adj
    return entry + sl_adj, entry - tp_adj


def open_trade(
    next_id: int,
    cfg: BrokerConfig,
    side: str,
    raw_price: float,
    lots: float,
    sl_pips: float,
    tp_pips: float,
    t: pd.Timestamp,
) -> tuple[Trade, float]:
    entry = apply_spread(cfg, side, raw_price)
    sl, tp = calc_sl_tp(entry, side, sl_pips, tp_pips)

    tr = Trade(
        id=next_id,
        side=side,
        entry_price=entry,
        lot_size=lots,
        entry_time=t,
        sl=sl,
        tp=tp,
        highest_price_during_trade=entry,
        lowest_price_during_trade=entry,
        sl_first=sl,
        sl_last=sl,
        tp_first=tp,
        tp_last=tp,
    )

    fee = commission_open(cfg, lots)  # per-side fee at open
    tr.commission_paid += fee  # record on trade

    return tr, fee
