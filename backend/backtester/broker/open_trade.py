# open_trades.py

import pandas as pd
from . import Trade, BrokerConfig, PIP_SIZE
from .cost_engine import apply_spread, commission_open, fill_price_on_open


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
    strategy_id=None,
    magic=None,
) -> tuple[Trade, float]:
    entry = fill_price_on_open(cfg, side, raw_price)
    baseline_no_slip = apply_spread(cfg, side, raw_price)
    slip_open_pips = (entry - baseline_no_slip) / PIP_SIZE
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
        slippage_open_pips=float(slip_open_pips),
        strategy_id=strategy_id,
        magic_number=magic,
    )

    fee = commission_open(cfg, lots)  # charged now on account
    tr.commission_paid += fee  # recorded on trade
    return tr, fee
