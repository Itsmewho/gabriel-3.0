# Audit.py

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List
from . import Trade, PIP_SIZE


def log(events_log: List[dict], **kwargs):
    """
    Append an event into events_log.
    Example:
        log(self.events_log, type="open", time=t, side="buy", price=1.2345, id=1)
    """
    events_log.append(dict(**kwargs))


def audit_trades(trades: List[Trade], filename: str = "results/audit/trade_audit.csv"):
    """
    Export trade audit with SL/TP modification info.
    """
    if not trades:
        return

    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for tr in trades:
        sl_delta = None
        tp_delta = None
        if tr.sl_first is not None and tr.sl_last is not None:
            sl_delta = (tr.sl_last - tr.sl_first) / PIP_SIZE
        if tr.tp_first is not None and tr.tp_last is not None:
            tp_delta = (tr.tp_last - tr.tp_first) / PIP_SIZE

        rows.append(
            dict(
                id=tr.id,
                side=tr.side,
                lots=tr.lot_size,
                open_time=tr.entry_time,
                close_time=tr.exit_time,
                entry_price=tr.entry_price,
                exit_price=tr.exit_price,
                sl_first=tr.sl_first,
                sl_last=tr.sl_last,
                sl_mod_count=tr.sl_mod_count,
                tp_first=tr.tp_first,
                tp_last=tr.tp_last,
                tp_mod_count=tr.tp_mod_count,
                highest_price=tr.highest_price_during_trade,
                lowest_price=tr.lowest_price_during_trade,
                commission=tr.commission_paid,
                swap=tr.swap_paid,
                gross_pnl=tr.pnl + tr.commission_paid + tr.swap_paid,  # reconstructed
                net_pnl=tr.pnl,
                exit_reason=tr.exit_reason or "Open",
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Audit log saved to {filename}")
    return df
