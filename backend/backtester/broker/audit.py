# Audit broker_functions

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List
from . import Trade


def log(events_log: List[dict], **kwargs):
    events_log.append(dict(**kwargs))


def _combine_reasons(tr: Trade) -> str | None:
    reasons = []
    if getattr(tr, "sl_reason", None):
        reasons.append(tr.sl_reason)
    if getattr(tr, "tp_reason", None):
        reasons.append(tr.tp_reason)
    if getattr(tr, "be_applied", False):
        reasons.append("break_even")
    if not reasons:
        return None
    # deduplicate while preserving order
    seen = set()
    out = []
    for r in reasons:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return "+".join(out)


def audit_trades(
    trades: List[Trade],
    filename: str = "results/audit/trade_audit.csv",
    initial_balance: float | None = None,
    final_balance: float | None = None,
):
    if not trades:
        return
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for tr in trades:
        rows.append(
            dict(
                id=tr.id,
                side=tr.side,
                lots=tr.lot_size,
                initial_balance=tr.balance_at_open,
                account_balance_after=tr.balance_at_close,
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
                sl_reason=getattr(tr, "sl_reason", None),
                tp_reason=getattr(tr, "tp_reason", None),
                combined_reasons=_combine_reasons(tr),
                be_applied=getattr(tr, "be_applied", False),
                be_price=getattr(tr, "be_price", None),
                be_trigger_pips=getattr(tr, "be_trigger_pips", None),
                be_offset_pips=getattr(tr, "be_offset_pips", 0.0),
                trailing_sl_distance=getattr(tr, "trailing_sl_distance", None),
                near_tp_buffer_pips=getattr(tr, "near_tp_buffer_pips", None),
                tp_extension_pips=getattr(tr, "tp_extension_pips", None),
                highest_price=tr.highest_price_during_trade,
                lowest_price=tr.lowest_price_during_trade,
                slippage_open_pips=getattr(tr, "slippage_open_pips", 0.0),
                slippage_close_pips=getattr(tr, "slippage_close_pips", 0.0),
                commission=tr.commission_paid,
                swap=tr.swap_paid,
                gross_pnl=tr.pnl + tr.commission_paid + tr.swap_paid,
                net_pnl=tr.pnl,
                exit_reason=tr.exit_reason or "Open",
                strategy_id=getattr(tr, "strategy_id", None),
                magic_number=getattr(tr, "magic_number", None),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Audit log saved to {filename}")
    return df
