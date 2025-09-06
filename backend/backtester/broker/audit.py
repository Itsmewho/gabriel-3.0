# Audit broker_functions

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List
from . import Trade


def log(events_log: List[dict], **kwargs):
    events_log.append(dict(**kwargs))


def _collect_mgmt_reasons(tr: Trade) -> list[str]:
    out: list[str] = []
    if getattr(tr, "be_applied", False):
        out.append("break_even")
    if (
        getattr(tr, "sl_mod_count", 0) > 0
        and getattr(tr, "sl_reason", None) == "trailing_sl"
    ):
        out.append("trailing_sl")
    if (
        getattr(tr, "tp_mod_count", 0) > 0
        and getattr(tr, "tp_reason", None) == "tp_extend"
    ):
        out.append("tp_extend")
    return out


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
        mgmt_reasons = _collect_mgmt_reasons(tr)
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
                # summaries
                mgmt_reasons=",".join(mgmt_reasons) if mgmt_reasons else None,
                mgmt_reason_count=len(mgmt_reasons),
                # last-state details
                sl_reason=getattr(tr, "sl_reason", None),
                tp_reason=getattr(tr, "tp_reason", None),
                # BE audit
                be_applied=getattr(tr, "be_applied", False),
                be_price=getattr(tr, "be_price", None),
                be_trigger_pips=getattr(tr, "be_trigger_pips", None),
                be_offset_pips=getattr(tr, "be_offset_pips", 0.0),
                # per-trade trailing overrides
                trailing_sl_distance=getattr(tr, "trailing_sl_distance", None),
                near_tp_buffer_pips=getattr(tr, "near_tp_buffer_pips", None),
                tp_extension_pips=getattr(tr, "tp_extension_pips", None),
                # stats
                highest_price=tr.highest_price_during_trade,
                lowest_price=tr.lowest_price_during_trade,
                # costs & PnL
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
