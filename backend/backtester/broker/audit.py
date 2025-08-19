from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List
from . import Trade, PIP_SIZE


def log(events_log: List[dict], **kwargs):
    events_log.append(dict(**kwargs))


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
                gross_pnl=tr.pnl + tr.commission_paid + tr.swap_paid,
                net_pnl=tr.pnl,
                account_balance_after=tr.balance_at_close,
                exit_reason=tr.exit_reason or "Open",
            )
        )
    df = pd.DataFrame(rows)
    if initial_balance is not None:
        df.attrs["initial_balance"] = initial_balance
    if final_balance is not None:
        df.attrs["final_balance"] = final_balance
    df.to_csv(filename, index=False)
    print(f"Audit log saved to {filename}")
    return df


def audit_rejections(
    rejections: list[dict], filename: str = "results/audit/rejected_trades.csv"
):
    if not rejections:
        return
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id",
        "time",
        "side",
        "lots",
        "price",
        "account_balance",
        "running_balance",
        "equity",
        "used_margin",
        "req_margin",
        "available_margin",
        "free_margin_after",
        "needed_balance",
        "reason",
    ]
    rows = []
    for r in rejections:
        rows.append(
            {
                "id": r.get("id"),
                "time": r.get("time"),
                "side": r.get("side"),
                "lots": r.get("lots"),
                "price": r.get("price"),
                "account_balance": r.get("account_balance"),
                "running_balance": r.get("running_balance"),
                "equity": r.get("equity"),
                "used_margin": r.get("used_margin"),
                "req_margin": r.get("req_margin"),
                "available_margin": r.get("free_margin_before"),
                "free_margin_after": r.get("free_margin_after"),
                "needed_balance": r.get("needed_balance"),
                "reason": r.get("reason"),
            }
        )
    pd.DataFrame(rows, columns=cols).to_csv(filename, index=False)
    print(f"Rejected trades log saved to {filename}")
