# Audit broker_functions

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
                # balances
                initial_balance=tr.balance_at_open,
                account_balance_after=tr.balance_at_close,
                # times & prices
                open_time=tr.entry_time,
                close_time=tr.exit_time,
                entry_price=tr.entry_price,
                exit_price=tr.exit_price,
                # SL/TP evolution
                sl_first=tr.sl_first,
                sl_last=tr.sl_last,
                sl_mod_count=tr.sl_mod_count,
                tp_first=tr.tp_first,
                tp_last=tr.tp_last,
                tp_mod_count=tr.tp_mod_count,
                # reasons for last SL/TP state
                sl_reason=getattr(tr, "sl_reason", None),
                tp_reason=getattr(tr, "tp_reason", None),
                # BE audit
                be_applied=getattr(tr, "be_applied", False),
                be_price=getattr(tr, "be_price", None),
                be_trigger_pips=getattr(tr, "be_trigger_pips", None),
                be_offset_pips=getattr(tr, "be_offset_pips", 0.0),
                # per-trade trailing overrides (if any)
                trailing_sl_distance=getattr(tr, "trailing_sl_distance", None),
                near_tp_buffer_pips=getattr(tr, "near_tp_buffer_pips", None),
                tp_extension_pips=getattr(tr, "tp_extension_pips", None),
                # stats
                highest_price=tr.highest_price_during_trade,
                lowest_price=tr.lowest_price_during_trade,
                # costs & PnL
                commission=tr.commission_paid,
                swap=tr.swap_paid,
                gross_pnl=tr.pnl + tr.commission_paid + tr.swap_paid,
                net_pnl=tr.pnl,
                exit_reason=tr.exit_reason or "Open",
                # Trade ids
                strategy_id=getattr(tr, "strategy_id", None),
                magic_number=getattr(tr, "magic_number", None),
            )
        )

    df = pd.DataFrame(rows)
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
        "magic_number",
        "strategy_id",
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
                "magic_number": r.get("magic_number"),
                "strategy_id": r.get("strategy_id"),
            }
        )
    pd.DataFrame(rows, columns=cols).to_csv(filename, index=False)
    print(f"Rejected trades log saved to {filename}")


def audit_max_open_trades(
    trades: list[Trade],
    filename: str = "results/audit/max_open_trades.csv",
):
    """
    Compute the maximum number of trades open at the same time.
    Saves a CSV with: max_open_trades, first_time_reached
    """
    if not trades:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"max_open_trades": 0, "first_time_reached": None}]).to_csv(
            filename, index=False
        )
        return 0

    events = []
    for tr in trades:
        if tr.entry_time is not None:
            events.append((pd.to_datetime(tr.entry_time), 0, +1))
        if tr.exit_time is not None:
            events.append((pd.to_datetime(tr.exit_time), 1, -1))

    events.sort(key=lambda x: (x[0], x[1]))

    cur = 0
    mx = 0
    first_ts = None
    for ts, _, delta in events:
        cur += delta
        if cur > mx:
            mx = cur
            first_ts = ts

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"max_open_trades": mx, "first_time_reached": first_ts}]).to_csv(
        filename, index=False
    )
    print(f"MAX open trades log saved to {filename}")
    return mx
