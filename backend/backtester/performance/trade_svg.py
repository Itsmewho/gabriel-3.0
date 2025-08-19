from dataclasses import asdict, is_dataclass
from backtester.broker.main_broker import Trade
from pathlib import Path
from typing import List
import pandas as pd


def export_trades_csv(trade_history: List[Trade], filename: str):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dicts
    rows = [asdict(t) if is_dataclass(t) else t for t in trade_history]

    # Build DataFrame
    df = pd.DataFrame(rows)

    # Optional: order and format columns
    cols = [
        "id",
        "side",
        "balance_at_open",
        "lot_size",
        "entry_time",
        "sl",
        "tp",
        "entry_price",
        "lowest_price_during_trade",
        "highest_price_during_trade",
        "swap_paid",
        "exit_price",
        "exit_time",
        "pnl",
        "balance_at_close",
        "exit_reason",
    ]
    existing = [c for c in cols if c in df.columns]
    df = df[existing]

    # Format times as local strings (no UTC conversion)
    for c in ("entry_time", "exit_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    df.to_csv(filename, index=False)
    print(f"CSV trade report saved to {filename}")
