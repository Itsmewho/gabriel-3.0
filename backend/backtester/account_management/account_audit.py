# Account auditor

from __future__ import annotations
from pathlib import Path
import pandas as pd


def export_account_audit(df: pd.DataFrame, filename: str):
    """
    df: output of Ledger.snapshot_df()
    Also computes per-strategy drawdown stats.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    stats = []
    for sid, g in out.groupby("strategy_id"):
        eq = g["equity_after"].fillna(method="ffill").fillna(0.0)  # type: ignore
        peak = eq.cummax().replace(0, 1e-9)
        dd = 1.0 - (eq / peak)
        g = g.assign(drawdown_pct=dd.values)
        stats.append(
            {
                "strategy_id": sid,
                "final_equity": float(eq.iloc[-1]),
                "max_drawdown_pct": float(dd.max()),
                "events": len(g),
            }
        )
        out.loc[g.index, "drawdown_pct"] = g["drawdown_pct"]

    out.to_csv(filename, index=False)
    pd.DataFrame(stats).to_csv(filename.replace(".csv", "_stats.csv"), index=False)
    print(f"Account audit saved to {filename}")
