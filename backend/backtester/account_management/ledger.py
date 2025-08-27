# Ledger

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
from .types import LedgerEntry


@dataclass
class StrategyBook:
    equity: float = 0.0
    history: List[LedgerEntry] = field(default_factory=list)


class Ledger:
    """
    Tracks per-strategy equity via realized PnL and cash flows.
    You call:
      - on_open(strategy_id, time, trade_id)   [no equity change]
      - on_close(strategy_id, time, trade_id, pnl)
      - deposit/withdraw/transfer
      - snapshot_df() for audit/equity curves
    """

    def __init__(self, initial_allocations: Dict[str, float]):
        # equity starts at allocation budgets
        self.books: Dict[str, StrategyBook] = {
            sid: StrategyBook(equity=eq) for sid, eq in initial_allocations.items()
        }

    def _ensure_book(self, sid: str):
        if sid not in self.books:
            self.books[sid] = StrategyBook(equity=0.0)

    def _push(self, sid: str, e: LedgerEntry):
        self._ensure_book(sid)
        self.books[sid].history.append(e)

    def on_open(self, sid: str, time: pd.Timestamp, trade_id: Optional[int] = None):
        self._ensure_book(sid)
        b = self.books[sid]
        self._push(
            sid, LedgerEntry(time, sid, "open", 0.0, b.equity, b.equity, trade_id)
        )

    def on_close(
        self, sid: str, time: pd.Timestamp, pnl: float, trade_id: Optional[int] = None
    ):
        self._ensure_book(sid)
        b = self.books[sid]
        before = b.equity
        after = before + pnl
        b.equity = after
        self._push(sid, LedgerEntry(time, sid, "close", pnl, before, after, trade_id))

    def deposit(self, sid: str, time: pd.Timestamp, amount: float):
        b = self.books[sid]
        before = b.equity
        after = before + amount
        b.equity = after
        self._push(sid, LedgerEntry(time, sid, "deposit", amount, before, after))

    def withdraw(self, sid: str, time: pd.Timestamp, amount: float):
        self.deposit(sid, time, -abs(amount))

    def transfer(self, from_sid: str, to_sid: str, time: pd.Timestamp, amount: float):
        self.withdraw(from_sid, time, amount)
        self.deposit(to_sid, time, amount)

    def equity(self, sid: str) -> float:
        return self.books[sid].equity

    def snapshot_df(self) -> pd.DataFrame:
        rows = []
        for sid, b in self.books.items():
            for e in b.history:
                rows.append(
                    {
                        "time": e.time,
                        "strategy_id": sid,
                        "kind": e.kind,
                        "amount": e.amount,
                        "equity_before": e.equity_before,
                        "equity_after": e.equity_after,
                        "trade_id": e.trade_id,
                    }
                )
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "time",
                    "strategy_id",
                    "kind",
                    "amount",
                    "equity_before",
                    "equity_after",
                    "trade_id",
                ]
            )
        return df.sort_values("time")
