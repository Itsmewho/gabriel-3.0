import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Trade:
    """A dataclass to store the complete state of a single trade."""

    id: int
    side: str
    entry_price: float
    lot_size: float
    entry_time: pd.Timestamp
    sl: Optional[float]
    tp: Optional[float]

    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0

    highest_price_during_trade: float = -np.inf
    lowest_price_during_trade: float = np.inf


class OrderEngine:
    """Manages account state, trade execution, and P&L calculations."""

    def __init__(self, config: dict):
        self.config = config
        self.pip_size = 0.0001
        self.balance = config["INITIAL_BALANCE"]
        self.open_trades: List[Trade] = []
        self.trade_history: List[Trade] = []
        self._next_trade_id = 1

    def _value_per_pip(self, lot):
        return lot * self.config["CONTRACT_SIZE"] * self.pip_size

    def _floating_pnl(self, trade: Trade, price: float) -> float:
        pips = (
            (price - trade.entry_price) / self.pip_size
            if trade.side == "buy"
            else (trade.entry_price - price) / self.pip_size
        )
        return pips * self._value_per_pip(trade.lot_size)

    def _used_margin(self) -> float:
        return sum(
            abs(t.lot_size) * self.config["MARGIN_PER_LOT"] for t in self.open_trades
        )

    def _margin_level_pct(self, price: float) -> float:
        eq = self.balance + sum(self._floating_pnl(t, price) for t in self.open_trades)
        used = self._used_margin()
        return 999999.0 if used == 0 else (eq / used) * 100.0

    def open_trade(
        self,
        side: str,
        price: float,
        lot_size: float,
        sl_pips: float,
        tp_pips: float,
        timestamp: pd.Timestamp,
    ):
        spread_adj = self.config["SPREAD_PIPS"] * self.pip_size
        entry_price = price + spread_adj if side == "buy" else price - spread_adj

        sl_adj = sl_pips * self.pip_size
        tp_adj = tp_pips * self.pip_size
        sl = entry_price - sl_adj if side == "buy" else entry_price + sl_adj
        tp = entry_price + tp_adj if side == "buy" else entry_price - tp_adj

        trade = Trade(
            id=self._next_trade_id,
            side=side,
            entry_price=entry_price,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            entry_time=timestamp,
        )
        trade.highest_price_during_trade = entry_price
        trade.lowest_price_during_trade = entry_price

        commission = self.config["COMMISSION_PER_LOT_RT"] * lot_size
        self.balance -= commission

        self.open_trades.append(trade)
        self._next_trade_id += 1

    def close_trade(
        self, trade: Trade, exit_price: float, reason: str, timestamp: pd.Timestamp
    ):
        if trade not in self.open_trades:
            return

        pips_moved = (
            (exit_price - trade.entry_price) / self.pip_size
            if trade.side == "buy"
            else (trade.entry_price - exit_price) / self.pip_size
        )
        pnl = pips_moved * self._value_per_pip(trade.lot_size)

        trade.pnl = pnl
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.exit_time = timestamp
        self.balance += pnl

        self.open_trades.remove(trade)
        self.trade_history.append(trade)

    def close_all_open_trades(self, final_price: float, timestamp: pd.Timestamp):
        for trade in list(self.open_trades):
            self.close_trade(trade, final_price, "End of Backtest", timestamp)

    def on_bar(self, high: float, low: float, close: float, timestamp: pd.Timestamp):
        # update extremes
        for trade in list(self.open_trades):
            trade.highest_price_during_trade = max(
                trade.highest_price_during_trade, high
            )
            trade.lowest_price_during_trade = min(trade.lowest_price_during_trade, low)
            if trade.side == "buy":
                if low <= trade.sl:  # type: ignore
                    self.close_trade(trade, trade.sl, "Stop Loss", timestamp)  # type: ignore
                elif high >= trade.tp:  # type: ignore
                    self.close_trade(trade, trade.tp, "Take Profit", timestamp)  # type: ignore
            else:
                if high >= trade.sl:  # type: ignore
                    self.close_trade(trade, trade.sl, "Stop Loss", timestamp)  # type: ignore
                elif low <= trade.tp:  # type: ignore
                    self.close_trade(trade, trade.tp, "Take Profit", timestamp)  # type: ignore

        # simple stop-out: if margin level below threshold, close all at market
        if self.open_trades:
            ml = self._margin_level_pct(close)
            if ml < self.config["STOP_OUT_LEVEL_PCT"]:
                for t in list(self.open_trades):
                    self.close_trade(t, close, "Stop Out", timestamp)
