# main_broker.py


import pandas as pd
from typing import List
from . import BrokerConfig, Trade
from .open_trade import open_trade as _open
from .close_orders import close_trade as _close
from .margin_calls import (
    can_open_order,
    needs_warning,
    needs_stop_out,
    pick_worst_trade,
)
from .trailing_sl import update_trailing_sl
from .cost_engine import swap_cost
from .audit import log


class Broker:
    def __init__(self, cfg: BrokerConfig):
        self.cfg = cfg
        self.balance = cfg.INITIAL_BALANCE
        self.open_trades: List[Trade] = []
        self.trade_history: List[Trade] = []
        self.events_log: List[dict] = []
        self.rejections: list[dict] = []
        self._last_swap_day = None
        self._next_id = 1

    def open_trade(
        self,
        side: str,
        price: float,
        lots: float,
        sl_pips: float,
        tp_pips: float,
        t: pd.Timestamp,
    ):
        diag = can_open_order(self.cfg, self.balance, self.open_trades, lots, price)
        if not diag["ok"]:
            rej = {
                "id": self._next_id,
                "time": t,
                "side": side,
                "lots": lots,
                "price": price,
                "account_balance": self.balance,
                "equity": diag["equity"],
                "used_margin": diag["used_margin"],
                "req_margin": diag["req_margin"],
                "available_margin": diag["free_margin_before"],
                "free_margin_after": diag["free_margin_after"],
                "needed_balance": diag["needed_balance"],
                "running_balance": self.balance,
                "reason": "Insufficient margin",
            }
            self.rejections.append(rej)
            log(self.events_log, type="rejection", **rej)
            return None

        tr, fee = _open(self._next_id, self.cfg, side, price, lots, sl_pips, tp_pips, t)
        tr.balance_at_open = self.balance  # BEFORE deducting open fee
        self.balance -= fee  # apply open commission
        self.open_trades.append(tr)
        self._next_id += 1

        log(
            self.events_log,
            type="open",
            time=t,
            side=side,
            price=tr.entry_price,
            id=tr.id,
            lots=lots,
        )
        return tr

    def close_trade(self, tr: Trade, price: float, reason: str, t: pd.Timestamp):
        bal_delta, full_net = _close(self.cfg, tr, price, reason, t)
        self.balance += bal_delta
        tr.balance_at_close = self.balance

        self.open_trades.remove(tr)
        self.trade_history.append(tr)
        log(
            self.events_log,
            type="close",
            time=t,
            side=tr.side,
            price=tr.exit_price,
            id=tr.id,
            reason=reason,
            pnl=tr.pnl,
        )

    def close_all(self, price: float, t: pd.Timestamp, reason="End"):
        for tr in list(self.open_trades):
            self.close_trade(tr, price, reason, t)

    def on_bar(
        self,
        high: float,
        low: float,
        close: float,
        t: pd.Timestamp,
        trail_pips: float | None = None,
    ):
        # update running stats + trailing SL + handle exits
        for tr in list(self.open_trades):
            tr.highest_price_during_trade = max(tr.highest_price_during_trade, high)
            tr.lowest_price_during_trade = min(tr.lowest_price_during_trade, low)
            update_trailing_sl(
                tr, high, low, trail_pips, events_log=self.events_log, t=t
            )

            if tr.side == "buy":
                if low <= (tr.sl or -1e9):
                    self.close_trade(tr, tr.sl, "Stop Loss", t)  # type: ignore
                    continue
                if high >= (tr.tp or 1e9):
                    self.close_trade(tr, tr.tp, "Take Profit", t)  # type: ignore
                    continue
            else:
                if high >= (tr.sl or 1e9):
                    self.close_trade(tr, tr.sl, "Stop Loss", t)  # type: ignore
                    continue
                if low <= (tr.tp or -1e9):
                    self.close_trade(tr, tr.tp, "Take Profit", t)  # type: ignore
                    continue

        # margin warning
        if self.open_trades and needs_warning(
            self.cfg, self.balance, self.open_trades, close
        ):
            log(self.events_log, type="margin_warning", time=t)

        # stop-out loop at 50% (largest loss first)
        while self.open_trades and needs_stop_out(
            self.cfg, self.balance, self.open_trades, close
        ):
            worst = pick_worst_trade(self.cfg, self.open_trades, close)
            if worst is None:
                break
            self.close_trade(worst, close, "Margin Call", t)
            log(self.events_log, type="margin_call", time=t)

        # daily rollover swap once per day
        cur_day = t.date()
        if cur_day != self._last_swap_day:
            for tr in self.open_trades:
                fee = swap_cost(self.cfg, tr, t)
                if fee:
                    tr.swap_paid += fee
                    self.balance -= fee
                    log(
                        self.events_log,
                        type="swap",
                        trade_id=tr.id,
                        fee=fee,
                        time=t,
                        side=tr.side,
                        lots=tr.lot_size,
                    )
            self._last_swap_day = cur_day
