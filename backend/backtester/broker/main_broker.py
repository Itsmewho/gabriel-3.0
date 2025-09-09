# main_broker.py

import pandas as pd
from typing import List
from . import BrokerConfig, Trade, PIP_SIZE
from .open_trade import open_trade as _open
from .close_orders import close_trade as _close
from .margin_calls import (
    can_open_order,
    pick_worst_trade,
    margin_level_pct,
    pick_biggest_lot,
    margin_level_pct_worst_bar,
    should_force_liquidation,
    used_margin,
    floating_pnl_worst_bar,
    free_margin,
)
from .break_even import update_break_even
from .trailing_sl import update_trailing_sl, update_trailing_tp
from .cost_engine import swap_cost, fill_price_on_close, apply_spread_at
from .audit import log


def liquidation_fill_price(
    cfg: BrokerConfig, tr: Trade, high: float, low: float, t
) -> float:
    if tr.side == "buy":
        side, ref = "sell", low
    else:
        side, ref = "buy", high
    exec_price = fill_price_on_close(cfg, side, ref, t=t)
    baseline = apply_spread_at(cfg, side, ref, t=t)
    tr.slippage_close_pips = float((exec_price - baseline) / PIP_SIZE)
    return exec_price


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
        self.trading_enabled = True
        self.resume_margin_level_pct = cfg.RESUME_MARGIN_LEVEL_PCT
        self._pending_strategy_id: str | None = None
        self._pending_magic: int | None = None

    def execute_trade(
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
                "strategy_id": getattr(self, "_pending_strategy_id", None),
                "magic_number": getattr(self, "_pending_magic", None),
            }
            self.rejections.append(rej)
            log(self.events_log, type="rejection", **rej)
            return None

        tr, fee = _open(
            self._next_id,
            self.cfg,
            side,
            price,
            lots,
            sl_pips,
            tp_pips,
            t,
            strategy_id=getattr(self, "_pending_strategy_id", None),
            magic=getattr(self, "_pending_magic", None),
        )
        tr.balance_at_open = self.balance  # BEFORE deducting open fee
        self.balance -= fee  # apply open commission
        self.open_trades.append(tr)
        self._next_id += 1
        self._pending_strategy_id = None
        self._pending_magic = None

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

    def _force_liquidation(
        self, high: float, low: float, close: float, t: pd.Timestamp
    ) -> None:
        while self.open_trades and should_force_liquidation(
            self.cfg, self.balance, self.open_trades, high, low, close
        ):
            um = used_margin(self.cfg, self.open_trades)
            eq_worst = self.balance + floating_pnl_worst_bar(
                self.cfg, self.open_trades, high, low
            )
            fm_worst = eq_worst - um
            ml_worst = 999999.0 if um == 0 else (eq_worst / um) * 100.0
            fm_cur = free_margin(self.cfg, self.balance, self.open_trades, close)

            # 1) current free margin < 0 → close biggest lot at close
            if fm_cur < 0.0:
                tr_to_close = pick_biggest_lot(self.open_trades)
                side = "sell" if tr_to_close.side == "buy" else "buy"
                exec_price = fill_price_on_close(self.cfg, side, close, t=t)
                baseline = apply_spread_at(self.cfg, side, close, t=t)
                tr_to_close.slippage_close_pips = float(
                    (exec_price - baseline) / PIP_SIZE
                )
                self.close_trade(tr_to_close, exec_price, "Close Low Margin", t)
                log(
                    self.events_log,
                    type="low_margin_close",
                    time=t,
                    ml_worst=ml_worst,
                    fm_worst=fm_worst,
                    fm_current=fm_cur,
                    trade_id=tr_to_close.id,
                )
                continue

            # 2) worst-bar stop-out → close worst PnL at adverse extreme
            if (ml_worst < self.cfg.STOP_OUT_LEVEL_PCT) or (fm_worst < 0.0):
                tr_to_close = pick_worst_trade(self.cfg, self.open_trades, high, low)
                if tr_to_close is None:
                    break
                exec_price = liquidation_fill_price(self.cfg, tr_to_close, high, low, t)
                self.close_trade(tr_to_close, exec_price, "Margin Call", t)
                log(
                    self.events_log,
                    type="margin_call",
                    time=t,
                    ml_worst=ml_worst,
                    fm_worst=fm_worst,
                    fm_current=fm_cur,
                    trade_id=tr_to_close.id,
                )
                continue

            break

    def open_trade(
        self,
        side: str,
        price: float,
        wanted_lots: float,
        sl_pips: float,
        tp_pips: float,
        t: pd.Timestamp,
        fallbacks: list[float] | None = None,
        strategy_id: str | None = None,
        magic: int | None = None,
    ):
        """
        Try wanted_lots, then fallbacks (fractions or absolute lots).
        If none pass margin checks, pause trading and log a 'trading_paused' event.
        """
        if not self.trading_enabled:
            self._pending_strategy_id = None
            self._pending_magic = None
            log(
                self.events_log,
                type="open_skipped_paused",
                time=t,
                reason="trading_paused",
            )
            return None

        # build fallback sizes
        if fallbacks is not None:
            candidates = [wanted_lots] + [
                l for l in fallbacks if l != wanted_lots  # noqa: E741
            ]  # noqa: E741
        elif self.cfg.FALLBACK_LOTS:
            candidates = [wanted_lots] + [
                l for l in self.cfg.FALLBACK_LOTS if l != wanted_lots  # noqa: E741
            ]
        else:
            # fractions of wanted size
            fracs = list(self.cfg.FALLBACK_FRACTIONS)
            candidates = [wanted_lots] + [wanted_lots * f for f in fracs]

        self._pending_strategy_id = strategy_id
        self._pending_magic = magic

        for lots in candidates:
            if lots <= 0:
                continue
            diag = can_open_order(self.cfg, self.balance, self.open_trades, lots, price)
            if diag["ok"]:
                # proceed with normal open
                return self.execute_trade(side, price, lots, sl_pips, tp_pips, t)

            # record each rejection attempt
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
                "reason": "Insufficient margin (fallback try)",
                "strategy_id": getattr(self, "_pending_strategy_id", None),
                "magic_number": getattr(self, "_pending_magic", None),
            }
            self.rejections.append(rej)
            log(self.events_log, type="rejection", **rej)

        # all failed → pause trading
        if self.cfg.AUTO_PAUSE_ON_REJECTION:
            self._pending_strategy_id = None
            self._pending_magic = None
            log(
                self.events_log,
                type="trading_paused",
                time=t,
                reason="all_fallbacks_rejected",
            )
            return None

    def maybe_resume_trading(self, price: float, t: pd.Timestamp):
        """Auto-resume when margin level is comfortably above a threshold."""
        if self.trading_enabled:
            return
        ml = margin_level_pct(self.cfg, self.balance, self.open_trades, price)
        if ml >= self.resume_margin_level_pct:
            self.trading_enabled = True
            log(self.events_log, type="trading_resumed", time=t, margin_level=ml)

    def set_break_even(self, trade_id: int, be_pips: float, offset_pips: float = 0.0):
        for tr in self.open_trades:
            if tr.id == trade_id:
                tr.be_trigger_pips = be_pips
                tr.be_offset_pips = offset_pips
                return True
        return False

    def on_bar(
        self,
        high: float,
        low: float,
        close: float,
        t: pd.Timestamp,
        trail_pips: float | None = None,
    ):
        for tr in list(self.open_trades):
            tr.highest_price_during_trade = max(tr.highest_price_during_trade, high)
            tr.lowest_price_during_trade = min(tr.lowest_price_during_trade, low)

            trig = tr.be_trigger_pips or (
                self.cfg.BREAK_EVEN_TRIGGER_PIPS if self.cfg.BREAK_EVEN_ENABLE else None
            )
            off = tr.be_offset_pips or self.cfg.BREAK_EVEN_OFFSET_PIPS

            if trig and trig > 0:
                update_break_even(
                    tr,
                    high,
                    low,
                    trigger_pips=trig,
                    offset_pips=off,
                    events_log=self.events_log,
                    t=t,
                )
                log(
                    self.events_log,
                    type="break_even_config",
                    trade_id=tr.id,
                    source="per_trade" if tr.be_trigger_pips else "global",
                    trigger_pips=trig,
                    offset_pips=off,
                    time=t,
                )

            update_trailing_sl(
                tr,
                high,
                low,
                trail_pips=tr.trailing_sl_distance or trail_pips,
                events_log=self.events_log,
                t=t,
            )

            if tr.near_tp_buffer_pips is not None or tr.tp_extension_pips is not None:
                update_trailing_tp(
                    tr,
                    high,
                    low,
                    cfg=self.cfg,
                    events_log=self.events_log,
                    t=t,
                    near_tp_buffer_pips=tr.near_tp_buffer_pips,
                    tp_extension_pips=tr.tp_extension_pips,
                )

            if tr.side == "buy":
                if tr.sl is not None and low <= tr.sl:
                    exec_price = fill_price_on_close(self.cfg, "sell", tr.sl, t=t)
                    baseline = apply_spread_at(self.cfg, "sell", tr.sl, t=t)
                    tr.slippage_close_pips = float((exec_price - baseline) / PIP_SIZE)
                    self.close_trade(tr, exec_price, tr.sl_reason or "Stop Loss", t)
                    continue
                if tr.tp is not None and high >= tr.tp:
                    exec_price = fill_price_on_close(self.cfg, "sell", tr.tp, t=t)
                    baseline = apply_spread_at(self.cfg, "sell", tr.tp, t=t)
                    tr.slippage_close_pips = float((exec_price - baseline) / PIP_SIZE)
                    self.close_trade(tr, exec_price, tr.tp_reason or "Take Profit", t)
                    continue
            else:
                if tr.sl is not None and high >= tr.sl:
                    exec_price = fill_price_on_close(self.cfg, "buy", tr.sl, t=t)
                    baseline = apply_spread_at(self.cfg, "buy", tr.sl, t=t)
                    tr.slippage_close_pips = float((exec_price - baseline) / PIP_SIZE)
                    self.close_trade(tr, exec_price, tr.sl_reason or "Stop Loss", t)
                    continue
                if tr.tp is not None and low <= tr.tp:
                    exec_price = fill_price_on_close(self.cfg, "buy", tr.tp, t=t)
                    baseline = apply_spread_at(self.cfg, "buy", tr.tp, t=t)
                    tr.slippage_close_pips = float((exec_price - baseline) / PIP_SIZE)
                    self.close_trade(tr, exec_price, tr.tp_reason or "Take Profit", t)
                    continue

        if self.open_trades:
            ml_worst = margin_level_pct_worst_bar(
                self.cfg, self.balance, self.open_trades, high, low
            )
            if 0 < ml_worst < 90.0:
                log(self.events_log, type="margin_warning", time=t, ml_worst=ml_worst)

        self._force_liquidation(high, low, close, t)

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

        self.maybe_resume_trading(close, t)
