from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from utils.helpers import setup_logger
import pandas as pd
import random


logger = setup_logger(__name__)
# Server-time engine. All distances in *pips* scaled by pip_size.


@dataclass
class Trade:
    id: int
    symbol: str
    side: str
    lot_size: float
    entry_price: float
    sl: Optional[float]
    tp: Optional[float]
    entry_time: Any
    initial_sl: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[Any] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    highest_price: float = field(default_factory=lambda: -math.inf)
    lowest_price: float = field(default_factory=lambda: math.inf)
    swap_accrued: float = 0.0
    be_activated: bool = False
    ts_activated: bool = False


class OrderEngine:
    def __init__(self, symbol: str, config: Dict[str, Any], pip_size: float = 0.0001):
        self.symbol = symbol
        self.config = config
        self.pip_size = pip_size

        # Accounting
        self.balance: float = float(config.get("INITIAL_BALANCE", 5000.0))
        self.equity: float = self.balance
        self.open_trades: List[Trade] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.concurrent_trades_log: List[int] = []
        self.risk_audit: List[Dict[str, Any]] = []

        # Streaks
        self.max_consecutive_losses = 0
        self.max_consecutive_loss_amount = 0.0
        self.max_consecutive_wins = 0
        self.max_consecutive_win_amount = 0.0
        self._cur_loss_streak = 0
        self._cur_win_streak = 0
        self._cur_loss_sum = 0.0
        self._cur_win_sum = 0.0

        # Sizing
        self.value_per_pip_per_lot: float = float(
            config.get("VALUE_PER_PIP_PER_LOT", 10.0)
        )
        self.use_compounding: bool = bool(config.get("USE_COMPOUNDING", True))
        self.fixed_lot: float = float(config.get("FIXED_TRADE_SIZE_LOTS", 0.01))
        self.risk_pct: float = float(config.get("RISK_PERCENTAGE", 1.0))
        self.max_lot: float = float(config.get("MAX_LOT_SIZE", 50.0))
        self.volume_step: float = float(config.get("VOLUME_STEP", 0.01))
        self.volume_min: float = float(config.get("VOLUME_MIN", 0.01))
        self.min_sizing_pips: float = float(config.get("MIN_SIZING_PIPS", 10.0))

        self.execution_latency_bars: int = int(config.get("EXECUTION_LATENCY_BARS", 1))
        self.max_deviation_pips: float = float(config.get("MAX_DEVIATION_PIPS", 2.0))

        # Costs
        self.use_spread: bool = bool(config.get("USE_SPREAD_COST", True))
        self.spread_pips: float = float(config.get("SPREAD_PIPS", 0.2))
        self.commission_rt: float = float(config.get("COMMISSION_PER_LOT_RT", 7.0))
        self.use_swap: bool = bool(config.get("USE_SWAP_COST", True))
        self.swap_long_pts: float = float(config.get("SWAP_LONG_POINTS", -9.89))
        self.swap_short_pts: float = float(config.get("SWAP_SHORT_POINTS", 5.44))
        self.use_equity_for_risk: bool = bool(
            self.config.get("USE_EQUITY_FOR_RISK", True)
        )

        # Margin
        self.margin_per_lot: float = float(config.get("MARGIN_PER_LOT", 3882.02))
        self.stop_out_level_pct: float = float(config.get("STOP_OUT_LEVEL_PCT", 50.0))
        self.max_concurrent_trades: int = int(config.get("MAX_CONCURRENT_TRADES", 3))
        self.max_risk_pct_total: float = float(
            self.config.get("MAX_RISK_PCT_TOTAL", 2.0)
        )
        self.resize_to_free_margin: bool = bool(
            self.config.get("RESIZE_TO_FREE_MARGIN", True)
        )

        # --- Blowout & Risk Management ---
        self.use_blowout_protection: bool = bool(
            config.get("USE_BLOWOUT_PROTECTION", False)
        )
        self.blowout_loss_pct: float = float(config.get("BLOWOUT_LOSS_PCT", 50.0))
        self.use_min_balance_stop: bool = bool(config.get("USE_MIN_BALANCE_STOP", True))
        self.min_balance_threshold: float = float(
            config.get("MIN_BALANCE_THRESHOLD", 200.0)
        )
        self.trading_enabled: bool = True
        self.blowout_activations = 0
        self.insufficient_funds_attempts = 0

        # Stops level constraints
        self.stops_level_pips: float = float(config.get("TRADE_STOPS_LEVEL_PIPS", 0.0))
        self.enforce_open: bool = bool(config.get("ENFORCE_STOPS_LEVEL_ON_OPEN", True))
        self.enforce_modify: bool = bool(
            config.get("ENFORCE_STOPS_LEVEL_ON_MODIFY", True)
        )

        # Slippage
        self.use_slippage: bool = bool(config.get("USE_SLIPPAGE", False))
        self.min_slip_pips: float = float(config.get("MIN_SLIPPAGE_PIPS", 0.0))
        self.max_slip_pips: float = float(config.get("MAX_SLIPPAGE_PIPS", 0.0))

        # Swap rollover config (server time)
        self.swap_rollover_hour: int = int(config.get("SWAP_ROLLOVER_HOUR", 0))
        self.triple_swap_weekday: int = int(config.get("TRIPLE_SWAP_WEEKDAY", 2))
        self.apply_triple_swap: bool = bool(config.get("APPLY_TRIPLE_SWAP", True))

        # BE/Trail (all in pips)
        self.use_trailing: bool = bool(config.get("USE_TRAILING_STOP", True))
        self.trail_dist_pips: float = float(
            config.get("TRAILING_STOP_DISTANCE_PIPS", 10)
        )
        self.use_be: bool = bool(config.get("USE_BREAK_EVEN_STOP", True))
        self.be_trigger_pips: float = float(config.get("BE_TRIGGER_PIPS", 20))
        self.be_offset_pips: float = float(config.get("BE_OFFSET_PIPS", 10))

        # Strategy handle (optional)
        self.strategy = None

        # Internal swap tracking
        self._last_swap_date_applied: Optional[pd.Timestamp] = None
        self.swap_history: List[Dict[str, Any]] = []

    def set_strategy(self, strategy):
        self.strategy = strategy
        strategy.set_backtester(self)

    def _round_volume(self, lots: float) -> float:
        step = self.volume_step
        lots = max(self.volume_min, min(self.max_lot, round(lots / step) * step))
        return float(lots)

    def _risk_cash_for(
        self, *, side: str, entry: float, sl: Optional[float], lots: float
    ) -> float:
        if sl is None:
            return 0.0
        if side == "buy":
            risk_pips = max(0.0, (entry - sl) / self.pip_size)
        else:
            risk_pips = max(0.0, (sl - entry) / self.pip_size)
        return float(risk_pips * self.value_per_pip_per_lot * lots)

    def _calc_lot_size(
        self, sl_price: Optional[float], entry_price: float, *, equity_ctx: float
    ) -> float:
        if not self.use_compounding:
            return self._round_volume(self.fixed_lot)
        if sl_price is None or sl_price == entry_price:
            return self._round_volume(self.fixed_lot)

        base = equity_ctx if self.use_equity_for_risk else self.balance
        risk_money = base * (self.risk_pct / 100.0)

        risk_pips = abs(entry_price - sl_price) / self.pip_size
        risk_pips = max(risk_pips, self.min_sizing_pips)

        if risk_pips <= 0:
            return self._round_volume(self.fixed_lot)

        lots = risk_money / (risk_pips * self.value_per_pip_per_lot)
        return self._round_volume(lots)

    def open_trade(
        self,
        side: str,
        entry_price: float,
        sl: Optional[float],
        tp: Optional[float],
        time: Any,
        lot_size: Optional[float] = None,
    ):
        if self.use_min_balance_stop:
            if self.balance < self.min_balance_threshold:
                if self.trading_enabled:
                    logger.warning(
                        f"MIN BALANCE REACHED at {time}: Balance ${self.balance:.2f} is below threshold. Halting new trades."
                    )
                    self.trading_enabled = False
                self.insufficient_funds_attempts += 1
                return
            elif not self.trading_enabled:
                logger.info(
                    f"TRADING RE-ENABLED at {time}: Balance ${self.balance:.2f} is back above threshold."
                )
                self.trading_enabled = True

        if not self.trading_enabled:
            return
        if len(self.open_trades) >= self.max_concurrent_trades:
            return
        if self.enforce_open and sl is not None and tp is not None:
            if not self._stops_distance_ok(entry_price, sl, tp):
                return

        open_pnl = sum(
            self._pnl_cash(t.side, t.lot_size, t.entry_price, entry_price)
            for t in self.open_trades
        )
        current_equity = self.balance + open_pnl
        used_margin = sum(t.lot_size for t in self.open_trades) * self.margin_per_lot
        free_margin = max(0.0, current_equity - used_margin)

        # 1) Base risk sizing (now using real-time equity)
        base_lots = (
            lot_size
            if lot_size is not None
            else self._calc_lot_size(sl, entry_price, equity_ctx=current_equity)
        )
        lots = base_lots
        cap_applied = False
        resized_for_margin = False

        # 2) Total exposure cap
        if self.max_risk_pct_total > 0 and sl is not None:
            already_at_risk = sum(
                self._risk_cash_for(
                    side=t.side, entry=t.entry_price, sl=t.sl, lots=t.lot_size
                )
                for t in self.open_trades
            )
            proposed_risk = self._risk_cash_for(
                side=side, entry=entry_price, sl=sl, lots=lots
            )
            cap_cash = current_equity * (self.max_risk_pct_total / 100.0)
            if already_at_risk + proposed_risk > cap_cash and proposed_risk > 0:
                remaining = max(0.0, cap_cash - already_at_risk)
                lots = self._round_volume(lots * (remaining / proposed_risk))

        # 3) Margin-aware resizing
        needed_margin = lots * self.margin_per_lot
        if needed_margin > free_margin:
            if self.resize_to_free_margin and self.margin_per_lot > 0:
                lots = self._round_volume(free_margin / self.margin_per_lot)
                needed_margin = lots * self.margin_per_lot
            if needed_margin > free_margin or lots < self.volume_min:
                self.insufficient_funds_attempts += 1
                return

        # Audit
        already_at_risk = sum(
            self._risk_cash_for(
                side=t.side, entry=t.entry_price, sl=t.sl, lots=t.lot_size
            )
            for t in self.open_trades
        )
        sl_pips = None if sl is None else abs(entry_price - sl) / self.pip_size
        risk_cash = (
            0.0 if sl_pips is None else sl_pips * self.value_per_pip_per_lot * lots
        )
        # Execute
        trade_id = len(self.trade_history) + len(self.open_trades) + 1
        fill = self._apply_entry_spread_slip(side, entry_price)
        self._record_risk_audit(
            {
                "time": time,
                "side": side,
                "event": "entry",
                "trade_id": trade_id,
                "entry_price_req": entry_price,
                "entry_price_fill": fill,
                "equity": current_equity,
                "entry_time": time,
                "free_margin": free_margin,
                "base_lots": base_lots,
                "lots": lots,
                "cap_applied": cap_applied,
                "resized_for_margin": resized_for_margin,
                "sl": sl,
                "sl_pips": sl_pips,
                "risk_cash": risk_cash,
                "risk_pct": (
                    (risk_cash / current_equity * 100.0) if current_equity > 0 else None
                ),
                "total_risk_cash_after": already_at_risk + risk_cash,
                "total_risk_pct_after": (
                    ((already_at_risk + risk_cash) / current_equity * 100.0)
                    if current_equity > 0
                    else None
                ),
            }
        )
        trade = Trade(
            id=trade_id,
            symbol=self.symbol,
            side=side,
            lot_size=lots,
            entry_price=fill,
            sl=sl,
            tp=tp,
            entry_time=time,
            initial_sl=sl,
        )
        trade.highest_price = fill
        trade.lowest_price = fill
        self.balance -= self._commission_cost(lots) / 2.0
        self.open_trades.append(trade)
        self._log_concurrency()

    def modify_trade(
        self, trade: Trade, *, sl: Optional[float] = None, tp: Optional[float] = None
    ):
        if sl is None and tp is None:
            return
        new_sl = trade.sl if sl is None else sl
        new_tp = trade.tp if tp is None else tp
        if self.enforce_modify and new_sl is not None and new_tp is not None:
            if not self._stops_distance_ok(trade.entry_price, new_sl, new_tp):
                return
        trade.sl = new_sl
        trade.tp = new_tp

    def close_trade(self, trade: Trade, reason: str, price: float, time: Any):
        if trade not in self.open_trades:
            return
        fill = self._apply_exit_spread(trade.side, price)
        pnl = self._pnl_cash(trade.side, trade.lot_size, trade.entry_price, fill)
        pnl -= self._commission_cost(trade.lot_size) / 2.0
        pnl += trade.swap_accrued
        # Refine reason
        reason_adj = reason
        if reason == "Stop Loss":
            if trade.ts_activated:
                reason_adj = "Trailing Stop"
            elif trade.be_activated and trade.sl is not None:
                if (trade.side == "buy" and trade.sl >= trade.entry_price) or (
                    trade.side == "sell" and trade.sl <= trade.entry_price
                ):
                    reason_adj = "Break-Even"

        trade.exit_price, trade.exit_time, trade.exit_reason, trade.pnl = (
            fill,
            time,
            reason_adj,
            pnl,
        )
        self.balance += pnl
        self.open_trades.remove(trade)
        self.trade_history.append(
            {
                "id": trade.id,
                "symbol": trade.symbol,
                "side": trade.side,
                "lot_size": trade.lot_size,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "exit_reason": trade.exit_reason,
                "swap_accrued": trade.swap_accrued,
                "pnl": trade.pnl,
                "be_activated": trade.be_activated,
                "ts_activated": trade.ts_activated,
            }
        )

        self._record_risk_audit(
            {
                "event": "exit",
                "time": time,
                "trade_id": trade.id,
                "entry_time": trade.entry_time,
                "exit_time": time,
                "entry_price_fill": trade.entry_price,
                "exit_price": fill,
                "exit_reason": trade.exit_reason,
                "swap_accrued": trade.swap_accrued,
                "pnl": pnl,
            }
        )
        self._update_streaks(pnl, time)
        self._log_concurrency()

    # ---------- Per-bar processing ----------
    def on_bar(self, row: pd.Series):
        price_high = (
            float(row["high"]) if "high" in row else float(row["close"])
        )  # safety
        price_low = float(row["low"]) if "low" in row else float(row["close"])  # safety
        price_close = float(row["close"])
        time = row.get("time")
        t = pd.to_datetime(time)  # type: ignore

        # Apply daily swap at rollover hour once per server day
        if t.hour == self.swap_rollover_hour and (
            self._last_swap_date_applied is None
            or t.normalize() != self._last_swap_date_applied.normalize()
        ):
            self._apply_daily_swaps(t)
            self._last_swap_date_applied = t

        for trade in list(self.open_trades):
            trade.highest_price = max(trade.highest_price, price_high)
            trade.lowest_price = min(trade.lowest_price, price_low)

            if self.use_be and trade.sl is not None:
                if trade.side == "buy":
                    gain_pips = (price_high - trade.entry_price) / self.pip_size
                    if gain_pips >= self.be_trigger_pips:
                        be_price = (
                            trade.entry_price + self.be_offset_pips * self.pip_size
                        )
                        if be_price > (trade.sl or -math.inf):
                            if not trade.be_activated:
                                trade.be_activated = True
                            self.modify_trade(trade, sl=be_price)
                else:
                    gain_pips = (trade.entry_price - price_low) / self.pip_size
                    if gain_pips >= self.be_trigger_pips:
                        be_price = (
                            trade.entry_price - self.be_offset_pips * self.pip_size
                        )
                        if be_price < (trade.sl or math.inf):
                            if not trade.be_activated:
                                trade.be_activated = True
                            self.modify_trade(trade, sl=be_price)
            # Trailing stop
            if self.use_trailing:
                dist = self.trail_dist_pips * self.pip_size
                if trade.side == "buy":
                    gain_pips = (
                        trade.highest_price - trade.entry_price
                    ) / self.pip_size
                    if gain_pips >= self.trail_dist_pips:
                        trail_sl = trade.highest_price - dist
                        # --- MODIFIED: Never worsen the SL ---
                        base_sl = trade.sl if trade.sl is not None else -math.inf
                        if trade.initial_sl is not None:
                            base_sl = max(base_sl, trade.initial_sl)
                        new_sl = max(trail_sl, base_sl)
                        if new_sl > (trade.sl or -math.inf):
                            if not trade.ts_activated:
                                trade.ts_activated = True
                            self.modify_trade(trade, sl=new_sl)
                else:  # Sell
                    gain_pips = (trade.entry_price - trade.lowest_price) / self.pip_size
                    if gain_pips >= self.trail_dist_pips:
                        trail_sl = trade.lowest_price + dist
                        # --- MODIFIED: Never worsen the SL ---
                        base_sl = trade.sl if trade.sl is not None else math.inf
                        if trade.initial_sl is not None:
                            base_sl = min(base_sl, trade.initial_sl)
                        new_sl = min(trail_sl, base_sl)
                        if new_sl < (trade.sl or math.inf):
                            if not trade.ts_activated:
                                trade.ts_activated = True
                            self.modify_trade(trade, sl=new_sl)
            # SL/TP checks
            if trade.side == "buy":
                hit_sl = trade.sl is not None and price_low <= trade.sl
                hit_tp = trade.tp is not None and price_high >= trade.tp
            else:
                hit_sl = trade.sl is not None and price_high >= trade.sl
                hit_tp = trade.tp is not None and price_low <= trade.tp

            if hit_sl and hit_tp:
                # Handle this specific case (e.g., prioritize SL)
                self.close_trade(trade, "Stop Loss", trade.sl, time)  # type: ignore
            elif hit_sl:
                self.close_trade(trade, "Stop Loss", trade.sl, time)  # type: ignore
            elif hit_tp:
                self.close_trade(trade, "Take Profit", trade.tp, time)  # type: ignore
            else:
                continue

        # Update equity curve
        open_pnl = sum(
            self._pnl_cash(t.side, t.lot_size, t.entry_price, price_close)
            for t in self.open_trades
        )
        self.equity = self.balance + open_pnl
        self.equity_curve.append({"time": time, "equity": self.equity})

        # Margin stop-out
        self._check_margin_stopout(time, price_close)

    # ---------- Costs, PnL, constraints ----------
    def _apply_daily_swaps(self, t: pd.Timestamp):
        if not self.use_swap or not self.open_trades:
            return
        triple = self.apply_triple_swap and (t.weekday() == self.triple_swap_weekday)
        multiplier = 3 if triple else 1
        for trade in self.open_trades:
            points = self.swap_long_pts if trade.side == "buy" else self.swap_short_pts
            # EURUSD: 1 pip = 10 points
            pips = points / 10.0
            cash = trade.lot_size * self.value_per_pip_per_lot * pips * multiplier
            trade.swap_accrued += cash
            self.balance += cash
            self.swap_history.append(
                {
                    "time": t,
                    "trade_id": trade.id,
                    "side": trade.side,
                    "lots": trade.lot_size,
                    "points": points,
                    "multiplier": multiplier,
                    "cash": cash,
                }
            )

    def _pnl_cash(self, side: str, lots: float, entry: float, exit: float) -> float:
        diff = (exit - entry) if side == "buy" else (entry - exit)
        pips = diff / self.pip_size
        return float(pips * self.value_per_pip_per_lot * lots)

    def _commission_cost(self, lots: float) -> float:
        return float(self.commission_rt * lots)

    def _apply_entry_spread_slip(self, side: str, price: float) -> float:
        fill = price
        if self.use_spread:
            adj = self.spread_pips * self.pip_size
            fill = price + adj if side == "buy" else price - adj
        if self.use_slippage and self.max_slip_pips > 0:
            slip_pips = random.uniform(self.min_slip_pips, self.max_slip_pips)
            slip = slip_pips * self.pip_size
            fill = fill + slip if side == "buy" else fill - slip
        return float(fill)

    def _apply_exit_spread(self, side: str, price: float) -> float:
        if not self.use_spread:
            return float(price)
        adj = self.spread_pips * self.pip_size
        return float(price - adj) if side == "buy" else float(price + adj)

    def _stops_distance_ok(self, entry: float, sl: float, tp: float) -> bool:
        min_dist = self.stops_level_pips * self.pip_size
        return (abs(entry - sl) >= min_dist) and (abs(entry - tp) >= min_dist)

    def _check_margin_stopout(self, time: Any, ref_price: float):
        if self.margin_per_lot <= 0 or not self.open_trades:
            return
        used_margin = sum(t.lot_size for t in self.open_trades) * self.margin_per_lot
        if used_margin <= 0:
            return
        equity_pct_of_margin = (self.equity / used_margin) * 100.0
        if equity_pct_of_margin < self.stop_out_level_pct:
            for t in sorted(
                self.open_trades,
                key=lambda x: self._pnl_cash(
                    x.side, x.lot_size, x.entry_price, ref_price
                ),
            ):
                self.close_trade(t, "Margin Call", ref_price, time)
                used_margin = (
                    sum(tt.lot_size for tt in self.open_trades) * self.margin_per_lot
                )
                if used_margin == 0:
                    break
                equity_pct_of_margin = (self.equity / used_margin) * 100.0
                if equity_pct_of_margin >= self.stop_out_level_pct:
                    break

    def _log_concurrency(self):
        self.concurrent_trades_log.append(len(self.open_trades))

    def _record_risk_audit(self, d: Dict[str, Any]) -> None:
        """Append a normalized audit row. Safe for CSV export."""
        row = {
            "event": (
                (d.get("event") or "").lower()
                if isinstance(d.get("event"), str)
                else d.get("event")
            ),
            "time": (
                pd.to_datetime(d.get("time")) if d.get("time") is not None else None  # type: ignore
            ),
            "trade_id": int(d.get("trade_id", d.get("trade_id_preview", 0) or 0)),
            "side": d.get("side"),
            "entry_time": (
                pd.to_datetime(d.get("entry_time"))  # type: ignore
                if d.get("entry_time") is not None
                else None
            ),
            "exit_time": (
                pd.to_datetime(d.get("exit_time"))  # type: ignore
                if d.get("exit_time") is not None
                else None
            ),
            "entry_price_req": (
                float(d.get("entry_price_req", float("nan")))
                if d.get("entry_price_req") is not None
                else None
            ),
            "entry_price_fill": (
                float(d.get("entry_price_fill", float("nan")))
                if d.get("entry_price_fill") is not None
                else None
            ),
            "exit_price": (
                float(d.get("exit_price", float("nan")))
                if d.get("exit_price") is not None
                else None
            ),
            "exit_reason": d.get("exit_reason"),
            "equity": (
                float(d.get("equity", float("nan")))
                if d.get("equity") is not None
                else None
            ),
            "free_margin": (
                float(d.get("free_margin", float("nan")))
                if d.get("free_margin") is not None
                else None
            ),
            "base_lots": (
                float(d.get("base_lots", float("nan")))
                if d.get("base_lots") is not None
                else None
            ),
            "lots": (
                float(d.get("lots", float("nan")))
                if d.get("lots") is not None
                else None
            ),
            "cap_applied": bool(d.get("cap_applied", False)),
            "resized_for_margin": bool(d.get("resized_for_margin", False)),
            "sl": float(d.get("sl", float("nan"))) if d.get("sl") is not None else None,
            "sl_pips": (
                float(d.get("sl_pips", float("nan")))
                if d.get("sl_pips") is not None
                else None
            ),
            "risk_cash": (
                float(d.get("risk_cash", float("nan")))
                if d.get("risk_cash") is not None
                else None
            ),
            "risk_pct": (
                float(d.get("risk_pct", float("nan")))
                if d.get("risk_pct") is not None
                else None
            ),
            "total_risk_cash_after": (
                float(d.get("total_risk_cash_after", float("nan")))
                if d.get("total_risk_cash_after") is not None
                else None
            ),
            "total_risk_pct_after": (
                float(d.get("total_risk_pct_after", float("nan")))
                if d.get("total_risk_pct_after") is not None
                else None
            ),
            "swap_accrued": (
                float(d.get("swap_accrued", float("nan")))
                if d.get("swap_accrued") is not None
                else None
            ),
            "pnl": (
                float(d.get("pnl", float("nan"))) if d.get("pnl") is not None else None
            ),
        }
        # --- FIX: Changed `d` to `row` to match the created dictionary ---
        self.risk_audit.append(row)

    def _update_streaks(self, pnl: float, time: Any):
        if pnl >= 0:
            self._cur_win_streak += 1
            self._cur_win_sum += pnl
            self._cur_loss_streak = 0
            self._cur_loss_sum = 0.0
        else:
            self._cur_loss_streak += 1
            self._cur_loss_sum += pnl
            self._cur_win_streak = 0
            self._cur_win_sum = 0.0

            blowout_threshold = self.balance * (self.blowout_loss_pct / 100.0)
            if (
                self.use_blowout_protection
                and abs(self._cur_loss_sum) >= blowout_threshold
            ):
                if self.trading_enabled:
                    logger.warning(
                        f"BLOWOUT TRIGGERED at {time}: Loss streak of ${abs(self._cur_loss_sum):.2f} exceeded {self.blowout_loss_pct}% of balance. Halting new trades."
                    )
                    self.trading_enabled = False
                    self.blowout_activations += 1

        self.max_consecutive_wins = max(self.max_consecutive_wins, self._cur_win_streak)
        self.max_consecutive_win_amount = max(
            self.max_consecutive_win_amount, self._cur_win_sum
        )
        self.max_consecutive_losses = max(
            self.max_consecutive_losses, self._cur_loss_streak
        )
        self.max_consecutive_loss_amount = min(
            self.max_consecutive_loss_amount, self._cur_loss_sum
        )
