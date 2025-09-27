from __future__ import annotations
from typing import Any, Optional, Dict, Set
import math
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest


def _mean(vals):
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _slope_norm(cur: float, prev: float, denom: float) -> float:
    # Normalize by ATR (or fallback denom) then map arctan to 0.1 where 0.5 is flat
    eps = 1e-9
    d = (cur - prev) / max(abs(denom), eps)
    return 0.5 + math.atan(d) / math.pi


class ThreeStageCrossover(BaseStrategy):
    """
    Implements staged long/short entry logic.
    See docstring for detailed rules.
    """

    def __init__(
        self,
        symbol: str,
        config: dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 9010,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # Packs
        self.fast_pack: Set[str] = set(
            config.get("FAST_PACK", ["sma_high_20", "sma_low_20"])
        )
        self.slow_pack: Set[str] = set(
            config.get("SLOW_PACK", ["ema_60", "ema_90", "ema_120"])
        )

        # Columns
        self.keltner_mid = config.get("KELTNER_MID", "kc_20_20_2.0_mid")
        self.keltner_upper = config.get("KELTNER_UP", "kc_20_20_2.0_upper")
        self.keltner_lower = config.get("KELTNER_LO", "kc_20_20_2.0_lower")
        self.rsi_col = config.get("RSI_COL", "rsi_14")
        self.stoch_k_col = config.get("STOCH_K_COL", "stoch_14_3_3_k")
        self.atr_col = config.get("ATR_COL", "atr_14")

        # Timing
        self.window_minutes = int(config.get("CONFIRM_WINDOW_MIN", 90))

        # Risk
        self.sl_short = float(config.get("SL_SHORT_PIPS", 15))
        self.tp_short = float(config.get("TP_SHORT_PIPS", 40))
        self.sl_long = float(config.get("SL_LONG_PIPS", 5))
        self.tp_long = float(config.get("TP_LONG_PIPS", 40))

        # Slope thresholds
        self.slope_up_thr = float(config.get("SLOPE_UP_THR", 0.6))
        self.slope_dn_thr = float(config.get("SLOPE_DN_THR", 0.4))

        # State
        self.prev_row: Optional[pd.Series] = None
        self.magic = magic
        self.pending: Optional[Dict[str, Any]] = None  # {side, stage, t0}

    def _get(self, row: pd.Series, col: str) -> float:
        return float(row[col])

    def _pack_level(self, row: pd.Series, cols: Set[str]) -> float:
        return _mean([row[c] for c in cols])

    def _pack_slope(
        self, row: pd.Series, prev: pd.Series, cols: Set[str], denom: float
    ) -> float:
        return _mean([_slope_norm(float(row[c]), float(prev[c]), denom) for c in cols])

    def _overall_slope(self, f_slope: float, s_slope: float, km_slope: float) -> float:
        return _mean([f_slope, s_slope, km_slope])

    def _km_slope(self, row: pd.Series, prev: pd.Series, denom: float) -> float:
        return _slope_norm(
            self._get(row, self.keltner_mid), self._get(prev, self.keltner_mid), denom
        )

    def _inside_short_trigger(self, row: pd.Series) -> bool:
        return self._get(row, "close") <= self._get(row, self.keltner_upper)

    # NEW: no-cross confirmation for long Stage-1
    def _long_stage1_state(self, row: pd.Series) -> bool:
        km_now = self._get(row, self.keltner_mid)
        e60_now = self._get(row, "ema_60")
        e90_now = self._get(row, "ema_90")
        e120_now = self._get(row, "ema_120")
        # Confirmation by LEVELS only (no crossover history):
        # KM above EMA60 AND EMA90 & EMA120 below EMA60
        return (km_now > e60_now) and (e90_now < e60_now) and (e120_now < e60_now)

    def _reset_pending(self, side: str, t0: pd.Timestamp):
        self.pending = {"side": side, "stage": 1, "t0": t0}

    def _maybe_timeout(self, now_ts: pd.Timestamp):
        if not self.pending:
            return False
        if (now_ts - self.pending["t0"]) > pd.Timedelta(minutes=self.window_minutes):
            self.pending = None
            return True
        return False

    def _fire(
        self,
        broker,
        t,
        row: pd.Series,
        side: str,
        sl_pips: float,
        tp_pips: Optional[float],
    ):
        price = float(row.get("close", row.get("Close")))
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        lots = self.sizer.size(req).lots
        if lots <= 0:
            return None
        tr = broker.open_trade(
            side=side,
            price=price,
            wanted_lots=lots,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            t=t,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)
        return tr

    def _has_open_side(self, broker, side: str) -> bool:
        return any(
            getattr(tr, "strategy_id", None) == self.name and tr.side == side
            for tr in broker.open_trades
        )

    def on_bar(self, broker, t, row: pd.Series):
        if self.prev_row is None:
            self.prev_row = row
            return None

        long_open = self._has_open_side(broker, "buy")  # noqa: F841
        short_open = self._has_open_side(broker, "sell")  # noqa: F841
        self._maybe_timeout(row.name)

        denom = float(row.get(self.atr_col, row.get("close", 1.0))) or 1e-6
        prev_denom = (
            float(self.prev_row.get(self.atr_col, self.prev_row.get("close", 1.0)))
            or 1e-6
        )
        denom = max(denom, prev_denom)

        f_slope = self._pack_slope(row, self.prev_row, self.fast_pack, denom)
        s_slope = self._pack_slope(row, self.prev_row, self.slow_pack, denom)
        km_slope = self._km_slope(row, self.prev_row, denom)
        overall = self._overall_slope(f_slope, s_slope, km_slope)

        km = self._get(row, self.keltner_mid)
        f_level = self._pack_level(row, self.fast_pack)
        s_level = self._pack_level(row, self.slow_pack)

        rsi = float(row.get(self.rsi_col, float("nan")))
        stoch = float(row.get(self.stoch_k_col, float("nan")))

        # Terminate trades
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) != self.name:
                continue
            if tr.side == "buy" and overall <= self.slope_dn_thr:
                broker.close_trade(tr, float(row["close"]), "Slope Termination", t)
            elif tr.side == "sell" and overall >= self.slope_up_thr:
                broker.close_trade(tr, float(row["close"]), "Slope Termination", t)

        # Stage-0 gates
        short_gate = (
            (self._get(row, "close") > self._get(row, self.keltner_upper))
            and (rsi >= 70)
            and (stoch >= 80)
        )
        long_gate = (
            (self._get(row, "close") < self._get(row, self.keltner_lower))
            and (rsi <= 30)
            and (stoch <= 20)
        )

        if short_gate:
            self._reset_pending("sell", row.name)
        if long_gate:
            self._reset_pending("buy", row.name)
        # allow pending signals even if opposite side is open (no mutual exclusion)

        if not self.pending:
            self.prev_row = row
            return None
        side = self.pending["side"]
        # do not cancel pending due to opposite open; allow hedging per user request

        if self.pending["stage"] == 1:
            if side == "sell":
                stage1_ok = self._inside_short_trigger(row) and (km_slope < 0.5)
            else:
                # Replaced crossover with level-based confirmation
                stage1_ok = self._long_stage1_state(row)
            if stage1_ok:
                self.pending["stage"] = 2

        if self.pending and self.pending["stage"] == 2:
            if side == "sell":
                stage2_ok = (f_slope < 0.5) and (s_slope <= 0.6)
            else:
                stage2_ok = (
                    (s_level > km)
                    and (f_level < km)
                    and (f_slope >= 0.5)
                    and (s_slope >= 0.5)
                )
            if stage2_ok:
                self.pending["stage"] = 3

        if self.pending and self.pending["stage"] == 3:
            if side == "sell":
                tp = None if (s_level > km and f_level < km) else self.tp_short
                tr = self._fire(broker, t, row, "sell", self.sl_short, tp)
                self.pending = None
                self.prev_row = row
                return tr
            else:
                tp = None if (f_level > km and s_level < km) else self.tp_long
                tr = self._fire(broker, t, row, "buy", self.sl_long, tp)
                self.pending = None
                self.prev_row = row
                return tr

        self.prev_row = row
        return None
