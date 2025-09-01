from __future__ import annotations
from typing import Any, Optional, Deque
from collections import deque
import math
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE, Trade


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class EmaBurst150(BaseStrategy):
    """
    EMA momentum-burst strategy, designed for streaming on_bar execution.

    Long
      - f > m > s and close > e150
      - Slopes up: df >= FAST_MIN_SLOPE, dm >= MID_MIN_SLOPE, ds >= SLOW_MIN_SLOPE
      - Cluster tight: max(f,m,s) - min(f,m,s) <= CLUSTER_MAX_PIPS * PIP_SIZE (set <=0 to disable)
      - Trigger: recent cross of f over e150 within X bars with nearly-flat e150,
                 OR volume spike (configurable via TRIGGER_MODE)

    Short is symmetric.
    """

    def __init__(
        self,
        symbol: str,
        config: dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 3315,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # EMAs
        self.fast = int(config.get("FAST_EMA", 14))
        self.mid = int(config.get("MID_EMA", 30))
        self.slow = int(config.get("SLOW_EMA", 50))
        self.trend = int(config.get("TREND_EMA", 150))

        # Slope and cluster
        self.cluster_max_pips = float(
            config.get("CLUSTER_MAX_PIPS", 2.0)
        )  # <=0 disables
        self.fast_min_slope = float(config.get("FAST_MIN_SLOPE", 0.00010))
        self.mid_min_slope = float(config.get("MID_MIN_SLOPE", 0.00005))
        self.slow_min_slope = float(config.get("SLOW_MIN_SLOPE", 0.00003))
        self.trend_flat_max = float(config.get("TREND_FLAT_MAX", 0.00001))
        # Optional EMA spread gate (set 0 to disable)
        self.min_ema_spread_pips = float(config.get("MIN_EMA_SPREAD_PIPS", 0.0))

        # Trigger combiner: "or" | "and" | "cross_only" | "vol_only"
        self.trigger_mode = str(config.get("TRIGGER_MODE", "or")).lower()

        # Optional staggered opposite entry (disabled by default)
        self.dual_stagger_min: Optional[int] = config.get("DUAL_STAGGER_MIN", None)
        self._pending_opposite: Optional[dict] = None

        # Cross logic
        self.recent_cross_lookback = int(config.get("RECENT_CROSS_LOOKBACK", 6))

        # Volume confirmations (Better Volume style or SMA spike)
        self.use_volume = bool(config.get("USE_VOLUME", True))
        self.vol_lookback = int(config.get("VOL_LOOKBACK", 20))
        self.vol_high_mult = float(config.get("VOL_HIGH_MULT", 1.8))

        self.cooldown_bars = int(config.get("COOLDOWN_BARS", 2))
        self.confirm_bars = int(config.get("CONFIRM_BARS", 1))

        self.magic = magic
        self.prev_row: Optional[pd.Series] = None
        self.cooldown = 0
        self._confirm_side: Optional[str] = None
        self._confirm_count = 0

        # Rolling buffers for recent-cross and volume SMA when feature not present
        self._buf_fast: Deque[float] = deque(maxlen=self.recent_cross_lookback + 2)
        self._buf_trend: Deque[float] = deque(maxlen=self.recent_cross_lookback + 2)
        self._buf_vol: Deque[float] = deque(maxlen=max(self.vol_lookback, 5) + 2)

    # -------- utils --------
    def _open_count_for_me(self, broker) -> int:
        return sum(
            getattr(tr, "strategy_id", None) == self.name for tr in broker.open_trades
        )

    def _ema_vals(self, s: pd.Series) -> tuple[float, float, float, float]:
        f = float(s.get(f"ema_{self.fast}"))
        m = float(s.get(f"ema_{self.mid}"))
        sl = float(s.get(f"ema_{self.slow}"))
        e150 = float(s.get(f"ema_{self.trend}"))
        return f, m, sl, e150

    def _slopes(
        self, prev: pd.Series, cur: pd.Series
    ) -> tuple[float, float, float, float]:
        pf, pm, psl, pe = self._ema_vals(prev)
        f, m, sl, e = self._ema_vals(cur)
        return f - pf, m - pm, sl - psl, e - pe

    @staticmethod
    def _cluster_tight(f: float, m: float, sl: float, max_pips: float) -> bool:
        if max_pips is None or max_pips <= 0:
            return True
        spread = max(f, m, sl) - min(f, m, sl)
        return spread <= max_pips * PIP_SIZE

    def _spread_ok(self, f: float, m: float, sl: float) -> bool:
        if self.min_ema_spread_pips <= 0:
            return True
        d1 = abs(f - m) / PIP_SIZE
        d2 = abs(m - sl) / PIP_SIZE
        return (d1 >= self.min_ema_spread_pips) and (d2 >= self.min_ema_spread_pips)

    def _recent_cross_flat_trend(self, side: str) -> bool:
        # Requires at least 2 samples
        if len(self._buf_fast) < 2 or len(self._buf_trend) < 2:
            return False
        f = list(self._buf_fast)
        e = list(self._buf_trend)
        n = len(f)
        # average slope magnitude of trend over buffer
        avg_slope = (e[-1] - e[0]) / max(n - 1, 1)
        if abs(avg_slope) > self.trend_flat_max:
            return False
        # search a recent cross within window
        rng = range(max(1, n - self.recent_cross_lookback), n)
        if side == "buy":
            return any(f[i - 1] <= e[i - 1] and f[i] > e[i] for i in rng)
        else:
            return any(f[i - 1] >= e[i - 1] and f[i] < e[i] for i in rng)

    def _volume_spike(self, row: pd.Series) -> bool:
        if not self.use_volume:
            return False
        # try common volume fields
        v = row.get("tick_volume")
        if v is None:
            v = row.get("Volume")
        if v is None:
            v = row.get("volume")
        if v is None:
            return False
        v = float(v)
        # prefer precomputed vol_sma_{lookback}
        col = f"vol_sma_{self.vol_lookback}"
        if col in row:
            v_sma = float(row[col])
        else:
            # fallback: rolling mean from buffer
            self._buf_vol.append(v)
            if len(self._buf_vol) < self.vol_lookback:
                return False
            v_sma = sum(list(self._buf_vol)[-self.vol_lookback :]) / float(
                self.vol_lookback
            )
        if v_sma <= 0:
            return False
        return v >= self.vol_high_mult * v_sma

    @staticmethod
    def _fav_move_pips(tr: Trade, row: pd.Series) -> float:
        hi = float(row.get("high", row.get("High", float("nan"))))
        lo = float(row.get("low", row.get("Low", float("nan"))))
        return (
            (hi - tr.entry_price) / PIP_SIZE
            if tr.side == "buy"
            else (tr.entry_price - lo) / PIP_SIZE
        )

    def _manage(self, broker, row: pd.Series):
        """Management delegated to broker. Only arm BE/trailing if configured.
        Avoids referencing per-strategy attributes that may be removed.
        """
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) != self.name:
                continue
            fav = self._fav_move_pips(tr, row)

            # Break-even: use per-trade config if present, else optional global
            be_thr = getattr(tr, "be_trigger_pips", None)
            be_off = getattr(tr, "be_offset_pips", 0.0)
            if be_thr is None:
                be_thr = self.config.get("BE_TRIGGER_PIPS")  # optional
                be_off = self.config.get("BE_OFFSET_PIPS", 0.0)
            if be_thr and not getattr(tr, "be_applied", False) and fav >= be_thr:
                broker.set_break_even(
                    trade_id=tr.id, be_pips=float(be_thr), offset_pips=float(be_off)
                )
                tr.be_applied = True

            # Trailing: arm distance from config if requested; broker moves it
            if getattr(tr, "trailing_sl_distance", None) is None and self.config.get(
                "USE_TRAILING_STOP"
            ):
                dist = self.config.get("TRAILING_STOP_DISTANCE_PIPS")
                if dist:
                    tr.trailing_sl_distance = float(dist)

    # -------- main hook --------
    def on_bar(self, broker, t, row: pd.Series):
        # manage open trades first
        self._manage(broker, row)

        # Optional dual-order: try to execute any pending opposite order
        if self.dual_stagger_min and self._pending_opposite is not None:
            sched_t = self._pending_opposite.get("time")
            opp_side = self._pending_opposite.get("side")
            if sched_t is not None and t >= sched_t:
                if self._open_count_for_me(
                    broker
                ) < self.strat_cfg.max_concurrent_trades and (
                    not self.governor or self.governor.allow_new_trade(self.name).ok
                ):
                    sl_pips = float(self.config.get("SL_PIPS", 0))
                    tp_pips = float(self.config.get("TP_PIPS", 0))
                    if sl_pips > 0 and tp_pips > 0:
                        price = float(row["close"])
                        req = SizeRequest(
                            balance=broker.balance,
                            sl_pips=sl_pips,
                            value_per_pip=self._value_per_pip_1lot(broker),
                        )
                        lots = _round_to_step(
                            self.sizer.size(req).lots,
                            broker.cfg.VOLUME_STEP,
                            broker.cfg.VOLUME_MIN,
                            9999.0,
                        )
                        if lots > 0:
                            broker.open_trade(
                                side=opp_side,
                                price=price,
                                wanted_lots=lots,
                                sl_pips=sl_pips,
                                tp_pips=tp_pips,
                                t=t,
                                strategy_id=self.name,
                                magic=self.magic,
                            )
                # clear regardless to avoid repeated attempts
                self._pending_opposite = None

        # cooldown handling
        if self.cooldown > 0:
            self.cooldown -= 1

        # risk limits
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            # still update buffers for future logic
            self._update_buffers(row)
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            self._update_buffers(row)
            return None

        if self.prev_row is None:
            self.prev_row = row
            self._update_buffers(row)
            return None

        try:
            pf, pm, psl, pe = self._ema_vals(self.prev_row)
            f, m, sl, e150 = self._ema_vals(row)
            close = float(row["close"])
        except Exception:
            self.prev_row = row
            self._update_buffers(row)
            return None

        # push buffers for cross/volume logic
        self._update_buffers(row, push_fast=f, push_trend=e150)

        # raw side based on alignment + regime
        raw_side = None
        if f > m > sl and close > e150:
            raw_side = "buy"
        elif f < m < sl and close < e150:
            raw_side = "sell"

        # confirmation logic across consecutive bars
        if self.confirm_bars > 0:
            if raw_side is None:
                self._confirm_side, self._confirm_count = None, 0
            elif raw_side == self._confirm_side:
                self._confirm_count += 1
            else:
                self._confirm_side, self._confirm_count = raw_side, 1
        else:
            self._confirm_side = raw_side
            self._confirm_count = 1 if raw_side else 0

        ready = (
            self._confirm_side is not None
            and self._confirm_count >= max(1, self.confirm_bars)
            and self.cooldown == 0
        )

        side = self._confirm_side if ready else None
        if side is None:
            self.prev_row = row
            return None

        # slopes
        df, dm, dsl, de = self._slopes(self.prev_row, row)
        if side == "buy":
            slope_ok = (
                (df >= self.fast_min_slope)
                and (dm >= self.mid_min_slope)
                and (dsl >= self.slow_min_slope)
            )
        else:
            slope_ok = (
                (df <= -self.fast_min_slope)
                and (dm <= -self.mid_min_slope)
                and (dsl <= -self.slow_min_slope)
            )

        cluster_ok = self._cluster_tight(f, m, sl, self.cluster_max_pips)
        spread_ok = self._spread_ok(f, m, sl)

        # triggers: recent cross with flat trend vs volume spike
        cross_ok = self._recent_cross_flat_trend(side)
        vol_ok = self._volume_spike(row)
        if self.trigger_mode == "and":
            trigger_ok = cross_ok and vol_ok
        elif self.trigger_mode == "cross_only":
            trigger_ok = cross_ok
        elif self.trigger_mode == "vol_only":
            trigger_ok = vol_ok
        else:  # default OR
            trigger_ok = cross_ok or vol_ok

        if slope_ok and cluster_ok and spread_ok and trigger_ok:
            # Pull SL/TP from config each time to avoid missing attributes
            sl_pips = float(self.config.get("SL_PIPS", 0))
            tp_pips = float(self.config.get("TP_PIPS", 0))
            if sl_pips <= 0 or tp_pips <= 0:
                self.prev_row = row
                return None

            req = SizeRequest(
                balance=broker.balance,
                sl_pips=sl_pips,
                value_per_pip=self._value_per_pip_1lot(broker),
            )
            lots = _round_to_step(
                self.sizer.size(req).lots,
                broker.cfg.VOLUME_STEP,
                broker.cfg.VOLUME_MIN,
                9999.0,
            )
            if lots > 0:
                trd = broker.open_trade(
                    side=side,
                    price=close,
                    wanted_lots=lots,
                    sl_pips=sl_pips,
                    tp_pips=tp_pips,
                    t=t,
                    strategy_id=self.name,
                    magic=self.magic,
                )
                # schedule opposite order only if enabled
                if self.dual_stagger_min:
                    try:
                        sched_t = t + pd.Timedelta(minutes=int(self.dual_stagger_min))
                    except Exception:
                        sched_t = None
                    if sched_t is not None:
                        opp = "sell" if side == "buy" else "buy"
                        self._pending_opposite = {"time": sched_t, "side": opp}

                self.prev_row = row
                self.cooldown = self.cooldown_bars
                self._confirm_side, self._confirm_count = None, 0
                return trd

        self.prev_row = row
        return None

    # ------ helpers ------
    def _update_buffers(
        self,
        row: pd.Series,
        push_fast: Optional[float] = None,
        push_trend: Optional[float] = None,
    ):
        if push_fast is None or push_trend is None:
            try:
                f, _, _, e150 = self._ema_vals(row)
                push_fast = f
                push_trend = e150
            except Exception:
                pass
        if push_fast is not None:
            self._buf_fast.append(push_fast)
        if push_trend is not None:
            self._buf_trend.append(push_trend)
        # volume handled inside _volume_spike via _buf_vol when needed


"""
Feature requirements:
  - ema_{FAST_EMA, MID_EMA, SLOW_EMA, TREND_EMA}
  - close, high, low
  - One of: tick_volume | Volume | volume
  - Optional: vol_sma_{VOL_LOOKBACK} (else computed on the fly)

Example wiring:
from backtester.strategies.ema_burst_150 import EmaBurst150
...
    STRATEGY_NAME = "EMA_BURST"
    feature_spec = {"ema": [14, 30, 50, 150], "vol_sma": [14]}

    cfg = BrokerConfig(**BACKTEST_CONFIG)
    broker = Broker(cfg)
    cfg_map = {
        STRATEGY_NAME: StrategyConfig(
            risk_mode=RiskMode.FIXED,
            risk_pct=0.1,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.1,
            max_drawdown_pct=0.3,
            max_concurrent_trades=5,
        ),
    }
    governor = RiskGovernor(cfg_map)
    strategies = [
        EmaBurst150(
            symbol=symbol,
            config={
                "name": STRATEGY_NAME,
                "FAST_EMA": 14,
                "MID_EMA": 30,
                "SLOW_EMA": 50,
                "TREND_EMA": 150,
                "CLUSTER_MAX_PIPS": 2.0,
                "FAST_MIN_SLOPE": 0.0001,
                "MID_MIN_SLOPE": 0.00005,
                "SLOW_MIN_SLOPE": 0.00003,
                "TREND_FLAT_MAX": 0.00001,
                "RECENT_CROSS_LOOKBACK": 20,
                "USE_VOLUME": True,
                "VOL_LOOKBACK": 14,
                "VOL_HIGH_MULT": 1.8,
                "COOLDOWN_BARS": 2,
                "CONFIRM_BARS": 1,
                ## Hedging
                "HEDGE_ENABLE": True,
                "HEDGE_DELAY_MIN": 1,
                "HEDGE_SIZE_FRACTION": 0.5,
                "HEDGE_ONLY_IF_ADVERSE_PIPS": 6,  # place only if we’re wrong fast
                "HEDGE_CANCEL_IF_FAVOR_PIPS": 4,  # cancel hedge if right quickly
                ## More stuff
                "TRIGGER_MODE": "or",  # or|and|cross_only|vol_only
                "MIN_EMA_SPREAD_PIPS": 0.0,  # start disabled
                ## Risk -> ( only use when baseline is good)
                "SL_PIPS": 10,
                "TP_PIPS": 50,
                "USE_BREAK_EVEN_STOP": False,
                "BE_TRIGGER_PIPS": 8,
                "BE_OFFSET_PIPS": 1,
                "USE_TRAILING_STOP": False,
                "TRAILING_STOP_DISTANCE_PIPS": 10,
                "USE_TP_EXTENSION": False,
                "NEAR_TP_BUFFER_PIPS": 2,
                "TP_EXTENSION_PIPS": 3,
            },
            strat_cfg=cfg_map[STRATEGY_NAME],  # or your own StrategyConfig
            governor=governor,
        )
    ]
"""
