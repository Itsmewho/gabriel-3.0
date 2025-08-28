# strategies/ichimoku_trend.py
from __future__ import annotations
from typing import Any, Dict, Optional
import math
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE, Trade


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


class IchimokuTrendStrategy(BaseStrategy):
    """
    Modernized Ichimoku strategy with broker-compatible sizing and
    *delayed* management (BE & trailing only after +1R).

    Expects these columns in your features frame (names configurable via config):
      - tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span, close, high, low
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any],
        strat_cfg=None,
        governor=None,
        magic: Optional[int] = 3001,
    ):
        super().__init__(symbol, config, strat_cfg, governor=governor)

        # Column names (allow override)
        self.col_tenkan = config.get("TENKAN_COL", "ichimoku_tenkan")
        self.col_kijun = config.get("KIJUN_COL", "ichimoku_kijun")
        self.col_span_a = config.get("SPAN_A_COL", "ichimoku_senkou_a")
        self.col_span_b = config.get("SPAN_B_COL", "ichimoku_senkou_b")
        self.col_chikou = config.get("CHIKOU_COL", "ichimoku_chikou")
        self.col_close = config.get("CLOSE_COL", "close")
        self.col_high = config.get("HIGH_COL", "high")
        self.col_low = config.get("LOW_COL", "low")

        # Core risk params
        self.sl_pips = float(config.get("SL_PIPS", 10))
        self.tp_pips = float(config.get("TP_PIPS", 60))

        # Chikou confirmation: look back N bars to compare with prior close (classic is 26)
        self.chikou_lookback = int(config.get("CHIKOU_LOOKBACK", 26))

        # Delayed management
        self.be_trigger_extra = float(config.get("BE_TRIGGER_EXTRA_PIPS", 1))
        self.be_offset = float(config.get("BE_OFFSET_PIPS", 2))
        self.use_trailing = bool(config.get("USE_TRAILING_STOP", True))
        self.trail_distance_pips = float(config.get("TRAILING_STOP_DISTANCE_PIPS", 10))

        self.magic = magic
        self.prev_row: Optional[pd.Series] = None

    # ---------------- helpers ----------------
    def _open_count_for_me(self, broker) -> int:
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    @staticmethod
    def _favorable_move_pips(tr: Trade, row: pd.Series) -> float:
        high = float(row.get("high", row.get("High", float("nan"))))
        low = float(row.get("low", row.get("Low", float("nan"))))
        if tr.side == "buy":
            return (high - tr.entry_price) / PIP_SIZE
        else:
            return (tr.entry_price - low) / PIP_SIZE

    def _manage_open_positions(self, broker, row: pd.Series) -> None:
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) != self.name:
                continue

            fav = self._favorable_move_pips(tr, row)
            be_threshold = self.sl_pips + self.be_trigger_extra

            if not getattr(tr, "be_applied", False) and fav >= be_threshold:
                broker.set_break_even(
                    trade_id=tr.id,
                    be_pips=self.sl_pips + self.be_trigger_extra,
                    offset_pips=self.be_offset,
                )
                tr.be_applied = True

            if (
                self.use_trailing
                and getattr(tr, "trailing_sl_distance", None) is None
                and fav >= self.sl_pips
            ):
                tr.trailing_sl_distance = self.trail_distance_pips

    # ---------------- main hook ----------------
    def on_bar(self, broker, t, row: pd.Series):
        # Always manage open positions first
        self._manage_open_positions(broker, row)

        # Respect concurrency and governor
        if self._open_count_for_me(broker) >= self.strat_cfg.max_concurrent_trades:
            self.prev_row = row
            return None
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            self.prev_row = row
            return None

        if self.prev_row is None:
            self.prev_row = row
            return None

        # Pull fields safely
        try:
            prev_tenkan = float(self.prev_row[self.col_tenkan])  # type: ignore
            prev_kijun = float(self.prev_row[self.col_kijun])  # type: ignore
            tenkan = float(row[self.col_tenkan])  # type: ignore
            kijun = float(row[self.col_kijun])  # type: ignore
            span_a = float(row[self.col_span_a])  # type: ignore
            span_b = float(row[self.col_span_b])  # type: ignore
            chikou = float(row[self.col_chikou])  # type: ignore
            close = float(row[self.col_close])  # type: ignore
        except Exception:
            self.prev_row = row
            return None

        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)

        # Chikou vs price N bars back (if available)
        chikou_ok_bull = False
        chikou_ok_bear = False
        if self.chikou_lookback > 0 and isinstance(row.name, (pd.Timestamp,)):
            # We assume caller feeds rows in time order; use index-based lookback via stored previous rows if needed.
            try:
                # Row comes from DataFrame.itertuples/iterrows in broker loop: the underlying frame index is DatetimeIndex.
                # We'll rely on BaseStrategy keeping the same DataFrame row with index for lookback slice if accessible via self.data.
                # If self.data is not available, fall back to simple current-close comparison.
                src_df: Optional[pd.DataFrame] = getattr(
                    self, "_last_df", None
                )  # BaseStrategy may set this in your engine
                if src_df is not None and row.name in src_df.index:
                    pos = src_df.index.get_loc(row.name)
                    if isinstance(pos, int) and pos - self.chikou_lookback >= 0:
                        past_close = float(
                            src_df.iloc[pos - self.chikou_lookback][self.col_close]  # type: ignore
                        )
                        chikou_ok_bull = chikou > past_close
                        chikou_ok_bear = chikou < past_close
                else:
                    chikou_ok_bull = chikou > close
                    chikou_ok_bear = chikou < close
            except Exception:
                chikou_ok_bull = chikou > close
                chikou_ok_bear = chikou < close
        else:
            chikou_ok_bull = chikou > close
            chikou_ok_bear = chikou < close

        # Crosses
        bull_cross = prev_tenkan <= prev_kijun and tenkan > kijun
        bear_cross = prev_tenkan >= prev_kijun and tenkan < kijun
        price_above_cloud = close > cloud_top
        price_below_cloud = close < cloud_bot

        side: Optional[str] = None
        if bull_cross and price_above_cloud and chikou_ok_bull:
            side = "buy"
        elif bear_cross and price_below_cloud and chikou_ok_bear:
            side = "sell"

        if side:
            # Risk sizing
            req = SizeRequest(
                balance=broker.balance,
                sl_pips=self.sl_pips,
                value_per_pip=self._value_per_pip_1lot(broker),
            )
            sized = self.sizer.size(req)
            lots = _round_to_step(
                sized.lots, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
            )

            if lots > 0:
                tr = broker.open_trade(
                    side=side,
                    price=close,  # spread handled in cost engine
                    wanted_lots=lots,
                    sl_pips=self.sl_pips,
                    tp_pips=self.tp_pips,
                    t=t,
                    strategy_id=self.name,
                    magic=self.magic,
                )
                if tr:
                    # BE & trailing are armed later via _manage_open_positions once +1R
                    pass
                self.prev_row = row
                return tr

        self.prev_row = row
        return None


# Example wiring in your runner
"""
# Risk/strategy config
cfg_map = {
    "ICHIMOKU": StrategyConfig(
        risk_mode=RiskMode.FIXED,
        risk_pct=0.1,
        lot_min=cfg.VOLUME_MIN,
        lot_step=cfg.VOLUME_STEP,
        lot_max=100.0,
        max_risk_pct_per_trade=0.1,
        max_drawdown_pct=0.3,
        max_concurrent_trades=2,
    ),
}

governor = RiskGovernor(cfg_map)

strategies = [
    IchimokuTrendStrategy(
        symbol=SYMBOL,
        config={
            "name": "ICHIMOKU",
            # columns only if your names differ from defaults
            # "TENKAN_COL": "tenkan_sen", "KIJUN_COL": "kijun_sen",
            # "SPAN_A_COL": "senkou_span_a", "SPAN_B_COL": "senkou_span_b",
            # "CHIKOU_COL": "chikou_span",
            "SL_PIPS": 12,
            "TP_PIPS": 48,  # 4R baseline
            "CHIKOU_LOOKBACK": 26,
            "USE_TRAILING_STOP": True,
            "TRAILING_STOP_DISTANCE_PIPS": 12,
            "BE_TRIGGER_EXTRA_PIPS": 1,
            "BE_OFFSET_PIPS": 2,
        },
        strat_cfg=cfg_map["ICHIMOKU"],
        governor=governor,
    )
]

# feature_spec to draw lines in your charts (optional)
feature_spec = {"ichimoku": True}
"""
