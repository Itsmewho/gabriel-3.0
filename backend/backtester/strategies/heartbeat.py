# Drop-in replacement for your current MyOwnHeartbeat with strict BV+trend gating
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import yaml
import pandas as pd
from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest, StrategyConfig
from backtester.broker import PIP_SIZE

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

ALLOWED_BV = {
    "downtrend": ["lime", "magenta", "red"],
    "uptrend": ["lime", "magenta", "white"],
}
BV_SIDE_MAP = {
    "lime": ["buy", "sell"],
    "magenta": ["buy", "sell"],
    "red": ["sell"],
    "white": ["buy"],
}
NEUTRALS = {"deepskyblue", "yellow"}


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = max(0.0, math.floor(x / step) * step)
    return max(min_lot, min(lots, max_lot))


@dataclass
class PTLOsParams:
    PTL_COL: str = CONFIG["strategy_params"]["PTL_TRENA_COL"]
    EMA130_COL: str = CONFIG["strategy_params"]["EMA130_COL"]
    EMA110_COL: str = CONFIG["strategy_params"]["EMA110_COL"]
    EMA400_COL: str = CONFIG["strategy_params"]["EMA400_COL"]
    KC_MID_COL: str = CONFIG["strategy_params"]["KC_MID_COL"]
    KC_UP_COL: str = CONFIG["strategy_params"]["KC_UP_COL"]
    KC_LOW_COL: str = CONFIG["strategy_params"]["KC_LOW_COL"]
    SL_PIPS: float = CONFIG["strategy_params"]["SL_PIPS"]
    TP_PIPS: float = CONFIG["strategy_params"]["TP_PIPS"]
    DEBUG: bool = CONFIG["strategy_params"].get("DEBUG", False)
    BUY_ON_UP: bool = CONFIG["strategy_params"].get("BUY_ON_UP", True)
    BV_COL: str = CONFIG["strategy_params"].get("BV_COL", "bv_color")
    TREND_COL: str = CONFIG["strategy_params"].get("TREND_COL", "trend_regime")


class MyOwnHeartbeat(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any] = CONFIG["strategy_params"],
        strat_cfg: Optional[StrategyConfig] = None,
        params: Optional[PTLOsParams] = None,
        governor=None,
        magic: Optional[int] = 101,
    ):
        super().__init__(symbol, config, strat_cfg=strat_cfg, governor=governor)
        self.params = params or PTLOsParams()
        self._required_cols = [
            self.params.PTL_COL,
            self.params.EMA130_COL,
            self.params.EMA110_COL,
            self.params.EMA400_COL,
            self.params.KC_MID_COL,
            self.params.KC_UP_COL,
            self.params.KC_LOW_COL,
            self.params.BV_COL,
            self.params.TREND_COL,
            "open",
            "close",
        ]
        self.magic = magic
        self.sl_pips = float(self.config.get("SL_PIPS", self.params.SL_PIPS))
        self.tp_pips = float(self.config.get("TP_PIPS", self.params.TP_PIPS))
        self.prev_row: Optional[pd.Series] = None
        self._validated = False
        self._oscillating = True
        self._force_sell = False
        self._force_buy = False
        self._hold_sell_by_ema400 = False
        self._hold_buy_by_ema400 = False

    @staticmethod
    def _value_per_pip_1lot(broker) -> float:
        return broker.cfg.CONTRACT_SIZE * PIP_SIZE

    def _flip_to_side(self, prev_val: int, cur_val: int) -> Optional[str]:
        if prev_val == cur_val:
            return None
        if self.params.BUY_ON_UP:
            if prev_val == 0 and cur_val == 1:
                return "buy"
            if prev_val == 1 and cur_val == 0:
                return "sell"
        else:
            if prev_val == 0 and cur_val == 1:
                return "sell"
            if prev_val == 1 and cur_val == 0:
                return "buy"
        return None

    def _bv_guard_allows(self, row: pd.Series, side: str) -> bool:
        bv = str(row.get(self.params.BV_COL, "")).strip().lower()
        trend = str(row.get(self.params.TREND_COL, "")).strip().lower()
        strict = bool(self.config.get("BV_STRICT", True))
        allow_neutral = bool(self.config.get("BV_ALLOW_NEUTRAL", False))

        if not bv:
            return not strict
        if (bv in NEUTRALS) and not allow_neutral:
            return False
        if strict:
            return (bv in ALLOWED_BV.get(trend, [])) and (
                side in BV_SIDE_MAP.get(bv, [])
            )
        return side in BV_SIDE_MAP.get(bv, [])

    def _open_new_trade(self, broker, t, row: pd.Series, side: str):
        if not self._bv_guard_allows(row, side):
            if self.params.DEBUG:
                print(
                    f"{t} | {self.name} :: BV guard BLOCK {side} | trend={row.get(self.params.TREND_COL)} bv={row.get(self.params.BV_COL)}"
                )
            return
        if self.governor and not self.governor.allow_new_trade(self.name).ok:
            return
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=self.sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        lots_raw = self.sizer.size(req).lots
        lots = _round_to_step(
            lots_raw, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        if lots <= 0:
            return
        tr = broker.open_trade(
            side=side,
            price=float(row["close"]),
            wanted_lots=lots,
            sl_pips=self.sl_pips,
            tp_pips=self.tp_pips,
            t=t,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)

    def on_bar(self, broker, t, row: pd.Series):
        if not self._validated:
            missing = [c for c in self._required_cols if c not in row.index]
            if missing:
                if self.params.DEBUG:
                    print(f"{t} | {self.name} :: missing columns: {missing}")
                return
            self._validated = True
        if self.prev_row is None:
            self.prev_row = row
            return

        ema130 = float(row[self.params.EMA130_COL])
        ema110 = float(row[self.params.EMA110_COL])
        ema400 = float(row[self.params.EMA400_COL])
        kc_mid = float(row[self.params.KC_MID_COL])
        kc_up = float(row[self.params.KC_UP_COL])
        kc_low = float(row[self.params.KC_LOW_COL])
        prev_ptl = int(self.prev_row[self.params.PTL_COL])
        cur_ptl = int(row[self.params.PTL_COL])

        # SELL gating
        if ema130 > kc_mid:
            self._oscillating = False
            self._force_sell = True
            self._force_buy = False
            for tr in list(broker.open_trades):
                if getattr(tr, "strategy_id", None) == self.name and tr.side == "buy":
                    broker.close_trade(tr, float(row["close"]), "EMA130 gating", t)
            if not any(
                tr.strategy_id == self.name and tr.side == "sell"
                for tr in broker.open_trades
            ):
                self._open_new_trade(broker, t, row, "sell")
            self.prev_row = row
            return

        if self._force_sell:
            if ema400 > kc_up:
                self._hold_sell_by_ema400 = True
                self.prev_row = row
                return
            if ema130 <= kc_mid and ema400 <= kc_up:
                for tr in list(broker.open_trades):
                    if (
                        getattr(tr, "strategy_id", None) == self.name
                        and tr.side == "sell"
                    ):
                        broker.close_trade(
                            tr, float(row["close"]), "EMA130/EMA400 release", t
                        )
                self._force_sell = False
                self._hold_sell_by_ema400 = False
                if ema130 > kc_mid:
                    self._force_sell = True
                elif ema110 < kc_mid:
                    self._force_buy = True
                else:
                    self._oscillating = True

        # BUY gating
        if ema110 < kc_mid:
            self._oscillating = False
            self._force_buy = True
            self._force_sell = False
            for tr in list(broker.open_trades):
                if getattr(tr, "strategy_id", None) == self.name and tr.side == "sell":
                    broker.close_trade(tr, float(row["close"]), "EMA110 gating", t)
            if not any(
                tr.strategy_id == self.name and tr.side == "buy"
                for tr in broker.open_trades
            ):
                self._open_new_trade(broker, t, row, "buy")
            self.prev_row = row
            return

        if self._force_buy:
            if ema400 < kc_low:
                self._hold_buy_by_ema400 = True
                self.prev_row = row
                return
            if ema110 >= kc_mid and ema400 >= kc_low:
                if self._hold_buy_by_ema400 and ema400 < kc_low:
                    self.prev_row = row
                    return
                for tr in list(broker.open_trades):
                    if (
                        getattr(tr, "strategy_id", None) == self.name
                        and tr.side == "buy"
                    ):
                        broker.close_trade(
                            tr, float(row["close"]), "EMA110/EMA400 release", t
                        )
                self._force_buy = False
                self._hold_buy_by_ema400 = False
                if ema130 > kc_mid:
                    self._force_sell = True
                elif ema110 < kc_mid:
                    self._force_buy = True
                else:
                    self._oscillating = True

        # Oscillation
        if not self._oscillating:
            self.prev_row = row
            return
        flip_side = self._flip_to_side(prev_ptl, cur_ptl)
        if not flip_side:
            self.prev_row = row
            return
        for tr in list(broker.open_trades):
            if getattr(tr, "strategy_id", None) == self.name and tr.side == (
                "buy" if flip_side == "sell" else "sell"
            ):
                broker.close_trade(tr, float(row["close"]), "Heartbeat Flip", t)
        self._open_new_trade(broker, t, row, flip_side)
        self.prev_row = row


def ptl_osc_feature_spec() -> Dict[str, Any]:
    return {
        "columns": [
            CONFIG["strategy_params"]["PTL_TRENA_COL"],
            CONFIG["strategy_params"]["EMA130_COL"],
            CONFIG["strategy_params"]["EMA110_COL"],
            CONFIG["strategy_params"]["EMA400_COL"],
            CONFIG["strategy_params"]["KC_MID_COL"],
            CONFIG["strategy_params"]["KC_UP_COL"],
            CONFIG["strategy_params"]["KC_LOW_COL"],
            CONFIG["strategy_params"].get("BV_COL", "bv_color"),
            CONFIG["strategy_params"].get("TREND_COL", "trend_regime"),
            "open",
            "close",
        ]
    }
