# Not really profitible need alot of fine tuning and analysing before making it work.
# Move on.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE


@dataclass
class _Pending:
    event_ts: Optional[pd.Timestamp] = None
    ref_px: Optional[float] = None
    triggered: bool = False
    side: Optional[str] = None  # "buy" | "sell"


class EventBracketColumn(BaseStrategy):
    """
    Column-driven event breakout using fields already present in MARKET_DATA parquet:
      - event_name
      - event_impact_score
      - event_minutes_to_event (0 at the event bar, positive before)

    Flow per event bar E:
      1) Snapshot ref price p0 = close at E.
      2) For WINDOW_MIN min after E: if high >= p0+UP_PIPS -> BUY, if low <= p0-DN_PIPS -> SELL.
      3) Optional hard exit HOLD_MIN after entry.

    Robust single-bar detection via state: mark E once when
      name != 'none', impact >= MIN_IMPACT and minutes_to_event <= EVENT_TOL_MIN,
      and the previous bar was not within tol.
    """

    def __init__(
        self, symbol: str, config: Dict[str, Any], strat_cfg=None, governor=None
    ) -> None:
        super().__init__(symbol, config or {"name": "EV_BRK_COL"}, strat_cfg, governor)
        # thresholds
        self.up_pips: float = float(self.config.get("UP_PIPS", 10))
        self.dn_pips: float = float(self.config.get("DN_PIPS", 10))
        self.sl_pips: float = float(self.config.get("SL_PIPS", 8))
        self.tp_pips: float = float(self.config.get("TP_PIPS", 15))
        self.window_min: int = int(self.config.get("WINDOW_MIN", 10))
        self.hold_min: int = int(self.config.get("HOLD_MIN", 15))
        self.min_impact: int = int(self.config.get("MIN_IMPACT", 2))
        self.tol_min: float = float(self.config.get("EVENT_TOL_MIN", 0.5))
        self.fixed_lots = self.config.get("FIXED_LOTS", None)
        self.magic = self.config.get("MAGIC", None)

        # state
        self._pending = _Pending()
        self._active_entry_ts: Optional[pd.Timestamp] = None
        self._last_ev_name: str = "none"
        self._last_mte: Optional[float] = None

    # --- helpers ---
    def _size_lots(self, broker, sl_pips: float) -> float:
        if self.fixed_lots is not None:
            import math

            lots = (
                math.floor(float(self.fixed_lots) / broker.cfg.VOLUME_STEP)
                * broker.cfg.VOLUME_STEP
            )
            return max(broker.cfg.VOLUME_MIN, min(lots, self.strat_cfg.lot_max))
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        sized = self.sizer.size(req)
        import math

        lots = math.floor(sized.lots / broker.cfg.VOLUME_STEP) * broker.cfg.VOLUME_STEP
        return max(broker.cfg.VOLUME_MIN, min(lots, self.strat_cfg.lot_max))

    def _is_event_bar(self, row) -> bool:
        name = str(row.get("event_name", "none"))
        if name == "none":
            return False
        impact = int(row.get("event_impact_score", 0))
        if impact < self.min_impact:
            return False
        try:
            mte = float(row.get("event_minutes_to_event", float("inf")))
        except Exception:
            mte = float("inf")
        prev_in_tol = (
            (self._last_mte is not None)
            and (self._last_mte <= self.tol_min)
            and (self._last_ev_name != "none")
        )
        return (mte <= self.tol_min) and (not prev_in_tol)

    def _within_window(self, t: pd.Timestamp) -> bool:
        if self._pending.event_ts is None:
            return False
        return (t - self._pending.event_ts) <= pd.Timedelta(minutes=self.window_min)

    def _hold_exit_due(self, t: pd.Timestamp) -> bool:
        if self.hold_min <= 0 or self._active_entry_ts is None:
            return False
        return (t - self._active_entry_ts) >= pd.Timedelta(minutes=self.hold_min)

    # --- main ---
    def on_bar(self, broker, t, row):
        px_close = float(row["close"])  # server time
        px_high = float(row.get("high", px_close))
        px_low = float(row.get("low", px_close))

        # 1) Hard hold exit
        if self._active_entry_ts is not None and self._hold_exit_due(t):
            for tr in list(broker.open_trades):
                if getattr(tr, "strategy_id", None) == self.name:
                    broker.close_trade(tr, px_close, "time_exit", t)
            self._active_entry_ts = None
            # update trackers
            try:
                self._last_ev_name = str(row.get("event_name", "none"))
                self._last_mte = float(row.get("event_minutes_to_event", float("inf")))
            except Exception:
                self._last_mte = float("inf")
            return None

        # 2) Detect event on this row and allow same-bar trigger
        if self._pending.event_ts is None and self._is_event_bar(row):
            self._pending = _Pending(
                event_ts=t, ref_px=px_close, triggered=False, side=None
            )

            up_level = self._pending.ref_px + self.up_pips * PIP_SIZE  # type: ignore
            dn_level = self._pending.ref_px - self.dn_pips * PIP_SIZE  # type: ignore

            side: Optional[str] = None
            if px_high >= up_level:
                side = "buy"
            elif px_low <= dn_level:
                side = "sell"

            if side is not None:
                # governor
                if self.governor:
                    chk = self.governor.allow_new_trade(self.name)
                    if not chk.ok:
                        self._pending = _Pending()
                        try:
                            self._last_ev_name = str(row.get("event_name", "none"))
                            self._last_mte = float(
                                row.get("event_minutes_to_event", float("inf"))
                            )
                        except Exception:
                            self._last_mte = float("inf")
                        return None

                # concurrency cap
                mine_open = sum(
                    1
                    for tr in broker.open_trades
                    if getattr(tr, "strategy_id", None) == self.name
                )
                if mine_open >= max(1, self.strat_cfg.max_concurrent_trades):
                    self._pending = _Pending()
                    try:
                        self._last_ev_name = str(row.get("event_name", "none"))
                        self._last_mte = float(
                            row.get("event_minutes_to_event", float("inf"))
                        )
                    except Exception:
                        self._last_mte = float("inf")
                    return None

                lots = self._size_lots(broker, self.sl_pips)
                if lots <= 0:
                    self._pending = _Pending()
                    try:
                        self._last_ev_name = str(row.get("event_name", "none"))
                        self._last_mte = float(
                            row.get("event_minutes_to_event", float("inf"))
                        )
                    except Exception:
                        self._last_mte = float("inf")
                    return None

                tr = broker.open_trade(
                    side=side,
                    price=px_close,
                    wanted_lots=lots,
                    sl_pips=self.sl_pips,
                    tp_pips=self.tp_pips,
                    t=t,
                    fallbacks=[],
                    strategy_id=self.name,
                    magic=self.magic,
                )
                if tr:
                    self.setup_trade(broker, tr)
                    self._pending.triggered = True
                    self._pending.side = side
                    self._active_entry_ts = t
                    try:
                        self._last_ev_name = str(row.get("event_name", "none"))
                        self._last_mte = float(
                            row.get("event_minutes_to_event", float("inf"))
                        )
                    except Exception:
                        self._last_mte = float("inf")
                    return tr
            # if no same-bar trigger, proceed to window monitoring

        # 3) After event, for WINDOW_MIN, watch for breakout
        if self._pending.event_ts is not None and not self._pending.triggered:
            if not self._within_window(t):
                self._pending = _Pending()
                try:
                    self._last_ev_name = str(row.get("event_name", "none"))
                    self._last_mte = float(
                        row.get("event_minutes_to_event", float("inf"))
                    )
                except Exception:
                    self._last_mte = float("inf")
                return None

            up_level = self._pending.ref_px + self.up_pips * PIP_SIZE  # type: ignore
            dn_level = self._pending.ref_px - self.dn_pips * PIP_SIZE  # type: ignore

            side: Optional[str] = None
            if px_high >= up_level:
                side = "buy"
            elif px_low <= dn_level:
                side = "sell"

            if side is None:
                try:
                    self._last_ev_name = str(row.get("event_name", "none"))
                    self._last_mte = float(
                        row.get("event_minutes_to_event", float("inf"))
                    )
                except Exception:
                    self._last_mte = float("inf")
                return None

            if self.governor:
                chk = self.governor.allow_new_trade(self.name)
                if not chk.ok:
                    self._pending = _Pending()
                    try:
                        self._last_ev_name = str(row.get("event_name", "none"))
                        self._last_mte = float(
                            row.get("event_minutes_to_event", float("inf"))
                        )
                    except Exception:
                        self._last_mte = float("inf")
                    return None

            mine_open = sum(
                1
                for tr in broker.open_trades
                if getattr(tr, "strategy_id", None) == self.name
            )
            if mine_open >= max(1, self.strat_cfg.max_concurrent_trades):
                self._pending = _Pending()
                try:
                    self._last_ev_name = str(row.get("event_name", "none"))
                    self._last_mte = float(
                        row.get("event_minutes_to_event", float("inf"))
                    )
                except Exception:
                    self._last_mte = float("inf")
                return None

            lots = self._size_lots(broker, self.sl_pips)
            if lots <= 0:
                self._pending = _Pending()
                try:
                    self._last_ev_name = str(row.get("event_name", "none"))
                    self._last_mte = float(
                        row.get("event_minutes_to_event", float("inf"))
                    )
                except Exception:
                    self._last_mte = float("inf")
                return None

            tr = broker.open_trade(
                side=side,
                price=px_close,
                wanted_lots=lots,
                sl_pips=self.sl_pips,
                tp_pips=self.tp_pips,
                t=t,
                fallbacks=[],
                strategy_id=self.name,
                magic=self.magic,
            )
            if tr:
                self.setup_trade(broker, tr)
                self._pending.triggered = True
                self._pending.side = side
                self._active_entry_ts = t
                try:
                    self._last_ev_name = str(row.get("event_name", "none"))
                    self._last_mte = float(
                        row.get("event_minutes_to_event", float("inf"))
                    )
                except Exception:
                    self._last_mte = float("inf")
                return tr

        # update trackers at end of bar
        try:
            self._last_ev_name = str(row.get("event_name", "none"))
            self._last_mte = float(row.get("event_minutes_to_event", float("inf")))
        except Exception:
            self._last_mte = float("inf")
        return None

    # # Risk and strategy config (keys match strategy names)
    # cfg_map = {
    #     "EV_BRK_KELLY": StrategyConfig(
    #         risk_mode=RiskMode.HALF_KELLY,
    #         risk_pct=0.01,
    #         kelly_p=0.53,
    #         kelly_rr=1.6,
    #         kelly_cap_pct=0.02,
    #         lot_min=cfg.VOLUME_MIN,
    #         lot_step=cfg.VOLUME_STEP,
    #         lot_max=100.0,
    #         max_risk_pct_per_trade=0.02,
    #         max_concurrent_trades=1001,
    #     ),
    #     "EV_BRK_FIXED": StrategyConfig(
    #         risk_mode=RiskMode.FIXED,
    #         risk_pct=0.01,
    #         lot_min=cfg.VOLUME_MIN,
    #         lot_step=cfg.VOLUME_STEP,
    #         lot_max=100.0,
    #         max_risk_pct_per_trade=0.01,
    #         max_concurrent_trades=1001,
    #     ),
    # }

    # governor = RiskGovernor(cfg_map)

    # strategies = [
    #     EventBracketColumn(
    #         symbol=symbol,
    #         config={
    #             "name": "EV_BRK_KELLY",
    #             "AUTO_BUILD_EVENTS": True,
    #             "MARKET_DATA": market_data,  # pass your DataFrame
    #             "MIN_IMPACT": 2,
    #             "UP_PIPS": 10,
    #             "DN_PIPS": 10,
    #             "SL_PIPS": 8,
    #             "TP_PIPS": 15,
    #             "WINDOW_MIN": 10,
    #             "HOLD_MIN": 15,
    #             "FIXED_LOTS": None,
    #         },
    #         strat_cfg=cfg_map["EV_BRK_KELLY"],
    #         governor=governor,
    #     ),
    #     EventBracketColumn(
    #         symbol=symbol,
    #         config={
    #             "name": "EV_BRK_FIXED",
    #             "AUTO_BUILD_EVENTS": True,
    #             "MARKET_DATA": market_data,
    #             "MIN_IMPACT": 2,
    #             "UP_PIPS": 10,
    #             "DN_PIPS": 10,
    #             "SL_PIPS": 8,
    #             "TP_PIPS": 15,
    #             "WINDOW_MIN": 10,
    #             "HOLD_MIN": 15,
    #             "FIXED_LOTS": None,
    #         },
    #         strat_cfg=cfg_map["EV_BRK_FIXED"],
    #         governor=governor,
    #     ),
    # ]
