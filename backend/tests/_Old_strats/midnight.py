### TESTED =========== NOT PROFITIBLE!!!!!!


from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backtester.strategies.base_strat import BaseStrategy
from backtester.account_management.types import SizeRequest
from backtester.broker import PIP_SIZE


@dataclass
class _State:
    long_id: Optional[int] = None
    long_px: Optional[float] = None
    day_key: Optional[tuple] = None  # (Y, M, D)


def _round_to_step(x: float, step: float, min_lot: float, max_lot: float) -> float:
    lots = math.floor(x / step) * step
    lots = max(min_lot, lots)
    lots = min(max_lot, lots)
    return round(lots, 8)


class MidnightLong(BaseStrategy):
    """
    Long-only midnight opener.
    Opens one BUY at 00:25 server time per day.
    TP=15 pips, SL=5 pips. Margin-aware sizing. Force-close at 02:00.
    """

    def __init__(
        self,
        symbol: str,
        config: Dict[str, Any] | None = None,
        strat_cfg=None,
        magic: Optional[int] = 3101,
        governor=None,
    ) -> None:
        super().__init__(
            symbol, config or {"name": "MIDNIGHT_LONG"}, strat_cfg, governor
        )
        self.magic = magic
        self.open_hour = int(self.config.get("OPEN_HOUR", 0))
        self.open_minute = int(self.config.get("OPEN_MINUTE", 25))
        self.sl_pips = float(self.config.get("SL_PIPS", 5))
        self.tp_pips = float(self.config.get("TP_PIPS", 15))
        self.fixed_lots = self.config.get("FIXED_LOTS", None)
        self.state = _State()

    # --- helpers ---
    def _is_open_time(self, t) -> bool:
        return (
            getattr(t, "hour", None) == self.open_hour
            and getattr(t, "minute", None) == self.open_minute
        )

    def _is_force_close_time(self, t) -> bool:
        return getattr(t, "hour", None) == 2 and getattr(t, "minute", None) == 0

    def _day_key(self, t) -> tuple:
        return (t.year, t.month, t.day)

    def _reset_if_new_day(self, t) -> None:
        dk = self._day_key(t)
        if self.state.day_key != dk and self.state.long_id is None:
            self.state = _State(day_key=dk)
        elif self.state.day_key is None:
            self.state.day_key = dk

    def _open_count_for_me(self, broker) -> int:
        return sum(
            1
            for tr in broker.open_trades
            if getattr(tr, "strategy_id", None) == self.name
        )

    def _size_lots(self, broker, sl_pips: float) -> float:
        if self.fixed_lots is not None:
            return _round_to_step(
                float(self.fixed_lots),
                broker.cfg.VOLUME_STEP,
                broker.cfg.VOLUME_MIN,
                9999.0,
            )
        req = SizeRequest(
            balance=broker.balance,
            sl_pips=sl_pips,
            value_per_pip=self._value_per_pip_1lot(broker),
        )
        sized = self.sizer.size(req)
        return _round_to_step(
            sized.lots, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )

    def _cap_by_margin(self, broker, lots: float) -> float:
        free = getattr(broker, "free_margin", broker.balance)
        max_lots = max(0.0, free / getattr(broker.cfg, "MARGIN_PER_LOT", float("inf")))
        return _round_to_step(
            min(lots, max_lots), broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )

    def _maybe_open_long(self, broker, t, px: float):
        if self.governor:
            chk = self.governor.allow_new_trade(self.name)
            if not chk.ok:
                return None
        if self._open_count_for_me(broker) >= max(
            1, self.strat_cfg.max_concurrent_trades
        ):
            return None
        lots = self._cap_by_margin(broker, self._size_lots(broker, self.sl_pips))
        if lots <= 0:
            return None
        fb1 = _round_to_step(
            lots * 0.5, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        fb2 = _round_to_step(
            lots * 0.25, broker.cfg.VOLUME_STEP, broker.cfg.VOLUME_MIN, 9999.0
        )
        fallbacks = [x for x in (fb1, fb2) if broker.cfg.VOLUME_MIN <= x < lots]

        tr = broker.open_trade(
            side="buy",
            price=px,
            wanted_lots=lots,
            sl_pips=self.sl_pips,
            tp_pips=self.tp_pips,
            t=t,
            fallbacks=fallbacks,
            strategy_id=self.name,
            magic=self.magic,
        )
        if tr:
            self.setup_trade(broker, tr)
            self.state.long_id = tr.id
            self.state.long_px = tr.entry_price
            return tr
        return None

    def _gc_long_id(self, broker) -> None:
        ids = {tr.id for tr in broker.open_trades}
        if self.state.long_id is not None and self.state.long_id not in ids:
            self.state.long_id = None
            self.state.long_px = None

    def on_bar(self, broker, t, row):
        self._reset_if_new_day(t)
        self._gc_long_id(broker)

        # Force-close at 02:00 using broker.close_trade(trade, price, reason, t)
        if self.state.long_id is not None and self._is_force_close_time(t):
            tr = next(
                (x for x in broker.open_trades if x.id == self.state.long_id), None
            )
            if tr is not None:
                px = float(row["close"])  # use current close as exit
                broker.close_trade(tr, px, "time_exit", t)
            self.state.long_id = None
            self.state.long_px = None
            return None

        px = float(row["close"])

        if self._is_open_time(t) and self.state.long_id is None:
            return self._maybe_open_long(broker, t, px)

        return None


# ===================== USAGE IN BROKER ==================


# # --- Risk and strategy config ---
# cfg_map = {
#     "MIDNIGHT_KELLY": StrategyConfig(
#         risk_mode=RiskMode.HALF_KELLY,
#         risk_pct=0.01,
#         kelly_p=0.53,
#         kelly_rr=1.6,
#         kelly_cap_pct=0.02,
#         lot_min=cfg.VOLUME_MIN,
#         lot_step=cfg.VOLUME_STEP,
#         lot_max=10.0,
#         max_risk_pct_per_trade=0.02,
#         max_concurrent_trades=2,
#     ),
#     "MIDNIGHT_FIXED": StrategyConfig(
#         risk_mode=RiskMode.FIXED,
#         risk_pct=0.01,
#         lot_min=cfg.VOLUME_MIN,
#         lot_step=cfg.VOLUME_STEP,
#         lot_max=10.0,
#         max_risk_pct_per_trade=0.01,
#         max_concurrent_trades=2,
#     ),
# }

# governor = RiskGovernor(cfg_map)

# cfg_map = {
#     "MIDNIGHT_KELLY": StrategyConfig(
#         risk_mode=RiskMode.HALF_KELLY,
#         risk_pct=0.01,
#         kelly_p=0.53,
#         kelly_rr=1.6,
#         kelly_cap_pct=0.02,
#         lot_min=cfg.VOLUME_MIN,
#         lot_step=cfg.VOLUME_STEP,
#         lot_max=100.0,
#         max_risk_pct_per_trade=0.01,
#         max_concurrent_trades=2,
#     ),
#     "MIDNIGHT_FIXED": StrategyConfig(
#         risk_mode=RiskMode.FIXED,
#         risk_pct=0.01,
#         lot_min=cfg.VOLUME_MIN,
#         lot_step=cfg.VOLUME_STEP,
#         lot_max=100.0,
#         max_risk_pct_per_trade=0.01,
#         max_concurrent_trades=2,
#     ),
# }
# governor = RiskGovernor(cfg_map)

# strategies = [
#     MidnightLong(
#         symbol=symbol,
#         config={
#             "name": "MIDNIGHT_KELLY",
#             "OPEN_HOUR": 0,
#             "OPEN_MINUTE": 25,
#             "SL_PIPS": 5,
#             "TP_PIPS": 10,
#             "FIXED_LOTS": None,
#             "USE_BREAK_EVEN_STOP": True,
#             "BE_TRIGGER_PIPS": 3,
#             "BE_OFFSET_PIPS": 2,
#         },
#         strat_cfg=cfg_map["MIDNIGHT_KELLY"],
#         governor=governor,
#     ),
#     MidnightLong(
#         symbol=symbol,
#         config={
#             "name": "MIDNIGHT_FIXED",
#             "OPEN_HOUR": 0,
#             "OPEN_MINUTE": 25,
#             "SL_PIPS": 5,
#             "TP_PIPS": 10,
#             "FIXED_LOTS": None,
#             "USE_BREAK_EVEN_STOP": True,
#             "BE_TRIGGER_PIPS": 3,
#             "BE_OFFSET_PIPS": 2,
#             "USE_TRAILING_STOP": False,
#             "TRAILING_STOP_DISTANCE_PIPS": 5,
#             "USE_TP_EXTENSION": False,
#             "NEAR_TP_BUFFER_PIPS": 2,
#             "TP_EXTENSION_PIPS": 3,
#         },
#         strat_cfg=cfg_map["MIDNIGHT_FIXED"],
#         governor=governor,
#     ),
# ]

# alloc = cfg.INITIAL_BALANCE
# allocations = {"MIDNIGHT_KELLY": alloc * 0.5, "MIDNIGHT_FIXED": alloc * 0.5}
# ledger = Ledger(initial_allocations=allocations)

# trade_to_strategy: dict[int, str] = {}
