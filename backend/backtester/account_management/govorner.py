# Govorner

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping
from .types import StrategyConfig, GovCheck


@dataclass
class StratState:
    open_trades: int = 0
    peak_equity: float = 0.0
    paused: bool = False


class RiskGovernor:
    def __init__(self, cfgs: Dict[str, StrategyConfig]):
        self.cfgs = cfgs
        self.state: Dict[str, StratState] = {sid: StratState() for sid in cfgs}

    # Seed initial equity/peaks from allocations or ledger
    def seed_from_allocations(self, alloc: Mapping[str, float]):
        for sid, eq in alloc.items():
            st = self.state.setdefault(sid, StratState())
            st.peak_equity = max(st.peak_equity, float(eq))

    def on_equity(self, strategy_id: str, equity: float):
        st = self.state[strategy_id]
        cfg = self.cfgs[strategy_id]
        # initialize peak on first observation
        if st.peak_equity <= 0 and equity > 0:
            st.peak_equity = float(equity)
        # update peak
        st.peak_equity = max(st.peak_equity, float(equity))
        peak = max(st.peak_equity, 1e-9)
        dd = 1.0 - (float(equity) / peak)
        if dd >= cfg.max_drawdown_pct:
            st.paused = True
        # hysteresis resume threshold
        if st.paused and float(equity) >= cfg.max_drawdown_resume_pct * peak:
            st.paused = False

    def on_open(self, strategy_id: str):
        self.state[strategy_id].open_trades += 1

    def on_close(self, strategy_id: str):
        self.state[strategy_id].open_trades = max(
            0, self.state[strategy_id].open_trades - 1
        )

    def allow_new_trade(self, strategy_id: str) -> GovCheck:
        st = self.state[strategy_id]
        cfg = self.cfgs[strategy_id]
        if st.paused:
            return GovCheck(False, "paused_by_drawdown")
        if st.open_trades >= cfg.max_concurrent_trades:
            return GovCheck(False, "max_concurrent_reached")
        return GovCheck(True, None)
