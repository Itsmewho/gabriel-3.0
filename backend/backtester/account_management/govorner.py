# Govorner

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .types import StrategyConfig, GovCheck


@dataclass
class StratState:
    open_trades: int = 0
    peak_equity: float = 0.0
    paused: bool = False


class RiskGovernor:
    """
    Enforces:
    - max concurrent trades
    - per-strategy drawdown pause/resume
    """

    def __init__(self, cfgs: Dict[str, StrategyConfig]):
        self.cfgs = cfgs
        self.state: Dict[str, StratState] = {sid: StratState() for sid in cfgs}

    def on_equity(self, strategy_id: str, equity: float):
        st = self.state[strategy_id]
        st.peak_equity = max(st.peak_equity, equity if equity > 0 else 0.0)
        peak = max(st.peak_equity, 1e-9)
        dd = 1.0 - (equity / peak)
        if dd >= self.cfgs[strategy_id].max_drawdown_pct:
            st.paused = True

        # optional auto-resume if equity recovers above, say, 80% of peak
        if st.paused and equity >= 0.9 * peak:
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
