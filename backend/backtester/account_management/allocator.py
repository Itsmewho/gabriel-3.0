# Allocator

from __future__ import annotations
from typing import Dict, List
from .types import Allocation


class PortfolioAllocator:
    """
    Proportional allocator: budgets = total_equity * weights (auto-normalized).
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights: Dict[str, float] = dict(weights)

    def set_weights(self, weights: Dict[str, float]):
        self.weights = dict(weights)

    def budgets(self, total_equity: float) -> List[Allocation]:
        wsum = sum(max(0.0, w) for w in self.weights.values()) or 1.0
        out: List[Allocation] = []
        for sid, w in self.weights.items():
            adj = max(0.0, w) / wsum
            out.append(
                Allocation(
                    strategy_id=sid, weight=adj, equity_budget=total_equity * adj
                )
            )
        return out

    def budget_for(self, total_equity: float, strategy_id: str) -> float:
        for a in self.budgets(total_equity):
            if a.strategy_id == strategy_id:
                return a.equity_budget
        return 0.0
