# Sizer


from __future__ import annotations
from math import floor
from .types import StrategyConfig, SizeRequest, SizeResult, RiskMode


def _round_lot(x: float, step: float) -> float:
    # floor to step to avoid over-ordering
    return max(0.0, floor(x / step) * step)


def _kelly_fraction(p: float, rr: float) -> float:
    # Kelly = p - (1-p)/rr
    q = 1.0 - p
    if rr <= 0:
        return 0.0
    return max(0.0, p - q / rr)


class AccountSizer:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg

    def _size_fixed(self, req: SizeRequest) -> SizeResult:
        risk_amt = self.cfg.risk_pct * req.balance
        if req.sl_pips <= 0 or req.value_per_pip <= 0:
            return SizeResult(0.0, 0.0, "invalid_inputs")

        # value_per_pip must be for 1.0 lot. Risk in $ = sl_pips * value_per_pip * lots
        raw_lots = risk_amt / (req.sl_pips * req.value_per_pip)
        lots = _round_lot(raw_lots, self.cfg.lot_step)
        lots = min(max(lots, self.cfg.lot_min), self.cfg.lot_max)
        # recompute risk with clamped lots
        risk_amt_clamped = req.sl_pips * req.value_per_pip * lots
        return SizeResult(lots, risk_amt_clamped)

    def _size_half_kelly(self, req: SizeRequest) -> SizeResult:
        k = _kelly_fraction(self.cfg.kelly_p, self.cfg.kelly_rr) * 0.5
        if k <= 0:
            return SizeResult(0.0, 0.0, "kelly_zero")

        # cap Kelly risk by kelly_cap_pct
        risk_frac = min(k, self.cfg.kelly_cap_pct)
        risk_amt = risk_frac * req.balance

        if req.sl_pips <= 0 or req.value_per_pip <= 0:
            return SizeResult(0.0, 0.0, "invalid_inputs")

        raw_lots = risk_amt / (req.sl_pips * req.value_per_pip)
        lots = _round_lot(raw_lots, self.cfg.lot_step)
        lots = min(max(lots, self.cfg.lot_min), self.cfg.lot_max)
        risk_amt_clamped = req.sl_pips * req.value_per_pip * lots
        return SizeResult(lots, risk_amt_clamped)

    def size(self, req: SizeRequest) -> SizeResult:
        if self.cfg.risk_mode == RiskMode.FIXED:
            out = self._size_fixed(req)
        else:
            out = self._size_half_kelly(req)

        # enforce per-trade risk cap
        cap_amt = self.cfg.max_risk_pct_per_trade * req.balance
        if out.risk_amount > cap_amt and out.lots > 0:
            shrink = cap_amt / max(out.risk_amount, 1e-9)
            lots = _round_lot(out.lots * shrink, self.cfg.lot_step)
            lots = max(0.0, min(lots, self.cfg.lot_max))
            out = SizeResult(
                lots, req.sl_pips * req.value_per_pip * lots, "capped_by_per_trade_risk"
            )
        return out
