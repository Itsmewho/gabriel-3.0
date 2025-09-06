# cost_engine.py

import pandas as pd
import numpy as np
from . import PIP_SIZE, BrokerConfig, Trade

DEFAULT_COMMISSION_PER_LOT_PER_SIDE = 3.50  # USD per lot per side


def _is_night_session(cfg: BrokerConfig, t: pd.Timestamp | None) -> bool:
    if t is None or cfg.NIGHT_SPREAD_PIPS is None:
        return False
    # Server-time hours assumed. No UTC conversions per project rule.
    h = int(pd.to_datetime(t).hour)
    start_h = int(getattr(cfg, "NIGHT_SPREAD_START_H", 21))
    end_h = int(getattr(cfg, "NIGHT_SPREAD_END_H", 6))
    if start_h <= end_h:
        return start_h <= h < end_h
    # wraps midnight
    return h >= start_h or h < end_h


def apply_spread(cfg: BrokerConfig, side: str, raw_price: float) -> float:
    """Legacy time-agnostic spread. Kept for compatibility."""
    adj = cfg.SPREAD_PIPS * PIP_SIZE
    return raw_price + adj if side == "buy" else raw_price - adj


def apply_spread_at(
    cfg: BrokerConfig, side: str, raw_price: float, t: pd.Timestamp | None
) -> float:
    pips = cfg.NIGHT_SPREAD_PIPS if _is_night_session(cfg, t) else cfg.SPREAD_PIPS
    adj = pips * PIP_SIZE
    return raw_price + adj if side == "buy" else raw_price - adj


def _per_side_commission(cfg: BrokerConfig) -> float:
    return cfg.COMMISSION_PER_LOT_PER_SIDE or DEFAULT_COMMISSION_PER_LOT_PER_SIDE


def value_per_pip(cfg: BrokerConfig, lots: float) -> float:
    return lots * cfg.CONTRACT_SIZE * PIP_SIZE


def commission_open(cfg: BrokerConfig, lots: float) -> float:
    return _per_side_commission(cfg) * lots


def commission_close(cfg: BrokerConfig, lots: float) -> float:
    return _per_side_commission(cfg) * lots


def swap_cost(cfg: BrokerConfig, trade: Trade, t: pd.Timestamp) -> float:
    if not isinstance(t, pd.Timestamp):
        t = pd.to_datetime(t)
    swap_points = cfg.SWAP_LONG_POINTS if trade.side == "buy" else cfg.SWAP_SHORT_POINTS
    # 1 point = 1/10 pip on MT5
    points_to_price = PIP_SIZE / 10.0
    swap_in_price = swap_points * points_to_price
    fee = cfg.CONTRACT_SIZE * swap_in_price * trade.lot_size
    if t.weekday() == 2:  # Wednesday triple
        fee *= 3
    return fee


def sample_slippage_pips(cfg: BrokerConfig, rng=None) -> float:
    rng = rng or np.random
    lo = float(getattr(cfg, "MIN_SLIPPAGE_PIPS", 0) or 0)
    hi = float(getattr(cfg, "MAX_SLIPPAGE_PIPS", 0) or 0)
    if hi <= 0:
        return 0.0
    if lo < 0:
        lo = 0.0
    if hi < lo:
        hi = lo
    # Adverse-biased distribution: 70% skew to upper half of [lo, hi]
    u = rng.rand()
    if u < 0.7:
        base = (lo + hi) / 2.0
        return float(base + rng.rand() * (hi - base))
    return float(lo + rng.rand() * (hi - lo))


def apply_slippage(
    cfg: BrokerConfig,
    side: str,
    price: float,
    rng=None,
    favorable_prob: float | None = None,
) -> float:
    rng = rng or np.random
    pips = sample_slippage_pips(cfg, rng)
    if pips <= 0:
        return price
    sign = 1.0 if side == "buy" else -1.0  # adverse by default
    fp = cfg.SLIPPAGE_FAVORABLE_PROB if favorable_prob is None else favorable_prob
    if fp > 0 and rng.rand() < float(fp):
        sign *= -1.0
    return price + sign * pips * PIP_SIZE


def _apply_latency(t: pd.Timestamp | None, latency_ms: int) -> pd.Timestamp | None:
    if t is None or latency_ms <= 0:
        return t
    return pd.to_datetime(t) + pd.Timedelta(milliseconds=int(latency_ms))


def fill_price_on_open(
    cfg: BrokerConfig,
    side: str,
    raw_price: float,
    t: pd.Timestamp | None = None,
    rng=None,
) -> float:
    # model decision->fill delay
    t_fill = _apply_latency(t, int(getattr(cfg, "EXECUTION_LATENCY_MS", 0) or 0))
    px = apply_spread_at(cfg, side, raw_price, t_fill)
    return apply_slippage(cfg, side, px, rng=rng)


def fill_price_on_close(
    cfg: BrokerConfig,
    side: str,
    target_price: float,
    t: pd.Timestamp | None = None,
    rng=None,
) -> float:
    t_fill = _apply_latency(t, int(getattr(cfg, "EXECUTION_LATENCY_MS", 0) or 0))
    px = apply_spread_at(cfg, side, target_price, t_fill)
    return apply_slippage(cfg, side, px, rng=rng)
