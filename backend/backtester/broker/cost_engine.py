# cost_engine.py

import pandas as pd
import numpy as np
from . import PIP_SIZE, BrokerConfig, Trade

DEFAULT_COMMISSION_PER_LOT_PER_SIDE = 3.50  # USD per lot per side


def _per_side_commission(cfg: BrokerConfig) -> float:
    return cfg.COMMISSION_PER_LOT_PER_SIDE or DEFAULT_COMMISSION_PER_LOT_PER_SIDE


def apply_spread(cfg: BrokerConfig, side: str, raw_price: float) -> float:
    adj = cfg.SPREAD_PIPS * PIP_SIZE
    return raw_price + adj if side == "buy" else raw_price - adj


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
    return float(rng.uniform(lo, hi))


def apply_slippage(
    cfg: BrokerConfig, side: str, price: float, rng=None, favorable_prob: float = 0.0
) -> float:
    rng = rng or np.random
    pips = sample_slippage_pips(cfg, rng)
    if pips <= 0:
        return price
    sign = 1.0 if side == "buy" else -1.0  # adverse by default
    if favorable_prob > 0 and rng.rand() < favorable_prob:
        sign *= -1.0
    return price + sign * pips * PIP_SIZE


def fill_price_on_open(
    cfg: BrokerConfig, side: str, raw_price: float, rng=None
) -> float:
    return apply_slippage(cfg, side, apply_spread(cfg, side, raw_price), rng=rng)


def fill_price_on_close(
    cfg: BrokerConfig, side: str, target_price: float, rng=None
) -> float:
    # side is the action: 'sell' to close long, 'buy' to close short
    return apply_slippage(cfg, side, apply_spread(cfg, side, target_price), rng=rng)
