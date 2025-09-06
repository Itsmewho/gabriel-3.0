from __future__ import annotations
from . import BrokerConfig, PIP_SIZE
from .cost_engine import value_per_pip


def required_margin(cfg: BrokerConfig, lots: float) -> float:
    return abs(lots) * cfg.MARGIN_PER_LOT


def used_margin(cfg: BrokerConfig, trades) -> float:
    return sum(abs(t.lot_size) * cfg.MARGIN_PER_LOT for t in trades)


def _pips_from(tr, price: float) -> float:
    if tr.side == "buy":
        return (price - tr.entry_price) / PIP_SIZE
    return (tr.entry_price - price) / PIP_SIZE


def floating_pnl(cfg: BrokerConfig, trades, price: float) -> float:
    if not trades:
        return 0.0
    vpp_cache = {}
    out = 0.0
    for tr in trades:
        vpp = vpp_cache.get(tr.lot_size)
        if vpp is None:
            vpp = value_per_pip(cfg, tr.lot_size)
            vpp_cache[tr.lot_size] = vpp
        out += _pips_from(tr, price) * vpp
    return out


def equity(cfg: BrokerConfig, balance: float, trades, price: float) -> float:
    return balance + floating_pnl(cfg, trades, price)


def free_margin(cfg: BrokerConfig, balance: float, trades, price: float) -> float:
    return equity(cfg, balance, trades, price) - used_margin(cfg, trades)


def margin_level_pct(cfg: BrokerConfig, balance: float, trades, price: float) -> float:
    um = used_margin(cfg, trades)
    eq = equity(cfg, balance, trades, price)
    return 999999.0 if um == 0 else (eq / um) * 100.0


# --- Worst-bar variants (use intrabar extremes) ---


def floating_pnl_worst_bar(cfg: BrokerConfig, trades, high: float, low: float) -> float:
    if not trades:
        return 0.0
    out = 0.0
    vpp_cache = {}
    for tr in trades:
        vpp = vpp_cache.get(tr.lot_size)
        if vpp is None:
            vpp = value_per_pip(cfg, tr.lot_size)
            vpp_cache[tr.lot_size] = vpp
        price = low if tr.side == "buy" else high  # adverse extreme per side
        out += _pips_from(tr, price) * vpp
    return out


def worst_case_equity(
    cfg: BrokerConfig, balance: float, trades, high: float, low: float
) -> float:
    return balance + floating_pnl_worst_bar(cfg, trades, high, low)


def margin_level_pct_worst_bar(
    cfg: BrokerConfig, balance: float, trades, high: float, low: float
) -> float:
    um = used_margin(cfg, trades)
    eq = worst_case_equity(cfg, balance, trades, high, low)
    return 999999.0 if um == 0 else (eq / um) * 100.0


def can_open_order(cfg: BrokerConfig, balance, trades, lots, price):
    req = required_margin(cfg, lots)
    um = used_margin(cfg, trades)
    eq = equity(cfg, balance, trades, price)
    fm_before = eq - um
    fm_after = fm_before - req
    needed_balance = max(0.0, req - fm_before)
    return {
        "ok": fm_after >= 0.0,
        "req_margin": req,
        "used_margin": um,
        "equity": eq,
        "free_margin_before": fm_before,
        "free_margin_after": fm_after,
        "needed_balance": needed_balance,
    }


def needs_warning(cfg: BrokerConfig, balance: float, trades, price: float) -> bool:
    ml = margin_level_pct(cfg, balance, trades, price)
    return 0 < ml < 90.0


def needs_stop_out(cfg: BrokerConfig, balance: float, trades, price: float) -> bool:
    ml = margin_level_pct(cfg, balance, trades, price)
    return ml < cfg.STOP_OUT_LEVEL_PCT


def pick_worst_trade(cfg: BrokerConfig, trades, high: float, low: float):
    if not trades:
        return None
    losses = []
    for tr in trades:
        price = low if tr.side == "buy" else high  # adverse extreme
        pnl = _pips_from(tr, price) * value_per_pip(cfg, tr.lot_size)
        losses.append((pnl, tr))
    losses.sort(key=lambda x: x[0])  # most negative first
    return losses[0][1]


# --- New helpers for broker liquidation policy ---


def should_force_liquidation(
    cfg: BrokerConfig, balance: float, trades, high: float, low: float
) -> bool:
    """Trigger liquidation if either margin level or equity is unsafe at worst bar."""
    if not trades:
        return False
    ml_worst = margin_level_pct_worst_bar(cfg, balance, trades, high, low)
    eq_worst = worst_case_equity(cfg, balance, trades, high, low)
    return (ml_worst < cfg.STOP_OUT_LEVEL_PCT) or (eq_worst < 0.0)
