from __future__ import annotations
from . import BrokerConfig, PIP_SIZE
from .cost_engine import value_per_pip


def required_margin(cfg: BrokerConfig, lots: float) -> float:
    return abs(lots) * cfg.MARGIN_PER_LOT


def used_margin(cfg: BrokerConfig, trades) -> float:
    return sum(abs(t.lot_size) * cfg.MARGIN_PER_LOT for t in trades)


def pick_biggest_lot(trades):
    if not trades:
        return None
    return max(trades, key=lambda t: abs(getattr(t, "lot_size", 0.0)))


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


def should_force_liquidation(
    cfg: BrokerConfig, balance: float, trades, high: float, low: float, close: float
) -> bool:
    """Liquidate if any unsafe condition is present."""
    if not trades:
        return False

    um = used_margin(cfg, trades)

    # Worst-bar (intrabar extreme)
    eq_worst = balance + floating_pnl_worst_bar(cfg, trades, high, low)
    fm_worst = eq_worst - um
    ml_worst = 999999.0 if um == 0 else (eq_worst / um) * 100.0

    # Current bar (close)
    eq_cur = equity(cfg, balance, trades, close)
    fm_cur = eq_cur - um

    return (ml_worst < cfg.STOP_OUT_LEVEL_PCT) or (fm_worst < 0.0) or (fm_cur < 0.0)
