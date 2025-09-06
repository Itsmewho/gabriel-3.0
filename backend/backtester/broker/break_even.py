# Break-even

from . import Trade, PIP_SIZE
from .audit import log


def update_break_even(
    tr: Trade,
    high: float,
    low: float,
    trigger_pips: float | None,
    offset_pips: float = 0.0,
    events_log: list[dict] | None = None,
    t=None,
):
    """
    MT5-like break-even:
      - Require favorable move of `trigger_pips`.
      - Move SL to entry ± offset only if price actually traded strictly past that SL level on this bar.
      - Applies once per trade. Monotonic with trailing SL.
      - Does not touch TP (original or extended).
    """
    if not trigger_pips or tr.be_applied:
        return

    trig = float(trigger_pips) * PIP_SIZE
    off = float(offset_pips) * PIP_SIZE

    old_sl = tr.sl

    if tr.side == "buy":
        reached_trigger = (high - tr.entry_price) >= trig
        candidate = tr.entry_price + off
        # Strictly below/high guard: must be strictly below the traded high
        market_ok = high > candidate
        improve = (old_sl is None) or (candidate > old_sl)
        if reached_trigger and market_ok and improve:
            tr.sl = candidate
    else:  # sell
        reached_trigger = (tr.entry_price - low) >= trig
        candidate = tr.entry_price - off
        # Strictly above/low guard: must be strictly above the traded low
        market_ok = low < candidate
        improve = (old_sl is None) or (candidate < old_sl)
        if reached_trigger and market_ok and improve:
            tr.sl = candidate

    if tr.sl != old_sl:
        tr.sl_reason = "break_even"
        tr.sl_mod_count += 1
        if tr.sl_first is None:
            tr.sl_first = old_sl if old_sl is not None else tr.sl
        tr.sl_last = tr.sl
        tr.be_applied = True
        tr.be_price = tr.sl
        if events_log is not None and t is not None:
            log(
                events_log,
                type="break_even",
                trade_id=tr.id,
                side=tr.side,
                old_sl=old_sl,
                new_sl=tr.sl,
                time=t,
            )
