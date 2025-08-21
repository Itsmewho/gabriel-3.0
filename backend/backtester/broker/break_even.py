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
    Move SL to break-even (optionally with +offset pips) once price moves in favor
    by `trigger_pips`. Applies ONCE per trade (tr.be_applied guards repeat).
    Direction-aware:
      - BUY: trigger if high - entry >= trigger; SL -> entry + offset
      - SELL: trigger if entry - low >= trigger; SL -> entry - offset
    """
    if not trigger_pips or tr.be_applied:
        return

    trig = trigger_pips * PIP_SIZE
    off = offset_pips * PIP_SIZE

    if tr.side == "buy":
        reached = (high - tr.entry_price) >= trig
        new_sl = tr.entry_price + off
        improve = (tr.sl is None) or (new_sl > tr.sl)
    else:
        reached = (tr.entry_price - low) >= trig
        new_sl = tr.entry_price - off
        improve = (tr.sl is None) or (new_sl < tr.sl)

    if reached and improve:
        old_sl = tr.sl
        tr.sl = new_sl
        tr.sl_reason = "break_even"
        tr.sl_mod_count += 1
        if tr.sl_first is None:
            tr.sl_first = old_sl if old_sl is not None else new_sl
        tr.sl_last = new_sl
        tr.be_applied = True
        tr.be_price = new_sl
        if events_log is not None and t is not None:
            log(
                events_log,
                type="break_even",
                trade_id=tr.id,
                side=tr.side,
                old_sl=old_sl,
                new_sl=new_sl,
                time=t,
            )
