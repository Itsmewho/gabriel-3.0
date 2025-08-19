# trailing_sl.py

from . import Trade, PIP_SIZE
from .audit import log


def update_trailing_sl(
    tr: Trade,
    high: float,
    low: float,
    trail_pips: float | None,
    events_log=None,
    t=None,
):
    if trail_pips is None:
        return
    if tr.side == "buy":
        new_sl = max(tr.sl or -1e9, high - trail_pips * PIP_SIZE)
    else:
        new_sl = min(tr.sl or 1e9, low + trail_pips * PIP_SIZE)
    if new_sl != tr.sl:
        old_sl = tr.sl
        if tr.sl_first is None:
            tr.sl_first = old_sl
        tr.sl_last = new_sl
        tr.sl_mod_count += 1
        tr.sl = new_sl
        if events_log is not None and t is not None:
            log(
                events_log,
                type="sl_mod",
                trade_id=tr.id,
                old_sl=old_sl,
                new_sl=new_sl,
                time=t,
            )
