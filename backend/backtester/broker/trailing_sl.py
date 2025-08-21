# trailing_sl.py

from . import Trade, PIP_SIZE
from .audit import log


def update_trailing_sl(
    tr: Trade,
    high: float,
    low: float,
    trail_pips: float | None,
    events_log,
    t=None,
):
    if trail_pips is None:
        return
    if tr.side == "buy":
        candidate = high - trail_pips * PIP_SIZE
        if candidate > (tr.sl or -1e9):
            tr.sl = candidate
            tr.sl_reason = "trailing_sl"
            tr.sl_mod_count += 1
            tr.sl_last = candidate
            log(events_log, type="sl_update", id=tr.id, time=t, new_sl=candidate)
    else:
        candidate = low + trail_pips * PIP_SIZE
        if candidate < (tr.sl or 1e9):
            tr.sl = candidate
            tr.sl_reason = "trailing_sl"
            tr.sl_mod_count += 1
            tr.sl_last = candidate
            log(events_log, type="sl_update", id=tr.id, time=t, new_sl=candidate)


def update_trailing_tp(
    tr: Trade,
    high: float,
    low: float,
    cfg,
    events_log=None,
    t=None,
    near_tp_buffer_pips: float | None = None,
    tp_extension_pips: float | None = None,
):
    if tr.tp is None:
        return

    buf = (near_tp_buffer_pips or cfg.NEAR_TP_BUFFER_PIPS) * PIP_SIZE
    ext = (tp_extension_pips or cfg.TP_EXTENSION_PIPS) * PIP_SIZE
    old_tp = tr.tp

    if tr.side == "buy":
        if (old_tp - high) <= buf:
            tr.tp = old_tp + ext
    else:
        if (low - old_tp) <= buf:
            tr.tp = old_tp - ext

    if tr.tp != old_tp:
        tr.tp_reason = "tp_extend"
        if tr.tp_first is None:
            tr.tp_first = old_tp
        tr.tp_last = tr.tp
        tr.tp_mod_count += 1
        if events_log is not None and t is not None:
            log(
                events_log,
                type="tp_extend",
                trade_id=tr.id,
                side=tr.side,
                old_tp=old_tp,
                new_tp=tr.tp,
                time=t,
            )
