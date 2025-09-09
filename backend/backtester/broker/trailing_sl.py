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
    """Monotonic trailing SL using absolute price action (high/low),
    clamped so SL never moves against the initial or previous SL.
    Long: SL can only go UP. Short: SL can only go DOWN.
    """
    if not trail_pips:
        return

    # Ensure first marker exists for auditing
    if tr.sl_first is None and getattr(tr, "sl", None) is not None:
        tr.sl_first = tr.sl

    if tr.side == "buy":
        # Candidate at current bar's adverse-immune location
        candidate = high - trail_pips * PIP_SIZE
        prev = tr.sl if tr.sl is not None else float("-inf")
        floor_first = tr.sl_first if tr.sl_first is not None else float("-inf")
        candidate = max(candidate, prev, floor_first)
        if candidate > prev:
            tr.sl = candidate
            tr.sl_reason = "trailing_sl"
            tr.sl_mod_count += 1
            tr.sl_last = candidate
            log(events_log, type="sl_update", id=tr.id, time=t, new_sl=candidate)
    else:
        candidate = low + trail_pips * PIP_SIZE
        prev = tr.sl if tr.sl is not None else float("inf")
        ceil_first = tr.sl_first if tr.sl_first is not None else float("inf")
        candidate = min(candidate, prev, ceil_first)
        if candidate < prev:
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
    """
    Monotonic TP extension using absolute price action (high/low).

    OPT-IN: Only runs if strategy explicitly provides per-trade params
    (near_tp_buffer_pips and/or tp_extension_pips). If both are None,
    this function does nothing (ignores cfg defaults).
    """
    if tr.tp is None:
        return

    # --- OPT-IN GUARD ---
    if near_tp_buffer_pips is None and tp_extension_pips is None:
        return

    # Treat non-positive values as disabled
    if (near_tp_buffer_pips is not None and near_tp_buffer_pips <= 0) or (
        tp_extension_pips is not None and tp_extension_pips <= 0
    ):
        return

    # Prefer per-trade values; if only one provided, both must be valid to proceed
    buf_src = near_tp_buffer_pips
    ext_src = tp_extension_pips
    if buf_src is None or ext_src is None:
        # If one is missing, do not fall back to cfg — remain opt-in only
        return

    buf = buf_src * PIP_SIZE
    ext = ext_src * PIP_SIZE
    old_tp = tr.tp

    if tr.tp_first is None:
        tr.tp_first = old_tp

    new_tp = old_tp  # default: unchanged

    if tr.side == "buy":
        # only consider extension when we're near the current TP
        if (old_tp - high) <= buf:
            proposed = old_tp + ext
            floor = max(old_tp, tr.tp_first)
            cand = max(proposed, floor)
            # guard: TP for buys must remain above market (use bar high as proxy)
            if cand > high:
                new_tp = cand
    else:
        if (low - old_tp) <= buf:
            proposed = old_tp - ext
            ceil = min(old_tp, tr.tp_first)
            cand = min(proposed, ceil)
            # guard: TP for sells must remain below market (use bar low as proxy)
            if cand < low:
                new_tp = cand

    if new_tp != old_tp:
        tr.tp = new_tp
        tr.tp_reason = "tp_extend"
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
