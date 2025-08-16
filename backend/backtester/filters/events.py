import pandas as pd


def add_event_block(
    df: pd.DataFrame,
    *,
    before_minutes: int = 30,
    after_minutes: int = 30,
    minutes_col: str = "event_minutes_to_event",
    out_col: str = "blocked_by_event",
) -> pd.DataFrame:
    """Add a no-trade flag around economic events using server-time deltas.

    Logic: block if -after_minutes <= minutes_to_event <= before_minutes.
    If the minutes column is missing, returns zeros (no block).
    No look-ahead: uses per-bar signed minutes-to-event from your pipeline.
    """
    out = df.copy()
    if minutes_col not in out.columns:
        out[out_col] = 0
        return out

    m = pd.to_numeric(out[minutes_col], errors="coerce")
    mask = (m >= -abs(after_minutes)) & (m <= abs(before_minutes))
    out[out_col] = mask.astype(int)
    # Optional diagnostics
    out["event_block_before"] = before_minutes
    out["event_block_after"] = after_minutes
    return out
