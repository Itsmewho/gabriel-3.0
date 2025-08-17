# --- ICT feature set (modular, no look-ahead) ---
import numpy as np
import pandas as pd


def add_pivots(
    df: pd.DataFrame,
    *,
    left: int = 3,
    right: int = 3,
) -> pd.DataFrame:
    """Add confirmed swing highs/lows. Confirmation uses right bars, so we align
    the pivot flag at the confirmation time to avoid look-ahead.
    """
    out = df.copy()
    w = left + right + 1
    # Centered rolling extrema
    rh = out["high"].rolling(w, center=True).max()
    rl = out["low"].rolling(w, center=True).min()
    ph_center = (out["high"] == rh).astype(int)
    pl_center = (out["low"] == rl).astype(int)
    # Shift forward by right bars so the signal appears when confirmed
    out["pivot_high"] = ph_center.shift(right).fillna(0).astype(int)
    out["pivot_low"] = pl_center.shift(right).fillna(0).astype(int)
    return out


def add_previous_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous day high/low/close mapped to each minute bar.
    Uses server date from 'time'. No look-ahead. Correctly maps *previous* day.
    """
    out = df.copy()
    if "time" not in out.columns:
        raise ValueError("df requires 'time' for previous-day levels")

    # Server date buckets
    out["__date"] = pd.to_datetime(out["time"]).dt.floor("D")

    # Daily OHLC for each server date
    daily = (
        out.groupby("__date")
        .agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
        )
        .reset_index()
    )

    # Shift one day to align current date with *previous* day's stats
    daily["prev_date"] = daily["__date"].shift(1)

    # Map current minute rows by their server date to previous day's levels
    map_df = daily[["prev_date", "day_high", "day_low", "day_close"]].rename(
        columns={"prev_date": "map_date"}
    )
    out = out.merge(map_df, left_on="__date", right_on="map_date", how="left")

    # Rename and clean
    out.rename(
        columns={"day_high": "pdh", "day_low": "pdl", "day_close": "pdc"},
        inplace=True,
    )
    out.drop(columns=["map_date"], inplace=True)

    return out


def add_session_high_low(df: pd.DataFrame) -> pd.DataFrame:
    """Add running session high/low and distances. Requires session_id from add_time_and_session_features()."""
    out = df.copy()
    if "session_id" not in out.columns:
        raise ValueError(
            "Missing 'session_id'. Run add_time_and_session_features() first."
        )
    # Build a run id that increments when session_id changes
    sid = out["session_id"].to_numpy()
    run_id = np.zeros(len(out), dtype=int)
    rid = 0
    prev = sid[0] if len(sid) else 0
    for i, s in enumerate(sid):
        if i == 0:
            run_id[i] = rid
            continue
        if s != prev:
            rid += 1
        run_id[i] = rid
        prev = s
    out["session_run_id"] = run_id
    # Cum highs/lows within each run
    grp = out.groupby("session_run_id", sort=False, as_index=False)
    out["session_high"] = grp["high"].cummax().values
    out["session_low"] = grp["low"].cummin().values
    out["dist_to_session_high"] = out["close"] - out["session_high"]
    out["dist_to_session_low"] = out["close"] - out["session_low"]
    return out


def add_ict_fvg(
    df: pd.DataFrame,
    *,
    min_gap_points: float = 0.0,
) -> pd.DataFrame:
    """Detect fair value gaps (FVG) using classic 3-candle pattern.
    Bullish: low[t] > high[t-2]; Bearish: high[t] < low[t-2].
    Flags appear at bar t (confirmation on close of t). No look-ahead.
    Tracks simple mitigation: mid-tag and full-fill.
    """
    out = df.copy()
    h = out["high"].astype(float)
    l = out["low"].astype(float)

    up_gap = l > h.shift(2)
    dn_gap = h < l.shift(2)

    up_size = (l - h.shift(2)).where(up_gap, 0.0)
    dn_size = (h.shift(2) - l).where(dn_gap, 0.0)

    if min_gap_points > 0:
        up_gap = up_gap & (up_size >= min_gap_points)
        dn_gap = dn_gap & (dn_size >= min_gap_points)

    out["fvg_up"] = up_gap.astype(int)
    out["fvg_dn"] = dn_gap.astype(int)
    out["fvg_up_size"] = up_size.fillna(0.0)
    out["fvg_dn_size"] = dn_size.fillna(0.0)
    out["fvg_up_mid"] = ((l + h.shift(2)) / 2).where(up_gap)
    out["fvg_dn_mid"] = ((h + l.shift(2)) / 2).where(dn_gap)

    # Track active and mitigation sequentially
    n = len(out)
    up_active = np.zeros(n, dtype=int)
    dn_active = np.zeros(n, dtype=int)
    up_mid_tag = np.zeros(n, dtype=int)
    dn_mid_tag = np.zeros(n, dtype=int)
    up_filled = np.zeros(n, dtype=int)
    dn_filled = np.zeros(n, dtype=int)

    cur_up_low = np.nan  # lower bound = high[t-2]
    cur_up_high = np.nan  # upper bound = low[t]
    cur_up_mid = np.nan
    cur_dn_low = np.nan  # lower bound = low[t]
    cur_dn_high = np.nan  # upper bound = high[t-2]
    cur_dn_mid = np.nan

    for i in range(n):
        # Activate new gaps at i
        if out["fvg_up"].iat[i] == 1:
            cur_up_low = float(h.shift(2).iat[i])
            cur_up_high = float(l.iat[i])
            cur_up_mid = (cur_up_low + cur_up_high) / 2
        if out["fvg_dn"].iat[i] == 1:
            cur_dn_low = float(l.shift(2).iat[i])
            cur_dn_high = float(h.iat[i])
            cur_dn_mid = (cur_dn_low + cur_dn_high) / 2

        # Evaluate mitigation for active gaps
        lo = float(l.iat[i])
        hi = float(h.iat[i])
        # Bullish gap gets mitigated when price trades back into [cur_up_low, cur_up_high]
        if not np.isnan(cur_up_low):
            up_active[i] = 1
            if lo <= cur_up_mid <= hi:
                up_mid_tag[i] = 1
            if lo <= cur_up_low:  # full fill when low pierces lower bound
                up_filled[i] = 1
                cur_up_low = np.nan
                cur_up_high = np.nan
                cur_up_mid = np.nan
        # Bearish gap mitigation
        if not np.isnan(cur_dn_low):
            dn_active[i] = 1
            if lo <= cur_dn_mid <= hi:
                dn_mid_tag[i] = 1
            if hi >= cur_dn_high:  # full fill when high pierces upper bound
                dn_filled[i] = 1
                cur_dn_low = np.nan
                cur_dn_high = np.nan
                cur_dn_mid = np.nan

    out["fvg_up_active"] = up_active
    out["fvg_dn_active"] = dn_active
    out["fvg_up_mid_tag"] = up_mid_tag
    out["fvg_dn_mid_tag"] = dn_mid_tag
    out["fvg_up_filled"] = up_filled
    out["fvg_dn_filled"] = dn_filled
    return out


def add_ict_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate order blocks using last opposite candle before displacement that creates an FVG.
    - Bullish OB: last down candle before an up FVG
    - Bearish OB: last up candle before a down FVG
    OB zone uses candle body [min(open,close), max(open,close)]. Tracks touch/mitigation.
    """
    out = df.copy()
    if "fvg_up" not in out.columns or "fvg_dn" not in out.columns:
        raise ValueError("Run add_ict_fvg() first")

    o = out["open"].astype(float)
    c = out["close"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)

    n = len(out)
    bull_low = np.full(n, np.nan)
    bull_high = np.full(n, np.nan)
    bear_low = np.full(n, np.nan)
    bear_high = np.full(n, np.nan)
    bull_active = np.zeros(n, dtype=int)
    bear_active = np.zeros(n, dtype=int)
    bull_mitigated = np.zeros(n, dtype=int)
    bear_mitigated = np.zeros(n, dtype=int)

    last_down_body = None  # (low_body, high_body)
    last_up_body = None

    for i in range(n):
        # Track most recent opposite bodies
        # --- FIX: Use .iat[] for positional access ---
        is_down = c.iat[i] < o.iat[i]
        is_up = c.iat[i] > o.iat[i]
        if is_down:
            last_down_body = (min(o.iat[i], c.iat[i]), max(o.iat[i], c.iat[i]))
        if is_up:
            last_up_body = (min(o.iat[i], c.iat[i]), max(o.iat[i], c.iat[i]))
        # --- END FIX ---

        # On FVG up, set bullish OB from last down body
        if out["fvg_up"].iat[i] == 1 and last_down_body is not None:
            bull_low[i], bull_high[i] = last_down_body
        # On FVG down, set bearish OB from last up body
        if out["fvg_dn"].iat[i] == 1 and last_up_body is not None:
            bear_low[i], bear_high[i] = last_up_body

        # Active flags if an OB exists and not mitigated yet
        if not np.isnan(bull_low[i]) and not np.isnan(bull_high[i]):
            bull_active[i] = 1
        if not np.isnan(bear_low[i]) and not np.isnan(bear_high[i]):
            bear_active[i] = 1

        # Mitigation check: price trades into the zone
        # --- FIX: Use .iat[] for positional access ---
        if bull_active[i] == 1 and (
            l.iat[i] <= bull_high[i] and h.iat[i] >= bull_low[i]
        ):
            bull_mitigated[i] = 1
        if bear_active[i] == 1 and (
            l.iat[i] <= bear_high[i] and h.iat[i] >= bear_low[i]
        ):
            bear_mitigated[i] = 1
        # --- END FIX ---

    out["ob_bull_low"] = bull_low
    out["ob_bull_high"] = bull_high
    out["ob_bull_active"] = bull_active
    out["ob_bull_mitigated"] = bull_mitigated

    out["ob_bear_low"] = bear_low
    out["ob_bear_high"] = bear_high
    out["ob_bear_active"] = bear_active
    out["ob_bear_mitigated"] = bear_mitigated
    return out


def add_liquidity_sweeps(
    df: pd.DataFrame,
    *,
    swing_left: int = 3,
    swing_right: int = 3,
) -> pd.DataFrame:
    """Detect sweeps of prior swing highs/lows with close back inside.
    Uses confirmed pivots with right-bar confirmation. No look-ahead.
    """
    out = df.copy()
    # Ensure pivots exist
    if "pivot_high" not in out.columns or "pivot_low" not in out.columns:
        out = add_pivots(out, left=swing_left, right=swing_right)

    # Track last confirmed swing levels
    n = len(out)
    last_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)
    cur_high = np.nan
    cur_low = np.nan

    for i in range(n):
        if out["pivot_high"].iat[i] == 1:
            cur_high = float(out["high"].iat[i])
        if out["pivot_low"].iat[i] == 1:
            cur_low = float(out["low"].iat[i])
        last_high[i] = cur_high
        last_low[i] = cur_low

    out["last_pivot_high"] = last_high
    out["last_pivot_low"] = last_low

    # Sweep definitions
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)

    sweep_up = (
        (h > out["last_pivot_high"])
        & (c < out["last_pivot_high"])
        & pd.notna(out["last_pivot_high"])
    )  # took highs, closed back below
    sweep_dn = (
        (l < out["last_pivot_low"])
        & (c > out["last_pivot_low"])
        & pd.notna(out["last_pivot_low"])
    )  # took lows, closed back above

    out["sweep_up"] = sweep_up.astype(int)
    out["sweep_dn"] = sweep_dn.astype(int)
    return out


def add_premium_discount(
    df: pd.DataFrame,
    *,
    swing_left: int = 3,
    swing_right: int = 3,
) -> pd.DataFrame:
    """Compute 50% equilibrium of most recent swing range (from last low to last high in sequence).
    Premium if close above EQ, discount if below. No look-ahead.
    """
    out = df.copy()
    if "pivot_high" not in out.columns or "pivot_low" not in out.columns:
        out = add_pivots(out, left=swing_left, right=swing_right)

    n = len(out)
    last_low_idx = -1
    last_high_idx = -1
    eq = np.full(n, np.nan)
    prem = np.zeros(n, dtype=int)
    disc = np.zeros(n, dtype=int)

    for i in range(n):
        if out["pivot_low"].iat[i] == 1:
            last_low_idx = i
        if out["pivot_high"].iat[i] == 1:
            last_high_idx = i
        # Define active range only if both exist and in sequence
        if last_low_idx != -1 and last_high_idx != -1:
            i1, i2 = sorted([last_low_idx, last_high_idx])
            lo = float(out["low"].iat[i1])
            hi = float(out["high"].iat[i2])
            if i1 < i2 and hi > lo:
                eq[i] = (hi + lo) / 2
                prem[i] = int(out["close"].iat[i] > eq[i])
                disc[i] = int(out["close"].iat[i] < eq[i])

    out["eq_50"] = eq
    out["in_premium"] = prem
    out["in_discount"] = disc
    return out
