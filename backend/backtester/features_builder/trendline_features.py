import numpy as np
import pandas as pd


def add_trendline_features(
    df: pd.DataFrame,
    *,
    daily_len: int = 1440,  # 1d in 1m bars
    session_len: int = 240,  # 4h session approx
    hourly_len: int = 60,
    short_len: int = 30,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Add simplified trendline features by regressing price over different horizons.

    Uses rolling linear regression slope of close price (OLS slope). Approximates trendline direction.
    """
    from numpy.linalg import lstsq

    out = df.copy()

    def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
        if window < 2:
            raise ValueError("window must be >=2")
        x = np.arange(window, dtype=float)
        x = x - x.mean()

        def _slope(y: np.ndarray) -> float:
            y = y.astype(float)
            y = y - y.mean()
            denom = (x**2).sum()
            if denom == 0:
                return 0.0
            return float((x * y).sum() / denom)

        return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)

    out["trendline_daily_slope"] = _rolling_slope(out["close"], daily_len)
    out["trendline_session_slope"] = _rolling_slope(out["close"], session_len)
    out["trendline_hourly_slope"] = _rolling_slope(out["close"], hourly_len)
    out["trendline_short_slope"] = _rolling_slope(out["close"], short_len)

    # Directional flags
    for col in [
        "trendline_daily_slope",
        "trendline_session_slope",
        "trendline_hourly_slope",
        "trendline_short_slope",
    ]:
        out[col + "_dir"] = np.sign(out[col]).fillna(0).astype(int)  # type: ignore

    if drop_warmup:
        out = out.dropna(
            subset=[
                "trendline_daily_slope",
                "trendline_session_slope",
                "trendline_hourly_slope",
                "trendline_short_slope",
            ]
        )

    return out


# --- Trendline and time-based features (modular, no look-ahead) ---


def add_time_and_session_features(
    df: pd.DataFrame,
    *,
    sessions: dict | None = None,
) -> pd.DataFrame:
    """Add day/hour/session features using server time.

    sessions format (defaults shown):
    {
      "asia": (0, 7),       # inclusive start hour, exclusive end hour
      "london": (7, 16),
      "newyork": (13, 22)
    }
    """
    out = df.copy()
    if "time" not in out.columns:
        raise ValueError("df requires a 'time' column for session features")

    if sessions is None:
        sessions = {"asia": (0, 7), "london": (7, 16), "newyork": (13, 22)}

    t = pd.to_datetime(out["time"])  # server time
    out["day_of_week"] = t.dt.dayofweek  # 0=Mon
    out["hour"] = t.dt.hour
    out["minute"] = t.dt.minute

    # Session flags
    for name, (h0, h1) in sessions.items():
        out[f"is_{name}"] = ((out["hour"] >= h0) & (out["hour"] < h1)).astype(int)

    # Single session id: 0 none, 1 asia, 2 london, 3 newyork (priority by dict order)
    sid = np.zeros(len(out), dtype=int)
    sid_val = 1
    for name in sessions.keys():
        sid = np.where(out[f"is_{name}"].to_numpy(dtype=bool), sid_val, sid)
        sid_val += 1
    out["session_id"] = sid

    # Minutes into current session (reset when session changes)
    in_session = sid > 0
    minutes = np.where(in_session, 1, 0)
    # Accumulate while session id stays constant
    acc = np.zeros(len(out), dtype=int)
    prev_sid = 0
    m = 0
    for i, s in enumerate(sid):
        if s == prev_sid and s != 0:
            m += 1
        elif s != 0:
            m = 1
        else:
            m = 0
        acc[i] = m
        prev_sid = s
    out["minutes_into_session"] = acc

    return out


def _rolling_linreg_slope_r2(y: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Compute rolling OLS slope and R^2 of y ~ x where x is 0..window-1 per window.
    Vectorized via cumulative sums. Returns two Series aligned with y.
    """
    if window < 2:
        raise ValueError("window must be >= 2")

    y = y.astype(float)
    n = window
    x = np.arange(n, dtype=float)
    Sx = x.sum()
    Sxx = (x * x).sum()
    denom = n * Sxx - Sx * Sx

    # Rolling sums of y and y*x
    y_vals = y.to_numpy(dtype=float)
    # Build rolling windows via convolution for efficiency
    k1 = np.ones(n, dtype=float)
    Sy = np.convolve(y_vals, k1, mode="full")[n - 1 : n - 1 + len(y_vals)]
    # y * x uses fixed x weights over the last n samples
    # Equivalent to sum_{i=0}^{n-1} y[t-n+1+i]*x[i]
    # Implement as convolution with x
    Syx = np.convolve(y_vals, x, mode="full")[n - 1 : n - 1 + len(y_vals)]

    # Rolling sums of y^2 for R^2
    Sy2 = np.convolve(y_vals * y_vals, k1, mode="full")[n - 1 : n - 1 + len(y_vals)]

    # Mask for first n-1 positions (insufficient data)
    mask = np.arange(len(y_vals)) >= (n - 1)

    slope = np.full(len(y_vals), np.nan, dtype=float)
    r2 = np.full(len(y_vals), np.nan, dtype=float)

    # Compute slope and R^2 only where window is full
    idx = np.where(mask)[0]
    # Numerator for slope: n*Syx - Sx*Sy
    num = n * Syx[idx] - Sx * Sy[idx]
    beta1 = num / denom
    slope[idx] = beta1

    # Intercept: (Sy - beta1*Sx)/n  (not returned)

    # R^2 computation: 1 - SSE/SST
    # SSE = sum((y - (a + b*x))^2) = Sy2 - 2*a*Sy - 2*b*Syx + n*a^2 + 2*a*b*Sx + b^2*Sxx
    # But easier: use correlation between x and y: R^2 = corr(x,y)^2
    # corr = cov(x,y) / sqrt(var(x)*var(y))
    mean_y = Sy[idx] / n
    # var(y) over window: E[y^2] - (E[y])^2
    var_y = (Sy2[idx] / n) - (mean_y**2)
    # cov(x,y) = (Syx/n) - (Sx/n)*(Sy/n)
    cov_xy = (Syx[idx] / n) - (Sx / n) * (Sy[idx] / n)
    var_x = (Sxx / n) - (Sx / n) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        corr2 = (cov_xy * cov_xy) / (var_x * var_y)
    r2[idx] = np.clip(corr2, 0.0, 1.0)

    slope_s = pd.Series(slope, index=y.index, name=f"slope_{window}")
    r2_s = pd.Series(r2, index=y.index, name=f"r2_{window}")
    return slope_s, r2_s


def add_hourly_trendline_features(
    df: pd.DataFrame,
    *,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """Linear-regression slope and R^2 over rolling 60 minutes as trendline proxy.
    No look-ahead. Uses index-local x=0..n-1 per window.
    """
    out = df.copy()
    slope, r2 = _rolling_linreg_slope_r2(out["close"], window_minutes)
    out[f"linreg_slope_{window_minutes}m"] = slope
    out[f"linreg_r2_{window_minutes}m"] = r2
    return out


def add_daily_trendline_features(
    df: pd.DataFrame,
    *,
    window_minutes: int = 1440,
) -> pd.DataFrame:
    """Linear-regression slope and R^2 over rolling 1 day (1440 minutes)."""
    out = df.copy()
    slope, r2 = _rolling_linreg_slope_r2(out["close"], window_minutes)
    out[f"linreg_slope_{window_minutes}m"] = slope
    out[f"linreg_r2_{window_minutes}m"] = r2
    return out


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


def add_trendline_from_pivots(
    df: pd.DataFrame,
    *,
    use_highs: bool = True,
    lookback_bars: int = 500,
) -> pd.DataFrame:
    """Derive a dynamic trendline from last two confirmed pivots (highs or lows).
    Computes line value at current bar and distance of close to that line.
    Updates causally at each bar. No look-ahead.
    """
    out = df.copy()
    piv_col = "pivot_high" if use_highs else "pivot_low"
    price_col = "high" if use_highs else "low"

    if piv_col not in out.columns:
        raise ValueError("Run add_pivots() first")

    n = len(out)
    tl_val = np.full(n, np.nan, dtype=float)
    tl_slope = np.full(n, np.nan, dtype=float)
    tl_intercept = np.full(n, np.nan, dtype=float)
    dist = np.full(n, np.nan, dtype=float)
    last_idx = []

    for i in range(n):
        if out[piv_col].iat[i] == 1:
            last_idx.append(i)
            if len(last_idx) > 2:
                last_idx.pop(0)
        # Use only bars within lookback
        last_idx = [j for j in last_idx if i - j <= lookback_bars]
        if len(last_idx) == 2:
            i1, i2 = last_idx[0], last_idx[1]
            y1 = float(out[price_col].iat[i1])
            y2 = float(out[price_col].iat[i2])
            if i2 != i1:
                b = (y2 - y1) / (i2 - i1)  # slope per bar
                a = y2 - b * i2  # intercept
                tl_slope[i] = b
                tl_intercept[i] = a
                tl_val[i] = a + b * i
                dist[i] = float(out["close"].iat[i]) - tl_val[i]

    name = "highs" if use_highs else "lows"
    out[f"tl_{name}_value"] = tl_val
    out[f"tl_{name}_slope"] = tl_slope
    out[f"tl_{name}_intercept"] = tl_intercept
    out[f"dist_close_to_tl_{name}"] = dist
    # Break flags when close crosses the active trendline value
    out[f"tl_{name}_break_up"] = (
        (out["close"] > out[f"tl_{name}_value"]) & pd.notna(out[f"tl_{name}_value"])
    ).astype(int)
    out[f"tl_{name}_break_down"] = (
        (out["close"] < out[f"tl_{name}_value"]) & pd.notna(out[f"tl_{name}_value"])
    ).astype(int)
    return out
