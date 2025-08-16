import pandas as pd
import numpy as np
from typing import Optional


# Wilder ATR
def add_atr(
    df: pd.DataFrame,
    *,
    atr_len: int = 14,
    out_col: str = "atr",
) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out[out_col] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()
    return out


def add_atr_volatility_guard(
    df: pd.DataFrame,
    *,
    atr_col: str = "atr",
    q_low: float = 0.05,
    q_high: float = 0.95,
    window: int = 10080,  # ~1 week of minutes
    out_flag: str = "atr_guard_ok",
) -> pd.DataFrame:
    """Flag bars where ATR is within rolling [q_low, q_high].

    Uses rolling quantiles shifted by 1 bar to avoid look-ahead.
    If atr_col missing or insufficient history, flag = 0.
    """
    out = df.copy()
    if atr_col not in out.columns:
        out[out_flag] = 0
        return out

    atr = pd.to_numeric(out[atr_col], errors="coerce")
    # Compute rolling quantiles then shift to exclude current bar info
    p5 = (
        atr.rolling(window=window, min_periods=max(10, int(window * 0.1)))
        .quantile(q_low)
        .shift(1)
    )
    p95 = (
        atr.rolling(window=window, min_periods=max(10, int(window * 0.1)))
        .quantile(q_high)
        .shift(1)
    )

    ok = (atr >= p5) & (atr <= p95)
    out[out_flag] = ok.fillna(0).astype(int)
    out["atr_p5"] = p5
    out["atr_p95"] = p95
    return out
