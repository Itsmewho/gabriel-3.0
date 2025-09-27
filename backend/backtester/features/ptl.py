from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numba import njit


@dataclass
class PTLSpec:
    fast: int = 15
    slow: int = 18


@njit
def _ptl_loop(close, ptl_line, upper_band, lower_band):
    # This Numba-compiled loop correctly implements the core path-dependent
    # logic from MQL4: slowln[i] = (close[i]>slowln[i-1]) ? tlows : thighs;
    for i in range(1, len(close)):
        if close[i] > ptl_line[i - 1]:
            ptl_line[i] = lower_band[i]
        else:
            ptl_line[i] = upper_band[i]
    return ptl_line


def compute_ptl(df: pd.DataFrame, *, fast: int = 15, slow: int = 11) -> pd.DataFrame:
    """
    A perfect Python translation of the MQL4 'Perfect trend line' indicator.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must be indexed by DatetimeIndex")
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Use forward-filled data to prevent calculation errors from any missing values
    high = df["high"].astype(float).ffill()
    low = df["low"].astype(float).ffill()
    close = df["close"].astype(float).ffill()

    if close.empty:
        return pd.DataFrame(
            columns=["ptl_slow", "ptl_fast", "ptl_trend", "ptl_trena"],
            index=df.index,
        )

    # Corresponds to ArrayMaximum/ArrayMinimum in MQL4
    fastH = high.rolling(window=int(fast), min_periods=1).max()
    fastL = low.rolling(window=int(fast), min_periods=1).min()
    slowH = high.rolling(window=int(slow), min_periods=1).max()
    slowL = low.rolling(window=int(slow), min_periods=1).min()

    # Corresponds to thighs/tlows/thighf/tlowf in MQL4
    thighs = pd.Series(np.maximum(high.values, slowH.values), index=df.index)
    tlows = pd.Series(np.minimum(low.values, slowL.values), index=df.index)
    thighf = pd.Series(np.maximum(high.values, fastH.values), index=df.index)
    tlowf = pd.Series(np.minimum(low.values, fastL.values), index=df.index)

    # Initialize PTL lines and run the Numba loop
    ptl_slow = pd.Series(np.nan, index=df.index)
    ptl_slow.iloc[0] = close.iloc[0]
    ptl_fast = pd.Series(np.nan, index=df.index)
    ptl_fast.iloc[0] = close.iloc[0]
    ptl_slow = pd.Series(
        _ptl_loop(close.values, ptl_slow.values, thighs.values, tlows.values),
        index=df.index,
    )
    ptl_fast = pd.Series(
        _ptl_loop(close.values, ptl_fast.values, thighf.values, tlowf.values),
        index=df.index,
    )

    # MQL4 uses 'trend = 1' for BEARISH and 'trend = 0' for BULLISH.
    conditions_trend = [
        (close < ptl_slow) & (close < ptl_fast),
        (close > ptl_slow) & (close > ptl_fast),
    ]
    choices_trend = [1, 0]  # 1 for Bear, 0 for Bull
    ptl_trend = pd.Series(
        np.select(conditions_trend, choices_trend, default=-1), index=df.index
    )

    # MQL4 uses two 'if' statements, allowing the second to override the first.
    ptl_trena = pd.Series(np.nan, index=df.index)

    # MQL4 trend==1 is BEARISH, so this means "slow > fast OR bearish signal" -> trena=1 (Down Trend)
    go_down_condition = (ptl_slow > ptl_fast) | (ptl_trend == 1)

    # MQL4 trend==0 is BULLISH, so this means "slow < fast OR bullish signal" -> trena=0 (Up Trend)
    go_up_condition = (ptl_slow < ptl_fast) | (ptl_trend == 0)

    ptl_trena[go_down_condition] = 1  # MQL4 author uses 1 for Down Trend state
    ptl_trena[go_up_condition] = 0  # MQL4 author uses 0 for Up Trend state

    # Forward fill the state to carry it through time
    ptl_trena.iloc[0] = 0  # Match MQL4 initialization
    ptl_trena = ptl_trena.ffill().astype(int)

    return pd.DataFrame(
        {
            "ptl_slow": ptl_slow,
            "ptl_fast": ptl_fast,
            "ptl_trend": ptl_trend,
            "ptl_trena": ptl_trena,
        }
    )


def add_ptl_features(df: pd.DataFrame, *, fast: int = 3, slow: int = 7) -> pd.DataFrame:
    """Join PTL columns to df and return df."""
    ptl = compute_ptl(df, fast=fast, slow=slow)
    df[ptl.columns] = ptl
    return df
