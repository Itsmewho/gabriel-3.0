# Base_features.py

"""
Basic TA feature builders for SMA, EMA, RSI, ATR, MACD, Ichimoku, Bollinger Bands.

Assumes input DataFrame index is time and has columns: ['open','high','low','close','tick_volume'].

Usage:
    from basic_ta_features import apply_basic_features
    df = apply_basic_features(df)

All functions are vectorized. No external deps beyond pandas/numpy.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Iterable, Optional


# --------- core helpers ---------


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's smoothing via EWM(alpha=1/period)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()  # Wilder style
    return atr


def _macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "macd_signal": sig, "macd_hist": hist})


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = _sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return pd.DataFrame(
        {f"bb_{n}_mid": mid, f"bb_{n}_upper": upper, f"bb_{n}_lower": lower}
    )


# --------- public API ---------


def apply_basic_features(
    df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Enrich df with selected TA features.

    cfg schema (all keys optional):
        {
          "sma": [20, 50, 200],
          "ema": [12, 26, 50],
          "rsi": [14],
          "atr": [14],
          "macd": {"fast": 12, "slow": 26, "signal": 9},
          "bb": {"n": 20, "k": 2.0},
          "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52, "shift": 26}
        }
    """
    _require_cols(df, ["open", "high", "low", "close"])
    close = df["close"]

    cfg = cfg or {}

    # SMA
    for n in cfg.get("sma", [20, 50, 200]):
        df[f"sma_{n}"] = _sma(close, int(n))

    # EMA
    for n in cfg.get("ema", [12, 26, 50]):
        df[f"ema_{n}"] = _ema(close, int(n))

    # RSI
    for n in cfg.get("rsi", [14]):
        df[f"rsi_{n}"] = _rsi(close, int(n))

    # ATR
    for n in cfg.get("atr", [14]):
        df[f"atr_{n}"] = _atr(df["high"], df["low"], close, int(n))

    # Volume SMA
    for n in cfg.get("vol_sma", []):
        df[f"volume_sma_{int(n)}"] = (
            df["tick_volume"].rolling(int(n), min_periods=int(n)).mean()
        )

    # Bollinger
    bb_cfg = cfg.get("bb", {"n": 20, "k": 2.0})
    bb_df = _bollinger(close, int(bb_cfg.get("n", 20)), float(bb_cfg.get("k", 2.0)))
    df = df.join(bb_df)

    return df


__all__ = [
    "apply_basic_features",
]
