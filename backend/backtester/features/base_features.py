# --- backtester/features/base_features.py (drop-in) ---
"""
Basic TA feature builders for SMA, EMA, RSI, ATR, Bollinger Bands.
Adds SMA on highs and lows: keys `sma_high` and `sma_low`.
Assumes df columns: ['open','high','low','close','tick_volume'].
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
    n = int(n)
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    period = int(period)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    period = int(period)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    n = int(n)
    k = float(k)
    mid = _sma(close, n)
    std = close.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return pd.DataFrame(
        {f"bb_{n}_mid": mid, f"bb_{n}_upper": upper, f"bb_{n}_lower": lower}
    )


# --------- public API ---------


def apply_basic_features(
    df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Enrich df with selected TA features.

    cfg keys (all optional):
      "sma": [20, 50, 200]                 # close-based SMA
      "sma_high": [20, 50]                 # high-based SMA
      "sma_low": [20, 50]                  # low-based SMA
      "ema": [12, 26, 50]
      "rsi": [14]
      "atr": [14]
      "vol_sma": [10, 20]
      "bb": {"n": 20, "k": 2.0}
    """
    _require_cols(df, ["open", "high", "low", "close"])
    close = df["close"]
    high = df["high"]
    low = df["low"]

    cfg = cfg or {}

    # SMA (close)
    for n in cfg.get("sma", []):
        df[f"sma_{int(n)}"] = _sma(close, n)

    # SMA (high/low)
    for n in cfg.get("sma_high", []):
        df[f"sma_high_{int(n)}"] = _sma(high, n)
    for n in cfg.get("sma_low", []):
        df[f"sma_low_{int(n)}"] = _sma(low, n)

    # EMA (close)
    for n in cfg.get("ema", []):
        df[f"ema_{int(n)}"] = _ema(close, n)

    # RSI
    for n in cfg.get("rsi", []):
        df[f"rsi_{int(n)}"] = _rsi(close, n)

    # ATR
    for n in cfg.get("atr", []):
        df[f"atr_{int(n)}"] = _atr(high, low, close, n)

    # Volume SMA
    for n in cfg.get("vol_sma", []):
        df[f"volume_sma_{int(n)}"] = (
            df["tick_volume"].rolling(int(n), min_periods=int(n)).mean()
        )

    # Bollinger
    if "bb" in cfg:
        bb_df = _bollinger(
            close,
            int(cfg.get("bb", {}).get("n", 20)),
            float(cfg.get("bb", {}).get("k", 2.0)),
        )
        df = df.join(bb_df)

    return df


__all__ = ["apply_basic_features"]


# --- backtester/features/features_cache.py (patches only) ---
# Extend _spec_columns and _sub_spec_for_missing to support sma_high/sma_low

# in _spec_columns(spec):
#   add:
#     for n in spec.get("sma_high", []): out.add(f"sma_high_{int(n)}")
#     for n in spec.get("sma_low", []): out.add(f"sma_low_{int(n)}")

# in _sub_spec_for_missing(spec, missing):
#   add blocks mirroring sma:
#     want_sma_high = []
#     for n in spec.get("sma_high", []):
#         if f"sma_high_{int(n)}" in missing: want_sma_high.append(int(n))
#     if want_sma_high: sub["sma_high"] = want_sma_high
#
#     want_sma_low = []
#     for n in spec.get("sma_low", []):
#         if f"sma_low_{int(n)}" in missing: want_sma_low.append(int(n))
#     if want_sma_low: sub["sma_low"] = want_sma_low


# --- Example request specs ---
# feature_spec = {
#   "sma": [12,14,20,30,50,150],
#   "sma_high": [20,50,150],
#   "sma_low": [20,50,150],
#   "ema": [14,18,20,24,30,40,50,130,150],
#   "rsi": [14],
#   "atr": [14],
#   "vol_sma": [10,14,20,30,40,50],
# }
