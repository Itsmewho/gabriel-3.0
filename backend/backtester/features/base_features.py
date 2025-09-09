# --- backtester/features/base_features.py  ---
"""
Basic TA feature builders for SMA, EMA, RSI, ATR, Bollinger Bands, and Keltner Channels.
Adds SMA on highs and lows: keys `sma_high` and `sma_low`.
Assumes df columns: ['open','high','low','close','tick_volume'].

Keltner config accepts either a dict or a list of dicts under key "kc":
  kc = {
    "n": 20,          # moving average length for midline
    "m": 2.0,         # ATR multiplier
    "atr_n": 20,      # ATR length (defaults to n if omitted)
    "ma": "ema"       # midline type: "ema" or "sma" (default: ema)
  }
Outputs columns: kc_{n}_{atrn}_{m}_mid, kc_{n}_{atrn}_{m}_upper, kc_{n}_{atrn}_{m}_lower

Stochastic config accepts either a dict or a list of dicts under key "stoch":
  stoch = {
    "k_period": 14,     # K period
    "d_period": 3,      # D period
    "slowing": 3        # Slowing
  }
Outputs columns: stoch_{k_period}_{d_period}_{slowing}_k, stoch_{k_period}_{d_period}_{slowing}_d

"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Iterable, Optional, List, Union

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


def _keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 20,
    m: float = 2.0,
    atr_n: Optional[int] = None,
    ma: str = "ema",
) -> pd.DataFrame:
    n = int(n)
    atr_n = int(atr_n) if atr_n is not None else n
    m = float(m)

    if ma.lower() == "sma":
        mid = _sma(close, n)
    else:
        mid = _ema(close, n)

    rng = _atr(high, low, close, atr_n)
    upper = mid + m * rng
    lower = mid - m * rng

    cols = {
        f"kc_{n}_{atr_n}_{m}_mid": mid,
        f"kc_{n}_{atr_n}_{m}_upper": upper,
        f"kc_{n}_{atr_n}_{m}_lower": lower,
    }
    return pd.DataFrame(cols)


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    slowing: int = 3,
) -> pd.DataFrame:
    """Calculates the Slow Stochastic Oscillator (%K and %D)."""
    k_period = int(k_period)
    d_period = int(d_period)
    slowing = int(slowing)

    # Find the Highest High and Lowest Low over the K-Period
    highest_high = high.rolling(window=k_period).max()
    lowest_low = low.rolling(window=k_period).min()

    # Calculate numerator and denominator for the Slow %K
    num = (close - lowest_low).rolling(window=slowing).sum()
    den = (highest_high - lowest_low).rolling(window=slowing).sum()

    # Calculate Slow %K line, handling division by zero
    k_line = 100 * (num / den)
    k_line.replace([np.inf, -np.inf], 100, inplace=True)
    k_line.fillna(100, inplace=True)

    # Calculate Slow %D line (SMA of %K)
    d_line = _sma(k_line, d_period)

    # Construct DataFrame with named columns
    base_name = f"stoch_{k_period}_{d_period}_{slowing}"
    cols = {f"{base_name}_k": k_line, f"{base_name}_d": d_line}
    return pd.DataFrame(cols)


# --------- public API ---------


def apply_basic_features(
    df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Enrich df with selected TA features.

    cfg keys (all optional):
      ...
      "stoch": dict or list of dicts, see module docstring
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

    # Bollinger Bands
    if "bb" in cfg:
        bb = cfg.get("bb", {})
        bb_df = _bollinger(close, int(bb.get("n", 20)), float(bb.get("k", 2.0)))
        df = df.join(bb_df)

    # Keltner Channels
    if "kc" in cfg:
        kc_cfgs: Union[Dict[str, Any], List[Dict[str, Any]]] = cfg["kc"]
        if isinstance(kc_cfgs, dict):
            kc_cfgs = [kc_cfgs]
        for kc in kc_cfgs:
            n = int(kc.get("n", 20))
            m = float(kc.get("m", 2.0))
            atr_n = kc.get("atr_n")
            ma = kc.get("ma", "ema")
            kc_df = _keltner(high, low, close, n=n, m=m, atr_n=atr_n, ma=ma)
            df = df.join(kc_df)

    # Stochastic Oscillator
    if "stoch" in cfg:
        stoch_cfgs: Union[Dict[str, Any], List[Dict[str, Any]]] = cfg["stoch"]
        if isinstance(stoch_cfgs, dict):
            stoch_cfgs = [stoch_cfgs]
        for stoch in stoch_cfgs:
            k = int(stoch.get("k_period", 14))
            d = int(stoch.get("d_period", 3))
            s = int(stoch.get("slowing", 3))
            stoch_df = _stochastic(high, low, close, k_period=k, d_period=d, slowing=s)
            df = df.join(stoch_df)

    return df


__all__ = ["apply_basic_features"]


# --- backtester/features/features_cache.py (patch guidance) ---
# Extend _spec_columns and _sub_spec_for_missing to support sma_high/sma_low and kc
#
# in _spec_columns(spec):
#   add:
#     for n in spec.get("sma_high", []): out.add(f"sma_high_{int(n)}")
#     for n in spec.get("sma_low", []): out.add(f"sma_low_{int(n)}")
#   for kc in ensure_list(spec.get("kc", [])):
#       n = int(kc.get("n", 20)); atrn = int(kc.get("atr_n", n)); m = float(kc.get("m", 2.0))
#       out |= {f"kc_{n}_{atrn}_{m}_mid", f"kc_{n}_{atrn}_{m}_upper", f"kc_{n}_{atrn}_{m}_lower"}
#
# in _sub_spec_for_missing(spec, missing):
#   mirror sma blocks for sma_high and sma_low
#   for kc in ensure_list(spec.get("kc", [])):
#       n = int(kc.get("n", 20)); atrn = int(kc.get("atr_n", n)); m = float(kc.get("m", 2.0))
#       need_mid = f"kc_{n}_{atrn}_{m}_mid" in missing
#       need_up  = f"kc_{n}_{atrn}_{m}_upper" in missing
#       need_lo  = f"kc_{n}_{atrn}_{m}_lower" in missing
#       if need_mid or need_up or need_lo:
#           (sub.setdefault("kc", [])).append({"n": n, "atr_n": atrn, "m": m, "ma": kc.get("ma", "ema")})
#
# helper ensure_list(x): return x if isinstance(x, list) else ([x] if x else [])
#
# --- Example request spec ---
# feature_spec = {
#   "sma": [12,14,20,30,50,150],
#   "sma_high": [20,50,150],
#   "sma_low": [20,50,150],
#   "ema": [14,18,20,24,30,40,50,130,150],
#   "rsi": [14],
#   "atr": [14],
#   "vol_sma": [10,14,20,30,40,50],
#   "kc": [{"n":20, "m":2.0, "atr_n":20, "ma":"ema"}],
#   "bb": {"n": 20, "k": 2.0},
# }
