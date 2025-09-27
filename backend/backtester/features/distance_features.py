# backtester/features/distance_features.py
from __future__ import annotations
from typing import Iterable, Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np

PIP_SIZE_DEFAULT = 0.0001  # EURUSD


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")


# ---------------- EMA and Keltner distances ----------------


def add_ema_distances(
    df: pd.DataFrame,
    ema_pairs: Iterable[Tuple[int, int]],
    *,
    pip_size: float = PIP_SIZE_DEFAULT,
) -> pd.DataFrame:
    for f, s in ema_pairs:
        f, s = int(f), int(s)
        _ensure_cols(df, [f"ema_{f}", f"ema_{s}"])
        gap = df[f"ema_{f}"] - df[f"ema_{s}"]
        df[f"ema_gap_{f}_{s}_abs"] = gap
        df[f"ema_gap_{f}_{s}_pips"] = gap / pip_size
    return df


def add_kc_distances(
    df: pd.DataFrame,
    kc_specs: Iterable[Dict[str, Any]],
    *,
    pip_size: float = PIP_SIZE_DEFAULT,
) -> pd.DataFrame:
    _ensure_cols(df, ["close"])
    for spec in kc_specs:
        n, atrn, m = (
            int(spec.get("n", 30)),
            int(spec.get("atr_n", spec.get("n", 30))),
            float(spec.get("m", 2.5)),
        )
        lo, up = f"kc_{n}_{atrn}_{m}_lower", f"kc_{n}_{atrn}_{m}_upper"
        _ensure_cols(df, [lo, up])
        width = df[up] - df[lo]
        key = f"kc_{n}_{atrn}_{m}"
        df[f"kcW_{key}_pips"] = width / pip_size
        df[f"kc_pos_{key}"] = ((df["close"] - df[lo]) / width).clip(0, 1)
    return df


# ---------------- Slopes ----------------


def slope_norm(series: pd.Series, denom: float | pd.Series | None = None) -> pd.Series:
    """
    Normalize slope to 0..1 scale:
      0.0 = vertical down, 0.5 = flat, 1.0 = vertical up.
    denom: ATR or other volatility measure. If None, use series' own std.
    """
    eps = 1e-9
    diff = series.diff().fillna(0)
    if denom is None:
        denom = max(series.std(), eps)
    if isinstance(denom, (float, int)):
        norm = diff / max(abs(denom), eps)
    else:
        norm = diff / denom.replace(0, eps)
    return 0.5 + np.arctan(norm) / np.pi


# ---------------- Convenience wrapper ----------------


def attach_distance_features(
    df: pd.DataFrame,
    *,
    ema_pairs: Optional[Iterable[Tuple[int, int]]] = None,
    kc_specs: Optional[Iterable[Dict[str, Any]]] = None,
    ema_slopes: Optional[Iterable[int]] = None,
    atr_col: Optional[str] = None,
    pip_size: float = PIP_SIZE_DEFAULT,
) -> pd.DataFrame:
    if ema_pairs:
        df = add_ema_distances(df, ema_pairs, pip_size=pip_size)
    if kc_specs:
        df = add_kc_distances(df, kc_specs, pip_size=pip_size)
    if ema_slopes:
        for n in ema_slopes:
            col = f"ema_{int(n)}"
            _ensure_cols(df, [col])
            denom = df[atr_col] if atr_col and atr_col in df.columns else None
            df[f"ema_{n}_slope_norm"] = slope_norm(df[col], denom=denom)
    return df
