# backtester/features/regime_classifier.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from .distance_features import slope_norm


@dataclass
class RegimeThresholds:
    # slope around 0.5 is flat; farther from 0.5 is steeper
    strong_slope: float = 0.60  # >= strong
    trend_slope: float = 0.55  # >= trend
    flat_slope_hi: float = 0.525  # <= flat band upper
    flat_slope_lo: float = 0.475  # >= flat band lower

    # gap thresholds are ATR-normalized or KC-width-normalized
    strong_gap: float = 1.00  # >= strong
    trend_gap: float = 0.50  # >= trend
    flat_gap: float = 0.20  # <= consolidation


REGIME_COLUMNS = [
    "regime",
    "regime_score",
    "gap_fs_norm",
    "gap_fm_norm",
    "gap_ms_norm",
    "f_slope",
    "m_slope",
    "s_slope",
]


def _norm_gap(
    df: pd.DataFrame, fast: int, mid: int, slow: int, denom: str | None
) -> tuple[pd.Series, pd.Series, pd.Series]:
    gap_fs = df[f"ema_{fast}"] - df[f"ema_{slow}"]
    gap_fm = df[f"ema_{fast}"] - df[f"ema_{mid}"]
    gap_ms = df[f"ema_{mid}"] - df[f"ema_{slow}"]
    if denom is None:
        d = df[f"atr_{mid}"] if f"atr_{mid}" in df.columns else df[f"atr_{slow}"]
    else:
        d = df[denom]
    d = d.replace(0, np.nan)
    return gap_fs / d, gap_fm / d, gap_ms / d


def classify_regime(
    df: pd.DataFrame,
    *,
    ema_fast: int = 60,
    ema_mid: int = 90,
    ema_slow: int = 120,
    denom_col: Optional[str] = None,
    thr: RegimeThresholds = RegimeThresholds(),
    smooth_n: int = 3,
) -> pd.DataFrame:
    """
    Past-only regime labelling. Uses EMA ordering, normalized gaps, and normalized slopes.
    Outputs vectorized labels per bar without lookahead.
    """
    f, m, s = int(ema_fast), int(ema_mid), int(ema_slow)
    for c in [f"ema_{f}", f"ema_{m}", f"ema_{s}"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # --- Calculations for slopes and gaps (Unchanged) ---
    atr_candidate = denom_col if denom_col and denom_col in df.columns else None
    f_s = slope_norm(df[f"ema_{f}"], denom=df[atr_candidate] if atr_candidate else None)
    m_s = slope_norm(df[f"ema_{m}"], denom=df[atr_candidate] if atr_candidate else None)
    s_s = slope_norm(df[f"ema_{s}"], denom=df[atr_candidate] if atr_candidate else None)
    gap_fs, gap_fm, gap_ms = _norm_gap(df, f, m, s, denom_col)

    # --- Regime Classification Logic (Unchanged) ---
    up_order = (df[f"ema_{f}"] > df[f"ema_{m}"]) & (df[f"ema_{m}"] > df[f"ema_{s}"])
    dn_order = (df[f"ema_{f}"] < df[f"ema_{m}"]) & (df[f"ema_{m}"] < df[f"ema_{s}"])
    f_up = f_s >= thr.trend_slope
    m_up = m_s >= thr.trend_slope
    s_up = s_s >= thr.trend_slope
    f_dn = f_s <= 1.0 - thr.trend_slope
    m_dn = m_s <= 1.0 - thr.trend_slope
    s_dn = s_s <= 1.0 - thr.trend_slope
    strong_up = up_order & f_up & m_up & s_up & (gap_fs >= thr.strong_gap)
    strong_dn = dn_order & f_dn & m_dn & s_dn & (gap_fs <= -thr.strong_gap)
    trend_up = up_order & f_up & m_up & s_up & (gap_fs >= thr.trend_gap)
    trend_dn = dn_order & f_dn & m_dn & s_dn & (gap_fs <= -thr.trend_gap)
    flat_slopes = (
        (f_s.between(thr.flat_slope_lo, thr.flat_slope_hi))
        & (m_s.between(thr.flat_slope_lo, thr.flat_slope_hi))
        & (s_s.between(thr.flat_slope_lo, thr.flat_slope_hi))
    )
    tight_gaps = gap_fs.abs() <= thr.flat_gap
    consolidation = flat_slopes & tight_gaps
    crossing = ~up_order & ~dn_order
    mixed_slopes = (f_up & m_dn) | (f_dn & m_up)
    transition = (
        (crossing | mixed_slopes)
        & (~consolidation)
        & (~trend_up)
        & (~trend_dn)
        & (~strong_up)
        & (~strong_dn)
    )

    regime = pd.Series("CONS", index=df.index)
    regime[transition] = "TRANS"
    regime[trend_up] = "UP"
    regime[trend_dn] = "DOWN"
    regime[strong_up] = "S_UP"
    regime[strong_dn] = "S_DOWN"

    # --- NEW: Smoother Score Calculation (0 to 10) ---
    # The gap component is now smoothly scaled to a -1 to 1 range using tanh
    gap_component = np.tanh(gap_fs / thr.strong_gap)

    # The slope component is the average of the individual normalized slopes
    slope_component = (f_s + m_s + s_s) / 3.0 - 0.5  # Center around 0

    # Combine components with weights (60% gap, 40% slope) and scale to 0-10
    # 5 is the neutral midpoint.
    raw_score = (
        5.0 + (5.0 * 0.6 * gap_component) + (5.0 * 0.4 * (slope_component / 0.5))
    )
    score = raw_score.clip(0, 10)
    # ---------------------------------------------------

    out = pd.DataFrame(
        {
            "regime": regime,
            "regime_score": score,
            "gap_fs_norm": gap_fs,
            "gap_fm_norm": gap_fm,
            "gap_ms_norm": gap_ms,
            "f_slope": f_s,
            "m_slope": m_s,
            "s_slope": s_s,
        }
    )

    if smooth_n and smooth_n > 1:
        vote = out["regime"].copy()
        map_vals = {"S_DOWN": -2, "DOWN": -1, "TRANS": 0, "CONS": 0, "UP": 1, "S_UP": 2}
        inv_map = {v: k for k, v in map_vals.items()}
        v = vote.map(map_vals).fillna(0).astype(int)
        vv = v.rolling(window=smooth_n, min_periods=1).mean().round().astype(int)
        out["regime"] = vv.map(inv_map).fillna("TRANS")

    return out


def attach_regime(
    df: pd.DataFrame,
    *,
    ema_fast: int = 60,
    ema_mid: int = 90,
    ema_slow: int = 120,
    atr_col: Optional[str] = None,  # e.g. "atr_18"
    kc_width_col: Optional[str] = None,  # if using KC width as denom
    smooth_n: int = 3,
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> pd.DataFrame:
    denom = None
    if kc_width_col and kc_width_col in df.columns:
        denom = kc_width_col
    elif atr_col and atr_col in df.columns:
        denom = atr_col
    reg = classify_regime(
        df,
        ema_fast=ema_fast,
        ema_mid=ema_mid,
        ema_slow=ema_slow,
        denom_col=denom,
        thr=thresholds,
        smooth_n=smooth_n,
    )
    return df.join(reg)
