# backtester/features/feature_regime.py
from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from backtester.features.better_volume_indicator import add_better_volume_mql


def _prep_market(market_data: pd.DataFrame) -> pd.DataFrame:
    df = market_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


def compute_regime_indicators(
    df: pd.DataFrame,
    *,
    trend_fast: int = 50,
    trend_slow: int = 200,
    trend_eps: float = 0.0,
    vol_window: int = 60,
    bv_lookback: int = 30,
) -> pd.DataFrame:
    """Return DataFrame with columns identical to performance/regime_eval.prepare_regime_indicators.

    Columns: sma_fast, sma_slow, trend_score, trend_regime, vol_sigma, vol_bucket, bv_color,
             regime_label, regime_label_bv
    Index aligns to input df index.
    """
    base = _prep_market(df)
    close = base["close"].astype(float)

    sma_fast = close.rolling(trend_fast, min_periods=trend_fast // 2).mean()
    sma_slow = close.rolling(trend_slow, min_periods=trend_slow // 2).mean()
    trend_score = sma_fast - sma_slow

    tr = pd.Series(index=base.index, dtype="object")
    tr[trend_score > trend_eps] = "uptrend"
    tr[trend_score < -trend_eps] = "downtrend"
    tr[(trend_score >= -trend_eps) & (trend_score <= trend_eps)] = "flat"

    logret = np.log(close).diff()
    vol_sigma = logret.rolling(vol_window, min_periods=vol_window // 2).std()
    q1, q2 = vol_sigma.quantile([1 / 3, 2 / 3])
    vb = pd.Series(index=base.index, dtype="object")
    vb[vol_sigma <= q1] = "low_vol"
    vb[(vol_sigma > q1) & (vol_sigma <= q2)] = "mid_vol"
    vb[vol_sigma > q2] = "high_vol"

    try:
        bv_df = add_better_volume_mql(base, lookback=bv_lookback)
        bv_color = bv_df["bv_color"].reindex(base.index)
    except Exception:
        bv_color = pd.Series(index=base.index, dtype="object")

    out = pd.DataFrame(
        {
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "trend_score": trend_score,
            "trend_regime": tr,
            "vol_sigma": vol_sigma,
            "vol_bucket": vb,
            "bv_color": bv_color,
        },
        index=base.index,
    )
    out["regime_label"] = (
        out["trend_regime"].astype(str) + "+" + out["vol_bucket"].astype(str)
    )
    out["regime_label_bv"] = (
        out["trend_regime"].astype(str) + "+" + out["bv_color"].astype(str)
    )
    return out


def add_regime_columns(
    df: pd.DataFrame,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Append regime columns into df using same logic as performance/regime_eval.

    If df already contains 'bv_color' (from better_volume step), keep it and avoid
    column overlap when joining. Otherwise compute and add it from the regime block.
    """
    cfg = cfg or {}
    reg = compute_regime_indicators(
        df,
        trend_fast=int(cfg.get("trend_fast", 50)),
        trend_slow=int(cfg.get("trend_slow", 200)),
        trend_eps=float(cfg.get("trend_eps", 0.0)),
        vol_window=int(cfg.get("vol_window", 60)),
        bv_lookback=int(cfg.get("bv_lookback", 30)),
    )
    # only join the needed columns to avoid accidental overwrite of OHLCV
    cols = [
        "sma_fast",
        "sma_slow",
        "trend_score",
        "trend_regime",
        "vol_sigma",
        "vol_bucket",
        "bv_color",
        "regime_label",
        "regime_label_bv",
    ]
    # Prevent overlap: if df already has bv_color, don't join it from reg
    join_cols = [c for c in cols if not (c == "bv_color" and "bv_color" in df.columns)]
    return df.join(reg[join_cols])


# -----------------------------------------------------------------------------
# Cache integration snippet for backtester/features/features_cache.py
# Add inside ensure_feature_parquet after apply_basic_features/better_volume step:
#
#    if spec.get("regime"):
#        from backtester.features.feature_regime import add_regime_columns
#        reg_cfg = spec.get("regime") if isinstance(spec.get("regime"), dict) else {}
#        df = add_regime_columns(df, reg_cfg)
#        df.to_parquet(fname)
#
# And update _spec_columns to include requested regime columns when spec has "regime".
# Example: want trend_regime + bv_color only -> in spec:
#   regime: { trend_fast: 50, trend_slow: 200, vol_window: 60 }
#   columns: ["trend_regime", "bv_color"]
# -----------------------------------------------------------------------------
