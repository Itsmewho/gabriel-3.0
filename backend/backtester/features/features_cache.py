from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
import pandas as pd

# base fetchers/builders
from backtester.data.loaders import fetch_sql_market_data, fetch_event_features
from backtester.features.base_features import apply_basic_features
from backtester.features.better_volume_indicator import add_better_volume_mql

OHLC = ["open", "high", "low", "close", "tick_volume"]


def _ensure_list(x: Any) -> List[Dict[str, Any]]:
    if not x:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    raise TypeError("spec must be a dict or list of dicts")


def _spec_columns(spec: Optional[Dict[str, Any]]) -> Set[str]:
    if not spec:
        return set()
    out: Set[str] = set()
    # SMA/EMA/RSI/ATR lists
    for n in spec.get("sma", []):
        out.add(f"sma_{int(n)}")
    for n in spec.get("sma_high", []):
        out.add(f"sma_high_{int(n)}")
    for n in spec.get("sma_low", []):
        out.add(f"sma_low_{int(n)}")
    for n in spec.get("ema", []):
        out.add(f"ema_{int(n)}")
    for n in spec.get("rsi", []):
        out.add(f"rsi_{int(n)}")
    for n in spec.get("atr", []):
        out.add(f"atr_{int(n)}")
    # Volume SMA
    for n in spec.get("vol_sma", []):
        out.add(f"volume_sma_{int(n)}")
    # Bollinger
    if "bb" in spec:
        n = int(spec["bb"].get("n", 20))
        out.update([f"bb_{n}_mid", f"bb_{n}_upper", f"bb_{n}_lower"])
    # Keltner
    if "kc" in spec:
        for kc in _ensure_list(spec.get("kc")):
            n = int(kc.get("n", 20))
            atr_n = int(kc.get("atr_n", n))
            m = float(kc.get("m", 2.0))
            out.update(
                {
                    f"kc_{n}_{atr_n}_{m}_mid",
                    f"kc_{n}_{atr_n}_{m}_upper",
                    f"kc_{n}_{atr_n}_{m}_lower",
                }
            )

    # Stochastic Oscillator
    if "stoch" in spec:
        for stoch in _ensure_list(spec.get("stoch")):
            k = int(stoch.get("k_period", 14))
            d = int(stoch.get("d_period", 3))
            s = int(stoch.get("slowing", 3))
            out.add(f"stoch_{k}_{d}_{s}_k")
            out.add(f"stoch_{k}_{d}_{s}_d")

    # PTL
    if "ptl" in spec:
        if _ensure_list(spec.get("ptl")):
            out.update({"ptl_slow", "ptl_fast", "ptl_trend", "ptl_trena", "ptl_arrow"})

    # Better Volume context color
    if spec.get("better_volume"):
        out.add("bv_color")
    return out


def _sub_spec_for_missing(
    spec: Optional[Dict[str, Any]], missing: Set[str]
) -> Dict[str, Any]:
    if not spec:
        return {}
    sub: Dict[str, Any] = {}

    # SMA family
    want_sma = [int(n) for n in spec.get("sma", []) if f"sma_{int(n)}" in missing]
    if want_sma:
        sub["sma"] = want_sma

    want_sma_high = [
        int(n) for n in spec.get("sma_high", []) if f"sma_high_{int(n)}" in missing
    ]
    if want_sma_high:
        sub["sma_high"] = want_sma_high

    want_sma_low = [
        int(n) for n in spec.get("sma_low", []) if f"sma_low_{int(n)}" in missing
    ]
    if want_sma_low:
        sub["sma_low"] = want_sma_low

    want_ema = [int(n) for n in spec.get("ema", []) if f"ema_{int(n)}" in missing]
    if want_ema:
        sub["ema"] = want_ema

    want_rsi = [int(n) for n in spec.get("rsi", []) if f"rsi_{int(n)}" in missing]
    if want_rsi:
        sub["rsi"] = want_rsi

    want_atr = [int(n) for n in spec.get("atr", []) if f"atr_{int(n)}" in missing]
    if want_atr:
        sub["atr"] = want_atr

    want_vol_sma = [
        int(n) for n in spec.get("vol_sma", []) if f"volume_sma_{int(n)}" in missing
    ]
    if want_vol_sma:
        sub["vol_sma"] = want_vol_sma

    if "bb" in spec:
        n = int(spec["bb"].get("n", 20))
        cols = {f"bb_{n}_mid", f"bb_{n}_upper", f"bb_{n}_lower"}
        if cols & missing:
            sub["bb"] = spec["bb"]

    if "kc" in spec:
        need_kc: List[Dict[str, Any]] = []
        for kc in _ensure_list(spec.get("kc")):
            n = int(kc.get("n", 20))
            atr_n = int(kc.get("atr_n", n))
            m = float(kc.get("m", 2.0))
            cols = {
                f"kc_{n}_{atr_n}_{m}_mid",
                f"kc_{n}_{atr_n}_{m}_upper",
                f"kc_{n}_{atr_n}_{m}_lower",
            }
            if cols & missing:
                need_kc.append(kc)
        if need_kc:
            sub["kc"] = need_kc if len(need_kc) > 1 else need_kc[0]

    # Stochastic Oscillator
    if "stoch" in spec:
        need_stoch: List[Dict[str, Any]] = []
        for stoch in _ensure_list(spec.get("stoch")):
            k = int(stoch.get("k_period", 14))
            d = int(stoch.get("d_period", 3))
            s = int(stoch.get("slowing", 3))
            cols = {f"stoch_{k}_{d}_{s}_k", f"stoch_{k}_{d}_{s}_d"}
            if cols & missing:
                need_stoch.append(stoch)
        if need_stoch:
            sub["stoch"] = need_stoch if len(need_stoch) > 1 else need_stoch[0]

    # PTL
    if "ptl" in spec:
        ptl_cols = {"ptl_slow", "ptl_fast", "ptl_trend", "ptl_trena", "ptl_arrow"}
        if ptl_cols & missing:
            pts = _ensure_list(spec.get("ptl"))
            sub["ptl"] = pts[0] if pts else {"fast": 15, "slow": 11}

    return sub


def ensure_feature_parquet(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    *,
    spec: Optional[Dict[str, Any]] = None,
    with_events: bool = True,
    cache_dir: str = "results/cache",
) -> pd.DataFrame:
    """
    Column-aware cache. Fetch OHLCV once, persist to parquet, and lazily append
    missing feature columns on demand. No need to rebuild per strategy.

    Returns a DataFrame indexed by time with OHLCV + requested features (+ events opt.).
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fname = (
        cache_path / f"{symbol}_{timeframe}_{start_date}_{end_date}_features.parquet"
    )

    if fname.exists():
        df = pd.read_parquet(fname)
        # normalize index
        if "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index("time")
        df.index = pd.to_datetime(df.index)
    else:
        where_clause = f"time >= '{start_date}' AND time <= '{end_date}'"
        df = fetch_sql_market_data(symbol, timeframe, where_clause)
        if df.empty:
            return df
        df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
        df = df[OHLC]
        if with_events:
            tmp = df.reset_index()
            tmp = fetch_event_features(symbol, tmp)
            df = tmp.set_index("time").sort_index()
        df.to_parquet(fname)

    # Ensure requested features
    spec = spec or {}
    want_cols = _spec_columns(spec)
    missing = {c for c in want_cols if c not in df.columns}
    if missing:
        sub = _sub_spec_for_missing(spec, missing)
        if sub:
            base = df.copy()
            base = apply_basic_features(base, cfg=sub)
            df = base
            df.to_parquet(fname)

    # Better Volume computed after apply_basic_features
    if spec.get("better_volume") and "bv_color" not in df.columns:
        lookback = 20
        bv_cfg = spec.get("better_volume")
        if isinstance(bv_cfg, dict):
            lookback = int(bv_cfg.get("lookback", 20))
        # Compute and append
        bv_df = add_better_volume_mql(df.copy(), lookback=lookback)
        df["bv_color"] = bv_df["bv_color"].astype("string")
        df.to_parquet(fname)

    return df
