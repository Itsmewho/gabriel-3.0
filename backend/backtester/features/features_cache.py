from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Set, List
import pandas as pd

# base fetchers/builders (your existing modules)
from backtester.data.loaders import fetch_sql_market_data, fetch_event_features
from backtester.features.base_features import apply_basic_features


OHLC = ["open", "high", "low", "close", "tick_volume"]


def _spec_columns(spec: Optional[Dict[str, Any]]) -> Set[str]:
    if not spec:
        return set()
    out: Set[str] = set()
    # SMA/EMA/RSI/ATR lists
    for n in spec.get("sma", []):
        out.add(f"sma_{int(n)}")
    for n in spec.get("ema", []):
        out.add(f"ema_{int(n)}")
    for n in spec.get("rsi", []):
        out.add(f"rsi_{int(n)}")
    for n in spec.get("atr", []):
        out.add(f"atr_{int(n)}")
    # MACD
    if "macd" in spec:
        out.update(["macd", "macd_signal", "macd_hist"])
    # Bollinger
    if "bb" in spec:
        n = int(spec["bb"].get("n", 20))
        out.update([f"bb_{n}_mid", f"bb_{n}_upper", f"bb_{n}_lower"])
    # Ichimoku
    if "ichimoku" in spec:
        out.update(
            [
                "ichimoku_tenkan",
                "ichimoku_kijun",
                "ichimoku_senkou_a",
                "ichimoku_senkou_b",
                "ichimoku_chikou",
            ]
        )
    return out


def _sub_spec_for_missing(
    spec: Optional[Dict[str, Any]], missing: Set[str]
) -> Dict[str, Any]:
    if not spec:
        return {}
    sub: Dict[str, Any] = {}

    # helper to include only the parts we need
    def _need(keys: List[str]) -> bool:
        return any(k in missing for k in keys)

    # SMA family
    want_sma = []
    for n in spec.get("sma", []):
        if f"sma_{int(n)}" in missing:
            want_sma.append(int(n))
    if want_sma:
        sub["sma"] = want_sma

    want_ema = []
    for n in spec.get("ema", []):
        if f"ema_{int(n)}" in missing:
            want_ema.append(int(n))
    if want_ema:
        sub["ema"] = want_ema

    want_rsi = []
    for n in spec.get("rsi", []):
        if f"rsi_{int(n)}" in missing:
            want_rsi.append(int(n))
    if want_rsi:
        sub["rsi"] = want_rsi

    want_atr = []
    for n in spec.get("atr", []):
        if f"atr_{int(n)}" in missing:
            want_atr.append(int(n))
    if want_atr:
        sub["atr"] = want_atr

    if _need(["macd", "macd_signal", "macd_hist"]) and "macd" in spec:
        sub["macd"] = spec["macd"]

    if "bb" in spec:
        n = int(spec["bb"].get("n", 20))
        cols = {f"bb_{n}_mid", f"bb_{n}_upper", f"bb_{n}_lower"}
        if cols & missing:
            sub["bb"] = spec["bb"]

    if (
        _need(
            [
                "ichimoku_tenkan",
                "ichimoku_kijun",
                "ichimoku_senkou_a",
                "ichimoku_senkou_b",
                "ichimoku_chikou",
            ]
        )
        and "ichimoku" in spec
    ):
        sub["ichimoku"] = spec["ichimoku"]

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
    want_cols = _spec_columns(spec)
    missing = {c for c in want_cols if c not in df.columns}
    if missing:
        sub = _sub_spec_for_missing(spec, missing)
        if sub:
            base = df.copy()
            base = apply_basic_features(base, cfg=sub)
            df = base
            df.to_parquet(fname)

    return df
