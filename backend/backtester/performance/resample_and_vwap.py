# backtester/pipeline/resample_and_vwap.py
# Phase 1 + VWAP features. Server-time anchored. No UTC.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

# -----------------------------
# Server-time (Pepperstone) helpers
# -----------------------------


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1)
    add = (weekday - d.weekday()) % 7
    d = d + pd.Timedelta(days=add) + pd.Timedelta(weeks=n - 1)
    return d.normalize()


def _us_dst_bounds(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = _nth_weekday_of_month(year, 3, 6, 2)  # 2nd Sunday in March
    end = _nth_weekday_of_month(year, 11, 6, 1)  # 1st Sunday in November
    return start, end


def _is_us_dst(day: pd.Timestamp) -> bool:
    day = pd.Timestamp(day).normalize()
    start, end = _us_dst_bounds(day.year)
    return start <= day < end


def server_offset_minutes(day: pd.Timestamp) -> int:
    return 180 if _is_us_dst(day) else 120  # GMT+3 in US DST, else GMT+2


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(
            None
        )  # keep naive; server-time added separately
    return out


def _ensure_ohlcv(df: pd.DataFrame) -> str:
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        pass
    else:
        auto = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        df.rename(
            columns={
                k: v for k, v in auto.items() if k in df.columns and v not in df.columns
            },
            inplace=True,
        )
        if not {"Open", "High", "Low", "Close"}.issubset(df.columns):
            missing = [
                c for c in ["Open", "High", "Low", "Close"] if c not in df.columns
            ]
            raise KeyError(f"Missing OHLC: {missing}")
    for c in ("tick_volume", "Volume", "volume", "Vol", "VOL"):
        if c in df.columns:
            return c
    raise KeyError("No volume/tick_volume column found")


def add_server_time(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_index(df)
    days = pd.to_datetime(out.index.date)
    offsets = pd.Series(days).apply(server_offset_minutes).to_numpy()
    out["server_ts"] = out.index + pd.to_timedelta(offsets, unit="m")
    out["server_date"] = pd.to_datetime(out["server_ts"]).dt.date
    out = out.set_index("server_ts", drop=False)
    return out


# -----------------------------
# Phase 1: Resampling from M1
# -----------------------------

AGG_OHLC = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
}


@dataclass
class ResampleResult:
    frames: Dict[str, pd.DataFrame]


def resample_from_m1(
    df_m1: pd.DataFrame, out_rules: Dict[str, str] | None = None
) -> ResampleResult:
    """Resample M1 to multiple timeframes using server-time index.
    out_rules: mapping name -> pandas rule. Defaults to common FX TFs.
    Returns dict of DataFrames with aligned OHLCV.
    """
    if out_rules is None:
        out_rules = {
            "M1": "1T",
            "M5": "5T",
            "M15": "15T",
            "M30": "30T",
            "H1": "1H",
            "H4": "4H",
            "D1": "1D",
            "W1": "1W-MON",  # ISO weeks starting Monday, but anchored on server_ts
            "MN1": "1MS",  # Month start
        }
    d = add_server_time(df_m1)
    vol_col = _ensure_ohlcv(d)

    frames: Dict[str, pd.DataFrame] = {}
    for name, rule in out_rules.items():
        ohlc = d.resample(rule).agg(AGG_OHLC).dropna(how="all")
        vol = d[vol_col].resample(rule).sum(min_count=1)
        out = ohlc.copy()
        out[vol_col] = vol
        frames[name] = out.dropna(subset=["Open", "High", "Low", "Close"], how="any")
    return ResampleResult(frames=frames)


# -----------------------------
# Phase 2: VWAPs (Daily, 2-Day, Weekly, Monthly) + bands
# -----------------------------


def _group_keys_for(df: pd.DataFrame, kind: str) -> pd.Series:
    ts = pd.to_datetime(df["server_ts"])  # already exists from add_server_time
    if kind == "D":
        return ts.dt.date.astype(str)
    if kind == "2D":
        # two-day rolling label: pair consecutive server dates
        d = ts.dt.date
        day_num = pd.factorize(d)[0]
        return ((day_num // 2).astype(int)).astype(str)
    if kind == "W":
        iso = ts.dt.isocalendar()
        return iso.year.astype(str) + "-W" + iso.week.astype(str)
    if kind == "M":
        return ts.dt.to_period("M").astype(str)
    raise ValueError("kind must be one of 'D','2D','W','M'")


def _vwap_and_bands(
    df: pd.DataFrame, vol_col: str, kind: str, band_sigmas: List[float] = [1.0, 2.0]
) -> pd.DataFrame:
    gkey = _group_keys_for(df, kind)
    pv = df["Close"] * pd.to_numeric(df[vol_col], errors="coerce").fillna(0)
    cum_vol = pv.groupby(gkey).transform(  # noqa: F841
        lambda _: _ * 0
    )  # placeholder to align index  # noqa: F841
    # Compute running sums within group without slow apply: use groupby.cumsum directly
    g = df.groupby(gkey, sort=False)
    csum_vol = g[vol_col].cumsum()
    csum_pv = g.apply(
        lambda x: (x["Close"] * pd.to_numeric(x[vol_col], errors="coerce")).cumsum()
    ).reset_index(level=0, drop=True)
    vwap = (csum_pv / pd.to_numeric(csum_vol, errors="coerce")).astype(float)

    # Bands: use rolling std within group around VWAP
    def _intra_std(x: pd.Series) -> pd.Series:
        return x.expanding(min_periods=5).std()

    std_intra = g["Close"].apply(_intra_std).reset_index(level=0, drop=True)

    out = df.copy()
    out[f"VWAP_{kind}"] = vwap
    for s in band_sigmas:
        out[f"VWAP_{kind}_up_{int(s)}s"] = vwap + s * std_intra
        out[f"VWAP_{kind}_dn_{int(s)}s"] = vwap - s * std_intra
    return out


def add_multi_vwap(
    df_m1: pd.DataFrame, kinds: List[str] = ["D", "2D", "W", "M"]
) -> pd.DataFrame:
    d = add_server_time(df_m1)
    vol_col = _ensure_ohlcv(d)
    out = d.copy()
    for k in kinds:
        out = _vwap_and_bands(out, vol_col=vol_col, kind=k)
    return out


# -----------------------------
# Convenience: end-to-end build
# -----------------------------


def build_timeframes_and_vwaps(df_m1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    res = resample_from_m1(df_m1)
    vw = add_multi_vwap(df_m1)
    res.frames["M1_with_VWAPs"] = vw
    return res.frames


# Usage:
# frames = build_timeframes_and_vwaps(df_m1)
# frames["D1"], frames["W1"], frames["M1_with_VWAPs"];  # all server-time aligned
