# backtester/performance/weekly_break_analysis.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Literal, List, Set
import numpy as np
import pandas as pd

# All metrics in %. No ATR.

# -----------------------------
# Helpers
# -----------------------------


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)  # type: ignore
    return out


def _ensure_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        return df
    auto = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
    mapping = {k: v for k, v in auto.items() if k in df.columns and v not in df.columns}
    if mapping:
        df = df.rename(columns=mapping)
    missing = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing OHLC columns: {missing}")
    return df


def _resample_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"close": df["Close"].resample("B").last().dropna()})


def _week_id(ts: pd.Timestamp) -> tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def _slice_after_week(
    df_bday: pd.DataFrame, week_end: pd.Timestamp, n: int
) -> pd.DataFrame:
    i = df_bday.index.searchsorted(week_end, side="right")
    j = min(i + n, len(df_bday))
    return df_bday.iloc[i:j]


# -----------------------------
# Weekly lookback ranges
# -----------------------------


def _get_lookback_hilo(
    df_daily_hl: pd.DataFrame, lookback_periods: Set[int]
) -> pd.DataFrame:
    di = df_daily_hl.copy()
    need = {"High", "Low"}
    if not need.issubset(di.columns):
        raise KeyError(f"Expected daily DataFrame with columns {need}.")

    di["iso_year"], di["iso_week"] = zip(*di.index.map(_week_id))
    grp = di.groupby(["iso_year", "iso_week"], sort=True)

    wk = grp.agg(high=("High", "max"), low=("Low", "min"))
    wk["week_end_time"] = grp.apply(lambda x: x.index.max())
    wk = wk.reset_index().sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

    for period in lookback_periods:
        wk[f"high_{period}w"] = wk["high"].shift(1).rolling(window=period).max()
        wk[f"low_{period}w"] = wk["low"].shift(1).rolling(window=period).min()

    return wk


# -----------------------------
# Break detection & evaluation (percent-based)
# -----------------------------


def _detect_break(
    df_wf: pd.DataFrame, prev_low: float, prev_high: float
) -> tuple[bool, bool]:
    if df_wf.empty or not np.isfinite(prev_low) or not np.isfinite(prev_high):
        return False, False
    up = (df_wf["close"] > prev_high).any()
    dn = (df_wf["close"] < prev_low).any()
    return bool(up), bool(dn)


def _analyze_post_break(
    eval_slice: pd.DataFrame,
    prev_low: float,
    prev_high: float,
    direction: Literal["up", "down"],
) -> Dict[str, Any]:
    """Return simple percentage outcomes relative to prior range, plus touch-probabilities."""
    out: Dict[str, Any] = {
        "re_mid": False,
        "hit_opposite": False,
        "overshoot": False,
        "deeper_pull": False,
        "move_pct": np.nan,  # follow-through beyond break level as % of range
        "pullback_pct": np.nan,  # adverse excursion back toward/through range as % of range
    }

    if (
        eval_slice.empty
        or not all(np.isfinite([prev_low, prev_high]))
        or prev_high <= prev_low
    ):
        return out

    rng = prev_high - prev_low
    if rng <= 0:
        return out

    closes = eval_slice["close"]

    # Band buckets for simple mid/opposite/overshoot counts
    mid_lo, mid_hi = 0.40, 0.60

    def _band_bucket(p: float) -> str:
        r = (p - prev_low) / rng
        if r < 0.0:
            return "below_low"
        if r < mid_lo:
            return "lower_band"
        if r <= mid_hi:
            return "mid_band"
        if r <= 1.0:
            return "upper_band"
        return "above_high"

    bands = closes.apply(_band_bucket)

    if direction == "up":
        move = closes.max() - prev_high
        pull = prev_high - closes.min()
        out["re_mid"] = bool((bands == "mid_band").any())
        out["hit_opposite"] = bool((bands == "lower_band").any())
        out["deeper_pull"] = bool((bands == "below_low").any())
        out["overshoot"] = bool((bands == "above_high").any())
    else:
        move = prev_low - closes.min()
        pull = closes.max() - prev_low
        out["re_mid"] = bool((bands == "mid_band").any())
        out["hit_opposite"] = bool((bands == "upper_band").any())
        out["deeper_pull"] = bool((bands == "above_high").any())
        out["overshoot"] = bool((bands == "below_low").any())

    out["move_pct"] = float(max(0.0, move) / rng * 100.0)
    out["pullback_pct"] = float(max(0.0, pull) / rng * 100.0)
    return out


def _process_week_for_experiments(
    week_data: pd.Series, df_b: pd.DataFrame, experiments: List[tuple[int, int, int]]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    week_end = pd.Timestamp(week_data["week_end_time"])

    for lookback_w, detection_d, evaluation_d in experiments:
        prev_high = float(week_data.get(f"high_{lookback_w}w", np.nan))
        prev_low = float(week_data.get(f"low_{lookback_w}w", np.nan))
        break_slice = _slice_after_week(df_b, week_end, detection_d)
        broke_up, broke_dn = _detect_break(break_slice, prev_low, prev_high)
        if not broke_up and not broke_dn:
            continue

        eval_slice = _slice_after_week(df_b, week_end, evaluation_d)
        base = {
            "iso_year": int(week_data["iso_year"]),
            "iso_week": int(week_data["iso_week"]),
            "week_end_time": pd.Timestamp(week_data["week_end_time"]),
            "lookback_weeks": lookback_w,
            "detection_days": detection_d,
            "evaluation_days": evaluation_d,
            "broke_up": broke_up,
            "broke_dn": broke_dn,
        }

        if broke_up:
            base.update(
                {
                    f"{k}_after_up": v
                    for k, v in _analyze_post_break(
                        eval_slice, prev_low, prev_high, "up"
                    ).items()
                }
            )
        if broke_dn:
            base.update(
                {
                    f"{k}_after_dn": v
                    for k, v in _analyze_post_break(
                        eval_slice, prev_low, prev_high, "down"
                    ).items()
                }
            )

        rows.append(base)

    return rows


# -----------------------------
# Reversal landing zones (percent-based)
# -----------------------------


def _swing_points_daily(close: pd.Series, w: int = 3) -> pd.DataFrame:
    s = close.copy()
    roll_max = s.rolling(2 * w + 1, center=True).max()
    roll_min = s.rolling(2 * w + 1, center=True).min()
    is_high = (s == roll_max) & s.notna()
    is_low = (s == roll_min) & s.notna()
    return pd.DataFrame({"close": s, "is_high": is_high, "is_low": is_low})


def _reversal_stats_percent(
    df_daily: pd.DataFrame, w: int = 3, horizon_days: int = 5
) -> pd.DataFrame:
    # df_daily must have High, Low, Close at daily frequency
    swings = _swing_points_daily(df_daily["Close"], w=w)
    rows: List[Dict[str, Any]] = []
    for ts, row in swings.iterrows():
        if not (bool(row["is_high"]) or bool(row["is_low"])):
            continue
        if ts not in df_daily.index:
            continue
        day_hi = float(df_daily.loc[ts, "High"])
        day_lo = float(df_daily.loc[ts, "Low"])
        ref = float(df_daily.loc[ts, "Close"])
        day_rng = day_hi - day_lo
        if not np.isfinite(day_rng) or day_rng <= 0:
            continue
        loc = df_daily.index.get_indexer([ts])[0]
        fut = df_daily.iloc[loc + 1 : loc + 1 + horizon_days]
        if fut.empty:
            continue
        dist = fut["Close"] - ref
        p25, p50, p75 = np.nanpercentile(dist, [25, 50, 75])
        worst = float(np.nanmin(dist)) if np.isfinite(dist.min()) else np.nan
        rows.append(
            {
                "timestamp": ts,
                "type": "high" if bool(row["is_high"]) else "low",
                "p25_pct": float(p25 / day_rng * 100.0),
                "p50_pct": float(p50 / day_rng * 100.0),
                "p75_pct": float(p75 / day_rng * 100.0),
                "max_adverse_pct": float(worst / day_rng * 100.0),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------
# Next-day continuation
# -----------------------------


def _next_day_hilo_stats_percent(df_daily_hl: pd.DataFrame) -> pd.DataFrame:
    d = df_daily_hl.copy()
    d["next_high"] = d["High"].shift(-1)
    d["next_low"] = d["Low"].shift(-1)
    p_high = float((d["next_high"] > d["High"]).mean()) * 100.0
    p_low = float((d["next_low"] < d["Low"]).mean()) * 100.0
    return pd.DataFrame(
        {
            "p_next_exceeds_high_pct": [round(p_high, 1)],
            "p_next_exceeds_low_pct": [round(p_low, 1)],
        }
    )


# -----------------------------
# Scenarios
# -----------------------------
SCENARIOS: List[tuple[int, int]] = [
    (1, 3),
    (2, 5),
    (3, 5),
    (4, 5),
    (6, 10),
]
DEFAULT_EXPERIMENTS: List[tuple[int, int, int]] = [
    (lookback, lookback * 5, evaluation) for lookback, evaluation in SCENARIOS
]

# -----------------------------
# Volume Profile (tick-volume proxy)
# -----------------------------


def _get_volume_column(df: pd.DataFrame) -> str | None:
    for c in ("tick_volume", "Volume", "volume", "Vol"):
        if c in df.columns:
            return c
    return None


def _build_volume_profile(
    df_slice: pd.DataFrame,
    n_bins: int = 120,
) -> Dict[str, Any] | None:
    """Approximate volume profile over df_slice using tick-volume.
    Uses Close price for binning. Returns POC, VAH, VAL, bin edges, and volumes.
    """
    if df_slice.empty:
        return None
    vol_col = _get_volume_column(df_slice)
    if vol_col is None:
        return None

    prices = df_slice["Close"].astype(float).values
    vols = pd.to_numeric(df_slice[vol_col], errors="coerce").fillna(0.0).values
    pmin, pmax = float(np.nanmin(prices)), float(np.nanmax(prices))
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return None

    edges = np.linspace(pmin, pmax, n_bins + 1)
    idx = np.clip(np.digitize(prices, edges) - 1, 0, n_bins - 1)
    hist = np.bincount(idx, weights=vols, minlength=n_bins).astype(float)
    if hist.sum() <= 0:
        return None

    poc_idx = int(np.argmax(hist))
    total = float(hist.sum())
    left = right = poc_idx
    cum = float(hist[poc_idx])
    # Expand around POC until ~70% volume captured
    while cum / total < 0.70 and (left > 0 or right < n_bins - 1):
        next_left = hist[left - 1] if left > 0 else -1.0
        next_right = hist[right + 1] if right < n_bins - 1 else -1.0
        if next_right >= next_left:
            if right < n_bins - 1:
                right += 1
                cum += float(hist[right])
            elif left > 0:
                left -= 1
                cum += float(hist[left])
            else:
                break
        else:
            if left > 0:
                left -= 1
                cum += float(hist[left])
            elif right < n_bins - 1:
                right += 1
                cum += float(hist[right])
            else:
                break

    # Map bin index to price (mid of bin)
    mids = (edges[:-1] + edges[1:]) / 2.0
    poc = float(mids[poc_idx])
    vah = float(edges[right + 1] if right + 1 < len(edges) else edges[-1])
    val = float(edges[left])
    profile_range = max(1e-12, vah - val)

    # Low-volume threshold (30th percentile of non-zero bins)
    nz = hist[hist > 0]
    lv_thresh = float(np.percentile(nz, 30)) if len(nz) else 0.0

    return {
        "edges": edges,
        "hist": hist,
        "mids": mids,
        "poc": poc,
        "vah": vah,
        "val": val,
        "poc_idx": poc_idx,
        "lv_thresh": lv_thresh,
        "range": profile_range,
    }


def _first_true_idx(mask: pd.Series) -> int | None:
    nz = np.flatnonzero(mask.values)
    return int(nz[0]) if len(nz) else None


def _vp_break_metrics(
    eval_slice: pd.DataFrame,
    vp: Dict[str, Any],
    direction: Literal["up", "down"],
) -> Dict[str, Any]:
    out = {
        "retest_poc": None,
        "t_retest_poc": None,
        "move_pct_vp": None,
        "hold_outside_value": None,
    }
    if eval_slice.empty or vp is None:
        return out
    poc, vah, val, rng = vp["poc"], vp["vah"], vp["val"], float(vp["range"]) or 1e-12
    closes = eval_slice["close"]

    if direction == "up":
        move = float(closes.max() - vah)
        hold = bool((closes > vah).all())
    else:
        move = float(val - closes.min())
        hold = bool((closes < val).all())

    retest = (closes - poc).abs() <= 0  # placeholder replaced below  # noqa: F841
    # Allow a small tolerance around POC equal to one bin height
    edges = vp["edges"]
    bin_height = float(edges[1] - edges[0]) if len(edges) > 1 else 0.0
    poc_touch = (closes - poc).abs() <= bin_height

    out["retest_poc"] = bool(poc_touch.any())
    out["t_retest_poc"] = _first_true_idx(poc_touch)
    out["move_pct_vp"] = max(0.0, move) / rng * 100.0
    out["hold_outside_value"] = hold
    return out


# -----------------------------
# Public API
# -----------------------------


def run_weekly_break_analysis(
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str,
) -> None:
    """Report with three sections in %:
    1) Break outcomes: probabilities + median move/pullback % of prior range.
    2) Reversal landing zones: p25/p50/p75 and max adverse over 5D as % of swing-day range.
    3) Next-day continuation: P(next high breaks / next low breaks).
    """
    df = _normalize_index(market_data)
    df = _ensure_ohlc_columns(df)

    # Series
    df_b = _resample_to_business_days(df)  # business-day Close for windows
    df_daily_hl = df.resample("D").agg({"High": "max", "Low": "min"}).dropna()
    df_daily = (
        df.resample("D").agg({"High": "max", "Low": "min", "Close": "last"}).dropna()
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md_path = Path(out_dir) / f"{symbol}_weekly_break_report_{period_tag}.md"
    # Weekly lookbacks
    lookback_periods = {exp[0] for exp in DEFAULT_EXPERIMENTS}
    wk = _get_lookback_hilo(df_daily_hl, lookback_periods)

    # Break events
    # Pre-compute 200D MA regime on daily closes
    daily_close = df.resample("D")["Close"].last().dropna()
    sma200 = daily_close.rolling(200, min_periods=200).mean()

    def _regime_at(ts: pd.Timestamp) -> str:
        if ts.normalize() in sma200.index:
            ma = (
                float(sma200.loc[ts.normalize()])
                if ts.normalize() in sma200.index
                else np.nan
            )
        else:
            # find last available day <= ts
            pos = sma200.index.searchsorted(ts.normalize(), side="right") - 1
            ma = float(sma200.iloc[pos]) if pos >= 0 else np.nan
        px = (
            float(
                daily_close.iloc[
                    daily_close.index.searchsorted(ts.normalize(), side="right") - 1
                ]
            )
            if len(daily_close)
            else np.nan
        )
        if np.isfinite(ma) and np.isfinite(px):
            return "bull" if px >= ma else "bear"
        return "unknown"

    all_rows: List[Dict[str, Any]] = []
    all_vp_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for _, week_data in wk.iterrows():
        # Existing price-range break logic
        all_rows.extend(
            _process_week_for_experiments(week_data, df_b, DEFAULT_EXPERIMENTS)
        )

        # Volume-profile break logic per experiment
        week_end = pd.Timestamp(week_data["week_end_time"]).to_pydatetime()
        for lookback_w, detection_d, evaluation_d in DEFAULT_EXPERIMENTS:
            start = pd.Timestamp(week_end) - pd.Timedelta(weeks=lookback_w)
            hist_slice = df[(df.index > start) & (df.index <= week_end)]
            vp = _build_volume_profile(hist_slice)
            if not vp:
                continue
            # Windows for detection/evaluation
            det = _slice_after_week(df_b, pd.Timestamp(week_end), detection_d)
            eva = _slice_after_week(df_b, pd.Timestamp(week_end), evaluation_d)
            if det.empty:
                continue
            vah, val = float(vp["vah"]), float(vp["val"])
            broke_up = bool((det["close"] > vah).any())
            broke_dn = bool((det["close"] < val).any())
            if not broke_up and not broke_dn:
                continue

            regime = _regime_at(pd.Timestamp(week_end))

            base = {
                "iso_year": int(week_data["iso_year"]),
                "iso_week": int(week_data["iso_week"]),
                "week_end_time": pd.Timestamp(week_end),
                "lookback_weeks": lookback_w,
                "detection_days": detection_d,
                "evaluation_days": evaluation_d,
                "regime": regime,
                "broke_up_vp": broke_up,
                "broke_dn_vp": broke_dn,
            }

            if broke_up:
                m = _vp_break_metrics(eva, vp, "up")
                # low-volume breakout flag: first close above VAH bin volume below 30th pct
                edges, hist = vp["edges"], vp["hist"]
                bin_h = float(edges[1] - edges[0]) if len(edges) > 1 else 0.0
                first_up_idx = _first_true_idx(det["close"] > vah)
                if first_up_idx is not None:
                    price = float(det.iloc[first_up_idx]["close"])
                    b = int(np.clip(np.digitize(price, edges) - 1, 0, len(hist) - 1))
                    low_vol_break = hist[b] <= vp["lv_thresh"]
                else:
                    low_vol_break = False
                base.update(
                    {
                        "retest_poc_after_up": m["retest_poc"],
                        "t_retest_poc_after_up": m["t_retest_poc"],
                        "move_pct_vp_after_up": m["move_pct_vp"],
                        "hold_outside_value_after_up": m["hold_outside_value"],
                        "low_volume_break_after_up": bool(low_vol_break),
                    }
                )

            if broke_dn:
                m = _vp_break_metrics(eva, vp, "down")
                edges, hist = vp["edges"], vp["hist"]
                bin_h = (  # noqa: F841
                    float(edges[1] - edges[0]) if len(edges) > 1 else 0.0
                )  # noqa: F841
                first_dn_idx = _first_true_idx(det["close"] < val)
                if first_dn_idx is not None:
                    price = float(det.iloc[first_dn_idx]["close"])
                    b = int(np.clip(np.digitize(price, edges) - 1, 0, len(hist) - 1))
                    low_vol_break = hist[b] <= vp["lv_thresh"]
                else:
                    low_vol_break = False
                base.update(
                    {
                        "retest_poc_after_dn": m["retest_poc"],
                        "t_retest_poc_after_dn": m["t_retest_poc"],
                        "move_pct_vp_after_dn": m["move_pct_vp"],
                        "hold_outside_value_after_dn": m["hold_outside_value"],
                        "low_volume_break_after_dn": bool(low_vol_break),
                    }
                )

            all_vp_rows.append(base)

        if not all_rows:
            breaks_df = pd.DataFrame()
        else:
            breaks_df = pd.DataFrame(all_rows)

    # Aggregate price-range breaks
    group_cols = ["lookback_weeks", "detection_days", "evaluation_days"]
    rows: List[Dict[str, Any]] = []
    for keys, g in breaks_df.groupby(group_cols):
        d: Dict[str, Any] = dict(zip(group_cols, keys))
        d["n_breaks_up"] = int(g["broke_up"].sum())
        d["n_breaks_dn"] = int(g["broke_dn"].sum())
        for side in ("up", "dn"):
            for ev in ("re_mid", "hit_opposite", "overshoot", "deeper_pull"):
                col = f"{ev}_after_{side}"
                if col in g:
                    d[f"p_{ev}_{side}_pct"] = round(
                        float(pd.to_numeric(g[col]).mean() * 100.0), 1
                    )
            for ev in ("move_pct", "pullback_pct"):
                col = f"{ev}_after_{side}"
                if col in g:
                    d[f"median_{ev}_{side}"] = round(
                        float(pd.to_numeric(g[col]).median()), 1
                    )
        rows.append(d)
    break_agg = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    # Aggregate VP breaks by regime
    if all_vp_rows:
        vp_df = pd.DataFrame(all_vp_rows)
        vp_group = ["lookback_weeks", "detection_days", "evaluation_days", "regime"]
        vp_rows: List[Dict[str, Any]] = []
        for keys, g in vp_df.groupby(vp_group):
            d = dict(zip(vp_group, keys))
            d["n_up_vp"] = int(g.get("broke_up_vp", pd.Series(dtype=bool)).sum())
            d["n_dn_vp"] = int(g.get("broke_dn_vp", pd.Series(dtype=bool)).sum())
            for side in ("up", "dn"):
                for ev, agg, scale in (
                    ("retest_poc", "mean", 100.0),
                    ("hold_outside_value", "mean", 100.0),
                    ("low_volume_break", "mean", 100.0),
                    ("move_pct_vp", "median", 1.0),
                ):
                    col = f"{ev}_after_{side}"
                    if col in g:
                        s = pd.to_numeric(g[col], errors="coerce")
                        if agg == "mean":
                            d[f"p_{ev}_{side}_pct"] = round(float(s.mean() * 100.0), 1)
                        else:
                            d[f"median_{ev}_{side}"] = round(float(s.median()), 1)
            vp_rows.append(d)
        vp_agg = pd.DataFrame(vp_rows).sort_values(vp_group).reset_index(drop=True)
    else:
        vp_agg = pd.DataFrame(
            columns=[
                "lookback_weeks",
                "detection_days",
                "evaluation_days",
                "regime",
                "n_up_vp",
                "n_dn_vp",
                "p_retest_poc_up_pct",
                "p_retest_poc_dn_pct",
                "p_hold_outside_value_up_pct",
                "p_hold_outside_value_dn_pct",
                "p_low_volume_break_up_pct",
                "p_low_volume_break_dn_pct",
                "median_move_pct_vp_up",
                "median_move_pct_vp_dn",
            ]
        )

    break_agg = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

    # Reversal landing zones
    rev_df = _reversal_stats_percent(df_daily, w=3, horizon_days=5)
    if not rev_df.empty:
        rev_agg = (
            rev_df.groupby("type")[["p25_pct", "p50_pct", "p75_pct", "max_adverse_pct"]]
            .median()
            .reset_index()
        )
    else:
        rev_agg = pd.DataFrame(columns=["type", "p25_pct", "p50_pct", "p75_pct", "max_adverse_pct"])  # type: ignore

    # Next-day continuation
    nextday_df = _next_day_hilo_stats_percent(df_daily_hl)

    # Markdown report
    rev_df_2 = _reversal_stats_percent(df_daily, w=3, horizon_days=2)
    rev_df_3 = _reversal_stats_percent(df_daily, w=3, horizon_days=3)

    if not rev_df_2.empty:
        rev_agg_2 = (
            rev_df_2.groupby("type")[
                ["p25_pct", "p50_pct", "p75_pct", "max_adverse_pct"]
            ]
            .median()
            .reset_index()
        )
    else:
        rev_agg_2 = pd.DataFrame(columns=["type", "p25_pct", "p50_pct", "p75_pct", "max_adverse_pct"])  # type: ignore

    if not rev_df_3.empty:
        rev_agg_3 = (
            rev_df_3.groupby("type")[
                ["p25_pct", "p50_pct", "p75_pct", "max_adverse_pct"]
            ]
            .median()
            .reset_index()
        )
    else:
        rev_agg_3 = pd.DataFrame(columns=["type", "p25_pct", "p50_pct", "p75_pct", "max_adverse_pct"])  # type: ignore

    md_lines = [
        f"# {symbol} Weekly Break Analysis ({period_tag})",
        "## Break Outcomes (percent)",
        break_agg.to_markdown(index=False),
        "## Reversal Landing Zones — 5D (percent of swing-day range)",
        (
            rev_agg.to_markdown(index=False)
            if not rev_agg.empty
            else "_No swing points detected._"
        ),
        "## Reversal Landing Zones — 3D (percent of swing-day range)",
        (
            rev_agg_3.to_markdown(index=False)
            if not rev_agg_3.empty
            else "_No swing points detected for 3D._"
        ),
        "## Reversal Landing Zones — 2D (percent of swing-day range)",
        (
            rev_agg_2.to_markdown(index=False)
            if not rev_agg_2.empty
            else "_No swing points detected for 2D._"
        ),
        "## Next-Day Continuation (percent)",
        nextday_df.to_markdown(index=False),
        "## Volume-Profile Breakouts by Regime (percent)",
        (
            vp_agg.to_markdown(index=False)
            if not vp_agg.empty
            else "_No VP breakouts detected or no volume column._"
        ),
    ]

    # Corrected the final line which had two encoding arguments
    md_path.write_text("\n\n".join(md_lines), encoding="utf-8")
