# backtester/performance/pullback_summary.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from math import sqrt

BreakMode = Literal["close", "intraday"]
OvershootMode = Literal["range_pct", "atr_k"]


# --------------------------
# Parameters / configuration
# --------------------------
@dataclass(frozen=True)
class SummaryParams:
    # Consolidation detection (pre-break)
    cons_lookback_bdays: int = 10  # business days before month_end
    cons_range_atr_k: float = 0.8  # (high-low over lookback) / ATR20 ≤ k → flat
    cons_mid_occupancy_min: float = 0.55  # ≥55% closes in mid band → flat
    cons_mid_range: Tuple[float, float] = (
        0.40,
        0.60,
    )  # use a tighter 40–60% neutral band

    # Quintile buckets for weekly/2-weekly landing
    bucket_edges: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # Windows / bands
    wf5_range: Tuple[int, int] = (3, 7)
    wf10_range: Tuple[int, int] = (5, 15)
    break_mode: BreakMode = "close"
    mid_band_list: Sequence[Tuple[float, float]] = (
        (0.45, 0.55),
        (0.40, 0.60),
        (0.35, 0.65),
        (0.30, 0.70),
    )

    # Overshoot definition
    overshoot_mode: OvershootMode = "range_pct"
    overshoot_k: float = 0.15

    # Reporting controls
    min_n: int = 10

    # Top set-ups section
    top_k: int = 10
    top_min_n: int = 30  # N ≥ Y
    top_max_ci_width: float = 0.20  # CI width ≤ X (e.g., 0.20 = 20pp)
    top_prob_cols: Tuple[str, ...] = (  # which probabilities to rank
        "p_opp_up",
        "p_opp_dn",
        "p_ovr_up",
        "p_ovr_dn",
        "p_re_mid_up",
        "p_re_mid_dn",
        # keep conditional columns if you later compute them
        # "p_opp_up|band5=lower_band",
        # "p_opp_up|band5=upper_band",
        # "p_opp_dn|band5=lower_band",
        # "p_opp_dn|band5=upper_band",
    )


# --------------------------
# Generic helpers
# --------------------------
def _to_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
    return df.sort_index().tz_localize(None)


def _resample_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"close": df["close"].resample("B").last().dropna()})


def _first_idx(mask: pd.Series) -> Optional[float]:
    nz = np.flatnonzero(mask.values)
    return float(nz[0]) if len(nz) else None


def _band_bucket(p: float, lo: float, hi: float, mid_lo: float, mid_hi: float) -> str:
    if not np.isfinite(p) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return "nan"
    r = (p - lo) / (hi - lo)
    if r < 0.0:
        return "below_low"
    if r < mid_lo:
        return "lower_band"
    if r <= mid_hi:
        return "mid_band"
    if r <= 1.0:
        return "upper_band"
    return "above_high"


def _bucket_label(p: float, lo: float, hi: float, edges: Sequence[float]) -> str:
    if not np.isfinite(p) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return "nan"
    r = (p - lo) / (hi - lo)
    for i in range(1, len(edges)):
        if r <= edges[i]:
            return f"Q{i}"  # Q1..Q{len(edges)-1}
    return f"Q{len(edges)-1}"


def _slice_after_point(df_bday: pd.DataFrame, t0: pd.Timestamp, n: int) -> pd.DataFrame:
    i = df_bday.index.searchsorted(t0, side="right")
    j = min(i + n, len(df_bday))
    return df_bday.iloc[i:j]


def _detect_break(
    df_wf: pd.DataFrame, lo: float, hi: float, mode: BreakMode
) -> tuple[bool, bool]:
    if df_wf.empty or not np.isfinite(lo) or not np.isfinite(hi):
        return (False, False)
    if mode == "close":
        return (bool((df_wf["close"] > hi).any()), bool((df_wf["close"] < lo).any()))
    # intraday (fallback if high/low missing)
    high = df_wf["high"] if "high" in df_wf.columns else df_wf["close"]
    low = df_wf["low"] if "low" in df_wf.columns else df_wf["close"]
    return (bool((high > hi).any()), bool((low < lo).any()))


# --------------------------
# Monthly / Weekly windows
# --------------------------
def _prev_month_hilo(df_daily: pd.DataFrame) -> pd.DataFrame:
    g = df_daily.groupby([df_daily.index.year, df_daily.index.month], sort=True)
    mon = g.agg(high=("close", "max"), low=("close", "min"))
    mon["month_start"] = g.apply(lambda s: s.index.min())
    mon["month_end"] = g.apply(lambda s: s.index.max())
    mon = (
        mon.reset_index(names=["year", "month"])
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    return mon


def _prev_week_hilo(df_daily: pd.DataFrame) -> pd.DataFrame:
    # ISO week
    iso = df_daily.index.isocalendar()
    g = df_daily.groupby([iso.year, iso.week], sort=True)
    wk = g.agg(high=("close", "max"), low=("close", "min"))
    wk["week_start"] = g.apply(lambda s: s.index.min())
    wk["week_end"] = g.apply(lambda s: s.index.max())
    wk = (
        wk.reset_index(names=["iso_year", "iso_week"])
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )
    return wk


def _two_week_hilo(df_daily: pd.DataFrame) -> pd.DataFrame:
    # Rolling 2 weeks based on ISO week blocks
    w = _prev_week_hilo(df_daily)
    rows = []
    for i in range(1, len(w)):  # need (i-1, i) combined to form a complete 2-week block
        lo = float(min(w.loc[i - 1, "low"], w.loc[i, "low"]))
        hi = float(max(w.loc[i - 1, "high"], w.loc[i, "high"]))
        end = pd.Timestamp(w.loc[i, "week_end"])
        start = pd.Timestamp(w.loc[i - 1, "week_start"])
        rows.append(dict(tw_start=start, tw_end=end, low=lo, high=hi))
    return pd.DataFrame(rows)


# --------------------------
# ATR / Overshoot threshold
# --------------------------
def _atr_from_daily(df_daily: pd.DataFrame, period: int = 20) -> pd.Series:
    # If high/low missing, fall back to True Range using close
    high = df_daily["high"] if "high" in df_daily.columns else df_daily["close"]
    low = df_daily["low"] if "low" in df_daily.columns else df_daily["close"]
    close = df_daily["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _overshoot_mask(
    series_close: pd.Series,
    hi: float,
    lo: float,
    mode: OvershootMode,
    k: float,
    atr_val: Optional[float],
) -> pd.Series:
    if mode == "range_pct":
        thr_up = hi + k * (hi - lo)
        thr_dn = lo - k * (hi - lo)
    else:
        # atr_k
        a = atr_val if atr_val and np.isfinite(atr_val) else k * (hi - lo)  # fallback
        thr_up = hi + k * a
        thr_dn = lo - k * a
    return pd.Series(
        (series_close > thr_up) | (series_close < thr_dn), index=series_close.index
    )


def _is_consolidation(
    df_bday: pd.DataFrame,
    t_end: pd.Timestamp,
    lookback: int,
    lo_m: float,
    hi_m: float,
    atr20: pd.Series,
    mid_lo_cons: float,
    mid_hi_cons: float,
    range_atr_k: float,
    mid_occ_min: float,
) -> bool:
    if df_bday.empty:
        return False
    # Slice last `lookback` business days ending at month_end
    j = df_bday.index.searchsorted(t_end, side="right")
    i = max(0, j - lookback)
    sl = df_bday.iloc[i:j]
    if sl.empty:
        return False

    # Range vs ATR20
    r = float(sl["close"].max() - sl["close"].min())
    atr_val = (
        float(atr20.loc[:t_end].iloc[-1]) if not atr20.loc[:t_end].empty else np.nan
    )
    range_ok = np.isfinite(atr_val) and atr_val > 0 and (r / atr_val) <= range_atr_k

    # Mid-band occupancy using a *fixed* neutral band (e.g., 40–60%) on the monthly range
    bands = (
        sl["close"]
        .astype(float)
        .apply(lambda p: _band_bucket(p, lo_m, hi_m, mid_lo_cons, mid_hi_cons))
    )
    mid_occ = (bands == "mid_band").mean() if len(bands) else 0.0
    mid_ok = mid_occ >= mid_occ_min

    return bool(range_ok and mid_ok)


# --------------------------
# Wilson CI + rate helper
# --------------------------
def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate_with_ci(mask: pd.Series) -> tuple[float, int, float, float]:
    m = mask.dropna().astype(bool)
    n = int(m.size)
    s = int(m.sum())
    p = (s / n) if n else np.nan
    lo, hi = _wilson_ci(s, n) if n else (np.nan, np.nan)
    return (p, n, lo, hi)


# --------------------------
# Main driver
# --------------------------
def run_pullback_summary(
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: Optional[str],
    params: SummaryParams,
) -> str:
    """
    One compressed, easy-to-read report:
      - Monthly break → continuation / pullback (with overshoot threshold)
      - Where price lands at 5D/10D vs prior **1-week** and **2-week** bands (both 3-band & quintiles)
      - Wilson CIs + N; rows with N < min_n suppressed
      - Condition on pre-break consolidation (CONS vs NONCONS)
    """
    df = _to_time_index(market_data.copy())
    df_b = _resample_to_business_days(df)
    df_d = df.resample("1D").last().dropna()

    # Prepare ranges
    mon = _prev_month_hilo(df_d)
    wk = _prev_week_hilo(df_d)
    tw = _two_week_hilo(df_d)
    atr20 = _atr_from_daily(df_d)

    rows: list[Dict[str, Any]] = []
    wf5_min, wf5_max = params.wf5_range
    wf10_min, wf10_max = params.wf10_range

    # Helper to find most recent completed week/two-week ending <= t0
    def _nearest_week_end(t0: pd.Timestamp) -> Optional[pd.Series]:
        cand = wk[wk["week_end"] <= t0]
        return cand.iloc[-1] if not cand.empty else None

    def _nearest_2w_end(t0: pd.Timestamp) -> Optional[pd.Series]:
        cand = tw[tw["tw_end"] <= t0]
        return cand.iloc[-1] if not cand.empty else None

    for _, m in mon.iloc[:-1].iterrows():  # last month has no forward window
        lo_m, hi_m = float(m["low"]), float(m["high"])
        m_end = pd.Timestamp(m["month_end"])
        atr_at_end = (
            float(atr20.loc[:m_end].iloc[-1]) if not atr20.loc[:m_end].empty else np.nan
        )

        # Reference weekly & bi-weekly ranges anchored at month end
        wk_end_row = _nearest_week_end(m_end)
        tw_end_row = _nearest_2w_end(m_end)
        if wk_end_row is None or tw_end_row is None:
            continue

        lo_w, hi_w = float(wk_end_row["low"]), float(wk_end_row["high"])
        lo_2w, hi_2w = float(tw_end_row["low"]), float(tw_end_row["high"])

        for wf5 in range(wf5_min, wf5_max + 1):
            sl5 = _slice_after_point(df_b, m_end, wf5)
            up5, dn5 = _detect_break(sl5, lo_m, hi_m, params.break_mode)

            for wf10 in range(wf10_min, wf10_max + 1):
                sl10 = _slice_after_point(df_b, m_end, wf10)

                # Overshoot mask vs monthly extremes with threshold
                if not sl10.empty:
                    o_mask = _overshoot_mask(
                        sl10["close"],
                        hi_m,
                        lo_m,
                        params.overshoot_mode,
                        params.overshoot_k,
                        atr_at_end,
                    )
                else:
                    o_mask = pd.Series(dtype=bool)

                for mid_lo, mid_hi in params.mid_band_list:
                    row: Dict[str, Any] = dict(
                        year=int(m["year"]),
                        month=int(m["month"]),
                        wf5=wf5,
                        wf10=wf10,
                        mid_lo=mid_lo,
                        mid_hi=mid_hi,
                        broke_up_5d=up5,
                        broke_dn_5d=dn5,
                    )

                    # Consolidation flag (using fixed neutral band cons_mid_range)
                    is_cons = _is_consolidation(
                        df_b,  # business-day closes
                        m_end,  # month end timestamp
                        params.cons_lookback_bdays,
                        lo_m,
                        hi_m,  # monthly low/high
                        atr20,  # daily ATR20 series
                        params.cons_mid_range[0],
                        params.cons_mid_range[1],  # fixed neutral band (e.g., 40–60%)
                        params.cons_range_atr_k,
                        params.cons_mid_occupancy_min,
                    )
                    row["pre_cons"] = is_cons

                    # Where did price land at 5D/10D vs WEEK and 2-WEEK bands (3-band & quintiles)?
                    def _band_at_end(sl: pd.DataFrame, lo: float, hi: float) -> str:
                        if sl.empty:
                            return "nan"
                        p = float(sl["close"].iloc[-1])
                        return _band_bucket(p, lo, hi, mid_lo, mid_hi)

                    def _q_end(sl: pd.DataFrame, lo: float, hi: float) -> str:
                        if sl.empty:
                            return "nan"
                        return _bucket_label(
                            float(sl["close"].iloc[-1]), lo, hi, params.bucket_edges
                        )

                    row["wk_band_5d_end"] = _band_at_end(sl5, lo_w, hi_w)
                    row["wk_band_10d_end"] = _band_at_end(sl10, lo_w, hi_w)
                    row["tw_band_5d_end"] = _band_at_end(sl5, lo_2w, hi_2w)
                    row["tw_band_10d_end"] = _band_at_end(sl10, lo_2w, hi_2w)

                    row["wk_q_5d_end"] = _q_end(sl5, lo_w, hi_w)
                    row["wk_q_10d_end"] = _q_end(sl10, lo_w, hi_w)
                    row["tw_q_5d_end"] = _q_end(sl5, lo_2w, hi_2w)
                    row["tw_q_10d_end"] = _q_end(sl10, lo_2w, hi_2w)

                    # Post-break booleans in 10D
                    if up5:
                        close10 = sl10["close"].astype(float)
                        bands10_m = close10.apply(
                            lambda p: _band_bucket(p, lo_m, hi_m, mid_lo, mid_hi)
                        )
                        row["re_mid_after_up"] = bool((bands10_m == "mid_band").any())
                        row["opp_after_up"] = bool((bands10_m == "lower_band").any())
                        row["deeper_after_up"] = bool((bands10_m == "below_low").any())
                        row["overshoot_after_up"] = bool(o_mask.any())
                    if dn5:
                        close10 = sl10["close"].astype(float)
                        bands10_m = close10.apply(
                            lambda p: _band_bucket(p, lo_m, hi_m, mid_lo, mid_hi)
                        )
                        row["re_mid_after_dn"] = bool((bands10_m == "mid_band").any())
                        row["opp_after_dn"] = bool((bands10_m == "upper_band").any())
                        row["deeper_after_dn"] = bool((bands10_m == "above_high").any())
                        row["overshoot_after_dn"] = bool(o_mask.any())

                    rows.append(row)

    df_all = pd.DataFrame(rows)
    return _export_summary_md(df_all, out_dir, symbol, period_tag, params)


def _standardize_uncond_cols(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Map internal names like 'opp_up_p', 'opp_up_lo', 'opp_up_hi', 'opp_up_n'
    to standardized names 'p_opp_up', 'ci_low_p_opp_up', 'ci_high_p_opp_up', 'N_p_opp_up'
    so the Top Set-ups block can work generically.
    """
    out = agg.copy()
    maps = [
        ("re_mid_up", "p_re_mid_up"),
        ("re_mid_dn", "p_re_mid_dn"),
        ("opp_up", "p_opp_up"),
        ("opp_dn", "p_opp_dn"),
        ("deeper_up", "p_deeper_up"),
        ("deeper_dn", "p_deeper_dn"),
        ("overshoot_up", "p_ovr_up"),
        ("overshoot_dn", "p_ovr_dn"),
    ]
    for base, std in maps:
        p, lo, hi, n = f"{base}_p", f"{base}_lo", f"{base}_hi", f"{base}_n"
        if p in out.columns:
            out[std] = out[p]
            out[f"ci_low_{std}"] = out.get(lo, np.nan)
            out[f"ci_high_{std}"] = out.get(hi, np.nan)
            out[f"N_{std}"] = out.get(n, np.nan)
    return out


def _append_top_setups_block(
    lines: list[str], top_src: pd.DataFrame, params: SummaryParams
) -> None:
    if top_src is None or top_src.empty:
        return

    def _fmt(df: pd.DataFrame, col: str) -> pd.DataFrame:
        need = [col, f"ci_low_{col}", f"ci_high_{col}", f"N_{col}"]
        if not all(c in df.columns for c in need):
            return pd.DataFrame()
        t = df[["wf5", "wf10", "mid_lo", "mid_hi", *need]].dropna().copy()
        if t.empty:
            return t
        t["ci_width"] = t[f"ci_high_{col}"] - t[f"ci_low_{col}"]
        t = t[
            (t[f"N_{col}"] >= params.top_min_n)
            & (t["ci_width"] <= params.top_max_ci_width)
        ]
        if t.empty:
            return t
        t[col] = (t[col] * 100).round(1)
        t[f"ci_low_{col}"] = (t[f"ci_low_{col}"] * 100).round(1)
        t[f"ci_high_{col}"] = (t[f"ci_high_{col}"] * 100).round(1)
        t = t.rename(
            columns={
                col: "p(%)",
                f"ci_low_{col}": "CI_low(%)",
                f"ci_high_{col}": "CI_high(%)",
                f"N_{col}": "N",
            }
        )
        return t.sort_values("p(%)", ascending=False).head(params.top_k)

    lines.append("")
    lines.append("## Top set-ups (highest p with narrow CI & enough samples)")
    lines.append(
        f"- Criteria: N ≥ **{params.top_min_n}**, CI width ≤ **{int(params.top_max_ci_width*100)}pp**, "
        f"top **{params.top_k}** per metric."
    )
    for col in params.top_prob_cols:
        t = _fmt(top_src, col)
        if t is None or t.empty:
            continue
        lines.append(f"### {col}")
        lines.append(t.to_markdown(index=False))
        lines.append("")  # spacer


def _build_conditional_top_source(
    df: pd.DataFrame, keys: list[str], params: SummaryParams
) -> pd.DataFrame:
    """Compute conditional probabilities per (wf5,wf10,mid_lo,mid_hi) across
    several conditions (quintiles, 3-band, CONS/NONCONS) and return a wide table
    with standardized columns like: p_ovr_up|wk_q_5d_end=Q5, plus CI & N columns.
    """
    cond_specs: list[tuple[str, list[Any]]] = [
        ("wk_q_5d_end", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("wk_q_10d_end", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("tw_q_5d_end", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("tw_q_10d_end", ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        ("wk_band_5d_end", ["lower_band", "mid_band", "upper_band"]),
        ("wk_band_10d_end", ["lower_band", "mid_band", "upper_band"]),
        ("tw_band_5d_end", ["lower_band", "mid_band", "upper_band"]),
        ("tw_band_10d_end", ["lower_band", "mid_band", "upper_band"]),
        ("pre_cons", [True, False]),
    ]

    out_rows: list[dict[str, Any]] = []
    for k, g in df.groupby(keys):
        d: dict[str, Any] = dict(zip(keys, k))
        for cond_key, values in cond_specs:
            if cond_key not in g.columns:
                continue
            for val in values:
                sub_up = g[(g[cond_key] == val) & (g["broke_up_5d"].astype(bool))]
                sub_dn = g[(g[cond_key] == val) & (g["broke_dn_5d"].astype(bool))]
                for side, sub in (("up", sub_up), ("dn", sub_dn)):
                    for metric in ("re_mid", "opp", "deeper", "overshoot"):
                        col = f"{metric}_after_{side}"
                        if col not in sub.columns:
                            p = n = lo = hi = np.nan
                        else:
                            p, n, lo, hi = _rate_with_ci(
                                pd.to_numeric(sub[col], errors="coerce")
                            )
                        base = f"p_{metric}_{side}|{cond_key}={val}"
                        d[base] = p
                        d[f"ci_low_{base}"] = lo
                        d[f"ci_high_{base}"] = hi
                        d[f"N_{base}"] = n
        out_rows.append(d)
    return pd.DataFrame(out_rows).sort_values(keys).reset_index(drop=True)


def _append_conditional_top_setups_block(
    lines: list[str], cond_src: pd.DataFrame, params: SummaryParams
) -> None:
    """Append ranked conditional top setups. Uses a curated default list of columns
    if present. You can later expose this list in params if you prefer."""
    if cond_src is None or cond_src.empty:
        return

    # Curated conditional targets
    cols = [
        # Continuations after strong 1w/2w pushes
        "p_ovr_up|wk_q_5d_end=Q5",
        "p_ovr_dn|wk_q_5d_end=Q1",
        "p_ovr_up|wk_q_10d_end=Q5",
        "p_ovr_dn|wk_q_10d_end=Q1",
        "p_ovr_up|tw_q_10d_end=Q5",
        "p_ovr_dn|tw_q_10d_end=Q1",
        # Fades from extremes
        "p_opp_up|wk_q_5d_end=Q5",
        "p_opp_dn|wk_q_5d_end=Q1",
        # 3-band conditioned
        "p_ovr_up|wk_band_5d_end=upper_band",
        "p_ovr_dn|wk_band_5d_end=lower_band",
        "p_opp_up|wk_band_5d_end=upper_band",
        "p_opp_dn|wk_band_5d_end=lower_band",
        # Consolidation conditioned
        "p_ovr_up|pre_cons=True",
        "p_ovr_dn|pre_cons=True",
        "p_opp_up|pre_cons=True",
        "p_opp_dn|pre_cons=True",
    ]

    def _fmt(df: pd.DataFrame, col: str) -> pd.DataFrame:
        need = [col, f"ci_low_{col}", f"ci_high_{col}", f"N_{col}"]
        if not all(c in df.columns for c in need):
            return pd.DataFrame()
        t = df[["wf5", "wf10", "mid_lo", "mid_hi", *need]].dropna().copy()
        if t.empty:
            return t
        t["ci_width"] = t[f"ci_high_{col}"] - t[f"ci_low_{col}"]
        t = t[
            (t[f"N_{col}"] >= params.top_min_n)
            & (t["ci_width"] <= params.top_max_ci_width)
        ]
        if t.empty:
            return t
        t[col] = (t[col] * 100).round(1)
        t[f"ci_low_{col}"] = (t[f"ci_low_{col}"] * 100).round(1)
        t[f"ci_high_{col}"] = (t[f"ci_high_{col}"] * 100).round(1)
        t = t.rename(
            columns={
                col: "p(%)",
                f"ci_low_{col}": "CI_low(%)",
                f"ci_high_{col}": "CI_high(%)",
                f"N_{col}": "N",
            }
        )
        return t.sort_values("p(%)", ascending=False).head(params.top_k)

    lines.append("")
    lines.append("## Top set-ups — **Conditional** (ranked, CI-filtered)")
    lines.append(
        f"- Criteria: N ≥ **{params.top_min_n}**, CI width ≤ **{int(params.top_max_ci_width*100)}pp**; "
        f"top **{params.top_k}** per conditional metric."
    )

    for col in cols:
        t = _fmt(cond_src, col)
        if t is None or t.empty:
            continue
        lines.append(f"### {col}")
        lines.append(t.to_markdown(index=False))
        lines.append("")  # --------------------------


# Export
# --------------------------
def _export_summary_md(
    df: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: Optional[str],
    params: SummaryParams,
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{period_tag}" if period_tag else ""
    md_path = Path(out_dir) / f"{symbol}{tag}_pullback_summary.md"

    lines: list[str] = [f"# {symbol} Monthly → Weekly / 2-Weekly Pullback Summary"]
    if period_tag:
        lines.append(f"**Period:** {period_tag}")
    lines.append(
        f"_Settings_: wf5={params.wf5_range}, wf10={params.wf10_range}, "
        f"break={params.break_mode}, overshoot={params.overshoot_mode}({params.overshoot_k}), "
        f"min_n={params.min_n}\n"
    )

    if df.empty:
        lines.append("No data in range.")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return str(md_path)

    # Ensure needed cols exist
    for c in [
        "re_mid_after_up",
        "opp_after_up",
        "deeper_after_up",
        "overshoot_after_up",
        "re_mid_after_dn",
        "opp_after_dn",
        "deeper_after_dn",
        "overshoot_after_dn",
        "wk_band_5d_end",
        "wk_band_10d_end",
        "tw_band_5d_end",
        "tw_band_10d_end",
        "wk_q_5d_end",
        "wk_q_10d_end",
        "tw_q_5d_end",
        "tw_q_10d_end",
        "pre_cons",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    # --- CONS vs NONCONS conditioning (pre-break)
    def _cond_on_cons(df_in: pd.DataFrame) -> str:
        keys = ["wf5", "wf10", "mid_lo", "mid_hi"]
        out_rows = []
        for k, g in df_in.groupby(keys):
            base = dict(zip(keys, k))
            for side in ("up", "dn"):
                for metric in ("re_mid", "opp", "deeper", "overshoot"):
                    for flag, label in ((True, "CONS"), (False, "NONCONS")):
                        m = g[g["pre_cons"] == flag].get(f"{metric}_after_{side}")
                        if m is None:
                            p = n = lo = hi = np.nan
                        else:
                            p, n, lo, hi = _rate_with_ci(
                                pd.to_numeric(m, errors="coerce")
                            )
                        out_rows.append(
                            {
                                **base,
                                "side": side,
                                "metric": metric,
                                "state": label,
                                "p": p,
                                "n": n,
                                "lo": lo,
                                "hi": hi,
                            }
                        )
        t = pd.DataFrame(out_rows)
        if t.empty:
            return "_No data._"
        t = t[t["n"] >= params.min_n]
        if t.empty:
            return "_No rows ≥ min_n._"
        disp = t.copy()
        for c in ("p", "lo", "hi"):
            disp[c] = (disp[c] * 100).round(1)
        return disp.sort_values(
            ["wf5", "wf10", "mid_lo", "mid_hi", "metric", "side", "state"]
        ).to_markdown(index=False)

    lines.append(
        "\n## Post-break probabilities | conditioned on **pre-break consolidation**"
    )
    lines.append(_cond_on_cons(df))

    # --- Unconditional monthly post-break probabilities
    keys = ["wf5", "wf10", "mid_lo", "mid_hi"]
    agg_rows = []
    for k, g in df.groupby(keys):
        d = dict(zip(keys, k))
        for side in ("up", "dn"):
            for metric in ("re_mid", "opp", "deeper", "overshoot"):
                m = g.get(f"{metric}_after_{side}")
                if m is None:
                    p = n = lo = hi = np.nan
                else:
                    p, n, lo, hi = _rate_with_ci(pd.to_numeric(m, errors="coerce"))
                d[f"{metric}_{side}_p"] = p
                d[f"{metric}_{side}_n"] = n
                d[f"{metric}_{side}_lo"] = lo
                d[f"{metric}_{side}_hi"] = hi
        agg_rows.append(d)
    agg = pd.DataFrame(agg_rows).sort_values(keys).reset_index(drop=True)

    # Top setups (based on unconditional stats)
    n_cols = [c for c in agg.columns if c.endswith("_n")]
    top_src = _standardize_uncond_cols(agg)
    _append_top_setups_block(lines, top_src, params)

    # Build and append conditional top setups
    cond_top_src = _build_conditional_top_source(df, keys, params)
    _append_conditional_top_setups_block(lines, cond_top_src, params)

    def _row_ok(r) -> bool:
        return any(pd.notna(r[c]) and int(r[c]) >= params.min_n for c in n_cols)

    agg = agg[[_row_ok(r) for _, r in agg.iterrows()]].reset_index(drop=True)

    if not agg.empty:
        disp = agg.copy()
        for c in disp.columns:
            if c.endswith("_p") or c.endswith("_lo") or c.endswith("_hi"):
                disp[c] = (disp[c] * 100).round(1)
        lines.append("## Unconditional post-break probabilities (%, Wilson CI, N)")
        lines.append(disp.to_markdown(index=False))
    else:
        lines.append("## Unconditional post-break probabilities")
        lines.append("_No rows ≥ min_n; widen period or relax settings._")

    # --- Landing band (3-band) after monthly break → WEEK & TWO-WEEK
    def _band_table(level: str) -> str:
        out_rows = []
        for k, g in df.groupby(keys):
            base = dict(zip(keys, k))
            for end in ("5d_end", "10d_end"):
                for b in ("lower_band", "mid_band", "upper_band"):
                    mask_up = g["broke_up_5d"].astype(bool)
                    mask_dn = g["broke_dn_5d"].astype(bool)
                    col = f"{level}_band_{end}"
                    if col not in g.columns:
                        continue
                    m_up = g.loc[mask_up, col].eq(b)
                    m_dn = g.loc[mask_dn, col].eq(b)
                    p_up, n_up, lo_up, hi_up = _rate_with_ci(m_up)
                    p_dn, n_dn, lo_dn, hi_dn = _rate_with_ci(m_dn)
                    out_rows.append(
                        {
                            **base,
                            "window": end.replace("_", " ").upper(),
                            "band": b,
                            "p_up": p_up,
                            "n_up": n_up,
                            "lo_up": lo_up,
                            "hi_up": hi_up,
                            "p_dn": p_dn,
                            "n_dn": n_dn,
                            "lo_dn": lo_dn,
                            "hi_dn": hi_dn,
                        }
                    )
        tbl = pd.DataFrame(out_rows)
        if tbl.empty:
            return "_No data._"
        tbl = tbl[(tbl["n_up"] >= params.min_n) | (tbl["n_dn"] >= params.min_n)]
        if tbl.empty:
            return "_No rows ≥ min_n._"
        disp = tbl.copy()
        for c in ["p_up", "lo_up", "hi_up", "p_dn", "lo_dn", "hi_dn"]:
            disp[c] = (disp[c] * 100).round(1)
        return disp.sort_values(keys + ["window", "band"]).to_markdown(index=False)

    # --- Landing quintiles (Q1..Q5) after monthly break → WEEK & TWO-WEEK
    def _quant_table(level: str) -> str:
        out_rows = []
        for k, g in df.groupby(keys):
            base = dict(zip(keys, k))
            for end in ("5d_end", "10d_end"):
                col = f"{level}_q_{end}"
                if col not in g.columns:
                    continue
                for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
                    m_up = g.loc[g["broke_up_5d"].astype(bool), col].eq(q)
                    m_dn = g.loc[g["broke_dn_5d"].astype(bool), col].eq(q)
                    p_up, n_up, lo_up, hi_up = _rate_with_ci(m_up)
                    p_dn, n_dn, lo_dn, hi_dn = _rate_with_ci(m_dn)
                    out_rows.append(
                        {
                            **base,
                            "window": end.replace("_", " ").upper(),
                            "bucket": q,
                            "p_up": p_up,
                            "n_up": n_up,
                            "lo_up": lo_up,
                            "hi_up": hi_up,
                            "p_dn": p_dn,
                            "n_dn": n_dn,
                            "lo_dn": lo_dn,
                            "hi_dn": hi_dn,
                        }
                    )
        tbl = pd.DataFrame(out_rows)
        if tbl.empty:
            return "_No data._"
        tbl = tbl[(tbl["n_up"] >= params.min_n) | (tbl["n_dn"] >= params.min_n)]
        if tbl.empty:
            return "_No rows ≥ min_n._"
        disp = tbl.copy()
        for c in ["p_up", "lo_up", "hi_up", "p_dn", "lo_dn", "hi_dn"]:
            disp[c] = (disp[c] * 100).round(1)
        return disp.sort_values(keys + ["window", "bucket"]).to_markdown(index=False)

    lines.append("\n## Landing band after monthly break → **1-Week bands (3-band)**")
    lines.append(_band_table("wk"))
    lines.append("\n## Landing band after monthly break → **2-Week bands (3-band)**")
    lines.append(_band_table("tw"))

    lines.append("\n## Landing **quintiles** after monthly break → **1-Week (Q1..Q5)**")
    lines.append(_quant_table("wk"))
    lines.append("\n## Landing **quintiles** after monthly break → **2-Week (Q1..Q5)**")
    lines.append(_quant_table("tw"))

    # --- TL;DR
    lines.append("\n## TL;DR (compressed)")
    lines.append(
        "- After a **monthly up-break**, continuation vs pullback stats (with Wilson CIs) + where price lands in 1–2 weeks are shown above."
    )
    lines.append(
        "- After a **monthly down-break**, mirror the logic; downside continuation often dominates in EURUSD."
    )
    lines.append(
        f"- Rows filtered with **N < {params.min_n}** are hidden. Use longer periods or lower min_n to surface more cells."
    )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(md_path)
