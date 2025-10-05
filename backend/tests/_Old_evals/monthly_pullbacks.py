from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Tuple, Dict, Any
import numpy as np
import pandas as pd

BreakMode = Literal["close", "intraday"]


@dataclass(frozen=True)
class MonthlyParams:
    # 5D and 10D business-day windows to sweep after prior-month end
    wf5_range: Tuple[int, int] = (3, 7)
    wf10_range: Tuple[int, int] = (5, 15)
    break_mode: BreakMode = "close"
    # mid-band definitions over [0,1] range of prior-month (low..high)
    mid_band_list: Sequence[Tuple[float, float]] = (
        (0.45, 0.55),
        (0.40, 0.60),
        (0.35, 0.65),
        (0.30, 0.70),
    )


# ---------- small utilities reused ----------
def _to_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
        else:
            df.index = pd.to_datetime(df.index)
    return df.sort_index().tz_localize(None)


def _resample_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"close": df["close"].resample("B").last().dropna()})


def _first_idx(mask: pd.Series) -> float | None:
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


def _analyze_post(
    df_10: pd.DataFrame,
    lo: float,
    hi: float,
    direction: Literal["up", "down"],
    mid_lo: float,
    mid_hi: float,
) -> Dict[str, Any]:
    out = dict(
        re_mid=False,
        hit_opp=False,
        deeper=False,
        overshoot=False,
        t_mid=None,
        t_opp=None,
        t_deeper=None,
        t_ovr=None,
    )
    if df_10.empty or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out
    close = df_10["close"].astype(float)
    bands = close.apply(lambda p: _band_bucket(p, lo, hi, mid_lo, mid_hi))
    if direction == "up":
        mid_mask = bands.eq("mid_band")
        opp_mask = bands.eq("lower_band")
        deeper_mask = bands.eq("below_low")
        ovr_mask = bands.eq("above_high")
    else:
        mid_mask = bands.eq("mid_band")
        opp_mask = bands.eq("upper_band")
        deeper_mask = bands.eq("above_high")
        ovr_mask = bands.eq("below_low")
    out["re_mid"] = bool(mid_mask.any())
    out["hit_opp"] = bool(opp_mask.any())
    out["deeper"] = bool(deeper_mask.any())
    out["overshoot"] = bool(ovr_mask.any())
    out["t_mid"] = _first_idx(mid_mask)
    out["t_opp"] = _first_idx(opp_mask)
    out["t_deeper"] = _first_idx(deeper_mask)
    out["t_ovr"] = _first_idx(ovr_mask)
    return out


# ---------- monthly grouping ----------
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


# ---------- main driver ----------
def run_monthly_break_analysis(
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str | None,
    params: MonthlyParams,
) -> str:
    df = _to_time_index(market_data.copy())
    df_b = _resample_to_business_days(df)
    df_d = df.resample("1D").last().dropna()
    mon = _prev_month_hilo(df_d)

    rows: list[Dict[str, Any]] = []
    wf5_min, wf5_max = params.wf5_range
    wf10_min, wf10_max = params.wf10_range

    for _, m in mon.iloc[:-1].iterrows():  # last month has no forward window
        lo, hi = float(m["low"]), float(m["high"])
        m_end = pd.Timestamp(m["month_end"])

        for wf5 in range(wf5_min, wf5_max + 1):
            sl5 = _slice_after_point(df_b, m_end, wf5)
            up5, dn5 = _detect_break(sl5, lo, hi, params.break_mode)

            for wf10 in range(wf10_min, wf10_max + 1):
                sl10 = _slice_after_point(df_b, m_end, wf10)

                for mid_lo, mid_hi in params.mid_band_list:
                    row = dict(
                        year=int(m["year"]),
                        month=int(m["month"]),
                        prev_low=lo,
                        prev_high=hi,
                        wf5=wf5,
                        wf10=wf10,
                        mid_lo=mid_lo,
                        mid_hi=mid_hi,
                        broke_up_5d=up5,
                        broke_dn_5d=dn5,
                    )

                    # unconditional monthly post-break stats (10D horizon)
                    if up5:
                        post = _analyze_post(sl10, lo, hi, "up", mid_lo, mid_hi)
                        row.update(
                            {
                                "re_mid_after_up": post["re_mid"],
                                "opp_after_up": post["hit_opp"],
                                "deeper_after_up": post["deeper"],
                                "overshoot_after_up": post["overshoot"],
                                "t_mid_up": post["t_mid"],
                                "t_opp_up": post["t_opp"],
                                "t_deeper_up": post["t_deeper"],
                                "t_ovr_up": post["t_ovr"],
                            }
                        )
                    if dn5:
                        post = _analyze_post(sl10, lo, hi, "down", mid_lo, mid_hi)
                        row.update(
                            {
                                "re_mid_after_dn": post["re_mid"],
                                "opp_after_dn": post["hit_opp"],
                                "deeper_after_dn": post["deeper"],
                                "overshoot_after_dn": post["overshoot"],
                                "t_mid_dn": post["t_mid"],
                                "t_opp_dn": post["t_opp"],
                                "t_deeper_dn": post["t_deeper"],
                                "t_ovr_dn": post["t_ovr"],
                            }
                        )

                    # triangulation: where did price sit at 5D and 10D relative to monthly band?
                    def _band_at_end(sl: pd.DataFrame) -> str:
                        if sl.empty:
                            return "nan"
                        p = float(sl["close"].iloc[-1])
                        return _band_bucket(p, lo, hi, mid_lo, mid_hi)

                    row["band_5d_end"] = _band_at_end(sl5)
                    row["band_10d_end"] = _band_at_end(sl10)

                    rows.append(row)

    out = pd.DataFrame(rows)
    return _export_monthly_md(out, out_dir, symbol, period_tag)


# ---------- simple expectancy + sketch ----------
def _expectancy_from_probs(
    p_win: float, rr: float, costs_pips: float, pip_value_per_unit: float = 1.0
) -> float:
    """
    Toy expectancy per 1 unit of risk (R), net of costs.
    p_win: probability of the 'target' event (e.g., pullback to opposite band)
    rr: reward:risk (e.g., 1.0 means TP at 1R, SL at 1R)
    costs_pips: total round-trip costs in pips (spread+slippage+commissions)
    pip_value_per_unit: value per pip per unit sizing (keep 1.0 for normalized)
    """
    cost_R = costs_pips * pip_value_per_unit
    # EV = p*(+RR) + (1-p)*(-1) - cost_R
    return p_win * rr + (1.0 - p_win) * (-1.0) - cost_R


def _sketch_rules_md() -> str:
    return "\n".join(
        [
            "## Strategy Sketch (rules of thumb)",
            "- **Monthly UP-break**:",
            "  - If **5D ends in upper_band** → **fade** toward mid-band (mean reversion bias).",
            "  - If **5D ends in mid_band** → **wait**; bias neutral or continue if momentum conditions align.",
            "  - If **5D ends in lower_band** → continuation bias is weak; only trend-follow with confluence.",
            "- **Monthly DOWN-break**:",
            "  - If **5D ends in lower_band** → **fade** toward mid-band.",
            "  - If **5D ends in mid_band** → wait.",
            "  - If **5D ends in upper_band** → continuation bias is weak; only trend-follow with confluence.",
            "",
            "Stops/Targets (toy):",
            "- Fade: **TP** at mid-band, **SL** just beyond extreme (‘above_high’ after up-break, ‘below_low’ after down-break).",
            "- Continue: **TP** at ‘above_high’ (up) or ‘below_low’ (down), **SL** inside opposite band.",
            "",
        ]
    )


# ---------- export ----------
def _export_monthly_md(
    df: pd.DataFrame, out_dir: str, symbol: str, period_tag: str | None
) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{period_tag}" if period_tag else ""
    md_path = Path(out_dir) / f"{symbol}{tag}_monthly_pullbacks.md"

    lines: list[str] = [f"# {symbol} Monthly Break → Pullback Triangulation"]
    if period_tag:
        lines.append(f"**Period:** {period_tag}\n")

    if df.empty:
        lines.append("No data in range.")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return str(md_path)

    # Core probabilities by (wf5,wf10,band)
    def P(s) -> float:
        if s is None:
            return np.nan
        s = pd.to_numeric(s, errors="coerce")
        if s.empty:
            return np.nan
        s = s.dropna()
        return float(s.mean()) if len(s) else np.nan

    keys = ["wf5", "wf10", "mid_lo", "mid_hi"]
    agg_rows = []
    needed_cols = [
        "re_mid_after_up",
        "opp_after_up",
        "deeper_after_up",
        "overshoot_after_up",
        "re_mid_after_dn",
        "opp_after_dn",
        "deeper_after_dn",
        "overshoot_after_dn",
        "band_5d_end",
        "band_10d_end",
    ]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = np.nan

    for k, g in df.groupby(keys):
        d = dict(zip(keys, k))
        # unconditional pullback/continuation post up/down break
        for side in ("up", "dn"):
            d[f"p_re_mid_{side}"] = P(g.get(f"re_mid_after_{side}"))
            d[f"p_opp_{side}"] = P(g.get(f"opp_after_{side}"))
            d[f"p_deeper_{side}"] = P(g.get(f"deeper_after_{side}"))
            d[f"p_ovr_{side}"] = P(g.get(f"overshoot_after_{side}"))
        # conditional triangulation: probability given band at 5D/10D
        for side in ("up", "dn"):
            for b in ("lower_band", "mid_band", "upper_band"):
                g5 = g[g["band_5d_end"] == b]
                g10 = g[g["band_10d_end"] == b]
                d[f"p_opp_{side}|band5={b}"] = P(g5.get(f"opp_after_{side}"))
                d[f"p_opp_{side}|band10={b}"] = P(g10.get(f"opp_after_{side}"))
        agg_rows.append(d)

    agg = pd.DataFrame(agg_rows).sort_values(keys).reset_index(drop=True)

    # Pretty print
    pct_cols = [c for c in agg.columns if c.startswith("p_")]
    disp = agg.copy()
    for c in pct_cols:
        disp[c] = (disp[c] * 100).round(1)

    lines.append("## Parameter sweeps (probabilities in %)")
    lines.append(disp.to_markdown(index=False))

    # --- Strategy Sketch ---
    lines.append("")
    lines.append(_sketch_rules_md())

    # --- Cost-aware Expectancy examples (toy, configurable here) ---
    rr_fade = 1.0  # reward:risk for fade setups
    rr_cont = 1.0  # reward:risk for continuation setups
    costs_pips = 0.6  # round-trip costs in pips (spread+slippage+commissions)
    pip_val = 1.0  # normalized (1 pip ≈ 1/1R); adapt to your infra if needed

    ev_rows = []
    for _, r in agg.iterrows():
        base = {
            "wf5": r["wf5"],
            "wf10": r["wf10"],
            "mid_lo": r["mid_lo"],
            "mid_hi": r["mid_hi"],
        }
        # Unconditional fades: probability to hit opposite band after break
        if pd.notna(r.get("p_opp_up")):
            ev_rows.append(
                {
                    **base,
                    "side": "up",
                    "mode": "fade_uncond",
                    "p": r["p_opp_up"],
                    "ev": _expectancy_from_probs(
                        r["p_opp_up"], rr_fade, costs_pips, pip_val
                    ),
                }
            )
        if pd.notna(r.get("p_opp_dn")):
            ev_rows.append(
                {
                    **base,
                    "side": "dn",
                    "mode": "fade_uncond",
                    "p": r["p_opp_dn"],
                    "ev": _expectancy_from_probs(
                        r["p_opp_dn"], rr_fade, costs_pips, pip_val
                    ),
                }
            )
        # Unconditional continuation: probability to overshoot beyond extreme
        if pd.notna(r.get("p_ovr_up")):
            ev_rows.append(
                {
                    **base,
                    "side": "up",
                    "mode": "cont_uncond",
                    "p": r["p_ovr_up"],
                    "ev": _expectancy_from_probs(
                        r["p_ovr_up"], rr_cont, costs_pips, pip_val
                    ),
                }
            )
        if pd.notna(r.get("p_ovr_dn")):
            ev_rows.append(
                {
                    **base,
                    "side": "dn",
                    "mode": "cont_uncond",
                    "p": r["p_ovr_dn"],
                    "ev": _expectancy_from_probs(
                        r["p_ovr_dn"], rr_cont, costs_pips, pip_val
                    ),
                }
            )

        # Conditional (triangulation) using band_5d_end:
        for b in ("lower_band", "mid_band", "upper_band"):
            col_up = f"p_opp_up|band5={b}"
            col_dn = f"p_opp_dn|band5={b}"
            if col_up in r and pd.notna(r[col_up]):
                ev_rows.append(
                    {
                        **base,
                        "side": "up",
                        "mode": f"fade_band5={b}",
                        "p": r[col_up],
                        "ev": _expectancy_from_probs(
                            r[col_up], rr_fade, costs_pips, pip_val
                        ),
                    }
                )
            if col_dn in r and pd.notna(r[col_dn]):
                ev_rows.append(
                    {
                        **base,
                        "side": "dn",
                        "mode": f"fade_band5={b}",
                        "p": r[col_dn],
                        "ev": _expectancy_from_probs(
                            r[col_dn], rr_fade, costs_pips, pip_val
                        ),
                    }
                )

    ev_df = pd.DataFrame(ev_rows)
    if not ev_df.empty:
        ev_disp = ev_df.copy()
        ev_disp["p"] = (ev_disp["p"] * 100).round(1)
        ev_disp["ev"] = ev_disp["ev"].round(3)

        lines.append("## Cost-aware Expectancy (toy, per 1R, net of costs)")
        lines.append(
            f"- Assumptions: RR_fade={rr_fade}, RR_cont={rr_cont}, costs={costs_pips} pips\n"
        )

        lines.append("### Top EV (unconditional)")
        top_uncond = (
            ev_disp[ev_disp["mode"].str.contains("uncond")]
            .sort_values("ev", ascending=False)
            .head(12)
        )
        lines.append(top_uncond.to_markdown(index=False))

        lines.append("\n### Top EV (conditional on band_5d_end)")
        top_cond = (
            ev_disp[ev_disp["mode"].str.contains("fade_band5")]
            .sort_values("ev", ascending=False)
            .head(12)
        )
        lines.append(top_cond.to_markdown(index=False))

    md_path.write_text("\n".join([s for s in lines if s is not None]), encoding="utf-8")
    return str(md_path)


# monthly_params = MonthlyParams(
#     wf5_range=(3, 7),
#     wf10_range=(5, 15),
#     break_mode="close",
#     mid_band_list=[(0.45, 0.55), (0.40, 0.60), (0.35, 0.65), (0.30, 0.70)],
# )
