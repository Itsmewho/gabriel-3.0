# backtester/performance/weekly_break_analysis.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Literal
import numpy as np
import pandas as pd

# --- Helper Functions ---


def _week_id(ts: pd.Timestamp) -> tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)


def _resample_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"close": df["close"].resample("B").last().dropna()})


def _slice_after_week(
    df_bday: pd.DataFrame, week_end: pd.Timestamp, n: int
) -> pd.DataFrame:
    i = df_bday.index.searchsorted(week_end, side="right")
    j = min(i + n, len(df_bday))
    return df_bday.iloc[i:j]


def _get_lookback_hilo(
    df_day: pd.DataFrame, lookback_periods: set[int]
) -> pd.DataFrame:
    """Calculates the high/low over multiple preceding weeks."""
    di = df_day.copy()
    di["iso_year"], di["iso_week"] = zip(*di.index.map(_week_id))
    grp = di.groupby(["iso_year", "iso_week"], sort=True)
    wk = grp.agg(
        high=("close", "max"),
        low=("close", "min"),
        week_end_time=("close", "last_valid_index"),
    ).reset_index()
    wk = wk.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

    for period in lookback_periods:
        wk[f"high_{period}w"] = wk["high"].shift(1).rolling(window=period).max()
        wk[f"low_{period}w"] = wk["low"].shift(1).rolling(window=period).min()
    return wk


# --- Core Analysis Logic ---


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
    mid_lo, mid_hi = 0.4, 0.6  # Using a fixed 40-60% mid-band

    def _band_bucket(p, lo, hi, mlo, mhi):
        if not np.isfinite(p) or hi <= lo:
            return "nan"
        r = (p - lo) / (hi - lo)
        if r < 0.0:
            return "below_low"
        if r < mlo:
            return "lower_band"
        if r <= mhi:
            return "mid_band"
        if r <= 1.0:
            return "upper_band"
        return "above_high"

    def _first_idx(s):
        nz = np.flatnonzero(s.values)
        return int(nz[0]) if len(nz) else None

    out = {
        k: None
        for k in [
            "re_mid",
            "hit_opposite",
            "overshoot",
            "deeper_pull",
            "t_mid",
            "t_opp",
            "t_ovr",
            "t_deeper",
        ]
    }
    if (
        eval_slice.empty
        or not all(np.isfinite([prev_low, prev_high]))
        or prev_high <= prev_low
    ):
        return out

    bands = eval_slice["close"].apply(
        lambda p: _band_bucket(p, prev_low, prev_high, mid_lo, mid_hi)
    )
    masks = {}
    if direction == "up":
        masks = {
            "mid": bands == "mid_band",
            "opp": bands == "lower_band",
            "deeper": bands == "below_low",
            "ovr": bands == "above_high",
        }
    else:
        masks = {
            "mid": bands == "mid_band",
            "opp": bands == "upper_band",
            "deeper": bands == "above_high",
            "ovr": bands == "below_low",
        }

    out.update(
        {
            "re_mid": bool(masks["mid"].any()),
            "hit_opposite": bool(masks["opp"].any()),
            "deeper_pull": bool(masks["deeper"].any()),
            "overshoot": bool(masks["ovr"].any()),
            "t_mid": _first_idx(masks["mid"]),
            "t_opp": _first_idx(masks["opp"]),
            "t_deeper": _first_idx(masks["deeper"]),
            "t_ovr": _first_idx(masks["ovr"]),
        }
    )
    return out


def _process_week_for_experiments(
    week_data: pd.Series, df_b: pd.DataFrame, experiments: list[tuple[int, int, int]]
) -> list[Dict[str, Any]]:
    rows = []
    week_end = pd.Timestamp(week_data["week_end_time"])
    for lookback_w, detection_d, evaluation_d in experiments:
        prev_high = float(week_data.get(f"high_{lookback_w}w", np.nan))
        prev_low = float(week_data.get(f"low_{lookback_w}w", np.nan))

        break_slice = _slice_after_week(df_b, week_end, detection_d)
        broke_up, broke_dn = _detect_break(break_slice, prev_low, prev_high)
        if not broke_up and not broke_dn:
            continue

        eval_slice = _slice_after_week(df_b, week_end, evaluation_d)
        row = {
            "iso_year": int(week_data["iso_year"]),
            "iso_week": int(week_data["iso_week"]),
            "lookback_weeks": lookback_w,
            "detection_days": detection_d,
            "evaluation_days": evaluation_d,
            "broke_up": broke_up,
            "broke_dn": broke_dn,
        }
        if broke_up:
            row.update(
                {
                    f"{k}_after_up": v
                    for k, v in _analyze_post_break(
                        eval_slice, prev_low, prev_high, "up"
                    ).items()
                }
            )
        if broke_dn:
            row.update(
                {
                    f"{k}_after_dn": v
                    for k, v in _analyze_post_break(
                        eval_slice, prev_low, prev_high, "down"
                    ).items()
                }
            )
        rows.append(row)
    return rows


# --- Main Entry Point for the Module ---

SCENARIOS = [
    (1, 3),  # Break of 1-wk range -> eval next 3 days
    (2, 5),  # Break of 2-wk range -> eval next 5 days (1 week)
    (3, 5),  # Break of 3-wk range -> eval next 5 days
    (4, 5),  # Break of 4-wk range -> eval next 5 days
    (6, 10),  # Break of 6-wk range -> eval next 10 days (2 weeks)
]
# Programmatically create the full experiment list
DEFAULT_EXPERIMENTS = [
    (lookback, lookback * 5, evaluation) for lookback, evaluation in SCENARIOS
]


def run_weekly_break_analysis(
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str,
    # MODIFIED: experiments argument is removed
):
    """Main function to be called from the backtester. Uses DEFAULT_EXPERIMENTS."""
    # MODIFIED: The function now uses the module-level constant
    experiments = DEFAULT_EXPERIMENTS

    df_b = _resample_to_business_days(market_data)
    df_d = market_data.resample("D").last().dropna()
    lookback_periods = {exp[0] for exp in experiments}
    wk = _get_lookback_hilo(df_d, lookback_periods)

    all_rows = []
    for _, week_data in wk.iterrows():
        all_rows.extend(_process_week_for_experiments(week_data, df_b, experiments))

    if not all_rows:
        print("Weekly Break Analysis: No breaks were detected for any experiments.")
        return

    out_df = pd.DataFrame(all_rows)

    # --- Aggregate and Create Markdown Report ---
    agg_rows = []
    group_cols = ["lookback_weeks", "detection_days", "evaluation_days"]

    def _p(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.mean()) if len(s) else np.nan

    for keys, g in out_df.groupby(group_cols):
        d = dict(zip(group_cols, keys))
        d["n_breaks_up"] = g["broke_up"].sum()
        d["n_breaks_dn"] = g["broke_dn"].sum()
        for side in ("up", "dn"):
            for event in ["re_mid", "hit_opposite", "overshoot"]:
                d[f"p_{event}_{side}"] = _p(g.get(f"{event}_after_{side}"))
        agg_rows.append(d)

    agg_df = pd.DataFrame(agg_rows).sort_values(group_cols).reset_index(drop=True)

    # --- Export MD ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    md_path = Path(out_dir) / f"{symbol}_weekly_break_report_{period_tag}.md"
    lines = [f"# {symbol} Weekly Break Analysis ({period_tag})"]
    for c in [c for c in agg_df.columns if c.startswith("p_")]:
        agg_df[c] = (agg_df[c] * 100).round(1)
    lines.append(agg_df.to_markdown(index=False, floatfmt=".1f"))
    md_path.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"Weekly Break Analysis: Report saved to {md_path}")
