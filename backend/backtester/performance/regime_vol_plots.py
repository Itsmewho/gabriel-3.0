# backtester/plots/regime_vol_plots.py (FINAL VERSION w/ 3-MONTH CHUNKING)
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Callable

from pathlib import Path
import numpy as np
import pandas as pd
import mplfinance as mpf
from pandas.tseries.offsets import DateOffset

from backtester.features.base_features import apply_basic_features
from backtester.features.regime_classifier import attach_regime, RegimeThresholds
from concurrent.futures import ProcessPoolExecutor

# ============================
# Core helpers & Other Functions
# ============================


def _tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=False, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    need = ["open", "high", "low", "close"]
    miss = [n for n in need if n not in cols]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    out = df.rename(
        columns={
            cols["open"]: "Open",
            cols["high"]: "High",
            cols["low"]: "Low",
            cols["close"]: "Close",
        }
    )
    return out


def _hline_addplots(
    y: float, n: int, *, panel: int, color: str, linestyle: str, lw: float
):
    return [
        mpf.make_addplot(
            [y] * n, panel=panel, color=color, linestyle=linestyle, linewidths=lw
        )
    ]


# (Other helper functions like classify_volatility, _limit_bands, etc. remain here unchanged)
@dataclass
class VolatilityBands:
    low_q: float = 0.33
    high_q: float = 0.66


def classify_volatility(
    atr: pd.Series, *, low_q: float = 0.33, high_q: float = 0.66
) -> pd.Series:
    a = atr.copy().astype(float)
    ql = a.quantile(low_q)
    qh = a.quantile(high_q)
    out = pd.Series("MID", index=a.index)
    out[a <= ql] = "LOW"
    out[a >= qh] = "HIGH"
    return out


REGIME_COLORS = {
    "S_UP": "#00ff7f",
    "UP": "#7fff7f",
    "TRANS": "#da11bc",
    "CONS": "#cccccc",
    "DOWN": "#ff7f7f",
    "S_DOWN": "#ff4500",
}


def _limit_bands(df: pd.DataFrame, pad_frac: float = 0.03) -> Tuple[float, float]:
    price_range = df["High"].max() - df["Low"].min()
    pad = price_range * pad_frac
    return df["High"].max() + pad, df["Low"].min() - pad


def _bg_addplots(
    df: pd.DataFrame, labels: pd.Series, color_map: Dict[str, str], alpha: float = 0.10
) -> List[Any]:
    up, dn = _limit_bands(df)
    y_upper = np.full(len(df), up)
    y_lower = np.full(len(df), dn)
    adds = []
    for label, color in color_map.items():
        where = (labels == label).to_numpy()
        if not where.any():
            continue
        adds.append(
            mpf.make_addplot(
                y_upper,
                fill_between=dict(y1=y_lower, where=where, alpha=alpha, color=color),
            )
        )
    return adds


def _infer_minutes(ix: pd.DatetimeIndex) -> float:
    if len(ix) < 2:
        return 1.0
    return max(1.0, (ix[1] - ix[0]).total_seconds() / 60.0)


# ============================
# Feature Computation & Core Plotter
# ============================


def compute_regime_and_vol(
    df: pd.DataFrame,
    *,
    ema_fast_min: int,
    ema_mid_min: int,
    ema_slow_min: int,
    atr_period: int,
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    base = _ensure_ohlcv(_tz_naive_index(df))
    bar_min = (
        max(1.0, (base.index[1] - base.index[0]).total_seconds() / 60.0)
        if len(base.index) > 1
        else 1.0
    )

    def to_span(mins: int) -> int:
        return max(1, int(round(mins / bar_min)))

    feat_cfg = {
        "ema": [to_span(ema_fast_min), to_span(ema_mid_min), to_span(ema_slow_min)],
        "atr": [to_span(atr_period)],
    }
    enriched = apply_basic_features(
        base.rename(columns={c: c.lower() for c in base.columns}).copy(), feat_cfg
    )
    atr_col = f"atr_{to_span(atr_period)}"
    regime_df = attach_regime(
        enriched,
        ema_fast=to_span(ema_fast_min),
        ema_mid=to_span(ema_mid_min),
        ema_slow=to_span(ema_slow_min),
        atr_col=atr_col,
        smooth_n=3,
        thresholds=thresholds,
    )
    regime_df.index = base.index
    return base, regime_df, pd.Series()


def plot_combined_regime(
    df_in: pd.DataFrame,
    combined_df: pd.DataFrame,
    *,
    title: str,
    filename: str,
    fig_dpi: int = 300,
) -> str:
    """Plots the combined score, its components, and the short-term gap strength."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    df = _ensure_ohlcv(_tz_naive_index(df_in))
    combined_df.index = df.index
    n = len(df)
    combined_plot = mpf.make_addplot(
        combined_df["combined_score"],
        panel=1,
        color="cyan",
        ylabel="Combined\nScore",
        ylim=(0, 10),
    )
    hlines_p1 = _hline_addplots(
        3.0, n, panel=1, color="r", linestyle="--", lw=0.7
    ) + _hline_addplots(7.0, n, panel=1, color="g", linestyle="--", lw=0.7)
    short_plot = mpf.make_addplot(
        combined_df["short_score"],
        panel=2,
        color="lime",
        ylabel="Individual\nScores",
        ylim=(0, 10),
    )
    mid_plot = mpf.make_addplot(combined_df["mid_score"], panel=2, color="darkorange")
    long_plot = mpf.make_addplot(combined_df["long_score"], panel=2, color="magenta")
    hlines_p2 = _hline_addplots(5.0, n, panel=2, color="white", linestyle=":", lw=0.6)
    gap_raw = combined_df["short_gap_norm"]
    min_g, max_g = gap_raw.min(), gap_raw.max()
    gap_scaled = (
        pd.Series(50.0, index=gap_raw.index)
        if (max_g - min_g) == 0
        else 50 + 50 * (2 * (gap_raw - min_g) / (max_g - min_g) - 1)
    )
    gap_plot = mpf.make_addplot(
        gap_scaled,
        panel=3,
        type="bar",
        color="gray",
        alpha=0.7,
        ylabel="Short Gap\nStrength %",
        ylim=(0, 100),
    )
    gap_hline = _hline_addplots(50, n, panel=3, color="white", linestyle=":", lw=0.7)
    all_addplots = [
        combined_plot,
        *hlines_p1,
        short_plot,
        mid_plot,
        long_plot,
        *hlines_p2,
        gap_plot,
        *gap_hline,
    ]
    fig, axes = mpf.plot(
        df,
        type="candle",
        style="binancedark",
        title=title,
        ylabel="Price",
        addplot=all_addplots,
        panel_ratios=(8, 2, 2, 2),
        tight_layout=True,
        warn_too_much_data=len(df) + 1,
        returnfig=True,
        datetime_format="%d-%b",
    )
    axes[4].text(
        0.01,
        0.95,
        "Scores: Short (lime), Mid (orange), Long (magenta)",
        transform=axes[4].transAxes,
        fontsize=8,
        va="top",
        color="white",
    )
    axes[6].text(
        0.01,
        0.95,
        "Short-Term Gap Strength (0-100)",
        transform=axes[6].transAxes,
        fontsize=8,
        va="top",
        color="white",
    )
    fig.savefig(filename, dpi=fig_dpi)
    return filename


# ============================
# Parallel Chunking Helper with Close-ups
# ============================
def _plot_in_chunks_with_closeups(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    filename_template: str,
    plot_func: Callable,
    **kwargs,
):
    if not isinstance(base_df.index, pd.DatetimeIndex) or base_df.empty:
        return
    start_date, end_date = base_df.index.min(), base_df.index.max()
    tasks = []

    # Generate list of chunk start/end dates
    chunk_starts = pd.date_range(start=start_date, end=end_date, freq="3MS")
    if chunk_starts.empty or chunk_starts[0] > start_date:
        chunk_starts = chunk_starts.insert(0, start_date)

    for i, current_start in enumerate(chunk_starts):
        current_end = current_start + DateOffset(months=3) - pd.Timedelta(nanoseconds=1)
        chunk_base_df = base_df.loc[current_start:current_end]
        chunk_feature_df = feature_df.loc[current_start:current_end]
        if chunk_base_df.empty:
            continue

        # --- Task 1: The 3-month plot ---
        chunk_num = i + 1
        main_filename = filename_template.format(chunk_num=chunk_num)
        main_kwargs = kwargs.copy()
        main_kwargs["title"] = f"{kwargs.get('title', 'Plot')} (Part {chunk_num})"
        main_kwargs["df_in"] = chunk_base_df
        main_kwargs["combined_df"] = chunk_feature_df
        main_kwargs["filename"] = main_filename
        tasks.append(main_kwargs)

        # --- Task 2: The 2-week close-up plot ---
        mid_point = (
            chunk_base_df.index[0]
            + (chunk_base_df.index[-1] - chunk_base_df.index[0]) / 2
        )
        closeup_start = mid_point - pd.Timedelta(weeks=1)
        closeup_end = mid_point + pd.Timedelta(weeks=1)
        closeup_base_df = chunk_base_df.loc[closeup_start:closeup_end]
        closeup_feature_df = chunk_feature_df.loc[closeup_start:closeup_end]

        if not closeup_base_df.empty:
            p = Path(main_filename)
            closeup_filename = str(p.with_name(f"{p.stem}_closeup{p.suffix}"))
            closeup_kwargs = kwargs.copy()
            closeup_kwargs["title"] = (
                f"{kwargs.get('title', 'Plot')} (Part {chunk_num} - 2 Week Closeup)"
            )
            closeup_kwargs["df_in"] = closeup_base_df
            closeup_kwargs["combined_df"] = closeup_feature_df
            closeup_kwargs["filename"] = closeup_filename
            tasks.append(closeup_kwargs)

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(plot_func, **task) for task in tasks]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"A plotting chunk failed: {e}")


# ============================
# Main Preset Function
# ============================
LONG_EMAS_MIN = (16200, 22400, 44800)
LONG_ATR_MIN = 50
MID_EMAS_MIN = (3440, 7200, 14400)
MID_ATR_MIN = 24
SHORT_EMAS_MIN = (720, 1000, 1300)
SHORT_ATR_MIN = 18


def combined_regime_plot(df: pd.DataFrame, *, filename: str) -> str:
    """Calculates all three regime scores and plots their combined average and the short-term gap."""
    base, regdf_s, _ = compute_regime_and_vol(
        df,
        ema_fast_min=SHORT_EMAS_MIN[0],
        ema_mid_min=SHORT_EMAS_MIN[1],
        ema_slow_min=SHORT_EMAS_MIN[2],
        atr_period=SHORT_ATR_MIN,
    )
    _, regdf_m, _ = compute_regime_and_vol(
        df,
        ema_fast_min=MID_EMAS_MIN[0],
        ema_mid_min=MID_EMAS_MIN[1],
        ema_slow_min=MID_EMAS_MIN[2],
        atr_period=MID_ATR_MIN,
    )
    _, regdf_l, _ = compute_regime_and_vol(
        df,
        ema_fast_min=LONG_EMAS_MIN[0],
        ema_mid_min=LONG_EMAS_MIN[1],
        ema_slow_min=LONG_EMAS_MIN[2],
        atr_period=LONG_ATR_MIN,
    )

    combined_df = pd.DataFrame(
        {
            "short_score": regdf_s["regime_score"],
            "mid_score": regdf_m["regime_score"],
            "long_score": regdf_l["regime_score"],
        },
        index=base.index,
    )
    combined_df["combined_score"] = combined_df.mean(axis=1)
    combined_df["short_gap_norm"] = regdf_s["gap_fs_norm"]

    p = Path(filename)
    template = str(p.with_name(f"{p.stem}_part{{chunk_num}}{p.suffix}"))
    _plot_in_chunks_with_closeups(
        base_df=base,
        feature_df=combined_df,
        filename_template=template,
        plot_func=plot_combined_regime,
        title="Combined Multi-Timeframe Regime",
    )
    return filename
