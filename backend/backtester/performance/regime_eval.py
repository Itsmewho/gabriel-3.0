# backtester/performance/regime_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional
from backtester.features.better_volume_indicator import add_better_volume_mql
from backtester.broker import Trade
import numpy as np
import pandas as pd


def trades_to_df(trades: Iterable[Trade]) -> pd.DataFrame:
    """Converts an iterable of Trade objects into a pandas DataFrame."""
    rows = []
    for t in trades:
        rows.append(
            dict(
                id=getattr(t, "id", None),
                strategy_id=getattr(t, "strategy_id", None),
                side=getattr(t, "side", None),
                lots=getattr(t, "lot_size", np.nan),
                entry_time=pd.to_datetime(getattr(t, "entry_time", None)),  # type: ignore
                exit_time=pd.to_datetime(getattr(t, "exit_time", None)),  # type: ignore
                entry_price=getattr(t, "entry_price", np.nan),
                exit_price=getattr(t, "exit_price", np.nan),
                pnl=getattr(t, "pnl", 0.0),
                commission=getattr(t, "commission_paid", 0.0),
                swap=getattr(t, "swap_paid", 0.0),
                exit_reason=getattr(t, "exit_reason", None),
                balance_at_open=getattr(t, "balance_at_open", np.nan),
            )
        )

    if not rows:
        # Return an empty DataFrame but with the expected schema for downstream functions
        cols = [
            "id",
            "strategy_id",
            "side",
            "lots",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "pnl",
            "commission",
            "swap",
            "exit_reason",
            "balance_at_open",
            "duration_minutes",
            "is_win",
            "gross_profit_component",
            "gross_loss_component",
            "ret",
        ]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)

    # Coerce all financial columns to numeric types, turning any errors into Not-a-Number (NaN)
    numeric_cols = [
        "lots",
        "entry_price",
        "exit_price",
        "pnl",
        "commission",
        "swap",
        "balance_at_open",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure timestamps are timezone-naive to prevent calculation errors
    if "entry_time" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["entry_time"]
    ):
        df["entry_time"] = df["entry_time"].dt.tz_localize(None)
    if "exit_time" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["exit_time"]
    ):
        df["exit_time"] = df["exit_time"].dt.tz_localize(None)

    # CRITICAL: Drop open trades (where exit_time is NaT) before calculating metrics
    df.dropna(subset=["exit_time", "entry_time"], inplace=True)

    # Add calculated columns. This will work even if the DataFrame is empty.
    df["duration_minutes"] = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60.0
    df["is_win"] = df["pnl"] > 0
    df["gross_profit_component"] = df["pnl"].where(df["pnl"] > 0, 0.0)
    df["gross_loss_component"] = -df["pnl"].where(df["pnl"] < 0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ret"] = df["pnl"] / df["balance_at_open"].replace(0.0, np.nan)

    # Sort trades chronologically by exit time
    if not df.empty:
        df = df.sort_values(
            ["exit_time", "entry_time"], na_position="last"
        ).reset_index(drop=True)

    return df


# -----------------------------
# Regime indicator prep (EURUSD)
# -----------------------------
def _prep_market(market_data: pd.DataFrame) -> pd.DataFrame:
    df = market_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:  # type: ignore
        df.index = df.index.tz_localize(None)  # type: ignore
    df = df.sort_index()
    return df


def prepare_regime_indicators(
    market_data: pd.DataFrame,
    trend_fast: int = 50,
    trend_slow: int = 200,
    trend_eps: float = 0.0,
    vol_window: int = 60,
    bv_lookback: int = 30,
) -> pd.DataFrame:
    df = _prep_market(market_data)
    close = df["close"].astype(float)

    sma_fast = close.rolling(trend_fast, min_periods=trend_fast // 2).mean()
    sma_slow = close.rolling(trend_slow, min_periods=trend_slow // 2).mean()
    trend_score = sma_fast - sma_slow

    tr = pd.Series(index=df.index, dtype="object")
    tr[trend_score > trend_eps] = "uptrend"
    tr[trend_score < -trend_eps] = "downtrend"
    tr[(trend_score >= -trend_eps) & (trend_score <= trend_eps)] = "flat"

    logret = np.log(close).diff()
    vol_sigma = logret.rolling(vol_window, min_periods=vol_window // 2).std()
    q1, q2 = vol_sigma.quantile([1 / 3, 2 / 3])
    vb = pd.Series(index=df.index, dtype="object")
    vb[vol_sigma <= q1] = "low_vol"
    vb[(vol_sigma > q1) & (vol_sigma <= q2)] = "mid_vol"
    vb[vol_sigma > q2] = "high_vol"

    # BetterVolume color
    try:
        bv_df = add_better_volume_mql(df, lookback=bv_lookback)
        bv_color = bv_df["bv_color"].reindex(df.index)
    except Exception:
        bv_color = pd.Series(index=df.index, dtype="object")

    out = pd.DataFrame(
        {
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "trend_score": trend_score,
            "trend_regime": tr,
            "vol_sigma": vol_sigma,
            "vol_bucket": vb,
            "bv_color": bv_color,  # NEW
        },
        index=df.index,
    )
    out["regime_label"] = (
        out["trend_regime"].astype(str) + "+" + out["vol_bucket"].astype(str)
    )
    out["regime_label_bv"] = (
        out["trend_regime"].astype(str) + "+" + out["bv_color"].astype(str)
    )  # NEW
    return out


# -----------------------------
# Tag trades with regimes
# -----------------------------
def _nearest_index(ix: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_localize(None)
    pos = ix.get_indexer([ts], method="nearest")[0]
    return ix[pos]  # type: ignore


def tag_trades_with_regimes(
    trades_df: pd.DataFrame,
    market_data: pd.DataFrame,
    trend_fast: int = 50,
    trend_slow: int = 200,
    trend_eps: float = 0.0,
    vol_window: int = 60,
) -> pd.DataFrame:
    """
    Returns a copy of trades_df with columns:
      trend_regime, vol_bucket, regime_label
    Tagging uses the market bar nearest to each trade's entry_time.
    """
    if trades_df is None or trades_df.empty:
        return trades_df.copy()

    md = _prep_market(market_data)
    reg = prepare_regime_indicators(
        md,
        trend_fast=trend_fast,
        trend_slow=trend_slow,
        trend_eps=trend_eps,
        vol_window=vol_window,
    )

    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    if df["entry_time"].dt.tz is not None:  # type: ignore
        df["entry_time"] = df["entry_time"].dt.tz_localize(None)  # type: ignore

    trend_vals, vol_vals, lab_vals = [], [], []
    bv_vals, lab_bv_vals = [], []  # NEW

    for _, r in df.iterrows():
        et = r.get("entry_time")
        if pd.isna(et):
            trend_vals.append(None)
            vol_vals.append(None)
            lab_vals.append(None)
            bv_vals.append(None)
            lab_bv_vals.append(None)
            continue
        key = _nearest_index(reg.index, et)

        trv = reg.at[key, "trend_regime"]
        vvb = reg.at[key, "vol_bucket"]
        bvc = reg.at[key, "bv_color"] if "bv_color" in reg.columns else None

        trend_vals.append(trv if isinstance(trv, str) else None)
        vol_vals.append(vvb if isinstance(vvb, str) else None)
        lab_vals.append(
            f"{trv}+{vvb}" if isinstance(trv, str) and isinstance(vvb, str) else None
        )

        bv_vals.append(bvc if isinstance(bvc, str) else None)
        lab_bv_vals.append(
            f"{trv}+{bvc}" if isinstance(trv, str) and isinstance(bvc, str) else None
        )

    df["trend_regime"] = trend_vals
    df["vol_regime"] = vol_vals
    df["regime_label"] = lab_vals
    df["bv_color"] = bv_vals  # NEW
    df["regime_label_bv"] = lab_bv_vals  # NEW
    return df


# -----------------------------
# Summaries & correlation
# -----------------------------
def summarize_by_regime(df_trades_tagged: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if df_trades_tagged.empty:
        empty = pd.DataFrame()
        return dict(
            trend_summary=empty,
            vol_summary=empty,
            combo_summary=empty,
            heatmap_df=empty,
            bv_summary=empty,
            trend_bv_summary=empty,
            bv_heatmap=empty,
        )

    def _agg(g):
        n = len(g)
        wins = (g["pnl"] > 0).sum()
        return pd.Series(
            dict(
                trades=n,
                pnl_total=g["pnl"].sum(),
                pnl_avg=g["pnl"].mean(),
                pnl_med=g["pnl"].median(),
                winrate=(wins / n) if n else 0.0,
            )
        )

    trend_summary = (
        df_trades_tagged.groupby("trend_regime", dropna=True).apply(_agg).reset_index()
    )
    vol_summary = (
        df_trades_tagged.groupby("vol_regime", dropna=True).apply(_agg).reset_index()
    )
    combo_summary = (
        df_trades_tagged.groupby(["trend_regime", "vol_regime"], dropna=True)
        .apply(_agg)
        .reset_index()
    )

    # NEW: BetterVolume summaries
    bv_summary = (
        df_trades_tagged.groupby("bv_color", dropna=True).apply(_agg).reset_index()
    )
    trend_bv_summary = (
        df_trades_tagged.groupby(["trend_regime", "bv_color"], dropna=True)
        .apply(_agg)
        .reset_index()
    )

    heatmap_df = df_trades_tagged.pivot_table(
        index="trend_regime",
        columns="vol_regime",
        values="pnl",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(
        index=["uptrend", "flat", "downtrend"],
        columns=["low_vol", "mid_vol", "high_vol"],
    )

    # NEW: Trend × BetterVolume heatmap
    bv_heatmap = df_trades_tagged.pivot_table(
        index="trend_regime",
        columns="bv_color",
        values="pnl",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=["uptrend", "flat", "downtrend"])

    return dict(
        trend_summary=trend_summary,
        vol_summary=vol_summary,
        combo_summary=combo_summary,
        heatmap_df=heatmap_df,
        bv_summary=bv_summary,
        trend_bv_summary=trend_bv_summary,
        bv_heatmap=bv_heatmap,
    )


def regime_correlation(
    df_trades_tagged: pd.DataFrame, market_data: pd.DataFrame
) -> Tuple[float, float]:
    """
    Correlate DAILY close-to-close returns with DAILY strategy PnL.
    Returns (pearson_corr, spearman_corr). np.nan if insufficient data.
    """
    if df_trades_tagged.empty:
        return (np.nan, np.nan)

    md = _prep_market(market_data)
    # Daily market returns
    daily_px = md["close"].resample("1D").last().dropna()
    daily_ret = daily_px.pct_change().dropna()

    # Daily strategy pnl
    dtr = df_trades_tagged.copy()
    dtr = dtr.dropna(subset=["exit_time"])
    dtr["exit_time"] = pd.to_datetime(dtr["exit_time"])
    if dtr["exit_time"].dt.tz is not None:  # type: ignore
        dtr["exit_time"] = dtr["exit_time"].dt.tz_localize(None)  # type: ignore
    daily_pnl = dtr.set_index("exit_time")["pnl"].resample("1D").sum().dropna()

    aligned = pd.DataFrame({"ret": daily_ret, "pnl": daily_pnl}).dropna()
    if len(aligned) < 3:
        return (np.nan, np.nan)

    pear = float(aligned["ret"].corr(aligned["pnl"], method="pearson"))
    spear = float(aligned["ret"].corr(aligned["pnl"], method="spearman"))
    return (pear, spear)


# -----------------------------
# Markdown export (MD only, no CSV/PNG)
# -----------------------------
def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    return df.to_markdown(index=False)


def export_regime_report(
    df_trades_tagged: pd.DataFrame,
    summaries: Dict[str, pd.DataFrame],
    corr_tuple: Tuple[float, float],
    out_dir: str,
    symbol: str,
    period_tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a markdown report only. No CSV. No PNG."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{period_tag}" if period_tag else ""

    md_lines: list[str] = []
    md_lines.append(f"# {symbol} Regime Report")
    if period_tag:
        md_lines.append(f"**Period:** {period_tag}\n")

    pear, spear = corr_tuple
    md_lines.append("## Correlation (daily returns vs daily strategy PnL)")
    md_lines.append(f"- Pearson: {pear if not np.isnan(pear) else '-'}")
    md_lines.append(f"- Spearman: {spear if not np.isnan(spear) else '-'}\n")

    md_lines.append("## Summary by Trend")
    md_lines.append(_md_table(summaries.get("trend_summary")))  # type: ignore
    md_lines.append("\n## Summary by Volatility")
    md_lines.append(_md_table(summaries.get("vol_summary")))  # type: ignore
    md_lines.append("\n## Summary by Trend × Vol")
    md_lines.append(_md_table(summaries.get("combo_summary")))  # type: ignore

    md_lines.append("\n## Summary by BetterVolume")
    md_lines.append(_md_table(summaries.get("bv_summary")))

    md_lines.append("\n## Summary by Trend × BetterVolume")
    md_lines.append(_md_table(summaries.get("trend_bv_summary")))

    bv_heatmap = summaries.get("bv_heatmap")
    if bv_heatmap is not None and not bv_heatmap.empty:
        md_lines.append("\n## PnL by Trend × BetterVolume (table)")
        md_lines.append(bv_heatmap.to_markdown())

    # Include heatmap table directly in markdown (no image file)
    heatmap_df = summaries.get("heatmap_df")
    if heatmap_df is not None and not heatmap_df.empty:
        md_lines.append("\n## PnL by Regime (table)")
        md_lines.append(heatmap_df.to_markdown())

    md_path = f"{out_dir}/{symbol}{tag}_regime.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    return {"markdown": md_path}


# -----------------------------
# Orchestrator
# -----------------------------
def regime_report(
    trades_or_df: Iterable[Dict[str, Any]] | pd.DataFrame,
    market_data: pd.DataFrame,
    out_dir: str = "results/metrics/regime",
    symbol: str = "EURUSD",
    period_tag: Optional[str] = None,
    trend_fast: int = 50,
    trend_slow: int = 200,
    trend_eps: float = 0.0,
    vol_window: int = 60,
) -> Dict[str, Any]:
    """
    End-to-end:
      - ensure DataFrame of trades with columns: entry_time, exit_time, pnl, side
      - tag with regimes
      - summarize and export markdown (no CSV/PNG)
    """
    if isinstance(trades_or_df, pd.DataFrame):
        df_trades = trades_or_df.copy()
    elif trades_to_df is not None:
        df_trades = trades_to_df(trades_or_df)  # type: ignore
    else:
        # minimal conversion if trades_to_df not available
        rows = []
        for t in trades_or_df:  # type: ignore
            rows.append(
                dict(
                    entry_time=pd.to_datetime(getattr(t, "entry_time", None)),  # type: ignore
                    exit_time=pd.to_datetime(getattr(t, "exit_time", None)),  # type: ignore
                    pnl=float(getattr(t, "pnl", 0.0)),
                    side=getattr(t, "side", None),
                    strategy_id=getattr(t, "strategy_id", None),
                )
            )
        df_trades = pd.DataFrame(rows)

    # Robust empty path
    if df_trades.empty:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tag = f"_{period_tag}" if period_tag else ""
        md = f"{out_dir}/{symbol}{tag}_regime.md"
        with open(md, "w", encoding="utf-8") as f:
            f.write(f"# {symbol} Regime Report\n\nNo trades.\n")
        return {"markdown": md}

    tagged = tag_trades_with_regimes(
        df_trades,
        market_data,
        trend_fast=trend_fast,
        trend_slow=trend_slow,
        trend_eps=trend_eps,
        vol_window=vol_window,
    )
    sums = summarize_by_regime(tagged)
    corr = regime_correlation(tagged, market_data)
    return export_regime_report(tagged, sums, corr, out_dir, symbol, period_tag)
