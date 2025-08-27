# backtester/performance/regime_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from .evaluation import trades_to_df  # optional convenience
except Exception:
    trades_to_df = None  # not required; you can pass a DataFrame directly


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
    trend_eps: float = 0.0,  # flat threshold (in price units)
    vol_window: int = 60,  # in bars
) -> pd.DataFrame:
    """
    Returns a DataFrame aligned to market_data.index with:
      - sma_fast, sma_slow
      - trend_score = sma_fast - sma_slow
      - trend_regime: 'uptrend' | 'flat' | 'downtrend'
      - vol_sigma: rolling std of log returns
      - vol_bucket: 'low_vol' | 'mid_vol' | 'high_vol' (terciles)
    """
    df = _prep_market(market_data)
    close = df["close"].astype(float)

    sma_fast = close.rolling(trend_fast, min_periods=trend_fast // 2).mean()
    sma_slow = close.rolling(trend_slow, min_periods=trend_slow // 2).mean()
    trend_score = sma_fast - sma_slow

    # Trend label
    tr = pd.Series(index=df.index, dtype="object")
    tr[trend_score > trend_eps] = "uptrend"
    tr[trend_score < -trend_eps] = "downtrend"
    tr[(trend_score >= -trend_eps) & (trend_score <= trend_eps)] = "flat"

    # Volatility (std of log-returns)
    logret = np.log(close).diff()  # type: ignore
    vol_sigma = logret.rolling(vol_window, min_periods=vol_window // 2).std()

    # Vol buckets by global terciles over the period
    q1, q2 = vol_sigma.quantile([1 / 3, 2 / 3])
    vb = pd.Series(index=df.index, dtype="object")
    vb[vol_sigma <= q1] = "low_vol"
    vb[(vol_sigma > q1) & (vol_sigma <= q2)] = "mid_vol"
    vb[vol_sigma > q2] = "high_vol"

    out = pd.DataFrame(
        {
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "trend_score": trend_score,
            "trend_regime": tr,
            "vol_sigma": vol_sigma,
            "vol_bucket": vb,
        },
        index=df.index,
    )
    out["regime_label"] = (
        out["trend_regime"].astype(str) + "+" + out["vol_bucket"].astype(str)
    )
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

    trend_vals: list[Optional[str]] = []
    vol_vals: list[Optional[str]] = []
    lab_vals: list[Optional[str]] = []

    for _, r in df.iterrows():
        et = r.get("entry_time")
        if pd.isna(et):
            trend_vals.append(None)
            vol_vals.append(None)
            lab_vals.append(None)
            continue
        key = _nearest_index(reg.index, et)  # type: ignore
        tr = reg.at[key, "trend_regime"]
        vb = reg.at[key, "vol_bucket"]
        trend_vals.append(tr if isinstance(tr, str) else None)
        vol_vals.append(vb if isinstance(vb, str) else None)
        lab_vals.append(
            f"{tr}+{vb}" if isinstance(tr, str) and isinstance(vb, str) else None
        )

    df["trend_regime"] = trend_vals
    df["vol_regime"] = vol_vals
    df["regime_label"] = lab_vals
    return df


# -----------------------------
# Summaries & correlation
# -----------------------------
def summarize_by_regime(df_trades_tagged: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      - trend_summary
      - vol_summary
      - combo_summary  (trend x vol)
      - heatmap_df     (pivot: sum pnl)
    """
    if df_trades_tagged.empty:
        empty = pd.DataFrame()
        return dict(
            trend_summary=empty,
            vol_summary=empty,
            combo_summary=empty,
            heatmap_df=empty,
        )

    def _agg(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        wins = (g["pnl"] > 0).sum()
        wl = wins / n if n else 0.0
        return pd.Series(
            dict(
                trades=n,
                pnl_total=g["pnl"].sum(),
                pnl_avg=g["pnl"].mean(),
                pnl_med=g["pnl"].median(),
                winrate=wl,
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

    # Heatmap data (sum pnl)
    heatmap_df = df_trades_tagged.pivot_table(
        index="trend_regime",
        columns="vol_regime",
        values="pnl",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Reindex for consistent ordering
    idx_order = ["uptrend", "flat", "downtrend"]
    col_order = ["low_vol", "mid_vol", "high_vol"]
    heatmap_df = heatmap_df.reindex(index=idx_order, columns=col_order)

    return dict(
        trend_summary=trend_summary,
        vol_summary=vol_summary,
        combo_summary=combo_summary,
        heatmap_df=heatmap_df,
    )


def regime_correlation(
    df_trades_tagged: pd.DataFrame, market_data: pd.DataFrame
) -> Tuple[float, float]:
    """
    Correlate DAILY EURUSD returns with DAILY strategy PnL.
    Returns (pearson_corr, spearman_corr). np.nan if insufficient data.
    """
    if df_trades_tagged.empty:
        return (np.nan, np.nan)

    md = _prep_market(market_data)
    # Daily EURUSD returns
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
# Plots & report
# -----------------------------
def _save_heatmap(heatmap_df: pd.DataFrame, out_path: str) -> Optional[str]:
    if heatmap_df is None or heatmap_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(heatmap_df.values, aspect="auto")
    ax.set_xticks(range(heatmap_df.shape[1]))
    ax.set_xticklabels(list(heatmap_df.columns))
    ax.set_yticks(range(heatmap_df.shape[0]))
    ax.set_yticklabels(list(heatmap_df.index))
    ax.set_xlabel("Volatility regime")
    ax.set_ylabel("Trend regime")
    ax.set_title("PnL by regime")
    # add values
    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            val = heatmap_df.values[i, j]
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


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
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{period_tag}" if period_tag else ""

    # Save CSVs
    paths: Dict[str, Any] = {}
    fn_base = f"{symbol}{tag}"
    df_trades_tagged.to_csv(f"{out_dir}/{fn_base}_trades_tagged.csv", index=False)
    for key, df in summaries.items():
        df.to_csv(f"{out_dir}/{fn_base}_{key}.csv", index=False)

    # Heatmap
    heatmap_png = _save_heatmap(
        summaries["heatmap_df"], f"{out_dir}/{fn_base}_heatmap.png"
    )
    if heatmap_png:
        paths["heatmap"] = heatmap_png

    # Markdown
    md_lines: list[str] = []
    md_lines.append(f"# {symbol} Regime Report")
    if period_tag:
        md_lines.append(f"**Period:** {period_tag}\n")

    pear, spear = corr_tuple
    md_lines.append("## Correlation (daily EURUSD returns vs daily strategy PnL)")
    md_lines.append(f"- Pearson: {pear if not np.isnan(pear) else '-'}")
    md_lines.append(f"- Spearman: {spear if not np.isnan(spear) else '-'}\n")

    md_lines.append("## Summary by Trend")
    md_lines.append(_md_table(summaries["trend_summary"]))
    md_lines.append("\n## Summary by Volatility")
    md_lines.append(_md_table(summaries["vol_summary"]))
    md_lines.append("\n## Summary by Trend × Vol")
    md_lines.append(_md_table(summaries["combo_summary"]))

    if heatmap_png:
        md_lines.append(f"\n![PnL by regime]({Path(heatmap_png).name})")

    md_path = f"{out_dir}/{fn_base}_regime.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    paths["markdown"] = md_path
    return paths


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
      - summarize and export markdown + csv + heatmap
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
        md = f"{out_dir}/{symbol}_{'_'+period_tag if period_tag else ''}_regime.md"
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
