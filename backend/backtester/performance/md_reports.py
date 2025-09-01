# markdown_report.py

from __future__ import annotations
import math
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple
import pandas as pd
import numpy as np

from backtester.broker import Trade


# -----------------------------
# Core DataFrame Builders (self-contained in this file)
# -----------------------------


DEFAULT_METRICS = {
    "initial_deposit": 0.0,
    "total_net_profit": 0.0,
    "gross_profit": 0.0,
    "gross_loss": 0.0,
    "profit_factor": np.inf,
    "expected_payoff": 0.0,
    "recovery_factor": np.inf,
    "sharpe_ratio": np.nan,
    "total_trades": 0,
    "profit_trades": 0,
    "loss_trades": 0,
    "winrate": 0.0,
    "balance_drawdown_abs_max": 0.0,
    "balance_drawdown_pct_max": 0.0,
    "equity_drawdown_abs_from_initial": 0.0,
    "equity_drawdown_rel_max": 0.0,
    "largest_profit_trade": 0.0,
    "average_profit_trade": 0.0,
    "largest_loss_trade": 0.0,
    "average_loss_trade": 0.0,
    "max_consecutive_wins_count": 0,
    "max_consecutive_wins_sum": 0.0,
    "max_consecutive_losses_count": 0,
    "max_consecutive_losses_sum": 0.0,
    "long_trades_count": 0,
    "long_trades_win_pct": 0.0,
    "short_trades_count": 0,
    "short_trades_win_pct": 0.0,
}


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
        return pd.DataFrame()  # Return empty if no trades

    df = pd.DataFrame(rows)
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

    if "entry_time" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["entry_time"]
    ):
        df["entry_time"] = df["entry_time"].dt.tz_localize(None)
    if "exit_time" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["exit_time"]
    ):
        df["exit_time"] = df["exit_time"].dt.tz_localize(None)

    df.dropna(subset=["exit_time", "entry_time"], inplace=True)
    if df.empty:
        return df

    df["duration_minutes"] = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60.0
    df["is_win"] = df["pnl"] > 0
    df["gross_profit_component"] = df["pnl"].where(df["pnl"] > 0, 0.0)
    df["gross_loss_component"] = -df["pnl"].where(df["pnl"] < 0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ret"] = df["pnl"] / df["balance_at_open"].replace(0.0, np.nan)

    df = df.sort_values(["exit_time", "entry_time"], na_position="last").reset_index(
        drop=True
    )
    return df


def equity_curve(df_trades: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    """Calculates the equity curve and drawdown at each trade close."""
    if df_trades.empty:
        return pd.DataFrame(
            {"time": [], "equity": [], "drawdown_abs": [], "drawdown_pct": []}
        )

    df_sorted = df_trades.dropna(subset=["exit_time"]).sort_values("exit_time")
    df_sorted["cumulative_pnl"] = df_sorted["pnl"].cumsum()
    ec = pd.DataFrame(
        {
            "time": df_sorted["exit_time"],
            "equity": initial_balance + df_sorted["cumulative_pnl"],
        }
    )

    if ec.empty:
        return ec

    peak = ec["equity"].cummax()
    dd_abs = peak - ec["equity"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = dd_abs / peak.replace(0.0, np.nan)

    ec["drawdown_abs"] = dd_abs
    ec["drawdown_pct"] = dd_pct.fillna(0.0)
    return ec


def mt5_like_metrics(df_trades: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
    if df_trades is None or df_trades.empty:
        d = DEFAULT_METRICS.copy()
        d["initial_deposit"] = float(initial_balance)
        return d

    n = len(df_trades)
    pnl = pd.to_numeric(
        df_trades.get("pnl", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)

    total_net = float(pnl.sum())
    gross_profit = float(pnl.clip(lower=0).sum())
    gross_loss = float((-pnl.clip(upper=0)).sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    expected_payoff = (total_net / n) if n else 0.0

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    winrate = (wins / n) if n else 0.0
    rets = df_trades["ret"].dropna()
    sharpe = np.nan
    if len(rets) > 1 and rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * math.sqrt(n)
    ec = equity_curve(df_trades, initial_balance)
    bal_dd_abs_max, bal_dd_pct_max, eq_abs_from_initial, eq_dd_rel_max = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if not ec.empty:
        bal_dd_abs_max = float(ec["drawdown_abs"].max())
        bal_dd_pct_max = float(ec["drawdown_pct"].max())
        min_equity = ec["equity"].min()
        if min_equity < initial_balance:
            eq_abs_from_initial = float(initial_balance - min_equity)
            if initial_balance > 0:
                eq_dd_rel_max = eq_abs_from_initial / initial_balance
    recovery_factor = (total_net / bal_dd_abs_max) if bal_dd_abs_max > 0 else np.inf
    df_wins = df_trades[df_trades["pnl"] > 0]
    df_losses = df_trades[df_trades["pnl"] < 0]
    largest_profit_trade = df_wins["pnl"].max() if not df_wins.empty else 0.0
    largest_loss_trade = df_losses["pnl"].min() if not df_losses.empty else 0.0
    average_profit_trade = df_wins["pnl"].mean() if not df_wins.empty else 0.0
    average_loss_trade = df_losses["pnl"].mean() if not df_losses.empty else 0.0
    is_win = df_trades["pnl"] > 0
    is_loss = df_trades["pnl"] < 0
    win_streaks = (is_win != is_win.shift()).cumsum()[is_win]
    loss_streaks = (is_loss != is_loss.shift()).cumsum()[is_loss]
    max_consecutive_wins_count = (
        win_streaks.value_counts().max() if not win_streaks.empty else 0
    )
    max_consecutive_losses_count = (
        loss_streaks.value_counts().max() if not loss_streaks.empty else 0
    )
    max_consecutive_wins_sum = (
        df_trades.groupby(win_streaks)["pnl"].sum().max()
        if not win_streaks.empty
        else 0.0
    )
    max_consecutive_losses_sum = (
        df_trades.groupby(loss_streaks)["pnl"].sum().min()
        if not loss_streaks.empty
        else 0.0
    )

    # Side-specific metrics
    df_trades["side"] = df_trades["side"].fillna("")
    long_trades = df_trades[df_trades["side"].str.lower() == "buy"]
    short_trades = df_trades[df_trades["side"].str.lower() == "sell"]
    long_trades_count = len(long_trades)
    short_trades_count = len(short_trades)
    long_wins = (long_trades["pnl"] > 0).sum()
    short_wins = (short_trades["pnl"] > 0).sum()
    long_trades_win_pct = (
        (long_wins / long_trades_count) if long_trades_count > 0 else 0.0
    )
    short_trades_win_pct = (
        (short_wins / short_trades_count) if short_trades_count > 0 else 0.0
    )

    return {
        "initial_deposit": float(initial_balance),
        "total_net_profit": total_net,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expected_payoff": expected_payoff,
        "recovery_factor": recovery_factor,
        "sharpe_ratio": sharpe,
        "total_trades": int(n),
        "profit_trades": int(wins),
        "loss_trades": int(losses),
        "winrate": winrate,
        "balance_drawdown_abs_max": bal_dd_abs_max,
        "balance_drawdown_pct_max": bal_dd_pct_max,
        "equity_drawdown_abs_from_initial": eq_abs_from_initial,
        "equity_drawdown_rel_max": eq_dd_rel_max,
        "largest_profit_trade": largest_profit_trade,
        "average_profit_trade": average_profit_trade,
        "largest_loss_trade": largest_loss_trade,
        "average_loss_trade": average_loss_trade,
        "max_consecutive_wins_count": max_consecutive_wins_count,
        "max_consecutive_wins_sum": max_consecutive_wins_sum,
        "max_consecutive_losses_count": max_consecutive_losses_count,
        "max_consecutive_losses_sum": max_consecutive_losses_sum,
        "long_trades_count": long_trades_count,
        "long_trades_win_pct": long_trades_win_pct,
        "short_trades_count": short_trades_count,
        "short_trades_win_pct": short_trades_win_pct,
    }


def top_avg_worst(
    df_trades: pd.DataFrame, k: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Finds the best, worst, and most average trades."""
    if df_trades.empty:
        empty = df_trades.copy()
        return empty, empty, empty
    best = df_trades.sort_values("pnl", ascending=False).head(k)
    worst = df_trades.sort_values("pnl", ascending=True).head(k)
    mean_pnl = df_trades["pnl"].mean()
    mid = (
        df_trades.assign(dist=(df_trades["pnl"] - mean_pnl).abs())
        .sort_values("dist")
        .head(k)
        .drop(columns=["dist"])
    )
    return best, mid, worst


# -----------------------------
# Text/Markdown rendering
# -----------------------------


def _df_to_markdown(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return ""
    return df.to_markdown(index=False)


def _create_summary_table(metrics_dict: Dict[str, Any]) -> str:
    def fmt(x, is_pct=False):
        if pd.isna(x) or x is None:
            return "-"
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        if isinstance(x, (float, np.floating)):
            return f"{x:.2%}" if is_pct else f"{x:,.2f}"
        return str(x)

    ov = metrics_dict
    left = [
        ("Initial Deposit", ov.get("initial_deposit")),
        ("Total Net Profit", ov.get("total_net_profit")),
        ("Profit Factor", ov.get("profit_factor")),
        ("Recovery Factor", ov.get("recovery_factor")),
        ("Balance Drawdown Absolute ($)", ov.get("balance_drawdown_abs_max")),
        ("Equity Drawdown Absolute ($)", ov.get("equity_drawdown_abs_from_initial")),
        ("Total Trades", ov.get("total_trades")),
        (
            "Profit Trades (% of total)",
            f"{fmt(ov.get('profit_trades'))} ({fmt(ov.get('winrate'), is_pct=True)})",
        ),
        ("Largest profit trade", ov.get("largest_profit_trade")),
        ("Average profit trade", ov.get("average_profit_trade")),
        ("Consecutive wins (# trades)", ov.get("max_consecutive_wins_count")),
        ("Consecutive profit ($)", ov.get("max_consecutive_wins_sum")),
    ]
    right = [
        ("Gross Profit", ov.get("gross_profit")),
        ("Gross Loss", f"{-ov.get('gross_loss', 0):,.2f}"),
        ("Expected Payoff", ov.get("expected_payoff")),
        ("Sharpe Ratio", ov.get("sharpe_ratio")),
        (
            "Balance Drawdown Maximal (%)",
            f"{fmt(ov.get('balance_drawdown_pct_max'), is_pct=True)}",
        ),
        (
            "Equity Drawdown Relative (%)",
            f"{fmt(ov.get('equity_drawdown_rel_max'), is_pct=True)}",
        ),
        (
            "Short Trades (won %)",
            f"{fmt(ov.get('short_trades_count'))} ({fmt(ov.get('short_trades_win_pct'), is_pct=True)})",
        ),
        (
            "Long Trades (won %)",
            f"{fmt(ov.get('long_trades_count'))} ({fmt(ov.get('long_trades_win_pct'), is_pct=True)})",
        ),
        ("Largest loss trade", ov.get("largest_loss_trade")),
        ("Average loss trade", ov.get("average_loss_trade")),
        ("Consecutive losses (# trades)", ov.get("max_consecutive_losses_count")),
        ("Consecutive loss ($)", ov.get("max_consecutive_losses_sum")),
    ]
    rows = [
        "| Metric                         | Value         | Metric                         | Value           |",
        "|:-------------------------------|--------------:|:-------------------------------|----------------:|",
    ]
    for i in range(max(len(left), len(right))):
        l_metric, l_val = left[i] if i < len(left) else ("", "")
        r_metric, r_val = right[i] if i < len(right) else ("", "")
        l_val_str = l_val if isinstance(l_val, str) else fmt(l_val)
        r_val_str = r_val if isinstance(r_val, str) else fmt(r_val)
        rows.append(
            f"| {l_metric:<30} | {l_val_str:>13} | {r_metric:<30} | {r_val_str:>15} |"
        )
    return "\n".join(rows)


def _render_text_report(
    symbol: str,
    period_tag: str | None,
    overall_df: pd.DataFrame,
    per_df: pd.DataFrame | None,
    monthly_df: pd.DataFrame | None,
    best_df: pd.DataFrame | None,
    mid_df: pd.DataFrame | None,
    worst_df: pd.DataFrame | None,
    out_path: str,
) -> str:
    ov = (
        overall_df.iloc[0].to_dict()
        if overall_df is not None and not overall_df.empty
        else {}
    )
    parts = [f"# {symbol} Backtest Report"]
    if period_tag:
        parts.append(f"**Period:** {period_tag}\n")
    parts.append("## Summary Metrics")
    parts.append(_create_summary_table(ov))
    if per_df is not None and not per_df.empty:
        parts.append("\n## Per-Strategy Metrics")
        for _, row in per_df.iterrows():
            sid = row.get("strategy_id", "Unknown")
            parts.append(f"\n### Strategy: {sid}")
            parts.append(_create_summary_table(row.to_dict()))

    def add_section(df, title):
        return (
            f"\n## {title}\n\n{_df_to_markdown(df)}"
            if df is not None and not df.empty
            else ""
        )

    parts.append(add_section(monthly_df, "Monthly Metrics"))
    parts.append(add_section(best_df, "Best 5 Trades"))
    parts.append(add_section(mid_df, "Average 5 Trades"))
    parts.append(add_section(worst_df, "Worst 5 Trades"))
    report_content = "\n".join(p for p in parts if p)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return out_path


# -----------------------------
# Public API
# -----------------------------


def generate_markdown_report(
    trades: Iterable["Trade"],
    initial_balance: float,
    out_dir: str,
    symbol: str,
    period_tag: str | None = None,
) -> str:
    """Compute metrics and export a Markdown report."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{period_tag}" if period_tag else ""

    df = trades_to_df(trades)

    # Overall metrics
    overall = mt5_like_metrics(df, initial_balance)
    overall_df = pd.DataFrame([overall])

    # Per-strategy split
    per_df = None
    if not df.empty and "strategy_id" in df.columns and df["strategy_id"].notna().any():
        per_rows = []
        for sid, g in df.groupby("strategy_id"):
            per_rows.append(
                {"strategy_id": sid, **mt5_like_metrics(g, initial_balance)}
            )
        per_df = pd.DataFrame(per_rows)

    # Top/mid/worst trades (overall)
    best_df, mid_df, worst_df = top_avg_worst(df, k=5)

    # Monthly - Placeholder
    monthly_df = pd.DataFrame()

    # Markdown report
    md_path = Path(out_dir) / f"{symbol}_report{tag}.md"
    _render_text_report(
        symbol,
        period_tag,
        overall_df,
        per_df,
        monthly_df,
        best_df,
        mid_df,
        worst_df,
        str(md_path),
    )

    print(f"Markdown report saved to {md_path}")
    return str(md_path)
