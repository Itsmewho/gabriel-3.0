from __future__ import annotations
import math
from pathlib import Path
from typing import Iterable, Dict, Any, Tuple, Mapping, List
import pandas as pd
import numpy as np
import mplfinance as mpf
from concurrent.futures import ProcessPoolExecutor

from backtester.broker import Trade


# -----------------------------
# Core dataframe builders
# -----------------------------


def trades_to_df(trades: Iterable[Trade]) -> pd.DataFrame:
    """Converts an iterable of Trade objects into a pandas DataFrame."""
    rows = []
    for t in trades:
        rows.append(
            dict(
                id=getattr(t, "id", None),
                strategy_id=getattr(t, "strategy_id", None),
                side=getattr(t, "side", None),  # e.g., 'buy' or 'sell'
                lots=float(getattr(t, "lot_size", np.nan)),
                entry_time=pd.to_datetime(getattr(t, "entry_time", None)),  # type: ignore
                exit_time=pd.to_datetime(getattr(t, "exit_time", None)),  # type: ignore
                entry_price=float(getattr(t, "entry_price", np.nan)),
                exit_price=(
                    float(getattr(t, "exit_price", np.nan))
                    if getattr(t, "exit_price", None) is not None
                    else np.nan
                ),
                pnl=float(getattr(t, "pnl", 0.0)),
                commission=float(getattr(t, "commission_paid", 0.0)),
                swap=float(getattr(t, "swap_paid", 0.0)),
                exit_reason=getattr(t, "exit_reason", None),
                balance_at_open=float(
                    getattr(t, "balance_at_open", np.nan)
                    if getattr(t, "balance_at_open", None) is not None
                    else np.nan
                ),
            )
        )
    df = pd.DataFrame(rows)

    for col, val in [
        ("entry_time", pd.NaT),
        ("exit_time", pd.NaT),
        ("pnl", 0.0),
        ("balance_at_open", np.nan),
        ("side", ""),
        ("gross_profit_component", 0.0),
        ("gross_loss_component", 0.0),
        ("ret", np.nan),
    ]:
        if col not in df.columns:
            df[col] = val

    if df.empty:
        return df  # now has required columns, so downstream code won’t KeyError

    df = df.sort_values(["exit_time", "entry_time"], na_position="last").reset_index(
        drop=True
    )
    df["duration_minutes"] = (
        df["exit_time"] - df["entry_time"]
    ).dt.total_seconds() / 60.0
    df["is_win"] = df["pnl"] > 0
    df["gross_profit_component"] = df["pnl"].where(df["pnl"] > 0, 0.0)
    df["gross_loss_component"] = -df["pnl"].where(df["pnl"] < 0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ret"] = df["pnl"] / df["balance_at_open"].replace(0.0, np.nan)
    return df


def equity_curve(df_trades: pd.DataFrame, initial_balance: float) -> pd.DataFrame:
    """Calculates the equity curve and drawdown at each trade close."""
    if df_trades.empty:
        return pd.DataFrame(
            {"time": [], "equity": [], "drawdown_abs": [], "drawdown_pct": []}
        )

    # Ensure trades are sorted by exit time to correctly calculate equity
    df_sorted = df_trades.dropna(subset=["exit_time"]).sort_values("exit_time")

    # Calculate cumulative PnL and equity
    df_sorted["cumulative_pnl"] = df_sorted["pnl"].cumsum()
    ec = pd.DataFrame(
        {
            "time": df_sorted["exit_time"],
            "equity": initial_balance + df_sorted["cumulative_pnl"],
        }
    )

    if ec.empty:
        return ec

    # Calculate drawdown
    peak = ec["equity"].cummax()
    dd_abs = peak - ec["equity"]
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = dd_abs / peak.replace(0.0, np.nan)

    ec["drawdown_abs"] = dd_abs
    ec["drawdown_pct"] = dd_pct.fillna(0.0)

    return ec


# -----------------------------
# Metric calculators (MT5-like)
# -----------------------------


def mt5_like_metrics(df_trades: pd.DataFrame, initial_balance: float) -> Dict[str, Any]:
    """Calculates a comprehensive set of performance metrics."""
    n = len(df_trades)

    # Core Profit/Loss Metrics
    total_net = float(df_trades["pnl"].sum()) if n else 0.0
    gross_profit = float(df_trades["gross_profit_component"].sum())
    gross_loss = float(df_trades["gross_loss_component"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    expected_payoff = (total_net / n) if n else 0.0

    # Win/Loss Metrics
    wins = int((df_trades["pnl"] > 0).sum())
    losses = int((df_trades["pnl"] < 0).sum())
    winrate = (wins / n) if n else 0.0

    # Sharpe Ratio (simple proxy on per-trade returns)
    rets = df_trades["ret"].dropna()
    sharpe = np.nan
    if len(rets) > 1 and rets.std() > 0:
        sharpe = (rets.mean() / rets.std()) * math.sqrt(
            n
        )  # Annualized using sqrt(num_trades)

    # Equity Curve and Drawdowns
    ec = equity_curve(df_trades, initial_balance)
    bal_dd_abs_max = 0.0
    bal_dd_pct_max = 0.0
    eq_abs_from_initial = 0.0
    eq_dd_rel_max = 0.0
    if not ec.empty:
        bal_dd_abs_max = float(ec["drawdown_abs"].max())
        bal_dd_pct_max = float(ec["drawdown_pct"].max())
        # Equity drawdown relative to the initial deposit
        min_equity = ec["equity"].min()
        if min_equity < initial_balance:
            eq_abs_from_initial = float(initial_balance - min_equity)
            if initial_balance > 0:
                eq_dd_rel_max = eq_abs_from_initial / initial_balance

    recovery_factor = (total_net / bal_dd_abs_max) if bal_dd_abs_max > 0 else np.inf

    # Largest & Average Trades
    df_wins = df_trades[df_trades["pnl"] > 0]
    df_losses = df_trades[df_trades["pnl"] < 0]
    largest_profit_trade = df_wins["pnl"].max() if not df_wins.empty else 0.0
    largest_loss_trade = df_losses["pnl"].min() if not df_losses.empty else 0.0
    average_profit_trade = df_wins["pnl"].mean() if not df_wins.empty else 0.0
    average_loss_trade = df_losses["pnl"].mean() if not df_losses.empty else 0.0

    # Consecutive Wins and Losses
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

    # Long/Short Metrics
    # FIX: Handle potential None values in 'side' column and check for 'buy'/'sell'
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
# Text/Markdown rendering (terminal-friendly)
# -----------------------------


def _df_to_markdown(df: pd.DataFrame | None) -> str:
    """Converts a DataFrame to a Markdown table string."""
    if df is None or df.empty:
        return ""
    return df.to_markdown(index=False)


def _create_summary_table(metrics_dict: Dict[str, Any]) -> str:
    """Creates a two-column Markdown table for a dictionary of metrics."""

    def fmt(x, is_pct=False):
        if pd.isna(x) or x is None:
            return "-"
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}"
        if isinstance(x, (float, np.floating)):
            if is_pct:
                return f"{x:.2%}"
            return f"{x:,.2f}"
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
            f"Profit Trades (% of total)",
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
            f"Balance Drawdown Maximal (%)",
            f"{fmt(ov.get('balance_drawdown_pct_max'), is_pct=True)}",
        ),
        (
            f"Equity Drawdown Relative (%)",
            f"{fmt(ov.get('equity_drawdown_rel_max'), is_pct=True)}",
        ),
        (
            f"Short Trades (won %)",
            f"{fmt(ov.get('short_trades_count'))} ({fmt(ov.get('short_trades_win_pct'), is_pct=True)})",
        ),
        (
            f"Long Trades (won %)",
            f"{fmt(ov.get('long_trades_count'))} ({fmt(ov.get('long_trades_win_pct'), is_pct=True)})",
        ),
        ("Largest loss trade", ov.get("largest_loss_trade")),
        ("Average loss trade", ov.get("average_loss_trade")),
        ("Consecutive losses (# trades)", ov.get("max_consecutive_losses_count")),
        ("Consecutive loss ($)", ov.get("max_consecutive_losses_sum")),
    ]

    rows = []
    rows.append(
        "| Metric                       | Value         | Metric                         | Value           |"
    )
    rows.append(
        "|:-------------------------------|--------------:|:-------------------------------|----------------:|"
    )
    for i in range(max(len(left), len(right))):
        l_metric, l_val = left[i] if i < len(left) else ("", "")
        r_metric, r_val = right[i] if i < len(right) else ("", "")

        l_val_str = l_val if isinstance(l_val, str) else fmt(l_val)
        r_val_str = r_val if isinstance(r_val, str) else fmt(r_val)

        rows.append(
            f"| {l_metric:<30} | {l_val_str:>13} | {r_metric:<30} | {r_val_str:>15} |"
        )

    return "\n".join(rows)


def _render_text_simple(
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
    """Renders all dataframes into a single, comprehensive Markdown report."""
    ov = (
        overall_df.iloc[0].to_dict()
        if overall_df is not None and not overall_df.empty
        else {}
    )

    # Assemble the final report string
    parts: list[str] = []
    parts.append(f"# {symbol} Backtest Report")
    if period_tag:
        parts.append(f"**Period:** {period_tag}\n")

    # Overall Summary Section
    parts.append("## Summary Metrics")
    parts.append(_create_summary_table(ov))

    # Per-Strategy Section
    if per_df is not None and not per_df.empty:
        parts.append("\n## Per-Strategy Metrics")
        for _, strategy_row in per_df.iterrows():
            strategy_id = strategy_row.get("strategy_id", "Unknown Strategy")
            parts.append(f"\n### Strategy: {strategy_id}")
            strategy_metrics = strategy_row.to_dict()
            parts.append(_create_summary_table(strategy_metrics))

    # Other Sections
    def add_section(df: pd.DataFrame | None, title: str) -> str:
        if df is None or df.empty:
            return ""
        return f"\n## {title}\n\n{_df_to_markdown(df)}"

    parts.append(add_section(monthly_df, "Monthly Metrics"))
    parts.append(add_section(best_df, "Best 5 Trades"))
    parts.append(add_section(mid_df, "Average 5 Trades"))
    parts.append(add_section(worst_df, "Worst 5 Trades"))

    report_content = "\n".join([p for p in parts if p])

    # Write the report to the specified file
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return out_path


# -----------------------------
# Plotting
# -----------------------------


def _ensure_ohlc_columns(
    df: pd.DataFrame, columns: Mapping[str, str] | None = None
) -> pd.DataFrame:
    dfc = df.copy()
    want = ["Open", "High", "Low", "Close", "Volume"]
    # Check for standard names first
    if all(c in dfc.columns for c in want):
        return dfc

    # Auto-map common alternative names
    auto = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "tick_volume": "Volume",
        "vol": "Volume",
    }
    mapping = {} if columns is None else dict(columns)
    if columns is None:
        for k, v in auto.items():
            if k in dfc.columns and v not in dfc.columns:
                mapping[k] = v
    if mapping:
        dfc = dfc.rename(columns=mapping)

    # Check for OHLC, but Volume is optional
    ohlc_want = ["Open", "High", "Low", "Close"]
    missing = [c for c in ohlc_want if c not in dfc.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns after mapping: {missing}")
    return dfc


def plot_trades(
    market_data: pd.DataFrame,
    trades_or_events: Iterable[Any],
    filename: str = "results/plots/trade_plot.png",
    columns: Mapping[str, str] | None = None,
    markersize: int = 30,
    warn_cap: int | None = None,
    fig_dpi: int = 300,
    strategy_names: List[str] | None = None,
):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    strategy_names = strategy_names or []

    # --- normalize market data index ---
    df = market_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:  # type: ignore
        df.index = df.index.tz_localize(None)  # type: ignore

    df = _ensure_ohlc_columns(df, columns)

    # --- helpers ---
    def _norm_ts(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        return ts

    def _nearest(ts: pd.Timestamp) -> pd.Timestamp:
        ts = _norm_ts(ts)
        # Find the index position in the market data for the given timestamp
        pos = df.index.get_indexer([ts], method="nearest")[0]
        # Return the actual timestamp from the index
        return df.index[pos]  # type: ignore

    # --- series for markers ---
    buy = pd.Series(np.nan, index=df.index, dtype="float64")
    sell = pd.Series(np.nan, index=df.index, dtype="float64")
    exitp = pd.Series(np.nan, index=df.index, dtype="float64")

    # Use a DataFrame for easier processing
    if not isinstance(trades_or_events, pd.DataFrame):
        trades_df = pd.DataFrame(trades_or_events)
    else:
        trades_df = trades_or_events

    # Entries and Exits from the trades_df
    for _, trade_row in trades_df.iterrows():
        # Entry
        entry_time = trade_row.get("entry_time")
        side = trade_row.get("side")
        entry_price = trade_row.get("entry_price")

        if pd.notna(entry_time) and pd.notna(side) and pd.notna(entry_price):
            ts = _norm_ts(entry_time)
            if side.lower() == "buy":
                buy.loc[_nearest(ts)] = entry_price  # type: ignore
            elif side.lower() == "sell":
                sell.loc[_nearest(ts)] = entry_price  # type: ignore

        # Exit
        exit_time = trade_row.get("exit_time")
        exit_price = trade_row.get("exit_price")
        if pd.notna(exit_time) and pd.notna(exit_price):
            ts = _norm_ts(exit_time)
            exitp.loc[_nearest(ts)] = exit_price  # type: ignore

    # Entry→exit line segments
    segments: list[list[tuple[pd.Timestamp, float]]] = []
    seg_colors: list[str] = []

    for _, trade_row in trades_df.iterrows():
        entry_time = trade_row.get("entry_time")
        exit_time = trade_row.get("exit_time")
        side = trade_row.get("side")
        entry_price = trade_row.get("entry_price")
        exit_price = trade_row.get("exit_price")

        if not all(
            pd.notna(val)
            for val in [entry_time, exit_time, side, entry_price, exit_price]
        ):
            continue

        ts_entry = _nearest(entry_time)  # type: ignore
        ts_exit = _nearest(exit_time)  # type: ignore

        segments.append([(ts_entry, float(entry_price)), (ts_exit, float(exit_price))])  # type: ignore
        prof = (
            (exit_price > entry_price)  # type: ignore
            if side.lower() == "buy"  # type: ignore
            else (exit_price < entry_price)  # type: ignore
        )
        seg_colors.append("green" if prof else "red")

    # --- Auto-detect and plot features ---
    feature_plots = []
    next_panel = 1  # Start counting panels after the main price panel (0)

    # Volume (if available, gets its own panel)
    show_volume = "Volume" in df.columns
    if show_volume:
        next_panel += 1  # Reserve panel 1 for volume

    # SMAs and EMAs (on main panel)
    ma_cols = [c for c in df.columns if c.startswith("sma_") or c.startswith("ema_")]
    colors = ["orange", "purple", "brown", "pink", "olive", "cyan"]
    for i, col in enumerate(ma_cols):
        feature_plots.append(mpf.make_addplot(df[col], color=colors[i % len(colors)]))

    # Bollinger Bands (on main panel)
    bb_upper = next(
        (c for c in df.columns if c.startswith("bb_") and c.endswith("_upper")), None
    )
    bb_lower = next(
        (c for c in df.columns if c.startswith("bb_") and c.endswith("_lower")), None
    )
    if bb_upper and bb_lower:
        feature_plots.append(
            mpf.make_addplot(df[bb_upper], color="blue", linestyle="--")
        )
        feature_plots.append(
            mpf.make_addplot(df[bb_lower], color="blue", linestyle="--")
        )

    # Ichimoku Cloud (on main panel)
    ichimoku_fill = {}
    if "ichimoku_senkou_a" in df.columns and "ichimoku_senkou_b" in df.columns:
        feature_plots.append(mpf.make_addplot(df["ichimoku_tenkan"], color="cyan"))
        feature_plots.append(mpf.make_addplot(df["ichimoku_kijun"], color="magenta"))
        feature_plots.append(
            mpf.make_addplot(df["ichimoku_chikou"], color="gray", linestyle=":")
        )
        # Prepare fill_between dict for the main plot call
        ichimoku_fill = dict(
            y1=df["ichimoku_senkou_a"].values,
            y2=df["ichimoku_senkou_b"].values,
            where=df["ichimoku_senkou_a"] >= df["ichimoku_senkou_b"],
            color="green",
            alpha=0.1,
        )

    # RSI (on a new panel) - Conditional
    rsi_cols = [c for c in df.columns if c.startswith("rsi_")]
    if rsi_cols and any("RSI" in name.upper() for name in strategy_names):
        for col in rsi_cols:
            feature_plots.append(
                mpf.make_addplot(
                    df[col], panel=next_panel, ylabel="RSI", color="purple"
                )
            )
        # Add overbought/oversold lines
        feature_plots.append(
            mpf.make_addplot(
                pd.Series(70, index=df.index),
                panel=next_panel,
                color="red",
                linestyle="--",
            )
        )
        feature_plots.append(
            mpf.make_addplot(
                pd.Series(30, index=df.index),
                panel=next_panel,
                color="green",
                linestyle="--",
            )
        )
        next_panel += 1

    # MACD (on a new panel) - Conditional
    if "macd" in df.columns and any("MACD" in name.upper() for name in strategy_names):
        feature_plots.append(
            mpf.make_addplot(df["macd"], panel=next_panel, ylabel="MACD", color="blue")
        )
        feature_plots.append(
            mpf.make_addplot(df["macd_signal"], panel=next_panel, color="orange")
        )
        feature_plots.append(
            mpf.make_addplot(
                df["macd_hist"], type="bar", panel=next_panel, color="gray", alpha=0.5
            )
        )
        next_panel += 1

    # --- Combine all plots ---
    all_plots = []

    # Add trade markers first
    def _nonempty(s: pd.Series) -> bool:
        return np.isfinite(s.values).any()  # type: ignore

    if _nonempty(buy):
        all_plots.append(
            mpf.make_addplot(
                buy, type="scatter", marker="^", color="green", markersize=markersize
            )
        )
    if _nonempty(sell):
        all_plots.append(
            mpf.make_addplot(
                sell, type="scatter", marker="v", color="red", markersize=markersize
            )
        )
    if _nonempty(exitp):
        all_plots.append(
            mpf.make_addplot(
                exitp, type="scatter", marker="x", color="blue", markersize=markersize
            )
        )

    # Add feature plots
    all_plots.extend(feature_plots)

    aline_kwargs = {}
    if segments:
        aline_kwargs = dict(
            alines=dict(alines=segments, colors=seg_colors, linewidths=0.7, alpha=0.9)
        )

    wtd = warn_cap if warn_cap is not None else len(df) + 1
    mpf.plot(
        df,
        type="candle",
        style="yahoo",
        title="Trade Entries and Exits",
        ylabel="Price",
        addplot=all_plots if all_plots else None,
        volume=show_volume,
        figscale=1.4,
        tight_layout=True,
        warn_too_much_data=wtd,
        savefig=dict(fname=filename, dpi=fig_dpi),
        fill_between=ichimoku_fill if ichimoku_fill else None,
        **aline_kwargs,
    )
    print(f"PNG trade-plot report saved to {filename}")
    return filename


def _plot_best_worst_avg_periods(
    df_trades: pd.DataFrame,
    market_data: pd.DataFrame,
    out_dir: str,
    period_type: str,  # 'day' or 'week'
    symbol: str,
    period_tag: str,
    strategy_names: List[str],
):
    """Identifies and plots the best, worst, and average trading periods."""
    if df_trades.empty:
        return

    # Ensure market_data index is timezone-naive before any comparisons
    md = market_data.copy()
    if md.index.tz is not None:  # type: ignore
        md.index = md.index.tz_localize(None)  # type: ignore

    df = df_trades.copy()
    df["period"] = df["entry_time"].dt.to_period("D" if period_type == "day" else "W")

    period_pnl = df.groupby("period")["pnl"].sum().sort_values()

    if len(period_pnl) < 3:
        print(
            f"Not enough data for best/worst/avg {period_type} plots. Need at least 3 distinct periods."
        )
        return

    best_period = period_pnl.index[-1]
    worst_period = period_pnl.index[0]
    # For average, find the period with PnL closest to the median
    median_pnl = period_pnl.median()
    avg_period = (period_pnl - median_pnl).abs().idxmin()

    for p_type, p_val in [
        ("best", best_period),
        ("worst", worst_period),
        ("average", avg_period),
    ]:
        trades_in_period = df[df["period"] == p_val]

        start_time = p_val.start_time
        end_time = p_val.end_time

        market_data_period = md[(md.index >= start_time) & (md.index <= end_time)]

        if market_data_period.empty:
            continue

        filename = f"{out_dir}/{symbol}_{period_tag}_{p_type}_{period_type}.png"
        plot_trades(
            market_data_period,
            trades_in_period,
            filename=filename,
            strategy_names=strategy_names,
        )


# -----------------------------
# Public API
# -----------------------------


def evaluate(
    trades: Iterable["Trade"],
    initial_balance: float,
    out_dir: str,
    symbol: str,
    period_tag: str | None = None,
    market_data: pd.DataFrame | None = None,  # Pass market data for plotting
    strategies: List[Any] | None = None,  # Pass strategies for conditional plotting
) -> Dict[str, Any]:
    """Compute metrics, generate plots, and export a Markdown report."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{period_tag}" if period_tag else ""

    df = trades_to_df(trades)
    paths: Dict[str, Any] = {}

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
    md_path = f"{out_dir}/{symbol}_report{tag}.md"
    _render_text_simple(
        symbol,
        period_tag,
        overall_df,
        per_df,
        monthly_df,
        best_df,
        mid_df,
        worst_df,
        md_path,
    )
    paths["markdown_report"] = md_path

    # Generate plots if market data is available
    if market_data is not None:
        plot_dir = str(Path(out_dir) / "plots")
        Path(plot_dir).mkdir(exist_ok=True)

        strategy_names = [s.name for s in strategies] if strategies else []

        # Plot for the whole period
        overall_plot_path = f"{plot_dir}/{symbol}_{tag}_all_trades.png"
        plot_trades(
            market_data, df, filename=overall_plot_path, strategy_names=strategy_names
        )
        paths["overall_plot"] = overall_plot_path

        # PERFORMANCE: Use a process pool to generate these plots in parallel
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = []
            for period_type in ["day", "week"]:
                future = executor.submit(
                    _plot_best_worst_avg_periods,
                    df,
                    market_data,
                    plot_dir,
                    period_type,
                    symbol,
                    tag,
                    strategy_names,
                )
                futures.append(future)
            # Wait for all plot generation tasks to complete
            for future in futures:
                future.result()

    else:
        print(
            "Warning: 'market_data' not provided to evaluate(). Skipping plot generation."
        )

    return paths
