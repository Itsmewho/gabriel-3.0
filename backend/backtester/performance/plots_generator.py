# plot_generator.py

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Mapping
import pandas as pd
import numpy as np
import mplfinance as mpf
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from backtester.broker import Trade

# -----------------------------
# Core DataFrame Builders (self-contained in this file)
# -----------------------------


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
        return pd.DataFrame()

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

    df = df.sort_values(["exit_time", "entry_time"], na_position="last").reset_index(
        drop=True
    )
    return df


# -----------------------------
# Plotting
# -----------------------------


def _ensure_ohlc_columns(
    df: pd.DataFrame, columns: Mapping[str, str] | None = None
) -> pd.DataFrame:
    dfc = df.copy()
    want = ["Open", "High", "Low", "Close", "Volume"]
    if all(c in dfc.columns for c in want):
        return dfc

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

    ohlc_want = ["Open", "High", "Low", "Close"]
    missing = [c for c in ohlc_want if c not in dfc.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns after mapping: {missing}")
    return dfc


def _plot_trades_on_chart(
    market_data: pd.DataFrame,
    trades_df: pd.DataFrame,
    filename: str,
    feature_spec: Dict[str, Any] | None = None,
):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    feature_spec = feature_spec or {}

    df = market_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:  # type: ignore
        df.index = df.index.tz_localize(None)  # type: ignore

    df = _ensure_ohlc_columns(df)

    def _nearest(ts: pd.Timestamp) -> pd.Timestamp:
        pos = df.index.get_indexer([ts], method="nearest")[0]
        return df.index[pos]  # type: ignore

    # --- Plot trade entry/exit markers ---
    buy = pd.Series(np.nan, index=df.index, dtype="float64")
    sell = pd.Series(np.nan, index=df.index, dtype="float64")
    exitp = pd.Series(np.nan, index=df.index, dtype="float64")

    for _, trade in trades_df.iterrows():
        if pd.notna(trade.get("entry_time")) and pd.notna(trade.get("side")):
            ts = _nearest(pd.to_datetime(trade["entry_time"]).tz_localize(None))
            if trade["side"].lower() == "buy":
                buy.loc[ts] = trade["entry_price"]  # type: ignore
            elif trade["side"].lower() == "sell":
                sell.loc[ts] = trade["entry_price"]  # type: ignore
        if pd.notna(trade.get("exit_time")):
            ts = _nearest(pd.to_datetime(trade["exit_time"]).tz_localize(None))
            exitp.loc[ts] = trade["exit_price"]  # type: ignore

    segments, seg_colors = [], []
    for _, trade in trades_df.iterrows():
        if not all(
            pd.notna(trade.get(k))
            for k in ["entry_time", "exit_time", "side", "entry_price", "exit_price"]
        ):
            continue
        ts_entry = _nearest(pd.to_datetime(trade["entry_time"]).tz_localize(None))
        ts_exit = _nearest(pd.to_datetime(trade["exit_time"]).tz_localize(None))
        segments.append(
            [(ts_entry, trade["entry_price"]), (ts_exit, trade["exit_price"])]
        )
        prof = (
            (trade["exit_price"] > trade["entry_price"])
            if trade["side"].lower() == "buy"
            else (trade["exit_price"] < trade["entry_price"])
        )
        seg_colors.append("green" if prof else "red")

    # --- Assemble all additional plots ---
    add_plots = []
    if np.isfinite(buy.values).any():  # type: ignore
        add_plots.append(
            mpf.make_addplot(
                buy, type="scatter", marker="^", color="lightcyan", markersize=30
            )
        )
    if np.isfinite(sell.values).any():  # type: ignore
        add_plots.append(
            mpf.make_addplot(
                sell, type="scatter", marker="v", color="crimson", markersize=30
            )
        )
    if np.isfinite(exitp.values).any():  # type: ignore
        add_plots.append(
            mpf.make_addplot(
                exitp, type="scatter", marker="x", color="deepskyblue", markersize=30
            )
        )

    # <<< feature plotting >>>
    feature_plots = []
    next_panel = 1
    if "Volume" in df.columns:
        next_panel += 1

    # SMAs and EMAs
    colors = {
        "sma": ["dodgerblue", "palegreen", "gold", "bisque", "violet"],
        "ema": ["lightsteelblue", "magenta", "beige", "aqua", "darkorange"],
    }
    for ma_type in ["sma", "ema"]:
        if ma_type in feature_spec:
            for idx, period in enumerate(feature_spec[ma_type]):
                col = f"{ma_type}_{period}"
                if col in df.columns:
                    feature_plots.append(
                        mpf.make_addplot(
                            df[col],
                            color=colors[ma_type][idx % len(colors[ma_type])],
                            label=f"{ma_type.upper()} {period}",
                            width=0.5,
                        )
                    )

    # Bollinger Bands
    if "bb" in feature_spec:
        n = feature_spec["bb"].get("n", 20)
        bb_upper = f"bb_{n}_upper"
        bb_lower = f"bb_{n}_lower"
        if bb_upper in df.columns and bb_lower in df.columns:
            feature_plots.append(
                mpf.make_addplot(df[bb_upper], color="blue", linestyle="--")
            )
            feature_plots.append(
                mpf.make_addplot(df[bb_lower], color="blue", linestyle="--")
            )

    # RSI
    if "rsi" in feature_spec:
        for period in feature_spec["rsi"]:
            col = f"rsi_{period}"
            if col in df.columns:
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

    add_plots.extend(feature_plots)
    # <<< END of feature logic >>>

    plot_kwargs = dict(
        type="candle",
        style="binancedark",
        title="Trade Entries and Exits",
        ylabel="Price",
        addplot=add_plots if add_plots else None,
        volume="Volume" in df.columns,
        figscale=1.4,
        tight_layout=True,
        returnfig=True,
        alines=(
            dict(alines=segments, colors=seg_colors, linewidths=0.7, alpha=0.9)
            if segments
            else None
        ),
    )

    fig, axes = mpf.plot(df, **plot_kwargs)
    axes[0].legend()
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {filename}")


def _plot_best_worst_avg_periods(
    df_trades: pd.DataFrame,
    market_data: pd.DataFrame,
    out_dir: str,
    period_type: str,
    symbol: str,
    period_tag: str,
    feature_spec: Dict[str, Any] | None = None,
):
    md = market_data.copy()
    if md.index.tz is not None:  # type: ignore
        md.index = md.index.tz_localize(None)  # type: ignore

    period_char = "D" if period_type == "day" else "W"
    all_periods = md.index.to_period(period_char).unique()  # type: ignore

    if not df_trades.empty:
        df_trades["period"] = pd.to_datetime(df_trades["entry_time"]).dt.to_period(
            period_char
        )
        period_pnl = df_trades.groupby("period")["pnl"].sum()
    else:
        period_pnl = pd.Series(dtype=float)

    all_period_pnl = period_pnl.reindex(all_periods, fill_value=0).sort_values()
    if len(all_period_pnl) < 3:
        print(f"Not enough data for best/worst/avg {period_type} plots.")
        return

    periods_to_plot = {
        "best": all_period_pnl.index[-1],
        "worst": all_period_pnl.index[0],
        "average": (all_period_pnl - all_period_pnl.median()).abs().idxmin(),
    }

    for p_name, p_val in periods_to_plot.items():
        if "period" not in df_trades.columns and not df_trades.empty:
            df_trades["period"] = pd.to_datetime(df_trades["entry_time"]).dt.to_period(
                period_char
            )

        trades_in_period = (
            df_trades[df_trades["period"] == p_val]
            if not df_trades.empty
            else df_trades
        )
        market_data_period = md[p_val.start_time : p_val.end_time]
        if market_data_period.empty:
            continue

        filename = Path(out_dir) / f"{symbol}_{period_tag}_{p_name}_{period_type}.png"
        _plot_trades_on_chart(
            market_data_period, trades_in_period, str(filename), feature_spec
        )


def _plot_fixed_periods(
    df_trades: pd.DataFrame,
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str,
    feature_spec: Dict[str, Any] | None = None,
):
    md = market_data.copy()
    if md.index.tz is not None:  # type: ignore
        md.index = md.index.tz_localize(None)  # type: ignore

    unique_weeks = sorted(md.index.to_period("W").unique())  # type: ignore

    if len(unique_weeks) < 4:
        print("Less than 4 weeks of data available, skipping fixed period plots.")
        return

    fourth_week_period = unique_weeks[3]
    market_data_4th_week = md[
        fourth_week_period.start_time : fourth_week_period.end_time
    ]

    trades_in_4th_week = pd.DataFrame()
    if not df_trades.empty:
        trades_in_4th_week = df_trades[
            pd.to_datetime(df_trades["entry_time"]).dt.to_period("W")
            == fourth_week_period
        ]

    if not market_data_4th_week.empty:
        filename_week = Path(out_dir) / f"{symbol}_{period_tag}_fixed_4th_week.png"
        _plot_trades_on_chart(
            market_data_4th_week, trades_in_4th_week, str(filename_week), feature_spec
        )

    market_data_tuesday = market_data_4th_week[market_data_4th_week.index.weekday == 1]  # type: ignore

    if not market_data_tuesday.empty:
        tuesday_period = market_data_tuesday.index[0].to_period("D")

        trades_on_tuesday = pd.DataFrame()
        if not df_trades.empty:
            trades_on_tuesday = df_trades[
                pd.to_datetime(df_trades["entry_time"]).dt.to_period("D")
                == tuesday_period
            ]

        filename_tuesday = (
            Path(out_dir) / f"{symbol}_{period_tag}_fixed_4th_week_tuesday.png"
        )
        _plot_trades_on_chart(
            market_data_tuesday, trades_on_tuesday, str(filename_tuesday), feature_spec
        )


# -----------------------------
# Public API
# -----------------------------


def generate_plots(
    trades: Iterable["Trade"],
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str | None = None,
    feature_spec: Dict[str, Any] | None = None,
):
    """Generate plots for best, worst, average, and fixed trading periods."""
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{period_tag}" if period_tag else "full_period"

    df = trades_to_df(trades)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for period_type in ["day", "week"]:
            future = executor.submit(
                _plot_best_worst_avg_periods,
                df,
                market_data,
                str(plot_dir),
                period_type,
                symbol,
                tag,
                feature_spec,
            )
            futures.append(future)
        for future in futures:
            future.result()

    _plot_fixed_periods(df, market_data, str(plot_dir), symbol, tag, feature_spec)
