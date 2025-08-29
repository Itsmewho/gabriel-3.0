from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Mapping
import pandas as pd
import numpy as np
import mplfinance as mpf
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from backtester.broker import Trade
from backtester.features.better_volume_indicator import add_better_volume


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


def _ensure_ohlc_columns(
    df: pd.DataFrame, columns: Mapping[str, str] | None = None
) -> pd.DataFrame:
    dfc = df.copy()
    # Standardize to capitalized first
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "tick_volume": "Volume",
    }
    dfc = dfc.rename(columns={k: v for k, v in rename_map.items() if k in dfc.columns})

    # Check for required columns
    want = ["Open", "High", "Low", "Close"]
    missing = [c for c in want if c not in dfc.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns after mapping: {missing}")
    return dfc


def _plot_trades_on_chart(
    market_data: pd.DataFrame,
    trades_df: pd.DataFrame,
    filename: str,
    feature_spec: Dict[str, Any] | None = None,
    enable_better_volume: bool = True,
    warn_cap: int | None = None,
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

    use_better_volume = enable_better_volume and "Volume" in df.columns
    if use_better_volume:
        df = add_better_volume(df)

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

    feature_plots = []
    current_panel = 1
    volume_panel = -1
    rsi_panel = -1

    if use_better_volume:
        volume_panel = current_panel
        current_panel += 1

    has_rsi = "rsi" in feature_spec and any(
        f"rsi_{p}" in df.columns for p in feature_spec.get("rsi", [])
    )
    if has_rsi:
        rsi_panel = current_panel
        current_panel += 1

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
    if "bb" in feature_spec:
        n = feature_spec["bb"].get("n", 20)
        bb_upper, bb_lower = f"bb_{n}_upper", f"bb_{n}_lower"
        if bb_upper in df.columns and bb_lower in df.columns:
            feature_plots.append(
                mpf.make_addplot(df[bb_upper], color="blue", linestyle="--")
            )
            feature_plots.append(
                mpf.make_addplot(df[bb_lower], color="blue", linestyle="--")
            )

    if has_rsi:
        for period in feature_spec["rsi"]:
            col = f"rsi_{period}"
            if col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[col], panel=rsi_panel, ylabel="RSI", color="purple"
                    )
                )
        feature_plots.append(
            mpf.make_addplot(
                pd.Series(70, index=df.index),
                panel=rsi_panel,
                color="red",
                linestyle="--",
            )
        )
        feature_plots.append(
            mpf.make_addplot(
                pd.Series(30, index=df.index),
                panel=rsi_panel,
                color="green",
                linestyle="--",
            )
        )

    add_plots.extend(feature_plots)

    if use_better_volume:
        colors = df["bv_color"].unique()
        for color in colors:
            if pd.isna(color):
                continue
            vol_series = df.apply(
                lambda row: row["Volume"] if row["bv_color"] == color else np.nan,
                axis=1,
            )
            if not vol_series.dropna().empty:
                add_plots.append(
                    mpf.make_addplot(
                        vol_series,
                        type="bar",
                        panel=volume_panel,
                        color=color,
                        secondary_y=False,
                    )
                )

    wtd = warn_cap if warn_cap is not None else len(df) + 1

    plot_kwargs = dict(
        type="candle",
        style="binancedark",
        title="Trade Entries and Exits",
        ylabel="Price",
        addplot=add_plots if add_plots else None,
        volume=not use_better_volume and "Volume" in df.columns,
        figscale=1.4,
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=wtd,
    )

    num_panels = current_panel
    if num_panels > 1:
        ratios = (3,) + (1,) * (num_panels - 1)
        plot_kwargs["panel_ratios"] = ratios

    if segments:
        plot_kwargs["alines"] = dict(
            alines=segments, colors=seg_colors, linewidths=0.7, alpha=0.9
        )

    fig, axes = mpf.plot(df, **plot_kwargs)
    if axes:
        axes[0].legend(loc="best")
        if use_better_volume and volume_panel > 0 and volume_panel < len(axes):
            axes[volume_panel].set_title("Better Volume", color="white", size=10)

    fig.savefig(filename, dpi=450, bbox_inches="tight")
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
        print("Less than 4 weeks of data, skipping fixed period plots.")
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


def generate_plots(
    trades: Iterable["Trade"],
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str | None = None,
    feature_spec: Dict[str, Any] | None = None,
):
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{period_tag}" if period_tag else "full_period"
    df = trades_to_df(trades)

    with ProcessPoolExecutor(max_workers=3) as executor:
        # max_workers is tricky keep in mind the max cores of your CPU
        # 1 worker defined here will exe 4 worker as per process the plot_generator will take up 3
        # 4 x 3 = 12 is you have a 12 core 24 cpu you can bump it up till 6 == speed up backtesting results :D
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
