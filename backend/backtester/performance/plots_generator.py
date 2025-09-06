from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Mapping
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import random


from backtester.broker import Trade
from backtester.features.better_volume_indicator import add_better_volume_mql

DARK_BLUE = "#0d47a1"


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
        df = add_better_volume_mql(df)

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
                buy, type="scatter", marker="^", color="limegreen", markersize=20
            )
        )
    if np.isfinite(sell.values).any():  # type: ignore
        add_plots.append(
            mpf.make_addplot(
                sell, type="scatter", marker="v", color="crimson", markersize=20
            )
        )
    if np.isfinite(exitp.values).any():  # type: ignore
        add_plots.append(
            mpf.make_addplot(
                exitp, type="scatter", marker="x", color="deepskyblue", markersize=20
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
                            width=0.3,
                        )
                    )

    if "sma_high" in feature_spec:
        for period in feature_spec["sma_high"]:
            col = f"sma_high_{int(period)}"
            if col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[col],
                        color=DARK_BLUE,
                        width=0.3,
                        label=f"SMA High {int(period)}",
                    )
                )

    if "sma_low" in feature_spec:
        for period in feature_spec["sma_low"]:
            col = f"sma_low_{int(period)}"
            if col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[col],
                        color=DARK_BLUE,
                        width=0.3,
                        label=f"SMA Low {int(period)}",
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

    fig.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {filename}")


def _plot_selected_periods(
    df_trades: pd.DataFrame,
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str,
    feature_spec: Dict[str, Any] | None = None,
):
    """Plots fixed/random days/weeks and their 6/12-hour close-ups."""
    md = market_data.copy()
    if md.index.tz is not None:
        md.index = md.index.tz_localize(None)

    all_days = sorted(md.index.to_period("D").unique())
    all_weeks = sorted(md.index.to_period("W").unique())

    periods_to_plot = {}

    if len(all_days) > 2:
        periods_to_plot["3rd_day"] = all_days[2]
    if len(all_days) > 3:
        periods_to_plot["4th_day"] = all_days[3]
    if len(all_days) > 7:
        periods_to_plot["8th_day"] = all_days[7]
    if len(all_days) > 8:  # Middle and End only make sense with some data
        periods_to_plot["middle_day"] = all_days[len(all_days) // 2]
        periods_to_plot["end_day"] = all_days[-3]

    # --- Select Fixed Weeks ---
    if len(all_weeks) > 2:
        periods_to_plot["3rd_week"] = all_weeks[2]
    if len(all_weeks) > 3:
        periods_to_plot["4th_week"] = all_weeks[3]
    if len(all_weeks) > 7:
        periods_to_plot["8th_week"] = all_weeks[7]
    if len(all_weeks) > 8:
        periods_to_plot["middle_week"] = all_weeks[len(all_weeks) // 2]
        periods_to_plot["end_week"] = all_weeks[-3]

    # --- Select Random Periods ---
    if len(all_days) >= 2:
        for i, p in enumerate(random.sample(all_days, min(2, len(all_days))), 1):
            periods_to_plot[f"random_day_{i}"] = p
    if len(all_weeks) >= 2:
        for i, p in enumerate(random.sample(all_weeks, min(2, len(all_weeks))), 1):
            periods_to_plot[f"random_week_{i}"] = p
    # --- END OF UPDATED SELECTION ---

    # 2. Generate plots for each selected period and its close-ups
    for p_name, p_val in periods_to_plot.items():
        period_char = "W" if "week" in p_name else "D"
        market_data_period = md.loc[p_val.start_time : p_val.end_time]
        if market_data_period.empty:
            continue

        trades_in_period = pd.DataFrame()
        if not df_trades.empty:
            trades_in_period = df_trades[
                pd.to_datetime(df_trades["entry_time"]).dt.to_period(period_char)
                == p_val
            ]

        filename = Path(out_dir) / f"{symbol}_{period_tag}_{p_name}.png"
        _plot_trades_on_chart(
            market_data_period, trades_in_period, str(filename), feature_spec
        )

        anchor_time = market_data_period.index[0]
        base_filename = str(filename).removesuffix(".png")

        for hours in [6, 12]:
            start_time = anchor_time
            end_time = anchor_time + pd.Timedelta(hours=hours)
            market_data_slice = market_data_period.loc[start_time:end_time]
            if market_data_slice.empty:
                continue

            trades_slice = pd.DataFrame()
            if not trades_in_period.empty:
                trades_slice = trades_in_period[
                    (pd.to_datetime(trades_in_period["entry_time"]) >= start_time)
                    & (pd.to_datetime(trades_in_period["entry_time"]) < end_time)
                ]

            intraday_filename = f"{base_filename}_{hours}hr.png"
            _plot_trades_on_chart(
                market_data_slice, trades_slice, intraday_filename, feature_spec
            )


def generate_plots(
    trades: Iterable["Trade"],
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str | None = None,
    feature_spec: Dict[str, Any] | None = None,
):
    """Generates a comprehensive set of charts for strategy analysis."""
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{period_tag}" if period_tag else "full_period"
    df = trades_to_df(trades)

    # The ProcessPoolExecutor is no longer needed since we have a single, unified function.
    # This simplifies the logic and removes the error.
    print("Generating selected period plots and their intraday close-ups...")
    _plot_selected_periods(df, market_data, str(plot_dir), symbol, tag, feature_spec)
    print("All plots have been generated.")
