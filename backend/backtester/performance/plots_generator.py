from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, Mapping, List, Union
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import random
import warnings

from concurrent.futures import ProcessPoolExecutor

from backtester.broker import Trade
from backtester.features.better_volume_indicator import add_better_volume_mql

# This tells Python to ignore any UserWarning with that specific message
warnings.filterwarnings(
    "ignore",
    message='Creating legend with loc="best" can be slow with large amounts of data.',
)


# Import the executor for parallel processing
DARK_BLUE = "#0d47a1"
BROWN = "saddlebrown"


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


# --- ADDED: Helper for Stochastic configs ---
def _iter_stoch_configs(
    stoch_cfg: Union[Dict[str, Any], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if stoch_cfg is None:
        return []
    if isinstance(stoch_cfg, list):
        return stoch_cfg
    return [stoch_cfg]


def _plot_trades_on_chart(
    market_data: pd.DataFrame,
    trades_df: pd.DataFrame,
    filename: str,
    feature_spec: Dict[str, Any] | None = None,
    enable_better_volume: bool = True,
    warn_cap: int | None = None,
):
    """
    Designed for use by ProcessPoolExecutor.
    """
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
            if str(trade["side"]).lower() == "buy":
                buy.loc[ts] = trade["entry_price"]  # type: ignore
            elif str(trade["side"]).lower() == "sell":
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
            if str(trade["side"]).lower() == "buy"
            else (trade["exit_price"] < trade["entry_price"])
        )
        seg_colors.append("green" if prof else "red")

    add_plots: list = []
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
    stoch_panel = -1  # --- ADDED ---

    if use_better_volume:
        volume_panel = current_panel
        current_panel += 1

    has_rsi = "rsi" in feature_spec and any(
        f"rsi_{p}" in df.columns for p in feature_spec.get("rsi", [])
    )
    if has_rsi:
        rsi_panel = current_panel
        current_panel += 1

    # --- ADDED: Panel detection for Stochastic ---
    stoch_configs = _iter_stoch_configs(feature_spec.get("stoch"))
    has_stoch = stoch_configs and any(
        f"stoch_{s.get('k_period', 14)}_{s.get('d_period', 3)}_{s.get('slowing', 3)}_k"
        in df.columns
        for s in stoch_configs
    )
    if has_stoch:
        stoch_panel = current_panel
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

    # --- Keltner Channels overlay ---
    fill_between_cfg = []

    if "kc" in feature_spec:
        kc_list = feature_spec["kc"]
        if isinstance(kc_list, dict):
            kc_list = [kc_list]

        palette = ["white", DARK_BLUE, BROWN]

        for i, kc_cfg in enumerate(kc_list):
            n = kc_cfg.get("n", 20)
            atr_n = kc_cfg.get("atr_n", n)
            m = kc_cfg.get("m", 2.0)

            mid_col = f"kc_{n}_{atr_n}_{m}_mid"
            upper_col = f"kc_{n}_{atr_n}_{m}_upper"
            lower_col = f"kc_{n}_{atr_n}_{m}_lower"

            color = palette[i % len(palette)]

            if mid_col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[mid_col],
                        color=color,
                        linestyle="-",
                        width=0.6,
                        label=f"KC Mid {n}/{atr_n}/{m}",
                    )
                )
            if upper_col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[upper_col],
                        color=color,
                        linestyle="--",
                        width=0.6,
                    )
                )
            if lower_col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[lower_col],
                        color=color,
                        linestyle="--",
                        width=0.6,
                    )
                )

            # Optional shading between upper and lower for this KC, use same hue
            if upper_col in df.columns and lower_col in df.columns:
                add_plots.append(
                    mpf.make_addplot(
                        df[upper_col].combine(df[lower_col], max),
                        color=color,
                        alpha=0.10,
                        width=0.0,
                    )
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

        # --- PTL (Price-Trend-Line) Plotting ---
    # This creates a single line that changes color based on the trend state.
    # --- PTL PLOTTING LOGIC ---
    if "ptl" in feature_spec:
        # We only need these columns to plot the flip markers
        ptl_cols = ["ptl_trena", "High", "Low"]
        if all(col in df.columns for col in ptl_cols):

            # Find the exact candle where the trend value changes
            trend_changed = df["ptl_trena"].diff() != 0

            # Calculate the position for the marker where the trend *becomes* blue (UP)
            up_trend_start = np.where(
                (df["ptl_trena"] == 0) & (trend_changed),
                df["Low"] - (df["High"] - df["Low"]).mean() * 0.3,
                np.nan,
            )
            # Calculate the position for the marker where the trend *becomes* red (DOWN)
            down_trend_start = np.where(
                (df["ptl_trena"] == 1) & (trend_changed),
                df["High"] + (df["High"] - df["Low"]).mean() * 0.3,
                np.nan,
            )

            # Plot the up-trend start arrows (^)
            feature_plots.append(
                mpf.make_addplot(
                    up_trend_start,
                    type="scatter",
                    marker="^",
                    color="dodgerblue",
                    markersize=25,
                )
            )
            # Plot the down-trend start arrows (v)
            feature_plots.append(
                mpf.make_addplot(
                    down_trend_start,
                    type="scatter",
                    marker="v",
                    color="crimson",
                    markersize=25,
                )
            )
    # --- ADDED: Stochastic Oscillator Plotting ---
    if has_stoch:
        for stoch in stoch_configs:
            k = int(stoch.get("k_period", 14))
            d = int(stoch.get("d_period", 3))
            s = int(stoch.get("slowing", 3))

            k_col = f"stoch_{k}_{d}_{s}_k"
            d_col = f"stoch_{k}_{d}_{s}_d"

            # Plot %K line
            if k_col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[k_col],
                        panel=stoch_panel,
                        ylabel="Stoch",
                        color="lightseagreen",
                    )
                )
            # Plot %D line
            if d_col in df.columns:
                feature_plots.append(
                    mpf.make_addplot(
                        df[d_col], panel=stoch_panel, color="red", linestyle="--"
                    )
                )

        # Add overbought/oversold lines
        feature_plots.append(
            mpf.make_addplot(
                pd.Series(80, index=df.index),
                panel=stoch_panel,
                color="blue",
                linestyle=":",
                width=0.7,
            )
        )
        feature_plots.append(
            mpf.make_addplot(
                pd.Series(20, index=df.index),
                panel=stoch_panel,
                color="blue",
                linestyle=":",
                width=0.7,
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

    if fill_between_cfg:
        plot_kwargs["fill_between"] = fill_between_cfg

    fig, axes = mpf.plot(df, **plot_kwargs, block=False)
    if axes:
        if use_better_volume and volume_panel > 0 and volume_panel < len(axes):
            axes[volume_panel].set_title("Better Volume", color="white", size=10)

    fig.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_wrapper(args):
    """Helper function to unpack arguments for the executor map."""
    _plot_trades_on_chart(*args)


def _plot_selected_periods(
    df_trades: pd.DataFrame,
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str,
    feature_spec: Dict[str, Any] | None = None,
):
    """
    Plots days/weeks with trade activity and random months, along with close-ups.
    """
    md = market_data.copy()
    if md.index.tz is not None:
        md.index = md.index.tz_localize(None)

    periods_to_plot = {}

    # --- 1. Select days and weeks with active trades ---
    if not df_trades.empty:
        active_days, active_weeks = set(), set()
        for trade in df_trades.itertuples():
            # Generate a date range for the duration of the trade
            trade_duration = pd.date_range(
                start=trade.entry_time, end=trade.exit_time, freq="D"
            )
            active_days.update(trade_duration.to_period("D"))
            active_weeks.update(trade_duration.to_period("W"))

        # From the active periods, select a representative sample to plot
        for periods, name in [
            (sorted(list(active_days)), "day"),
            (sorted(list(active_weeks)), "week"),
        ]:
            if not periods:
                continue

            # Select first, middle, and last periods
            indices_to_get = {0, len(periods) // 2, len(periods) - 1}
            selected = {periods[i] for i in indices_to_get}

            # Add up to 2 random periods for variety
            num_random = min(7, len(periods))
            selected.update(random.sample(periods, num_random))

            for i, p in enumerate(sorted(list(selected))):
                periods_to_plot[f"active_{name}_{i+1}"] = p

    # --- 2. Prepare and execute plotting tasks in parallel ---
    plot_tasks = []
    for p_name, p_val in periods_to_plot.items():
        # Robustly get the period character ('D', 'W', or 'M')
        period_char = p_val.freqstr[0]
        market_data_period = md.loc[p_val.start_time : p_val.end_time]

        if market_data_period.empty:
            continue

        # Filter trades relevant to the selected period
        trades_in_period = pd.DataFrame()
        if not df_trades.empty:
            trades_in_period = df_trades[
                (pd.to_datetime(df_trades["entry_time"]) <= p_val.end_time)
                & (pd.to_datetime(df_trades["exit_time"]) >= p_val.start_time)
            ]

        filename = Path(out_dir) / f"{symbol}_{period_tag}_{p_name}.png"

        # Add the main period plot task
        plot_tasks.append(
            (market_data_period, trades_in_period, str(filename), feature_spec)
        )

        # --- Add close-up plot tasks (e.g., first 6/12 hours of the period) ---
        if period_char in ("D", "W"):  # Only create close-ups for days/weeks
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
                        (pd.to_datetime(trades_in_period["entry_time"]) < end_time)
                        & (pd.to_datetime(trades_in_period["exit_time"]) > start_time)
                    ]

                intraday_filename = f"{base_filename}_{hours}hr.png"
                plot_tasks.append(
                    (market_data_slice, trades_slice, intraday_filename, feature_spec)
                )

    # Execute all plotting tasks in parallel
    if plot_tasks:
        with ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(_plot_wrapper, plot_tasks)


def generate_plots(
    trades: Iterable["Trade"],
    market_data: pd.DataFrame,
    out_dir: str,
    symbol: str,
    period_tag: str | None = None,
    feature_spec: Dict[str, Any] | None = None,
):

    plot_dir = Path(out_dir) / "plots"
    """Generates a comprehensive set of charts for strategy analysis."""
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{period_tag}" if period_tag else "full_period"
    df = trades_to_df(trades)
    _plot_selected_periods(df, market_data, str(plot_dir), symbol, tag, feature_spec)
    print("All plots have been generated.")
