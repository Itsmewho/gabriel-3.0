from __future__ import annotations
import pandas as pd
import mplfinance as mpf
import numpy as np
from pathlib import Path
from typing import Iterable, Mapping, Any


def _ensure_ohlc_columns(
    df: pd.DataFrame, columns: Mapping[str, str] | None = None
) -> pd.DataFrame:
    dfc = df.copy()
    want = ["Open", "High", "Low", "Close"]
    if all(c in dfc.columns for c in want):
        return dfc
    auto = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vol": "Volume",
    }
    mapping = {} if columns is None else dict(columns)
    if columns is None:
        for k, v in auto.items():
            if k in dfc.columns and v not in dfc.columns:
                mapping[k] = v
    if mapping:
        dfc = dfc.rename(columns=mapping)
    missing = [c for c in want if c not in dfc.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns after mapping: {missing}")
    return dfc


def plot_trades(
    market_data: pd.DataFrame,
    trades_or_events: Iterable[Any],
    filename: str = "results/tradeplots/trade_plot.png",
    columns: Mapping[str, str] | None = None,
    markersize: int = 30,
    warn_cap: int | None = None,
    fig_dpi: int = 300,
):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # --- normalize market data index ---
    df = market_data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:  # type: ignore
        df.index = df.index.tz_convert("UTC").tz_localize(None)  # type: ignore

    df = _ensure_ohlc_columns(df, columns)

    # --- helpers ---
    def _norm_ts(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.to_datetime(ts)
        if getattr(ts, "tzinfo", None) is not None:
            try:
                ts = ts.tz_convert("UTC")
            except Exception:
                pass
            ts = ts.tz_localize(None)
        return ts

    def _price_at(ts: pd.Timestamp, key: str) -> float | None:
        ts = _norm_ts(ts)
        if ts in df.index:
            return float(df.loc[ts, key])  # type: ignore[index]
        pos = df.index.get_indexer([ts], method="nearest")[0]
        if pos == -1:
            return None
        return float(df.iloc[pos][key])

    def _nearest(ts: pd.Timestamp) -> pd.Timestamp:
        ts = _norm_ts(ts)
        pos = df.index.get_indexer([ts], method="nearest")[0]
        return df.index[pos]  # type: ignore

    # --- series for markers ---
    buy = pd.Series(np.nan, index=df.index, dtype="float64")
    sell = pd.Series(np.nan, index=df.index, dtype="float64")
    exitp = pd.Series(np.nan, index=df.index, dtype="float64")

    # Entries
    for it in trades_or_events:
        if isinstance(it, dict):
            ts = _norm_ts(it.get("time"))  # type: ignore
            et = it.get("type")
            side = it.get("side")
            if et == "open" and side == "buy":
                p = _price_at(ts, "Low")
                buy.loc[_nearest(ts)] = p * 0.999 if p is not None else np.nan  # type: ignore
            elif et == "open" and side == "sell":
                p = _price_at(ts, "High")
                sell.loc[_nearest(ts)] = p * 1.001 if p is not None else np.nan  # type: ignore
        else:
            ts = _norm_ts(getattr(it, "entry_time", None))  # type: ignore
            side = getattr(it, "side", None)
            if side == "buy":
                p = _price_at(ts, "Low")
                buy.loc[_nearest(ts)] = p * 0.999 if p is not None else np.nan  # type: ignore
            elif side == "sell":
                p = _price_at(ts, "High")
                sell.loc[_nearest(ts)] = p * 1.001 if p is not None else np.nan  # type: ignore

    # Exits
    for it in trades_or_events:
        if isinstance(it, dict):
            if it.get("type") == "close":
                ts = _norm_ts(it.get("time"))  # type: ignore
                px = it.get("price")
                if px is not None:
                    exitp.loc[_nearest(ts)] = float(px)  # type: ignore
        else:
            ts_exit = getattr(it, "exit_time", None)
            if ts_exit is not None:
                ts = _norm_ts(ts_exit)
                px = getattr(it, "exit_price", None)
                if px is not None:
                    exitp.loc[_nearest(ts)] = float(px)  # type: ignore
                else:
                    c = _price_at(ts, "Close")
                    if c is not None:
                        exitp.loc[_nearest(ts)] = c  # type: ignore

    # Entry→exit line segments
    segments: list[list[tuple[pd.Timestamp, float]]] = []
    seg_colors: list[str] = []

    for it in trades_or_events:
        if isinstance(it, dict) or getattr(it, "exit_time", None) is None:
            continue

        side = getattr(it, "side", None)
        ts_entry = _nearest(getattr(it, "entry_time"))
        ts_exit = _nearest(getattr(it, "exit_time"))

        if side == "buy":
            p_entry = _price_at(ts_entry, "Low")
            y_entry = p_entry * 0.999 if p_entry is not None else None
        elif side == "sell":
            p_entry = _price_at(ts_entry, "High")
            y_entry = p_entry * 1.001 if p_entry is not None else None
        else:
            continue

        y_exit = getattr(it, "exit_price", None)
        if y_exit is None:
            y_exit = _price_at(ts_exit, "Close")

        if y_entry is None or y_exit is None:
            continue

        segments.append([(ts_entry, float(y_entry)), (ts_exit, float(y_exit))])
        prof = (y_exit > y_entry) if side == "buy" else (y_exit < y_entry)
        seg_colors.append("green" if prof else "red")

    # Build overlays only if they have data
    def _nonempty(s: pd.Series) -> bool:
        v = s.values
        return np.isfinite(v).any()  # type: ignore

    plots: list[Any] = []
    if _nonempty(buy):
        plots.append(
            mpf.make_addplot(
                buy, type="scatter", marker="^", color="green", markersize=markersize
            )
        )
    if _nonempty(sell):
        plots.append(
            mpf.make_addplot(
                sell, type="scatter", marker="v", color="red", markersize=markersize
            )
        )
    if _nonempty(exitp):
        plots.append(
            mpf.make_addplot(
                exitp, type="scatter", marker="x", color="blue", markersize=markersize
            )
        )

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
        addplot=plots if plots else None,
        figscale=1.4,
        tight_layout=True,
        warn_too_much_data=wtd,
        savefig=dict(fname=filename, dpi=fig_dpi),
        **aline_kwargs,
    )
    print(f"PNG trade-plot report saved to {filename}")
    return filename
