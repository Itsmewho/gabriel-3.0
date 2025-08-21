from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from utils.helpers import setup_logger


# Logger
logger = setup_logger(__name__)

# Auditor
from backtester.broker.audit import (
    audit_trades,
    audit_rejections,
    audit_max_open_trades,
)

# Global Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# TODO: add a config for strategies ( so each stat can have his own trading params)

# Loaders
from backtester.data.loaders import fetch_event_features
from backtester.data.loaders import fetch_sql_market_data

# Evals
from backtester.performance.trade_plots import plot_trades
from backtester.performance.trade_svg import export_trades_csv

# Broker
from backtester.broker import BrokerConfig
from backtester.broker.main_broker import Broker


# --- Data Preparation & Caching  ---


def prepare_data(
    symbol, timeframe, start_date, end_date, force_reload=False
) -> pd.DataFrame:
    cache_dir = Path("results/cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    data_file = (
        cache_dir / f"{symbol}_{timeframe}_{start_date}_{end_date}_features.parquet"
    )

    if data_file.exists() and not force_reload:
        logger.info(f"Loading cached feature data from {data_file}...")
        return pd.read_parquet(data_file)

    logger.info("No cache found or reload forced. Fetching and processing new data...")
    where_clause = f"time >= '{start_date}' AND time <= '{end_date}'"
    df = fetch_sql_market_data(symbol, timeframe, where_clause)

    if df.empty:
        logger.error("Failed to fetch market data. Aborting.")
        return pd.DataFrame()

    # Normalize types and index. Use server time (no UTC conversion).
    df["time"] = pd.to_datetime(df["time"], utc=False, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")
    # Keep only what you need for now
    df = df[["open", "high", "low", "close", "tick_volume"]]

    # Cache
    df.to_parquet(data_file)
    logger.info(f"Saving features to cache file: {data_file}")
    return df


if __name__ == "__main__":
    np.random.seed(42)

    START_DATE, END_DATE = "2023-08-01", "2023-08-31"
    SYMBOL, TIMEFRAME = "EURUSD", "1m"

    market_data = prepare_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if market_data.empty:
        raise SystemExit("No market data fetched")

    Path("results/tradeplots").mkdir(parents=True, exist_ok=True)
    Path("results/evals").mkdir(parents=True, exist_ok=True)
    Path("results/audit").mkdir(parents=True, exist_ok=True)

    cfg = BrokerConfig(**BACKTEST_CONFIG)
    broker = Broker(cfg)

    did_manual_long = False
    did_margin_dummy = False
    did_weekend_dummy = False
    weekend_trade_id = None
    # before the loop
    did_be_1 = False
    did_be_2 = False
    did_be_3 = False

    for ts, row in market_data.iterrows():
        px = float(row["close"])

        # --- Example 1: once at a specific time ---
        if (
            not did_manual_long
            and ts.date() == pd.to_datetime("2023-08-09").date()  # type: ignore
            and ts.hour == 9  # type: ignore
            and ts.minute == 0  # type: ignore
        ):
            tr = broker.open_trade(
                "sell",
                float(row["close"]),
                wanted_lots=50,
                sl_pips=50,
                tp_pips=50,
                t=ts,  # type: ignore
            )
            if tr:
                did_manual_long = True  # ← correct flag here

        if (
            not did_be_1  # type: ignore
            and ts.date() == pd.to_datetime("2023-08-09").date()  # type: ignore
            and ts.hour == 9  # type: ignore
            and ts.minute == 0  # type: ignore
        ):
            tr = broker.open_trade(
                "sell",
                px,
                wanted_lots=1.0,
                sl_pips=50,
                tp_pips=50,
                t=ts,  # type: ignore
                fallbacks=[0.5, 0.25],
            )
            if tr:
                broker.set_break_even(tr.id, be_pips=8)  # 8 pips to BE
            did_be_1 = False

        # --- Example 2: weekend trade once at Fri 16:00 ---
        if not did_be_2 and ts.weekday() == 4 and ts.hour == 16 and ts.minute == 0:  # type: ignore
            tr = broker.open_trade(
                "buy",
                px,
                wanted_lots=0.01,
                sl_pips=5000,
                tp_pips=5000,
                t=ts,  # type: ignore
                fallbacks=[0.35, 0.15],
            )
            if tr:
                broker.set_break_even(tr.id, be_pips=5)  # 5 pips to BE
                did_be_2 = False

        # --- Example 3: 6-hour strategy once at first 6h tick after start ---
        if not did_be_3 and ts.hour % 6 == 0 and ts.minute == 0:  # type: ignore
            tr = broker.open_trade(
                side="buy" if np.random.rand() > 0.5 else "sell",
                price=px,
                wanted_lots=0.50,
                sl_pips=10,
                tp_pips=30,
                t=ts,  # type: ignore
                fallbacks=[0.35, 0.15],
            )
            if tr:
                broker.set_break_even(tr.id, be_pips=6)  # 6 pips to BE
                did_be_3 = False

        # normal bar handling
        broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)  # type: ignore

    # Reports
    audit_trades(broker.trade_history, f"results/audit/{SYMBOL}_trade_audit.csv")
    audit_rejections(broker.rejections, f"results/audit/{SYMBOL}_rejected_trades.csv")
    plot_trades(
        market_data, broker.trade_history, f"results/tradeplots/{SYMBOL}_trade_plot.png"
    )
    export_trades_csv(broker.trade_history, f"results/evals/{SYMBOL}_trade_report.csv")
    audit_max_open_trades(
        broker.trade_history, f"results/audit/{SYMBOL}_max_open_trades.csv"
    )
