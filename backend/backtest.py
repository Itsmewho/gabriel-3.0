from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional
from utils.helpers import setup_logger

# Logger
logger = setup_logger(__name__)

# Global Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# TODO: add a config for strategies ( so each stat can have his own trading params)

# Loaders
from backtester.data.loaders import fetch_event_features
from backtester.data.loaders import fetch_sql_market_data

# Evals
from backtester.performance.trade_plots import plot_trades
from backtester.performance.trade_svg import export_trades_csv

# order executions
from backtester.order_executions.main_order import OrderEngine


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
    # --- Setup ---
    START_DATE = "2023-08-01"
    END_DATE = "2023-08-31"
    SYMBOL = "EURUSD"
    TIMEFRAME = "1m"

    # --- Data Loading ---
    market_data = prepare_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)

    # --- Backtest ---
    engine = OrderEngine(BACKTEST_CONFIG)

    # Simple "trade every 6 hours" strategy
    for timestamp, row in market_data.iterrows():
        if timestamp.hour % 6 == 0 and timestamp.minute == 0 and not engine.open_trades:  # type: ignore
            if np.random.rand() > 0.5:
                engine.open_trade("buy", row["close"], 0.5, 50, 500, timestamp)  # type: ignore
            else:
                engine.open_trade("sell", row["close"], 0.1, 50, 100, timestamp)  # type: ignore

        engine.on_bar(row["high"], row["low"], row["close"], timestamp)  # type: ignore

    # --- Reporting ---
    engine.close_all_open_trades(market_data.iloc[-1]["close"], market_data.index[-1])

    plot_trades(market_data, engine.trade_history, f"results/tradeplots/{SYMBOL}_trade_plot.png")  # type: ignore
    export_trades_csv(engine.trade_history, f"results/evals/{SYMBOL}_trade_report.csv")
