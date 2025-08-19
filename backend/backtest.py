from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional
from utils.helpers import setup_logger


# Logger
logger = setup_logger(__name__)

# Auditor
from backtester.broker.audit import audit_trades, audit_rejections

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

    for ts, row in market_data.iterrows():
        # Manual long: 2023-08-09 09:00 local (naive)
        if (
            not did_manual_long
            and ts.date() == pd.to_datetime("2023-08-09").date()  # type: ignore
            and ts.hour == 9  # type: ignore
            and ts.minute == 0  # type: ignore
        ):
            broker.open_trade(
                "buy", float(row["close"]), lots=0.1, sl_pips=100, tp_pips=100, t=ts  # type: ignore
            )
            did_manual_long = True

        # Margin dummy: 2023-08-10 09:00 local
        if (
            not did_margin_dummy
            and ts.date() == pd.to_datetime("2023-08-10").date()  # type: ignore
            and ts.hour == 4  # type: ignore
            and ts.minute == 0  # type: ignore
        ):
            broker.open_trade(
                "sell", float(row["close"]), lots=500.0, sl_pips=5000, tp_pips=5000, t=ts  # type: ignore
            )
            did_margin_dummy = True

        # Weekend trade open: Friday 16:00
        if (
            not did_weekend_dummy
            and ts.weekday() == 4  # type: ignore
            and ts.hour == 16  # type: ignore
            and ts.minute == 0  # type: ignore
        ):
            broker.open_trade(
                "buy", float(row["close"]), lots=0.01, sl_pips=5000, tp_pips=5000, t=ts  # type: ignore
            )
            weekend_trade_id = broker.open_trades[-1].id
            did_weekend_dummy = True

        broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)  # type: ignore

        # Close weekend trade: Wednesday 12:00 (weekday 2)
        if (
            weekend_trade_id is not None
            and ts.weekday() == 2  # type: ignore
            and ts.hour == 8  # type: ignore
            and ts.minute == 0  # type: ignore
        ):
            for tr in list(broker.open_trades):
                if tr.id == weekend_trade_id:
                    broker.close_trade(
                        tr, float(row["close"]), "Manual Close (Wed Noon)", ts  # type: ignore
                    )
                    weekend_trade_id = None
                    break

        from collections import Counter

    broker.close_all(float(market_data.iloc[-1]["close"]), market_data.index[-1])

    # Reports
    audit_trades(broker.trade_history, f"results/audit/{SYMBOL}_trade_audit.csv")
    audit_rejections(broker.rejections, f"results/audit/{SYMBOL}_rejected_trades.csv")
    plot_trades(
        market_data, broker.trade_history, f"results/tradeplots/{SYMBOL}_trade_plot.png"
    )
    export_trades_csv(broker.trade_history, f"results/evals/{SYMBOL}_trade_report.csv")
