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
# Dummy  strategies
from backtester.strategies.test_config_strat import (
    RandomEntryStrategyConfig,
    RandomEntryStrategyFixed,
)

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

    first_ts = market_data.index[0]
    first_px = float(market_data.iloc[0]["close"])

    lot_sizes = [
        0.50,
        0.35,
        0.20,
        0.10,
    ]  # different sizes → should margin-call biggest loss first
    opened_ids = []
    for lots in lot_sizes:
        tr = broker.open_trade(
            side="buy",
            price=first_px,
            wanted_lots=lots,
            sl_pips=10000,  # very wide: ensure only margin closes them
            tp_pips=10000,
            t=first_ts,
            fallbacks=[],  # no fallbacks; if one size fails, we learn from rejection
        )
        if tr:
            opened_ids.append((tr.id, lots))
    print(f"Opened trades: {opened_ids}")

    # 2) Drive price sharply down to trigger margin calls
    steps = 80  # tune if needed
    shock_px = first_px
    for s in range(steps):
        shock_px -= 0.0010  # 10 pips down per step
        ts = first_ts + pd.Timedelta(minutes=s + 1)
        broker.on_bar(
            high=shock_px + 0.00005,
            low=shock_px - 0.00005,
            close=shock_px,
            t=ts,
        )

    # 3) Close any leftovers at final shock price
    broker.close_all(shock_px, first_ts + pd.Timedelta(minutes=steps + 1))

    # Instantiate both strategies
    strategies = [
        RandomEntryStrategyConfig(
            symbol=SYMBOL,
            config={
                "EVERY_N_MINUTES": 30,
                "LOTS": 0.15,
                "SL_PIPS": 18,
                "TP_PIPS": 27,
                "USE_BREAK_EVEN_STOP": True,
                "BE_TRIGGER_PIPS": 8,
                "BE_OFFSET_PIPS": 1,
                "USE_TRAILING_STOP": True,
                "TRAILING_STOP_DISTANCE_PIPS": 10,
                "USE_TP_EXTENSION": True,
                "NEAR_TP_BUFFER_PIPS": 2,
                "TP_EXTENSION_PIPS": 3,
                "FALLBACK_LOTS": [0.10, 0.05],
            },
        ),
        RandomEntryStrategyFixed(symbol=SYMBOL),
    ]

    for ts, row in market_data.iterrows():
        # feed each strategy
        for strat in strategies:
            strat.on_bar(broker, ts, row)
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
