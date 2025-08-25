# backtester

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple
from concurrent.futures import ProcessPoolExecutor
from utils.helpers import setup_logger


# Logger
logger = setup_logger(__name__)

# Global Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# Auditors
from backtester.account_management.account_audit import export_account_audit
from backtester.broker.audit import (
    audit_trades,
    audit_rejections,
    audit_max_open_trades,
)

# Account management
from backtester.account_management import (
    Ledger,
)
from backtester.account_management.types import StrategyConfig, RiskMode
from backtester.account_management.govorner import RiskGovernor

# TODO: add a config for strategies ( so each stat can have his own trading params)
# Dummy  strategies
from backtester.strategies.test_config_strat import (
    RandomEntryStrategyConfig,
    RandomEntryStrategyFixed,
)

# Loaders
from backtester.data.loaders import fetch_sql_market_data

# Evals
from backtester.performance.trade_plots import plot_trades
from backtester.performance.trade_svg import export_trades_csv
from backtester.performance.evaluation import evaluate

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


# --- Single period run  ---


def run_period(
    symbol: str, timeframe: str, start_date: str, end_date: str, seed: int | None = 42
) -> None:
    if seed is not None:
        np.random.seed(seed)

    # Ensure output dirs
    Path("results/tradeplots").mkdir(parents=True, exist_ok=True)
    Path("results/evals").mkdir(parents=True, exist_ok=True)
    Path("results/audit").mkdir(parents=True, exist_ok=True)

    period_tag = f"{start_date}_{end_date}"

    market_data = prepare_data(symbol, timeframe, start_date, end_date)
    if market_data.empty:
        logger.error(f"No market data for period {period_tag}. Skipping.")
        return

    cfg = BrokerConfig(**BACKTEST_CONFIG)
    broker = Broker(cfg)

    # --- Risk and strategy config ---
    cfg_map = {
        "RAND_CFG": StrategyConfig(
            risk_mode=RiskMode.HALF_KELLY,
            risk_pct=0.01,
            kelly_p=0.53,
            kelly_rr=1.6,
            kelly_cap_pct=0.02,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.02,
            max_concurrent_trades=1,
        ),
        "RAND_FIX": StrategyConfig(
            risk_mode=RiskMode.FIXED,
            risk_pct=0.01,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.02,
            max_concurrent_trades=1,
        ),
    }

    governor = RiskGovernor(cfg_map)

    strategies = [
        RandomEntryStrategyConfig(
            symbol=symbol,
            config={
                "name": "RAND_CFG",
                "EVERY_N_MINUTES": 30,
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
            },
            strat_cfg=cfg_map["RAND_CFG"],
            governor=governor,
        ),
        RandomEntryStrategyFixed(
            symbol=symbol,
            config={"name": "RAND_FIX"},
            strat_cfg=cfg_map["RAND_FIX"],
            governor=governor,
        ),
    ]

    # Initial allocations per strategy
    alloc = cfg.INITIAL_BALANCE
    allocations = {"RAND_CFG": alloc * 0.5, "RAND_FIX": alloc * 0.5}
    ledger = Ledger(initial_allocations=allocations)

    trade_to_strategy: dict[int, str] = {}

    # --- Backtest loop ---
    for ts, row in market_data.iterrows():
        for strat in strategies:
            tr = strat.on_bar(broker, ts, row)
            if tr:
                trade_to_strategy[tr.id] = strat.name
                ledger.on_open(strat.name, ts, trade_id=tr.id)  # type: ignore
        broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)  # type: ignore

    # Attribute closed PnL back to strategies in the ledger
    for tr in broker.trade_history:
        sid = trade_to_strategy.get(tr.id, "UNKNOWN")
        ledger.on_close(sid, tr.exit_time, pnl=tr.pnl, trade_id=tr.id)  # type: ignore

    # --- Reports (period-tagged) ---
    export_account_audit(
        ledger.snapshot_df(), f"results/audit/account_ledger_{period_tag}.csv"
    )
    audit_trades(
        broker.trade_history, f"results/audit/{symbol}_trade_audit_{period_tag}.csv"
    )
    audit_rejections(
        broker.rejections, f"results/audit/{symbol}_rejected_trades_{period_tag}.csv"
    )
    plot_trades(
        market_data,
        broker.trade_history,
        f"results/tradeplots/{symbol}_trade_plot_{period_tag}.png",
    )
    export_trades_csv(
        broker.trade_history, f"results/evals/{symbol}_trade_report_{period_tag}.csv"
    )
    audit_max_open_trades(
        broker.trade_history, f"results/audit/{symbol}_max_open_trades_{period_tag}.csv"
    )
    evaluate(
        broker.trade_history,
        initial_balance=cfg.INITIAL_BALANCE,
        out_dir="results/metrics",
        symbol=symbol,
        period_tag=period_tag,
    )

    logger.info(f"Finished period {period_tag} | Final balance: {broker.balance:.2f}")


# --- Multi-period driver  ---


def run_periods(
    symbol: str, timeframe: str, periods: Iterable[Tuple[str, str]], base_seed: int = 42
) -> None:
    for i, (start_date, end_date) in enumerate(periods):

        seed = None if base_seed is None else base_seed + i
        run_period(symbol, timeframe, start_date, end_date, seed=seed)


if __name__ == "__main__":
    # Define your periods here
    PERIODS = [
        ("2023-08-01", "2023-08-31"),  # period 1
        ("2023-09-01", "2023-09-30"),  # period 2
        ("2023-10-01", "2023-10-31"),  # period 3
        ("2023-11-01", "2023-11-29"),  # period 4
        ("2025-08-01", "2025-08-25"),  # period 5
    ]

    SYMBOL, TIMEFRAME = "EURUSD", "1m"
    run_periods(SYMBOL, TIMEFRAME, PERIODS, base_seed=42)
