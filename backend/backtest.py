# backtester

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple
from utils.helpers import setup_logger


# Logger
logger = setup_logger(__name__)

# Global Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# Auditors
# from backtester.account_management.account_audit import export_account_audit
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


# Strategies to test:
from backtester.strategies.sma import SmaConfluenceStrategy

# Loaders
from backtester.features.features_cache import ensure_feature_parquet

# Evals
from backtester.performance.evaluation import trades_to_df
from backtester.performance.trade_plots import plot_trades
from backtester.performance.trade_svg import export_trades_csv
from backtester.performance.evaluation import evaluate
from backtester.performance.regime_eval import regime_report

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

    if force_reload and data_file.exists():
        try:
            data_file.unlink()
            logger.info(f"Force reload: deleted {data_file}")
        except Exception as e:
            logger.warning(f"Could not delete cache {data_file}: {e}")

    # Ask for only what this run needs; missing cols are appended to the same parquet later.
    feature_spec = {"sma": [12, 20, 50, 150], "vol_sma": [20]}

    df = ensure_feature_parquet(
        symbol,
        timeframe,
        start_date,
        end_date,
        spec=feature_spec,
        with_events=True,
        cache_dir=str(cache_dir),
    )

    if df.empty:
        logger.error("Failed to fetch market data. Aborting.")
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" in df.columns:
            df = df.set_index("time")
    df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    df = df.dropna().sort_index()

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

    # # Risk and strategy config (keys match strategy names)
    cfg_map = {
        "SMA_KELLY": StrategyConfig(
            risk_mode=RiskMode.HALF_KELLY,
            risk_pct=0.01,
            kelly_p=0.53,
            kelly_rr=1.6,
            kelly_cap_pct=0.02,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.02,
            max_concurrent_trades=5,
        ),
        "SMA_FIXED": StrategyConfig(
            risk_mode=RiskMode.FIXED,
            risk_pct=0.01,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.01,
            max_concurrent_trades=5,
        ),
    }

    governor = RiskGovernor(cfg_map)

    strategies = [
        SmaConfluenceStrategy(
            symbol=symbol,
            config={
                "name": "SMA_KELLY",
                "FAST_MA": 12,
                "SHORT_MA": 20,
                "MEDIUM_MA": 50,
                "SLOW_MA": 150,
                # "SL_PIPS": 20,
                # "TP_PIPS": 40,
                "VOLUME_MA": 20,
                "VOLUME_FACTOR": 1.5,  # (volume must be 50% above average)
                "USE_BREAK_EVEN_STOP": False,
                "BE_TRIGGER_PIPS": 10,
                "BE_OFFSET_PIPS": 2,
                "USE_TRAILING_STOP": False,
                "TRAILING_STOP_DISTANCE_PIPS": 10,
                "USE_TP_EXTENSION": False,
                "NEAR_TP_BUFFER_PIPS": 2,
                "TP_EXTENSION_PIPS": 3,
            },
            strat_cfg=cfg_map["SMA_KELLY"],
            governor=governor,
        ),
        SmaConfluenceStrategy(
            symbol=symbol,
            config={
                "name": "SMA_FIXED",
                "FAST_MA": 12,
                "SHORT_MA": 20,
                "MEDIUM_MA": 50,
                "SLOW_MA": 150,
                "SL_PIPS": 20,
                "TP_PIPS": 40,
                "VOLUME_MA": 20,
                "VOLUME_FACTOR": 1.5,  # (volume must be 50% above average)
                "USE_BREAK_EVEN_STOP": False,
                "BE_TRIGGER_PIPS": 10,
                "BE_OFFSET_PIPS": 2,
                "USE_TRAILING_STOP": False,
                "TRAILING_STOP_DISTANCE_PIPS": 10,
                "USE_TP_EXTENSION": True,
                "NEAR_TP_BUFFER_PIPS": 2,
                "TP_EXTENSION_PIPS": 3,
            },
            strat_cfg=cfg_map["SMA_FIXED"],
            governor=governor,
        ),
    ]

    # Allocations
    alloc = cfg.INITIAL_BALANCE
    allocations = {"SMA_KELLY": alloc * 0.5, "SMA_FIXED": alloc * 0.5}
    ledger = Ledger(initial_allocations=allocations)

    trade_to_strategy: dict[int, str] = {}

    ## for plotting charts.
    feature_spec = {"sma": [12, 20, 50, 150]}

    # --- Backtest loop ---
    for ts, row in market_data.iterrows():
        for strat in strategies:
            tr = strat.on_bar(broker, ts, row)  # type: ignore
            if tr:
                trade_to_strategy[tr.id] = strat.name
                ledger.on_open(strat.name, ts, trade_id=tr.id)  # type: ignore
        broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)  # type: ignore

    # Attribute closed PnL back to strategies in the ledger
    for tr in broker.trade_history:
        sid = trade_to_strategy.get(tr.id, "UNKNOWN")
        ledger.on_close(sid, tr.exit_time, pnl=tr.pnl, trade_id=tr.id)  # type: ignore

    # --- Reports (period-tagged) ---
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
        market_data=market_data,
        strategies=strategies,
        feature_spec=feature_spec,
    )
    trade_df = trades_to_df(broker.trade_history)
    regime_report(
        trade_df,
        market_data,
        out_dir="results/metrics/regime",
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
        ("2008-05-01", "2009-02-09"),  # Bank collapse
        ("2017-04-01", "2018-04-01"),  # bull market
        ("2021-05-01", "2022-10-15"),  # bear (covid bullshit)
        ("2023-02-01", "2024-09-02"),  # consolidation periode ( after covid )
        (
            "2014-11-01",
            "2024-11-01",
        ),  # contains bull / bear and consolidation periodes.
    ]

    SYMBOL, TIMEFRAME = "EURUSD", "1m"
    run_periods(SYMBOL, TIMEFRAME, PERIODS, base_seed=42)
