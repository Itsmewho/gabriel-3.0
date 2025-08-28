# backtester

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple
from utils.helpers import setup_logger, green, reset


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


# Strategies to test:
from backtester.strategies.rsi import RSIOscillator
from backtester.strategies.sma import SmaCrossoverSimple
from backtester.strategies.ema import EmaCrossover

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
    feature_spec = {
        "sma": [
            5,
            8,
            12,
            14,
            18,
            20,
            24,
            40,
            50,
            100,
            130,
            150,
            200,
        ],
        "vol_sma": [10, 20, 30, 40, 50],
        "ema": [12, 14, 26, 30],
        "rsi": [14],
    }

    df = ensure_feature_parquet(
        symbol,
        timeframe,
        start_date,
        end_date,
        spec=feature_spec,
        with_events=False,
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

    cfg_map = {
        "EMA_X": StrategyConfig(
            risk_mode=RiskMode.FIXED,
            risk_pct=0.1,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.1,
            max_drawdown_pct=0.3,
            max_concurrent_trades=2,
        ),
    }
    governor = RiskGovernor(cfg_map)

    strategies = [
        EmaCrossover(
            symbol=SYMBOL,
            config={
                "name": "EMA_X",
                "FAST_EMA": 12,
                "SLOW_EMA": 26,
                "SL_PIPS": 12,
                "TP_PIPS": 48,
                "USE_TRAILING_STOP": True,
                "TRAILING_STOP_DISTANCE_PIPS": 12,
                "BE_TRIGGER_EXTRA_PIPS": 1,
                "BE_OFFSET_PIPS": 2,
            },
            strat_cfg=cfg_map["EMA_X"],
            governor=governor,
        )
    ]

    # Allocations
    alloc = cfg.INITIAL_BALANCE
    allocations = {"EMA_FXD": alloc * 1}
    ledger = Ledger(initial_allocations=allocations)

    trade_to_strategy: dict[int, str] = {}

    ## for plotting charts.
    feature_spec = {
        "ema": [14, 26],
        "sma": [150],
    }

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
    # export_account_audit(
    #     ledger.snapshot_df(), f"results/audit/account_ledger_{period_tag}.csv"
    # )
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
    # audit_max_open_trades(
    #     broker.trade_history, f"results/audit/{symbol}_max_open_trades_{period_tag}.csv"
    # )
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

    logger.info(
        green
        + f"Finished period {period_tag} | Final balance: {broker.balance:.2f}"
        + reset
    )


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
        (
            "2009-10-01",
            "2010-10-01",
        ),  # Bank collapse (End periode.  Starts with bull ends with bear)
        ("2014-01-01", "2015-01-01"),  # Brexit  (bear market)
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
