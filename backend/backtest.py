# backtester

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple
from utils.helpers import setup_logger, green, reset
from concurrent.futures import ProcessPoolExecutor

# Global Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# Auditors
from backtester.account_management.account_audit import export_account_audit
from backtester.broker.audit import audit_trades, audit_rejections, audit_margin_events

# Account management
from backtester.account_management import (
    Ledger,
)
from backtester.account_management.types import StrategyConfig, RiskMode
from backtester.account_management.govorner import RiskGovernor


# Strategies to test:
from backtester.strategies.test_config_strat import RandomEntryStrategyFixed

# Loaders
from backtester.features.features_cache import ensure_feature_parquet

# Evals

from backtester.performance.plots_generator import generate_plots
from backtester.performance.md_reports import generate_markdown_report
from backtester.performance.trade_svg import export_trades_csv
from backtester.performance.regime_eval import regime_report, trades_to_df
from backtester.performance.trade_plots import plot_trades

# Broker
from backtester.broker import BrokerConfig
from backtester.broker.main_broker import Broker


# Logger
logger = setup_logger(__name__)


# --- Data Preparation & Caching  ---


def prepare_data(
    symbol, timeframe, start_date, end_date, force_reload=True
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

    feature_spec = {
        "ema": [300, 500, 1440, 7200],
        "kc": [{"n": 40, "m": 4.0, "atr_n": 40, "ma": "ema"}],
        "atr": [14],
        "rsi": [24],
        "stoch": [
            {"k_period": 5, "d_period": 5, "slowing": 5},
        ],
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

    period_tag = f"{start_date}_{end_date}"
    market_data = prepare_data(symbol, timeframe, start_date, end_date)
    if market_data.empty:
        logger.error(f"No market data for period {period_tag}. Skipping.")
        return

    STRATEGY_NAME = "TEST"
    feature_spec = {
        "ema": [300, 500, 1440, 7200],
        "kc": [{"n": 40, "m": 4.0, "atr_n": 40, "ma": "ema"}],
        "atr": [14],
        "rsi": [24],
        "stoch": [
            {"k_period": 5, "d_period": 5, "slowing": 5},
        ],
    }
    cfg = BrokerConfig(**BACKTEST_CONFIG)
    broker = Broker(cfg)

    cfg_map = {
        STRATEGY_NAME: StrategyConfig(
            risk_mode=RiskMode.FIXED,
            risk_pct=0.01,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.01,
            max_drawdown_pct=0.30,
            max_concurrent_trades=100,
        )
    }
    governor = RiskGovernor(cfg_map)

    strategies = [
        RandomEntryStrategyFixed(
            symbol=symbol,
            config={
                "name": STRATEGY_NAME,
                "EVERY_N_MINUTES": 900,
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
            strat_cfg=cfg_map[STRATEGY_NAME],
            governor=governor,
        )
    ]
    if not strategies:
        logger.error("No strategies defined for this run. Skipping.")
        return

    # --- Setup: Dynamic Directory Creation ---
    strategy_folder_name = strategies[0].name
    base_out_dir = Path(f"results/{strategy_folder_name}")
    audit_dir = base_out_dir / "audit"
    evals_dir = base_out_dir / "evals"
    metrics_dir = base_out_dir / "metrics"
    regime_dir = metrics_dir / "regime"

    audit_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    regime_dir.mkdir(parents=True, exist_ok=True)

    # --- Backtest loop ---
    alloc = cfg.INITIAL_BALANCE
    allocations = {STRATEGY_NAME: alloc}
    ledger = Ledger(initial_allocations=allocations)
    trade_to_strategy: dict[int, str] = {}
    last_closed_idx = 0

    # Seed governor with initial equity so drawdown logic has a baseline
    if governor:
        for sid, eq0 in allocations.items():
            governor.on_equity(sid, eq0)

    for ts, row in market_data.iterrows():
        # 1) Strategy decisions
        for strat in strategies:
            tr = strat.on_bar(broker, ts, row)
            if tr:
                trade_to_strategy[tr.id] = strat.name
                ledger.on_open(strat.name, ts, trade_id=tr.id)
                if governor:
                    governor.on_open(strat.name)

        # 2) Broker lifecycle on this bar
        broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)

        # 3) Book any newly closed trades immediately and update governor/ledger
        if last_closed_idx < len(broker.trade_history):
            for trc in broker.trade_history[last_closed_idx:]:
                sid = trade_to_strategy.get(trc.id, STRATEGY_NAME)
                ledger.on_close(sid, trc.exit_time, pnl=trc.pnl, trade_id=trc.id)
                if governor:
                    governor.on_close(sid)
                    governor.on_equity(sid, ledger.equity(sid))
            last_closed_idx = len(broker.trade_history)

    # --- Reports Section ---
    trade_df = trades_to_df(broker.trade_history)

    dd_thresholds = {STRATEGY_NAME: cfg_map[STRATEGY_NAME].max_drawdown_pct}
    resume_thresholds = {
        STRATEGY_NAME: getattr(cfg_map[STRATEGY_NAME], "max_drawdown_resume_pct", 0.90)
    }

    export_account_audit(
        ledger.snapshot_df(),
        str(audit_dir / f"account_ledger_{period_tag}.csv"),
        dd_thresholds=dd_thresholds,
        resume_thresholds=resume_thresholds,
    )
    audit_trades(
        broker.trade_history, str(audit_dir / f"{symbol}_trade_audit_{period_tag}.csv")
    )
    audit_rejections(
        broker.rejections, str(audit_dir / f"{symbol}_rejected_trades_{period_tag}.csv")
    )
    audit_margin_events(
        broker.events_log, str(audit_dir / f"{symbol}_margin_events_{period_tag}.csv")
    )
    export_trades_csv(
        broker.trade_history, str(evals_dir / f"{symbol}_trade_report_{period_tag}.csv")
    )

    if not trade_df.empty:
        regime_report(
            trades_or_df=trade_df,
            market_data=market_data,
            out_dir=str(regime_dir),
            symbol=symbol,
            period_tag=period_tag,
        )

    generate_markdown_report(
        trades=broker.trade_history,
        initial_balance=cfg.INITIAL_BALANCE,
        out_dir=str(metrics_dir),
        symbol=symbol,
        period_tag=period_tag,
    )

    if market_data is not None:
        generate_plots(
            trades=broker.trade_history,
            market_data=market_data,
            out_dir=str(metrics_dir),
            symbol=symbol,
            period_tag=period_tag,
            feature_spec=feature_spec,
        )

    plot_trades(
        market_data,
        broker.trade_history,
        str(metrics_dir / f"{symbol}_trade_plot_{period_tag}.png"),
    )

    logger.info(
        green
        + f"Finished period {period_tag} | Final balance: {broker.balance:.2f}"
        + reset
    )


# --- Multi-period driver  ---


def run_periods(
    symbol: str,
    timeframe: str,
    periods: Iterable[Tuple[str, str]],
    base_seed: int = 42,
    max_workers: int = 8,
) -> None:
    """Runs multiple backtest periods in parallel."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (start_date, end_date) in enumerate(periods):
            seed = None if base_seed is None else base_seed + i
            # Submit each run_period call to the executor
            future = executor.submit(
                run_period, symbol, timeframe, start_date, end_date, seed=seed
            )
            futures.append(future)
            logger.info(f"Submitted backtest for period {start_date} to {end_date}")

        # Wait for all futures to complete and handle any exceptions
        for future in futures:
            try:
                future.result()  # This will raise any exceptions from the worker process
            except Exception as e:
                logger.error(f"A backtest period failed: {e}", exc_info=True)


if __name__ == "__main__":
    # Define your periods here
    PERIODS = [
        ("2010-06-04", "2010-09-09"),  # Testing periode (volitile)
        ("2014-06-29", "2015-01-01"),  # Testing periode (down )
        ("2009-10-01", "2010-10-01"),  # Bank collapse
        ("2014-01-01", "2015-01-01"),  # Brexit (bear market)
        ("2017-04-01", "2018-04-01"),  # Bull market
        ("2021-05-01", "2022-10-15"),  # Bear (covid)
        ("2023-02-01", "2024-09-02"),  # Consolidation (post-covid)
        ("2025-01-01", "2025-09-04"),  # Current year
        ("2023-09-01", "2025-09-04"),  # last 2 years
        ("2014-01-01", "2015-01-01"),  # Long run (mixed)
        ("2015-01-01", "2016-01-01"),  # Long run (mixed)
        ("2016-01-01", "2017-01-01"),  # Long run (mixed)
        ("2017-01-01", "2018-01-01"),  # Long run (mixed)
        ("2018-01-01", "2019-01-01"),  # Long run (mixed)
        ("2019-01-01", "2020-01-01"),  # Long run (mixed)
        ("2020-01-01", "2021-01-01"),  # Long run (mixed)
        ("2021-01-01", "2022-01-01"),  # Long run (mixed)
        ("2022-01-01", "2023-01-01"),  # Long run (mixed)
    ]

    SYMBOL, TIMEFRAME = "EURUSD", "1m"
    run_periods(SYMBOL, TIMEFRAME, PERIODS, base_seed=42)
