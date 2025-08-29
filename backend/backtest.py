# backtester

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, Tuple
from utils.helpers import setup_logger, green, reset


# Global Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# Auditors
from backtester.account_management.account_audit import export_account_audit
from backtester.broker.audit import (
    audit_trades,
    audit_rejections,
    audit_max_open_trades,  # noqa: F401
)

# Account management
from backtester.account_management import (
    Ledger,
)
from backtester.account_management.types import StrategyConfig, RiskMode
from backtester.account_management.govorner import RiskGovernor


# Strategies to test:
from backtester.strategies.rsi import RSIOscillator  # noqa: F401
from backtester.strategies.sma import SmaCrossoverSimple  # noqa: F401
from backtester.strategies.ema import EmaCrossover

# Loaders
from backtester.features.features_cache import ensure_feature_parquet

# Evals
from backend.backtester.performance.plots_generator import generate_plots
from backend.backtester.performance.md_reports import generate_markdown_report
from backtester.performance.trade_plots import plot_trades
from backtester.performance.trade_svg import export_trades_csv
from backtester.performance.regime_eval import regime_report, trades_to_df

# Broker
from backtester.broker import BrokerConfig
from backtester.broker.main_broker import Broker


# Logger
logger = setup_logger(__name__)


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
            30,
            40,
            50,
            130,
            150,
            200,
            300,
            400,
        ],
        "vol_sma": [10, 20, 30, 40, 50],
        "ema": [
            5,
            8,
            12,
            14,
            18,
            20,
            24,
            30,
            40,
            50,
            130,
            150,
            200,
            300,
            400,
        ],
        "rsi": [14],
        "atr": [14],
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

    # --- This part is the same ---
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
                "SL_PIPS": 10,
                "TP_PIPS": 40,
                "USE_BREAK_EVEN_STOP": False,
                "USE_TRAILING_STOP": False,
                "USE_TP_EXTENSION": False,
            },
            strat_cfg=cfg_map["EMA_X"],
            governor=governor,
        )
    ]

    if not strategies:
        logger.error("No strategies defined for this run. Skipping.")
        return
    strategy_folder_name = strategies[0].name

    base_out_dir = Path(f"results/{strategy_folder_name}")
    audit_dir = base_out_dir / "audit"
    evals_dir = base_out_dir / "evals"
    metrics_dir = base_out_dir / "metrics"
    regime_dir = metrics_dir / "regime"

    audit_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # --- The backtest loop is the same ---
    alloc = cfg.INITIAL_BALANCE
    allocations = {"EMA_FXD": alloc * 1}
    ledger = Ledger(initial_allocations=allocations)
    trade_to_strategy: dict[int, str] = {}
    feature_spec = {"ema": [14, 26], "sma": [150]}

    for ts, row in market_data.iterrows():
        for strat in strategies:
            tr = strat.on_bar(broker, ts, row)
            if tr:
                trade_to_strategy[tr.id] = strat.name
                ledger.on_open(strat.name, ts, trade_id=tr.id)  # type: ignore
        broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)  # type: ignore

    for tr in broker.trade_history:
        sid = trade_to_strategy.get(tr.id, "UNKNOWN")
        ledger.on_close(sid, tr.exit_time, pnl=tr.pnl, trade_id=tr.id)  # type: ignore

    export_account_audit(
        ledger.snapshot_df(), str(audit_dir / f"account_ledger_{period_tag}.csv")
    )
    audit_trades(
        broker.trade_history, str(audit_dir / f"{symbol}_trade_audit_{period_tag}.csv")
    )
    audit_rejections(
        broker.rejections, str(audit_dir / f"{symbol}_rejected_trades_{period_tag}.csv")
    )
    export_trades_csv(
        broker.trade_history, str(evals_dir / f"{symbol}_trade_report_{period_tag}.csv")
    )
    plot_trades(
        market_data,
        broker.trade_history,
        str(evals_dir) / f"results/tradeplots/{symbol}_trade_plot_{period_tag}.png",
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

    trade_df = trades_to_df(broker.trade_history)
    if not trade_df.empty:
        regime_report(
            trades_or_df=trade_df,
            market_data=market_data,
            out_dir=str(regime_dir),
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
        ),  # contains bull / bear and consolidation periodes. (long run)
        ("2025-01-01", "2025-08-29"),  # current
    ]

    SYMBOL, TIMEFRAME = "EURUSD", "1m"
    run_periods(SYMBOL, TIMEFRAME, PERIODS, base_seed=42)
