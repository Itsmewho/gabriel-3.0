from __future__ import annotations
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, Optional
from utils.helpers import setup_logger


# Config
from backtester.config.backtest_config import BACKTEST_CONFIG

# Order engine (swap-enabled)
from backtester.order_executions.order_exe import OrderEngine

# Loaders
from backtester.data.loaders import fetch_event_features
from backtester.data.loaders import fetch_sql_market_data

# Features
from backtester.features_builder.features import (
    build_features,
    add_keltner_features,
    add_keltner_renko_interactions,
    add_gaussian_channel,
)

# ICT + trendline live in their own files per your structure
from backtester.filters.events import add_event_block
from backtester.filters.atr import add_atr, add_atr_volatility_guard

# Strategies
from backtester.strategies.base_strat import BaseStrategy
from backtester.strategies.ema import EMACrossoverStrategy
from backtester.strategies.sma import SMACrossoverStrategy
from backtester.strategies.macd import MACDStrategy
from backtester.strategies.ichimoku import IchimokuStrategy
from backtester.strategies.rsi_occilator import RSIOscillatorStrategy
from backtester.strategies.gaussian import GaussianChannelStrategy
from backtester.strategies.keltner_renko import KeltnerRenkoStrategy
from backtester.strategies.renko_ichimoku import RenkoIchimokuStrategy
from backtester.strategies.ict import ICTStrategy
from backtester.strategies.trendline import TrendlineStrategy


logger = setup_logger(__name__)

# Performance
from backtester.performance.eval import (
    get_performance_report,
    print_performance_report,
    plot_equity_curve,
)


# ---------- Strategy factory ----------
STRATEGY_MAP = {
    "ema": EMACrossoverStrategy,
    "sma": SMACrossoverStrategy,
    "macd": MACDStrategy,
    "ichimoku": IchimokuStrategy,
    "rsi": RSIOscillatorStrategy,
    "gaussian": GaussianChannelStrategy,
    "keltner_renko": KeltnerRenkoStrategy,
    "renko_ichimoku": RenkoIchimokuStrategy,
    "ict": ICTStrategy,
    "trendline": TrendlineStrategy,
}


def make_strategy(name: str, symbol: str, config: Dict[str, Any]):
    key = name.lower()
    if key not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}")
    cls = STRATEGY_MAP[key]
    cfg = dict(config)
    cfg["name"] = name
    return cls(symbol, cfg)


# ---------- Default feature pipeline ----------
def default_feature_pipeline(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = build_features(df)
    out = add_keltner_features(out)
    out = add_gaussian_channel(out)
    out = add_keltner_renko_interactions(out)

    # Events and ATR guard
    out = fetch_event_features(cfg.get("SYMBOL", "EURUSD"), out)
    out = add_event_block(
        out,
        before_minutes=int(cfg.get("BEFORE_EVENT", 30)),
        after_minutes=int(cfg.get("AFTER_EVENT", 30)),
    )
    out = add_atr(out, atr_len=14)
    out = add_atr_volatility_guard(
        out, atr_col="atr", q_low=0.05, q_high=0.95, window=10080
    )

    return out


# --- Data Preparation & Caching  ---
def prepare_data(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    force_reload: bool = False,
    feature_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Loads, processes, and caches data WITH a full feature pipeline."""
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
    market_data = fetch_sql_market_data(symbol, timeframe, where_clause)
    if market_data.empty:
        logger.error("Failed to fetch market data. Aborting.")
        return pd.DataFrame()

    logger.info("Building feature pipeline...")
    cfg = {**BACKTEST_CONFIG, **(feature_config or {})}
    features_df = default_feature_pipeline(market_data, cfg)

    logger.info(f"Saving features to cache file: {data_file}")
    features_df.to_parquet(data_file)
    return features_df


# --- Backtester Class (New) ---
class Backtester:
    def __init__(
        self, data: pd.DataFrame, strategy: BaseStrategy, config: Dict[str, Any]
    ):
        self.features_df = data
        self.strategy = strategy
        self.config = config
        self.symbol = config.get("SYMBOL", "EURUSD")

    def run(self) -> Dict[str, Any]:
        logger.info(
            f"Running backtest for strategy: {self.strategy.get_name()} on symbol: {self.symbol}"
        )
        pip_size = 0.01 if "JPY" in self.symbol.upper() else 0.0001

        eng = OrderEngine(symbol=self.symbol, config=self.config, pip_size=pip_size)
        eng.set_strategy(self.strategy)

        for i in range(1, len(self.features_df)):
            window = self.features_df.iloc[i - 1 : i + 1]
            self.strategy.generate_signals(window)
            eng.on_bar(self.features_df.iloc[i])

        # CORRECT: Use the imported functions, passing the engine instance
        report = get_performance_report(eng)
        if report:
            print_performance_report(eng, report)
            plot_equity_curve(eng, report)

        return {"report": report, "engine": eng}


# --- Parallel Execution Helper ---
def run_backtest_for_strategy(
    strategy: BaseStrategy, data: pd.DataFrame, config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """A top-level function that can be pickled for parallel processing."""
    bt = Backtester(data=data.copy(), strategy=strategy, config=config)
    result = bt.run()
    return result.get("report")


# --- Main Execution Block (Refactored for Parallelism) ---
if __name__ == "__main__":
    SYMBOL, TIMEFRAME = "EURUSD", "1m"
    START_DATE, END_DATE = "2020-06-01", "2025-08-01"
    RUN_IN_PARALLEL = True
    MAX_WORKERS = 12  # MAX 24 Cores / 100gb ram

    # 1. Prepare and cache the data with all features ONCE.
    data = prepare_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE, force_reload=False)

    if data.empty:
        logger.error("Data loading failed. Exiting.")
        raise SystemExit(1)

    # 2. Define the list of strategies to test.
    strategies_to_run = [
        MACDStrategy(
            symbol=SYMBOL,
            config={"name": "MACD Crossover", "sl_pips": 10, "tp_pips": 70},
        ),
        IchimokuStrategy(
            symbol=SYMBOL,
            config={"name": "Ichimoku Cloud", "sl_pips": 10, "tp_pips": 60},
        ),
        EMACrossoverStrategy(
            symbol=SYMBOL,
            config={"name": "EMA Crossover", "sl_pips": 10, "tp_pips": 40},
        ),
        SMACrossoverStrategy(
            symbol=SYMBOL,
            config={"name": "SMA Crossover", "sl_pips": 10, "tp_pips": 70},
        ),
        RSIOscillatorStrategy(
            symbol=SYMBOL,
            config={"name": "RSI Oscillator", "sl_pips": 10, "tp_pips": 50},
        ),
    ]

    # 3. Run the backtests serially or in parallel.
    all_results = []
    if RUN_IN_PARALLEL:
        logger.info(f"Running {len(strategies_to_run)} strategies in parallel...")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a future for each strategy run
            futures = [
                executor.submit(run_backtest_for_strategy, strat, data, BACKTEST_CONFIG)
                for strat in strategies_to_run
            ]
            for future in futures:
                result_report = future.result()
                if result_report:
                    all_results.append(result_report)
    else:
        logger.info(f"Running {len(strategies_to_run)} strategies serially...")
        for strat in strategies_to_run:
            bt = Backtester(data=data.copy(), strategy=strat, config=BACKTEST_CONFIG)
            result = bt.run()
            if result and result["report"]:
                all_results.append(result["report"])

    # 4. Print a final summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "=" * 80)
        print("--- Final Backtest Summary ---")
        print(
            summary_df[["strategy_name", "net_profit", "final_balance"]].to_string(
                index=False
            )
        )
        print("=" * 80 + "\n")
