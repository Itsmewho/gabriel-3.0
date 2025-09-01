# Strategy integration: requirements and usage

This project runs streaming-style strategies on cached features with the internal Broker. Use this checklist to plug in any strategy class.

## 1) Data and features

- **OHLCV**: `open, high, low, close` and one of `tick_volume | Volume | volume`.
- **Time index**: `pd.DatetimeIndex` (server time), sorted ascending.
- **EMAs / SMAs**: whichever your strategy needs, present as `ema_{n}` or `sma_{n}`.
- **Optional**: `vol_sma_{n}` if you want volume spikes without local buffers.

### Feature cache request

Ask only for what you need. Example for EMA-based strats:

```python
feature_spec = {
  "ema": [14, 30, 50, 150],
  "vol_sma": [20],  # optional for volume spikes
}
```

The loader merges missing cols into the same parquet when re-requested.

## 2) Strategy config schema

Each strategy reads its own keys from `config`.

Common keys used across examples:

- Identity: `name`
- Periods: `FAST_EMA`, `MID_EMA`, `SLOW_EMA`, `TREND_EMA` (or SMA equivalents)
- Risk distances: `SL_PIPS`, `TP_PIPS` (required for sizing and order params)
- Management (optional, handled by Broker when armed):

  - `USE_TRAILING_STOP`
  - `TRAILING_STOP_DISTANCE_PIPS`
  - `BE_TRIGGER_PIPS`, `BE_OFFSET_PIPS`

Example (EMA burst):

```python
config = {
  "name": "EmaBurst150",
  "FAST_EMA": 14, "MID_EMA": 30, "SLOW_EMA": 50, "TREND_EMA": 150,
  "SL_PIPS": 5, "TP_PIPS": 20,
  # optional mgmt
  "USE_TRAILING_STOP": False,
  "TRAILING_STOP_DISTANCE_PIPS": 10,
  "BE_TRIGGER_PIPS": 7,
  "BE_OFFSET_PIPS": 2,
}
```

## 3) Per‑strategy risk config (StrategyConfig)

Use `StrategyConfig` to define sizing mode and limits. Example:

```python
from backtester.account_management.types import StrategyConfig, RiskMode

cfg = StrategyConfig(
  risk_mode=RiskMode.FIXED,
  risk_pct=0.10,                 # 10% per trade for testing
  lot_min=broker_cfg.VOLUME_MIN,
  lot_step=broker_cfg.VOLUME_STEP,
  lot_max=100.0,
  max_risk_pct_per_trade=0.10,
  max_drawdown_pct=0.30,
  max_concurrent_trades=2,
)
```

## 4) Governor (optional but recommended)

Limit concurrency and pause on drawdown.

```python
from backtester.account_management.govorner import RiskGovernor
cfg_map = {"EmaBurst150": cfg}
governor = RiskGovernor(cfg_map)
```

## 5) Broker wiring

`BrokerConfig` lives in your backtest config module. Broker enforces SL/TP, break‑even, trailing, margin, swap, and rejections.

```python
from backtester.broker import BrokerConfig
from backtester.broker.main_broker import Broker
broker_cfg = BrokerConfig(**BACKTEST_CONFIG)
broker = Broker(broker_cfg)
```

## 6) Register a strategy

Import your strategy and instantiate with `symbol`, `config`, `strat_cfg`, and optional `governor`.

```python
from backtester.strategies.ema_burst_150 import EmaBurst150

strategies = [
  EmaBurst150(
    symbol=SYMBOL,
    config=config,
    strat_cfg=cfg,
    governor=governor,
  )
]
```

## 7) Backtest loop contract

The engine calls `strat.on_bar(broker, ts, row)` per bar. Strategy may return a `Trade` on entry. The engine then calls `broker.on_bar(...)` to manage open trades and exits.

Skeleton used in `run_period`:

```python
for ts, row in market_data.iterrows():
    for strat in strategies:
        tr = strat.on_bar(broker, ts, row)
        if tr:
            trade_to_strategy[tr.id] = strat.name
            ledger.on_open(strat.name, ts, trade_id=tr.id)
    broker.on_bar(float(row["high"]), float(row["low"]), float(row["close"]), t=ts)

for tr in broker.trade_history:
    sid = trade_to_strategy.get(tr.id, "UNKNOWN")
    ledger.on_close(sid, tr.exit_time, pnl=tr.pnl, trade_id=tr.id)
```

## 8) Reports

Outputs are written under `results/{strategy_name}/`:

- `metrics/` markdown report, plots, regime analysis
- `audit/` CSV audits for trades, rejections, account ledger
- `evals/` trade export CSV

## 9) Example: full wiring for EmaBurst150

```python
SYMBOL, TIMEFRAME = "EURUSD", "1m"
feature_spec = {"ema": [14, 30, 50, 150], "vol_sma": [20]}

config = {
  "name": "EmaBurst150",
  "FAST_EMA": 14, "MID_EMA": 30, "SLOW_EMA": 50, "TREND_EMA": 150,
  "SL_PIPS": 5, "TP_PIPS": 20,
  # optional mgmt flags
  # "USE_TRAILING_STOP": True,
  # "TRAILING_STOP_DISTANCE_PIPS": 10,
  # "BE_TRIGGER_PIPS": 7, "BE_OFFSET_PIPS": 2,
}

broker_cfg = BrokerConfig(**BACKTEST_CONFIG)
broker = Broker(broker_cfg)

strat_cfg = StrategyConfig(
  risk_mode=RiskMode.FIXED, risk_pct=0.10,
  lot_min=broker_cfg.VOLUME_MIN, lot_step=broker_cfg.VOLUME_STEP, lot_max=100.0,
  max_risk_pct_per_trade=0.10, max_drawdown_pct=0.30, max_concurrent_trades=2,
)

governor = RiskGovernor({"EmaBurst150": strat_cfg})

strategies = [
  EmaBurst150(symbol=SYMBOL, config=config, strat_cfg=strat_cfg, governor=governor)
]
```

## 10) Running multi‑period

Use the existing `run_periods` driver. Add your strategy to the `strategies` list in `run_period` or build a dedicated runner function.

## 11) Common errors and fixes

- **AttributeError in strategy management**: remove direct references to deleted attrs; read BE/trailing from `config` or per‑trade fields.
- **Missing features**: ensure `feature_spec` includes all `ema_{n}/sma_{n}` you reference.
- **No volume column**: disable volume filters or add `tick_volume`/`Volume` to the dataset and optionally `vol_sma_{n}`.
- **Empty market data**: confirm symbol/timeframe/date range and that the cache path exists.

## 12) Time rules

Use server time for all operations. Avoid UTC in SQL. If you see UTC, it is intentional and accompanied by workarounds.

---

This guide is minimal. Extend per strategy by adding its specific keys under **2) Strategy config schema**.
