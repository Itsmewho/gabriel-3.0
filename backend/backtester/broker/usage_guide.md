### Broker Usage Guide

This guide documents how to use the broker features in backtests and strategies.

### Trade Opening

tr = broker.open_trade(
side="buy", # "buy" or "sell"
price=px, # current price
wanted_lots=0.50, # initial lot size
sl_pips=20, # stop-loss distance in pips
tp_pips=40, # take-profit distance in pips
t=ts, # timestamp of entry
fallbacks=[0.35, 0.15], # (optional) fallback lot sizes if margin fails
)

If margin is insufficient for wanted_lots, broker tries:

explicit fallbacks (if given),

else cfg.FALLBACK_LOTS (absolute sizes),

else fractions of wanted_lots (cfg.FALLBACK_FRACTIONS, default = [0.5, 0.25, 0.10]).

If all fail → trading is paused until margin recovers.

### Break-Even Rules

Break-even can be set per-trade (preferred via strategies) or globally (via config).

Per-trade (from strategy)
broker.set_break_even(tr.id, be_pips=40, offset_pips=10)

When price moves +40 pips in favor, SL is moved to entry + 10 pips (locking profit).

Config-based (global)
BrokerConfig(
BREAK_EVEN_ENABLE=True,
BREAK_EVEN_TRIGGER_PIPS=40,
BREAK_EVEN_OFFSET_PIPS=10,
)

### Trailing Stop-Loss (TSL)

Trailing SL adjusts dynamically as price moves.

Per-trade (preferred)
tr.trailing_sl_distance = 80 # trail 80 pips behind price

Global (passed in broker.on_bar())
broker.on_bar(high, low, close, t, trail_pips=100)

Per-trade tr.trailing_sl_distance always takes priority.

### TP Extension

Allows “stretching” take-profits once price comes near the target.

Config (global default)
BrokerConfig(
NEAR_TP_BUFFER_PIPS=3, # distance from TP where extension is triggered
TP_EXTENSION_PIPS=5, # amount TP is extended by
)

Per-trade (from strategy)
tr.near_tp_buffer_pips = 2
tr.tp_extension_pips = 6

### Fallback Lots & Fractions

Explicit per-call:
fallbacks=[0.35, 0.15]

Absolute list (config):

BrokerConfig(FALLBACK_LOTS=[1.0, 0.5, 0.25])

Fractions (default):

BrokerConfig(FALLBACK_FRACTIONS=(0.5, 0.25, 0.10))

If margin check fails, broker tries in order until a trade is accepted or all fail.

### Audit Logging

After backtests, CSVs are written under results/audit/:

trade_audit.csv: per-trade details (PNL, SL/TP mods, BE applied, trailing/TP extension usage).

rejected_trades.csv: margin-based rejections with full diagnostics.

max_open_trades.csv: peak concurrent trades + timestamp.

### Strategy Example

from backtester.strategies.base_strategy import BaseStrategy

class MyStrat(BaseStrategy):
def on_bar(self, broker, t, row): # Simple example: open a random trade once per hour
if t.minute == 0:
tr = broker.open_trade(
side="buy",
price=row["close"],
wanted_lots=0.50,
sl_pips=20,
tp_pips=60,
)
if tr:
self.setup_trade(broker, tr) # applies BE, TSL, TP ext.

Config usage:

MyStrat(
symbol="EURUSD",
config={
"USE_BREAK_EVEN_STOP": True,
"BE_TRIGGER_PIPS": 40,
"BE_OFFSET_PIPS": 10,
"USE_TRAILING_STOP": True,
"TRAILING_STOP_DISTANCE_PIPS": 80,
"USE_TP_EXTENSION": True,
"NEAR_TP_BUFFER_PIPS": 3,
"TP_EXTENSION_PIPS": 6,
}
)

### flowchart TD

A[Order Opened] --> B{Margin Check}
B -- Fail --> R[Rejected (audit logged)]
B -- Pass --> C[Trade Active]

C --> D[Break-Even Check]
D -- Triggered --> D1[Move SL to Entry + Offset]
D -- Not Yet --> E

D1 --> E[Trailing Stop Check]
E -- Active --> E1[Adjust SL closer to price]
E -- Inactive --> F

E1 --> F[TP Extension Check]
F -- Near TP --> F1[Extend TP further]
F -- Not Near --> G

F1 --> G[Exit Conditions]
G -- SL Hit --> H[Trade Closed - Stop Loss]
G -- TP Hit --> I[Trade Closed - Take Profit]
G -- Margin Call --> J[Trade Closed - Margin Call]
G -- Manual/End --> K[Trade Closed - Other]

H --> Z[Audit CSV Log]
I --> Z
J --> Z
K --> Z
R --> Z
