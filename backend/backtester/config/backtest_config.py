# Config


# 1pip = 0.0001 (EUR/USD = 1.00040 + 1 pips = 1.00050)
# if latency is >20ms a 2 pip sl is possible. (2pip safe -> 1pip can be rejected!)

BACKTEST_CONFIG = {
    # Account
    "INITIAL_BALANCE": 5000.00,
    "SYMBOL": "EURUSD",
    # Broker
    "SPREAD_PIPS": 0.2,
    "COMMISSION_PER_LOT_PER_SIDE": 3.50,
    "SWAP_LONG_POINTS": -9.89,
    "SWAP_SHORT_POINTS": 5.44,
    "CONTRACT_SIZE": 100000,
    "MARGIN_PER_LOT": 3882.02,
    "STOP_OUT_LEVEL_PCT": 50.0,
    # Slippage
    "MIN_SLIPPAGE_PIPS": 1,
    "MAX_SLIPPAGE_PIPS": 3,
    # Sizing & risk
    "VOLUME_STEP": 0.01,
    "VOLUME_MIN": 0.01,
}
