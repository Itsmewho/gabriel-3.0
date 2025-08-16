BACKTEST_CONFIG = {
    # Core
    "INITIAL_BALANCE": 5000.00,
    "SYMBOL": "EURUSD",
    "VALUE_PER_PIP_PER_LOT": 10.0,
    # Costs
    "USE_SPREAD_COST": True,
    "SPREAD_PIPS": 0.2,
    "COMMISSION_PER_LOT_RT": 7.00,
    "USE_SWAP_COST": True,
    "SWAP_LONG_POINTS": -9.89,
    "SWAP_SHORT_POINTS": 5.44,
    # Sizing & risk
    "USE_COMPOUNDING": True,
    "FIXED_TRADE_SIZE_LOTS": 0.05,
    "RISK_PERCENTAGE": 1.0,
    "MAX_LOT_SIZE": 50.0,
    "MARGIN_PER_LOT": 3882.02,
    "VOLUME_STEP": 0.01,
    "VOLUME_MIN": 0.01,
    "STOP_OUT_LEVEL_PCT": 50.0,  # typical broker stop-out 30–50%
    "ENTRY_MARGIN_BUFFER_PCT": 120.0,  # don't open if proj. ML < buffer
    "MAX_CONCURRENT_TRADES": 3,  # per strategy (mirrors live)
    # Trade management
    "USE_TRAILING_STOP": True,
    "TRAILING_STOP_DISTANCE_POINTS": 10,
    "USE_BREAK_EVEN_STOP": True,
    "BE_TRIGGER_PIPS": 5,
    "BE_OFFSET_PIPS": 2,
    # Slippage
    "USE_SLIPPAGE": True,
    "MIN_SLIPPAGE_PIPS": 0.1,
    "MAX_SLIPPAGE_PIPS": 0.3,
    # Broker stops-level simulation (pips)
    "TRADE_STOPS_LEVEL_PIPS": 2,  # set 0 to disable
    "ENFORCE_STOPS_LEVEL_ON_OPEN": True,
    "ENFORCE_STOPS_LEVEL_ON_MODIFY": True,
}
