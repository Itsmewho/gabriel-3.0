# Remember 1pip = 0.0001 (1.12345 + 10 pips = 1.12355)

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
    "FIXED_TRADE_SIZE_LOTS": 0.05,
    "RISK_PERCENTAGE": 1.0,
    "MAX_LOT_SIZE": 50.0,
    "MARGIN_PER_LOT": 3882.02,
    "VOLUME_STEP": 0.01,
    "VOLUME_MIN": 0.01,
    "STOP_OUT_LEVEL_PCT": 50.0,  # typical broker stop-out 30–50%
    "ENTRY_MARGIN_BUFFER_PCT": 120.0,
    "MAX_CONCURRENT_TRADES": 3,  # per strategy
    "USE_BLOWOUT_PROTECTION": True,
    "BLOWOUT_LOSS_PCT": 50.0,
    "MIN_BALANCE_THRESHOLD": 200.0,
    "USE_MIN_BALANCE_STOP": True,
    "MIN_SIZING_PIPS": 10,
    # Trade management
    "USE_COMPOUNDING": True,
    "USE_EQUITY_FOR_RISK": True,
    "MAX_RISK_PCT_TOTAL": 1.5,
    "RESIZE_TO_FREE_MARGIN": True,
    "USE_TRAILING_STOP": True,
    "TRAILING_STOP_DISTANCE_PIPS": 100,
    "USE_BREAK_EVEN_STOP": True,
    "BE_TRIGGER_PIPS": 50,  # Pips in profit to activate break-even.
    "BE_OFFSET_PIPS": 20,  # Pips to add to entry price for the new SL (e.g., to cover costs).
    "BEFORE_EVENT": 30,
    "AFTER_EVENT": 30,
    # Slippage
    "USE_SLIPPAGE": True,
    "MIN_SLIPPAGE_PIPS": 0.1,
    "MAX_SLIPPAGE_PIPS": 0.3,
    # Broker stops-level simulation (pips)
    "TRADE_STOPS_LEVEL_PIPS": 2,  # set 0 to disable
    "ENFORCE_STOPS_LEVEL_ON_OPEN": True,
    "ENFORCE_STOPS_LEVEL_ON_MODIFY": True,
}
