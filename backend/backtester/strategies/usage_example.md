### example mapping use backtester:

    market_data = prepare_data(symbol, timeframe, start_date, end_date)
    if market_data.empty:
        logger.error(f"No market data for period {period_tag}. Skipping.")
        return

    cfg = BrokerConfig(**BACKTEST_CONFIG)
    broker = Broker(cfg)

    # --- Risk and strategy config ---
    cfg_map = {
        "RAND_CFG": StrategyConfig(
            risk_mode=RiskMode.HALF_KELLY,
            risk_pct=0.01,
            kelly_p=0.53,
            kelly_rr=1.6,
            kelly_cap_pct=0.02,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.02,
            max_concurrent_trades=1,
        ),
        "RAND_FIX": StrategyConfig(
            risk_mode=RiskMode.FIXED,
            risk_pct=0.01,
            lot_min=cfg.VOLUME_MIN,
            lot_step=cfg.VOLUME_STEP,
            lot_max=100.0,
            max_risk_pct_per_trade=0.02,
            max_concurrent_trades=1,
        ),
    }

    governor = RiskGovernor(cfg_map)

    strategies = [
        RandomEntryStrategyConfig(
            symbol=symbol,
            config={
                "name": "RAND_CFG",
                "EVERY_N_MINUTES": 30,
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
            strat_cfg=cfg_map["RAND_CFG"],
            governor=governor,
        ),
        RandomEntryStrategyFixed(
            symbol=symbol,
            config={"name": "RAND_FIX"},
            strat_cfg=cfg_map["RAND_FIX"],
            governor=governor,
        ),
    ]

    # Initial allocations per strategy
    alloc = cfg.INITIAL_BALANCE
    allocations = {"RAND_CFG": alloc * 0.5, "RAND_FIX": alloc * 0.5}
    ledger = Ledger(initial_allocations=allocations)

    trade_to_strategy: dict[int, str] = {}
