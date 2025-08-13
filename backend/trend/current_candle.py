from datetime import datetime, timedelta


class RealTimeCandleBuilder:
    """
    Deze klasse bouwt real-time M1 kaarsen op basis van inkomende tick data.
    """

    def __init__(self):
        self.current_candles = {}  # {(symbol, timeframe): candle_dict}
        self.last_bar_time = {}  # {(symbol, timeframe): datetime from MT5}
        self.last_minute_key = {}

    def update(self, symbol, bid):
        key = (symbol.upper(), "M1")
        if key not in self.last_bar_time:
            return  # Wacht op de eerste bar time van on_bar_data

        candle_time = self.last_bar_time[key] + timedelta(minutes=1)

        if self.last_minute_key.get(key) and self.last_minute_key[key] != candle_time:
            pass

        if (
            key not in self.current_candles
            or self.last_minute_key.get(key) != candle_time
        ):
            self.current_candles[key] = {
                "time": candle_time,
                "open": bid,
                "high": bid,
                "low": bid,
                "close": bid,
                "tick_volume": 1,
                "spread": 0,
                "real_volume": 0,
            }
            self.last_minute_key[key] = candle_time
        else:
            candle = self.current_candles[key]
            candle["high"] = max(candle["high"], bid)
            candle["low"] = min(candle["low"], bid)
            candle["close"] = bid
            candle["tick_volume"] += 1

    def set_last_bar_time(self, symbol, timeframe, raw_time_str):
        key = (symbol.upper(), timeframe.upper())
        try:
            dt = datetime.strptime(raw_time_str, "%Y.%m.%d %H:%M")
            if self.last_bar_time.get(key) != dt:
                self.last_bar_time[key] = dt
        except Exception as e:
            print(f"[ERROR] Kan bar time niet parsen: {raw_time_str} - {e}")
