from flask import Blueprint, jsonify, current_app
from utils.helpers import normalize_timeframe
from operations.redis_operations import get_cache
import json


trend_bp = Blueprint("forex", __name__)


VALID_TIMEFRAMES = {"1m"}
VALID_SYMBOLS = {"EURUSD"}


def validate_timeframe(timeframe: str) -> tuple[bool, str | None]:
    if timeframe not in VALID_TIMEFRAMES:
        return (
            False,
            f"Invalid timeframe: {timeframe}. Allowed: {', '.join(VALID_TIMEFRAMES)}",
        )
    return True, None


def validate_symbol(symbol: str) -> tuple[bool, str | None]:
    if symbol.upper() not in VALID_SYMBOLS:
        return False, f"Invalid symbol: {symbol}. Allowed: {', '.join(VALID_SYMBOLS)}"
    return True, None


@trend_bp.route("/api/lastcandle/<symbol>/<timeframe>")
def get_last_candle(symbol: str, timeframe: str):

    key = (symbol.upper(), normalize_timeframe(timeframe))

    candle_builder = current_app.config["CANDLE_BUILDER"]
    candle = candle_builder.current_candles.get(key)

    if not candle:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"No data Found  {key}..",
                }
            ),
            404,
        )

    change = candle["close"] - candle["open"]
    change_percent = (change / candle["open"]) * 100 if candle["open"] != 0 else 0.0

    response_data = {
        "time": candle["time"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": candle["open"],
        "high": candle["high"],
        "low": candle["low"],
        "close": candle["close"],
        "tick_volume": candle["tick_volume"],
        "real_volume": candle.get("real_volume", 0),
        "spread": candle.get("spread", 0),
        "change": f"{change:.6f}",
        "change_percent": round(change_percent, 2),
    }

    return jsonify({"success": True, "data": response_data})


@trend_bp.route("/api/history/<symbol>/<timeframe>")
def get_market_history(symbol: str, timeframe: str):
    """
    Reads the historical market data directly from the Redis cache.
    Does NOT trigger a new data fetch.
    """
    is_valid_tf, error_tf = validate_timeframe(timeframe)
    is_valid_sym, error_sym = validate_symbol(symbol)
    if not is_valid_tf or not is_valid_sym:
        return jsonify({"success": False, "message": error_tf or error_sym}), 400

    history_key = f"market:previous:{symbol.upper()}:{timeframe}"

    raw_data = get_cache(history_key)
    if not raw_data:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"No historical data found in cache for {symbol} at {timeframe}",
                }
            ),
            404,
        )

    return jsonify({"success": True, "data": json.loads(raw_data)}), 200
