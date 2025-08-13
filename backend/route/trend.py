from flask import Blueprint, jsonify, current_app
from utils.helpers import validate_symbol, validate_timeframe, normalize_timeframe

trend_bp = Blueprint("forex", __name__)


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
