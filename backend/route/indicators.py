from flask import Blueprint, jsonify
from operations.sql_operations import get_indicator_data_from_db
from indicators.markov import build_markov_prediction

indicators_bp = Blueprint("indicators", __name__)

MAX_INDICATOR_POINTS = 9000  # its's over 9000 WHaaaahhhhhhhhh!!!!!!!!!!!


def get_data_or_404(indicator_name, symbol, timeframe):
    """Helper to fetch a large amount of indicator data and format the response."""

    data = get_indicator_data_from_db(
        indicator_name, symbol, timeframe, limit=MAX_INDICATOR_POINTS
    )

    if data is None:
        return (
            jsonify(
                {"success": False, "message": f"No data found for {indicator_name}"}
            ),
            404,
        )

    return jsonify({"success": True, "data": data})


@indicators_bp.route("/api/rsi/<symbol>/<timeframe>", methods=["GET"])
def get_rsi(symbol: str, timeframe: str):
    return get_data_or_404("rsi", symbol, timeframe)


@indicators_bp.route("/api/sma/<symbol>/<timeframe>", methods=["GET"])
def get_sma(symbol: str, timeframe: str):
    return get_data_or_404("sma", symbol, timeframe)


@indicators_bp.route("/api/ema/<symbol>/<timeframe>", methods=["GET"])
def get_ema(symbol: str, timeframe: str):
    return get_data_or_404("ema", symbol, timeframe)


@indicators_bp.route("/api/markov/historic/<symbol>/<timeframe>", methods=["GET"])
def get_markov_prediction_historic(symbol: str, timeframe: str):
    return get_data_or_404("markov", symbol, timeframe)


@indicators_bp.route("/api/bollinger/<symbol>/<timeframe>", methods=["GET"])
def get_bollinger(symbol: str, timeframe: str):
    return get_data_or_404("bb", symbol, timeframe)


@indicators_bp.route("/api/markov/<symbol>/<timeframe>", methods=["GET"])
def get_markov_prediction(symbol: str, timeframe: str):
    """
    Builds a live Markov Model prediction using pre-calculated historical states.
    """
    historical_states = get_indicator_data_from_db(
        "markov", symbol, timeframe, limit=60
    )

    if not historical_states or len(historical_states) < 2:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Not enough historical state data to build a model.",
                }
            ),
            404,
        )

    prediction_data = build_markov_prediction(historical_states)

    return jsonify({"success": True, "data": prediction_data})
