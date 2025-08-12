from flask import Blueprint, jsonify
import traceback
import pandas as pd
from indicators.trend_eval import analyze_window
from operations.sql_operations import execute_query

trend_eval_bp = Blueprint("trend_eval", __name__)


@trend_eval_bp.route("/api/trend-eval/<symbol>/<timeframe>", methods=["GET"])
def get_trend_evaluation(symbol, timeframe):
    try:
        # Fetch a large chunk of data to cover the 1-week timeframe
        table_name = f"market_data.{symbol.lower()}_{timeframe}"
        query = f"SELECT * FROM {table_name} ORDER BY time DESC LIMIT 11000"

        # Use the execute_query helper function
        records = execute_query(query, fetchall=True)
        if not records:
            return (
                jsonify({"error": f"No historical data found for {table_name}."}),
                404,
            )

        # Convert the list of records (assuming they are dicts) into a DataFrame
        df = pd.DataFrame(records)

        # Convert to list of dicts and reverse to be chronological (oldest first)
        all_candles = df.sort_values("time").to_dict("records")

        # Define all timeframes for analysis in minutes
        time_windows = {
            "two_min": 2,
            "three_min": 3,
            "four_min": 4,
            "five_min": 5,
            "ten_min": 10,
            "fifteen_min": 15,
            "thirty_min": 30,
            "one_hour": 60,
            "eight_hour": 480,
            "twenty_four_hour": 1440,
            "one_week": 10080,
        }

        results = {
            key: analyze_window(all_candles, minutes)
            for key, minutes in time_windows.items()
        }

        return jsonify({"success": True, "data": results})

    except Exception as e:
        print("--- AN UNEXPECTED ERROR OCCURRED IN TREND EVALUATION ---")
        traceback.print_exc()
        return (
            jsonify({"error": "An internal error occurred during trend evaluation."}),
            500,
        )
