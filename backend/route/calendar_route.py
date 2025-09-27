from flask import Blueprint, jsonify, request, Response
from operations.sql_operations import fetch_records
from datetime import datetime
from typing import List, Dict, Any
from flask_cors import CORS
from typing import Union, Tuple

calendar_bp = Blueprint("calendar", __name__)
CORS(calendar_bp, resources={r"/*": {"origins": "*"}})


@calendar_bp.route("/api/calendar", methods=["GET"])
def get_economic_calendar() -> Union[Response, Tuple[Response, int]]:
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))

    try:
        rows = fetch_records(
            "economic_events",
            "event_time, currency, event_name, impact, actual, forecast, previous",
            "event_date = %s",
            [date],
        )

        result: List[Dict[str, Any]] = []
        for row in rows:
            try:
                event_time = row["event_time"]
                result.append(
                    {
                        "time": event_time.strftime("%H:%M") if event_time else "N/A",
                        "currency": row["currency"],
                        "event": row["event_name"],
                        "impact": row["impact"],
                        "actual": row["actual"],
                        "forecast": row["forecast"],
                        "previous": row["previous"],
                    }
                )
            except Exception as e:
                print(f"Error formatting row: {row} -> {e}")

        return jsonify(result)

    except Exception as e:
        print("Error fetching calendar from PostgreSQL:", str(e))
        return jsonify({"error": "Could not fetch calendar events"}), 500
