import numpy as np


def get_trend(data_points):
    """
    Determines the trend of a series of data points using linear regression.
    """
    data_points = [p for p in data_points if p is not None]
    if len(data_points) < 2:
        return "stable"

    x = np.arange(len(data_points))
    y = np.array(data_points, dtype=float)

    if np.all(y == y[0]):
        return "stable"

    slope, _ = np.polyfit(x, y, 1)

    y_range = np.max(y) - np.min(y)
    if y_range == 0:
        return "stable"

    threshold = y_range * 0.05 / len(x)

    if abs(slope) < threshold:
        return "stable"

    return "rising" if slope > 0 else "falling"


def analyze_window(candles, window_minutes):
    """
    Analyzes a specific time window of candle and forecast data.
    """
    if not candles or len(candles) < window_minutes:
        return {"score": 0, "analysis": {}}

    candles_in_window = candles[-window_minutes:]

    def safe_float(value):
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    candle_metrics = {
        "upper_wick": [
            safe_float(c.get("high"))
            - max(safe_float(c.get("open")), safe_float(c.get("close")))  # type: ignore
            for c in candles_in_window
        ],
        "lower_wick": [
            min(safe_float(c.get("open")), safe_float(c.get("close")))  # type: ignore
            - safe_float(c.get("low"))
            for c in candles_in_window
        ],
        "body_size": [
            abs(safe_float(c.get("open")) - safe_float(c.get("close")))  # type: ignore
            for c in candles_in_window
        ],
        "close_price": [safe_float(c.get("close")) for c in candles_in_window],
    }

    analysis = {
        "Upper Wicks": get_trend(candle_metrics["upper_wick"]),
        "Lower Wicks": get_trend(candle_metrics["lower_wick"]),
        "Candle Body": get_trend(candle_metrics["body_size"]),
        "Close Price": get_trend(candle_metrics["close_price"]),
    }

    pressure_score = 0
    if analysis["Candle Body"] == "rising":
        pressure_score += 1
    if analysis["Candle Body"] == "falling":
        pressure_score -= 1
    if analysis["Lower Wicks"] == "rising":
        pressure_score += 1
    if analysis["Upper Wicks"] == "rising":
        pressure_score -= 1
    analysis["Price Pressure"] = pressure_score  # type: ignore

    score = 0
    if analysis["Close Price"] == "rising":
        score += 2
    if analysis["Close Price"] == "falling":
        score -= 2
    if analysis["Candle Body"] == "rising":
        score += 1

    if analysis["Upper Wicks"] == "rising":
        score -= 0.5
    if analysis["Lower Wicks"] == "rising":
        score -= 0.5

    return {"score": round(score, 2), "analysis": analysis}
