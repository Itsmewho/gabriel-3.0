import pandas as pd
import numpy as np


def get_trend_for_worker(data_points: pd.Series) -> int:
    """
    Calculates trend direction using linear regression slope.
    Returns:
        1 for rising, 0 for stable, -1 for falling
    """
    numeric_series = pd.to_numeric(data_points, errors="coerce").dropna()
    if len(numeric_series) < 2:
        return 0

    y = numeric_series.values
    x = np.arange(len(y))

    if np.all(y == y[0]):
        return 0

    slope, _ = np.polyfit(x, y, 1)  # type: ignore
    y_range = np.max(y) - np.min(y)  # type: ignore
    if y_range == 0:
        return 0

    threshold = y_range * 0.05 / len(x)
    if abs(slope) < threshold:
        return 0

    return 1 if slope > 0 else -1


def calculate_rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Calculates the rolling regression slope in a fast, vectorized way."""
    # The x-values are 0, 1, 2, ... window-1
    x = np.arange(window)
    # Pre-calculate sums for the slope formula
    sum_x = np.sum(x)
    sum_x2 = np.sum(x**2)

    # Calculate rolling sums of the series (y) and its product with x
    sum_y = series.rolling(window).sum()
    sum_xy = series.rolling(window).apply(lambda y: np.sum(y * x), raw=True)

    # Apply the linear regression slope formula
    numerator = (window * sum_xy) - (sum_x * sum_y)  # type: ignore
    denominator = (window * sum_x2) - (sum_x**2)

    return numerator / denominator


def process_single_window(args):
    """
    Worker function for calculating trend and pressure using a fast, vectorized slope.
    """
    df, window = args
    if window < 2:
        return f"eval_{window}m", pd.DataFrame(index=df.index)

    # Use the new vectorized function instead of .apply()
    upper_wick_slope = calculate_rolling_slope(df["upper_wick"], window)
    lower_wick_slope = calculate_rolling_slope(df["lower_wick"], window)
    body_slope = calculate_rolling_slope(df["body_size"], window)

    # Define a simple threshold to determine trend direction
    threshold = 1e-9

    analysis_df = pd.DataFrame(
        {
            "upper_wick_trend": np.select(
                [upper_wick_slope > threshold, upper_wick_slope < -threshold],
                ["rising", "falling"],
                default="stable",
            ),
            "lower_wick_trend": np.select(
                [lower_wick_slope > threshold, lower_wick_slope < -threshold],
                ["rising", "falling"],
                default="stable",
            ),
            "candle_body_trend": np.select(
                [body_slope > threshold, body_slope < -threshold],
                ["rising", "falling"],
                default="stable",
            ),
        },
        index=df.index,
    )

    analysis_df["price_pressure"] = (
        (analysis_df["candle_body_trend"] == "rising").astype(int)
        - (analysis_df["candle_body_trend"] == "falling").astype(int)
        + (analysis_df["lower_wick_trend"] == "rising").astype(int)
        - (analysis_df["upper_wick_trend"] == "rising").astype(int)
    )

    # Shift the final results by one step to prevent lookahead bias
    safe_analysis_df = analysis_df.shift(1)

    return f"eval_{window}m", safe_analysis_df
