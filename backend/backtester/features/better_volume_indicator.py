# Better volume

from __future__ import annotations
import pandas as pd
import numpy as np


def add_better_volume(
    df: pd.DataFrame,
    lookback: int = 14,
    high_vol_factor: float = 2.0,
    low_vol_factor: float = 0.5,
) -> pd.DataFrame:
    """
    Adds a 'bv_color' column to the DataFrame based on the "Better Volume" indicator logic.

    Args:
        df (pd.DataFrame): Input market data with OHLCV columns. Must contain 'High', 'Low', 'Open', 'Close', and 'Volume' columns.
        lookback (int): The lookback period for calculating moving averages.
        high_vol_factor (float): The multiplier for the volume SMA to be considered "high volume".
        low_vol_factor (float): The multiplier for the volume SMA to be considered "low volume".

    Returns:
        pd.DataFrame: The original DataFrame with the new 'bv_color' column.
    """
    dfc = df.copy()

    # Calculate necessary components using standardized, capitalized column names
    dfc["vol_sma"] = dfc["Volume"].rolling(window=lookback).mean()
    dfc["range"] = dfc["High"] - dfc["Low"]
    dfc["range_sma"] = dfc["range"].rolling(window=lookback).mean()

    # --- Define Conditions for each color ---

    # Condition 1: High volume churn (Green)
    cond_green = (dfc["Volume"] > dfc["vol_sma"] * high_vol_factor) & (
        dfc["range"] < dfc["range_sma"]
    )

    # Condition 2: Climax volume (Red for selling, White/Lime for buying)
    cond_high_vol_wide_range = (dfc["Volume"] > dfc["vol_sma"] * high_vol_factor) & (
        dfc["range"] > dfc["range_sma"]
    )

    # Use capitalized 'Close' and 'Open'
    cond_red = cond_high_vol_wide_range & (dfc["Close"] < dfc["Open"])
    cond_lime = cond_high_vol_wide_range & (dfc["Close"] > dfc["Open"])

    # Condition 3: Low volume (Blue)
    cond_blue = dfc["Volume"] < dfc["vol_sma"] * low_vol_factor

    default_color = "#646464"  # A neutral gray

    conditions = [cond_red, cond_lime, cond_green, cond_blue]
    choices = [
        "#FF0000",  # Red
        "#00FF00",  # Lime
        "#008000",  # Green
        "#0000FF",  # Blue
    ]

    dfc["bv_color"] = np.select(conditions, choices, default=default_color)

    dfc.drop(columns=["vol_sma", "range", "range_sma"], inplace=True)

    return dfc
