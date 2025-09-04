from __future__ import annotations
import pandas as pd
import numpy as np


def add_better_volume_mql(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    A direct Python translation of the logic from the BetterVolume.mq5 indicator.

    This function uses rolling windows to find the absolute highest/lowest values
    over a lookback period, mimicking the MQL5 implementation.

    Args:
        df (pd.DataFrame): Input market data. Must contain 'High', 'Low',
                           'Close', and 'Volume' columns (case-insensitive).
        lookback (int): The lookback period for finding max/min values.
                        The original MQL5 indicator uses a fixed value of 20.

    Returns:
        pd.DataFrame: The original DataFrame with a 'bv_color' column added.
    """
    dfc = df.copy()

    # --- Standardize column names (make them case-insensitive) ---
    dfc.rename(
        columns={
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "tick_volume": "Volume",
        },
        inplace=True,
    )

    # --- 1. Calculate the core values from the MQL5 code ---
    dfc["range"] = dfc["High"] - dfc["Low"]
    # Value2 in MQL5: Volume * Range
    dfc["vol_range"] = dfc["Volume"] * dfc["range"]
    # Value3 in MQL5: Volume / Range (handle division by zero)
    dfc["vol_per_range"] = (dfc["Volume"] / dfc["range"]).replace([np.inf, -np.inf], 0)

    # --- 2. Find the min/max over the rolling lookback window ---
    # Note: The window includes the current bar, so we shift to look at the past `lookback` bars
    dfc["min_vol_lookback"] = dfc["Volume"].rolling(window=lookback).min()
    dfc["max_vol_range_lookback"] = dfc["vol_range"].rolling(window=lookback).max()
    dfc["max_vol_per_range_lookback"] = (
        dfc["vol_per_range"].rolling(window=lookback).max()
    )

    # --- 3. Define the conditions based on the MQL5 logic ---
    # The MQL code uses a series of `if` statements. The last one to be true sets the color.
    # We replicate this priority using np.select, with conditions ordered from lowest to highest priority.

    # Default color (Neutral - blue)
    default_color = "deepskyblue"

    # Low volume condition (Yellow)
    cond_yellow = dfc["Volume"] == dfc["min_vol_lookback"]

    # Climax Selling condition (Red)
    cond_red = (dfc["vol_range"] == dfc["max_vol_range_lookback"]) & (
        dfc["Close"] > (dfc["High"] + dfc["Low"]) / 2
    )

    # High Volume Churn condition (Lime)
    cond_lime = dfc["vol_per_range"] == dfc["max_vol_per_range_lookback"]

    # Climax Buying condition (White)
    cond_white = (dfc["vol_range"] == dfc["max_vol_range_lookback"]) & (
        dfc["Close"] <= (dfc["High"] + dfc["Low"]) / 2
    )

    # Climax Churn condition (Magenta) - Highest priority
    cond_magenta = (dfc["vol_range"] == dfc["max_vol_range_lookback"]) & (
        dfc["vol_per_range"] == dfc["max_vol_per_range_lookback"]
    )

    # --- 4. Apply conditions in order of priority ---
    conditions = [
        cond_magenta,  # Highest priority
        cond_white,
        cond_lime,
        cond_red,
        cond_yellow,  # Lowest priority
    ]
    choices = [
        "magenta",
        "white",
        "lime",
        "red",
        "yellow",
    ]

    dfc["bv_color"] = np.select(conditions, choices, default=default_color)

    # --- Clean up temporary columns ---
    dfc.drop(
        columns=[
            "range",
            "vol_range",
            "vol_per_range",
            "min_vol_lookback",
            "max_vol_range_lookback",
            "max_vol_per_range_lookback",
        ],
        inplace=True,
    )

    return dfc
