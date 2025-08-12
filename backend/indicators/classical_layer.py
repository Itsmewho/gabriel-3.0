import pandas as pd
import numpy as np


def calculate_standard_indicators(df: pd.DataFrame) -> dict:
    indicators = {}
    close = df["close"]

    # SMA (14)
    sma = close.rolling(window=14).mean().shift(1)
    indicators["sma"] = sma

    # EMA (50)
    ema = close.ewm(span=50, adjust=False).mean().shift(1)
    indicators["ema"] = ema

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    rsi = (100 - (100 / (1 + rs))).shift(1)
    indicators["rsi"] = rsi

    return indicators


def calculate_bollinger_bands(
    df: pd.DataFrame, length: int = 20, std: int = 2
) -> pd.DataFrame:
    bb_df = pd.DataFrame(index=df.index)

    bb_df["middle"] = df["close"].rolling(window=length).mean()
    rolling_std = df["close"].rolling(window=length).std()

    bb_df["upper"] = bb_df["middle"] + (rolling_std * std)
    bb_df["lower"] = bb_df["middle"] - (rolling_std * std)

    bb_df["width"] = (bb_df["upper"] - bb_df["lower"]) / bb_df["middle"]
    bb_df["percent"] = (df["close"] - bb_df["lower"]) / (
        bb_df["upper"] - bb_df["lower"]
    )

    return bb_df.shift(1)


def calculate_markov_states(df: pd.DataFrame) -> pd.DataFrame:
    conditions = [df["close"] > df["open"], df["close"] < df["open"]]
    choices = ["Bullish", "Bearish"]
    states = np.select(conditions, choices, default="Neutral")
    return pd.DataFrame({"state": states}, index=df.index).shift(1)
