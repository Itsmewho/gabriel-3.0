import logging
import time

from logging import Logger
from colorama import Style, Fore
from typing import Dict, Any, Tuple, Optional
import pandas as pd


reset = Style.RESET_ALL
blue = Fore.BLUE
yellow = Fore.YELLOW
red = Fore.RED
green = Fore.GREEN
import os
from typing import Optional


VALID_TIMEFRAMES = {"1m"}
VALID_SYMBOLS = {"EURUSD"}


def get_env_str(key: str, default: Optional[str] = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable '{key}' is required but not set.")
    return value


def get_env_int(key: str, default: Optional[int] = None) -> int:
    value = os.getenv(key)
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Environment variable '{key}' is required but not set.")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable '{key}' must be an integer.")


def setup_logger(name: str, level: int = logging.INFO) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def format_market_data(candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Takes a single raw candle, validates it, calculates derived values,
    and ensures all fields are in the correct format.
    Returns None if the candle is invalid.
    """
    # Validate that essential keys exist. If not, reject the candle.
    if not all(k in candle for k in ["time", "open", "close"]):
        print(f"Skipping invalid candle due to missing key: {candle}")
        return None

    # 2. Get values and perform calculations
    open_price = float(candle["open"])
    close_price = float(candle["close"])
    change = close_price - open_price
    change_percent = (change / open_price) * 100 if open_price != 0 else 0.0

    # 3. Return the fully formatted and validated dictionary
    return {
        "time": candle[
            "time"
        ],  # Assume time is already a correct string from the client
        "open": open_price,
        "high": float(candle.get("high", 0.0)),
        "low": float(candle.get("low", 0.0)),
        "close": close_price,
        "change": f"{change:.6f}",
        "change_percent": round(change_percent, 2),
        "spread": int(candle.get("spread", 0)),
        "tick_volume": int(candle.get("tick_volume", 0)),
        "real_volume": int(candle.get("real_volume", 0)),
    }


def create_future_price_targets(
    df: pd.DataFrame, horizons: list = [1, 5, 15]
) -> pd.DataFrame:
    """
    Creates future price direction targets for classification.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'close' and 'time' index.
        horizons (list): A list of future time steps (in minutes) to predict.

    Returns:
        pd.DataFrame: The original DataFrame with new target columns added.
    """
    df_out = df.copy()
    for h in horizons:
        # 1 if future price is higher, 0 otherwise
        df_out[f"target_direction_{h}m"] = (
            df_out["close"].shift(-h) > df_out["close"]
        ).astype(int)

    # Also create a continuous target for price prediction
    df_out["target_price_1m_ahead"] = df_out["close"].shift(-1)

    return df_out


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


def normalize_timeframe(tf: str) -> str:
    tf_lower = tf.lower()
    if "m" in tf_lower:
        return f"M{tf_lower.replace('m', '')}"
    return tf.upper()
