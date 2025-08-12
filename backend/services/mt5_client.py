import time
import logging
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
from utils.helpers import red, blue, reset
from typing import Dict, Any, List, cast, Tuple, Optional
from config.configure import ACCOUNT, ACCOUNT_PASS, META_SERVER


logger = logging.getLogger(__name__)


# --- Configuration ---
TIMEFRAMES: Dict[str, int] = {"1m": mt5.TIMEFRAME_M1}
PROPERTIES_CACHE: Dict[str, Any] = {}


# --- Central Connection Manager ---
class connect_mt5:
    """A context manager to ensure MT5 is initialized, logged in, and properly shut down."""

    def __enter__(self):
        if not mt5.initialize():  # type: ignore
            raise ConnectionError(f"Failed to initialize MT5: {mt5.last_error()}")  # type: ignore
        if not mt5.login(int(ACCOUNT), ACCOUNT_PASS, META_SERVER):  # type: ignore
            mt5.shutdown()  # type: ignore
            raise ConnectionError(f"Failed to login to MT5: {mt5.last_error()}")  # type: ignore
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mt5.shutdown()  # type: ignore
        return False


def get_mt5_server_time(symbol: str = "EURUSD") -> datetime:
    """Returns the current server time according to MT5 (UTC)."""
    tick = mt5.symbol_info_tick(symbol)  # type: ignore
    if tick and tick.time:
        return datetime.fromtimestamp(tick.time, tz=timezone.utc)
    else:
        # Fallback to system UTC if MT5 fails
        return datetime.now(timezone.utc)


# --- DATA FETCHING FUNCTIONS (Using Standard UTC) --


def get_last_hours(
    symbol: str = "EURUSD", timeframe_key: str = "1m", minutes: int = 5760
) -> List[Dict[str, Any]]:
    """Fetches a large chunk of historical data, used for seeding the database."""
    timeframe = TIMEFRAMES[timeframe_key]

    try:
        with connect_mt5():
            now_utc = get_mt5_server_time(symbol)
            from_time = now_utc - timedelta(minutes=minutes)

            print(
                blue
                + f"Fetching last {minutes} minutes of data for {symbol} up to {now_utc}..."
                + reset
            )
            rates = mt5.copy_rates_range(symbol, timeframe, from_time, now_utc)  # type: ignore
    except ConnectionError as e:
        print(red + f"MT5 connection error in get_last_hours: {e}" + reset)
        return []

    if rates is None or len(rates) == 0:
        print(f"No historical data received for {symbol}.")
        return []

    data = pd.DataFrame(rates)
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return cast(List[Dict[str, Any]], data.to_dict(orient="records"))


def get_last_60_entries(
    symbol: str = "EURUSD", timeframe_key: str = "1m"
) -> List[Dict[str, Any]]:
    """Fetches the last 120 candles from the current moment."""
    timeframe = TIMEFRAMES[timeframe_key]
    print(f"Fetching last 120 entries for {symbol}...")
    try:
        with connect_mt5():
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 120)  # type: ignore
    except ConnectionError as e:
        print(red + f"MT5 connection error in get_last_60_entries: {e}" + reset)
        return []

    if rates is None or len(rates) == 0:
        print(f"No latest market data received for {symbol}.")
        return []

    data = pd.DataFrame(rates)
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return cast(List[Dict[str, Any]], data.to_dict(orient="records"))


def get_account_info() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Haalt de actuele accountinformatie op (equity, balance, etc.).
    Retourneert een dictionary met de info en een eventuele foutmelding.
    """
    if not mt5.initialize():  # type: ignore
        return None, "MT5 connection failed"

    account = mt5.account_info()  # type: ignore
    mt5.shutdown()  # type: ignore

    if not account:
        return None, "Failed to fetch account info"

    account_dict = {
        "balance": account.balance,
        "equity": account.equity,
        "margin_free": account.margin_free,
        "leverage": account.leverage,
    }
    return account_dict, None


def get_symbol_properties(symbol: str) -> Optional[Any]:
    """
    Fetches the static properties of a symbol (volume rules, digits, etc.),
    using a cache for performance as these rarely change.
    """
    if symbol in PROPERTIES_CACHE:
        return PROPERTIES_CACHE[symbol]

    logger.info(f"Properties cache miss for '{symbol}'. Fetching from broker.")
    try:
        with connect_mt5():
            # Use symbol_info() for static properties
            properties = mt5.symbol_info(symbol)  # type: ignore
            if properties:
                PROPERTIES_CACHE[symbol] = properties
                return properties
            else:
                logger.warning(f"Could not retrieve symbol_info for '{symbol}'.")
                return None
    except Exception as e:
        logger.error(f"Failed to get symbol_properties for {symbol}: {e}")
        return None
