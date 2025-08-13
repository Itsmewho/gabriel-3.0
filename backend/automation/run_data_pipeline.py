import json
from datetime import datetime, timedelta, timezone
from services.mt5_client import get_last_hours, get_last_60_entries
from operations.redis_operations import set_cache, persist_redis_key, get_cache
from connections.postSQL import get_db_connection
from utils.helpers import red, green, blue, reset, setup_logger, format_market_data

from .run_indicator_pipeline import run_indicator_pipeline
from .run_calander import calendar_fetch_service
from .run_clean_redis import cleanup_old_persisted_date_keys

# Logger
logger = setup_logger(__name__)

# --- Configuration ---
FORCE_RUN = False
RUN_DAILY_FETCH = False  # This can be toggled for manual runs
REDIS_DAILY_KEY = f"daily_fetch_done:{datetime.now().strftime('%Y-%m-%d')}"
SYMBOL = "EURUSD"
TIMEFRAME = "1m"
MINUTES_MAP = {
    "1m": 5760,
}

# --- Has pipeline run ---


def has_daily_fetch_run() -> bool:
    """Checks Redis to see if the daily fetch has run in the last 24 hours."""
    fetch_timestamp = get_cache(REDIS_DAILY_KEY)
    if fetch_timestamp:
        last_run = datetime.fromisoformat(fetch_timestamp)
        if datetime.now(timezone.utc) - last_run < timedelta(hours=24):
            return True
    return False


def mark_daily_fetch_completed() -> None:
    """Updates Redis to mark the daily fetch as complete."""
    set_cache(REDIS_DAILY_KEY, datetime.now(timezone.utc).isoformat())
    persist_redis_key(REDIS_DAILY_KEY)
    print(green + "Marked daily fetch as completed in Redis." + reset)


# --- Market Data Fetching ---


def store_data_in_db(symbol: str, timeframe: str, market_data: list):
    table_name = f"market_data.{symbol.lower()}_{timeframe.lower()}"

    query = f"""
        INSERT INTO {table_name} 
        (time, open, high, low, close, volume, tick_volume, real_volume, change, change_percent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (time) DO NOTHING;
    """

    records = [
        (
            datetime.fromisoformat(entry["time"].replace("Z", "+00:00")),
            entry.get("open", None),
            entry.get("high", None),
            entry.get("low", None),
            entry.get("close", None),
            entry.get("volume", 0),
            entry.get("tick_volume", 0),
            entry.get("real_volume", 0),
            entry.get("change", 0),
            entry.get("change_percent", 0),
        )
        for entry in market_data
    ]

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, records)
            conn.commit()
    except Exception as e:
        print(red + f"Failed to insert data for {symbol} [{timeframe}]: {e}" + reset)


def cache_and_store_market_data(symbol, timeframe):
    history_key = f"market:previous:{symbol.upper()}:{timeframe}"
    last_hour_key = f"market:last_hour:{symbol.upper()}:{timeframe}"

    market_data = get_last_hours(symbol, timeframe, MINUTES_MAP[timeframe])
    last_hour = get_last_60_entries(symbol, timeframe)

    if not market_data or len(market_data) < 2:
        print(blue + f"Not enough data for {symbol} [{timeframe}]." + reset)
        return

    formatted_data = [format_market_data(entry) for entry in market_data if entry]
    last_hour_formatted = [format_market_data(entry) for entry in last_hour if entry]

    if len(last_hour_formatted) > 1:
        set_cache(last_hour_key, json.dumps(last_hour_formatted[:-1]))

    combined_history = formatted_data + last_hour_formatted
    unique_combined_history = list(
        {entry["time"]: entry for entry in combined_history}.values()  # type: ignore
    )

    if unique_combined_history:
        unique_combined_history.pop()
        set_cache(history_key, json.dumps(unique_combined_history))
        store_data_in_db(symbol, timeframe, unique_combined_history)


# --- Main Execution Loop ---
def run_data_pipeline():
    print(green + "--- Starting Daily data fetch ---" + reset)
    if not RUN_DAILY_FETCH and has_daily_fetch_run():
        print(blue + "Daily fetch already performed. Skipping." + reset)
        return

    # 1. Fetch latest market data
    cache_and_store_market_data(symbol="EURUSD", timeframe="1m")

    # 2. Run indicators
    run_indicator_pipeline(mode="live")

    # 3. Check economic calendar
    calendar_fetch_service()

    # 4. Clean up redis
    cleanup_old_persisted_date_keys(days=5, dry_run=False)

    # last mark the key
    mark_daily_fetch_completed()


if __name__ == "__main__":
    run_data_pipeline()
