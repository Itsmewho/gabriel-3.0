import time
import json
from datetime import datetime, timedelta, timezone
from operations.redis_operations import set_cache
from operations.sql_operations import get_next_event_time
from connections.postSQL import get_db_connection

from .run_indicator_pipeline import run_indicator_pipeline
from .run_clean_redis import cleanup_old_persisted_date_keys

from services.mt5_client import get_last_hours, get_last_60_entries

from utils.helpers import red, green, blue, reset, setup_logger
from utils.helpers import format_market_data


# Logger
logger = setup_logger(__name__)

# --- Configuration ---
SYMBOL = "EURUSD"
TIMEFRAMES = "1m"

MINUTES_MAP = {
    "1m": 5760,
}


def sleep_until_next_aligned_minute(offset_seconds=1):
    now = datetime.now(timezone.utc)
    next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    target_time = next_minute + timedelta(seconds=offset_seconds)
    sleep_duration = (target_time - now).total_seconds()

    print(
        blue
        + f"Sleeping {sleep_duration:.2f}s until {target_time.strftime('%H:%M:%S')} UTC"
        + reset
    )
    time.sleep(sleep_duration)


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
        print(
            blue
            + f"Niet genoeg historische marktdata voor {symbol} [{timeframe}]."
            + reset
        )
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
        print(
            green
            + f"Cached {len(unique_combined_history)} historische entries."
            + reset
        )
        store_data_in_db(symbol, timeframe, unique_combined_history)


def run_continuous_data_pipeline():
    while True:
        start_time = time.time()
        now_utc = datetime.now(timezone.utc)
        print(
            f"\n{green}--- Running tasks for {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} ---{reset}"
        )
        # 1. Fetch latest market data
        cache_and_store_market_data(symbol="EURUSD", timeframe="1m")

        # 2. Fetch indicators
        run_indicator_pipeline(mode="live")

        # 3. Clean up redis
        cleanup_old_persisted_date_keys(days=5, dry_run=False)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{green}--- Cycle finished in {elapsed:.2f}s. ---{reset}")
        sleep_until_next_aligned_minute(offset_seconds=2)


if __name__ == "__main__":
    run_continuous_data_pipeline()
