from datetime import datetime, timedelta, timezone
from operations.redis_operations import set_cache, persist_redis_key, get_cache

from utils.helpers import green, blue, reset, setup_logger

from .run_calander import calendar_fetch_service


# Logger
logger = setup_logger(__name__)

# --- Configuration ---

RUN_DAILY_FETCH = False  # This can be toggled for manual runs
REDIS_DAILY_KEY = f"daily_fetch_done:{datetime.now().strftime('%Y-%m-%d')}"

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


# --- Main Execution Loop ---
def run_data_pipeline():
    print(green + "--- Starting Daily data fetch ---" + reset)
    # 1. Check economic calendar
    calendar_fetch_service()
    if not RUN_DAILY_FETCH and has_daily_fetch_run():
        print(blue + "Daily fetch already performed. Skipping." + reset)
        return

    # last mark the key
    mark_daily_fetch_completed()


if __name__ == "__main__":
    run_data_pipeline()
