import json
from datetime import datetime, timedelta, timezone, date, time
from typing import Any, Dict, List, Optional, Tuple

from operations.sql_operations import upsert_event
from operations.redis_operations import set_cache, persist_redis_key
from utils.helpers import green, reset, red


def detect_mt5_offset() -> int:
    today = datetime.now(timezone.utc)

    last_sunday_march = max(
        datetime(today.year, 3, d, tzinfo=timezone.utc)
        for d in range(25, 32)
        if datetime(today.year, 3, d).weekday() == 6
    )

    last_sunday_oct = max(
        datetime(today.year, 10, d, tzinfo=timezone.utc)
        for d in range(25, 32)
        if datetime(today.year, 10, d).weekday() == 6
    )

    return 1 if last_sunday_march <= today < last_sunday_oct else 2


def adjust_event_time(event: Dict[str, str], offset_hours: int) -> Tuple[date, time]:
    time_str = f"{event['date']} {event['time']}"
    dt_utc = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    dt_adjusted = dt_utc + timedelta(hours=offset_hours)
    return dt_adjusted.date(), dt_adjusted.time()


def parse_float(value: str) -> Optional[float]:
    try:
        if value in ["", "N/A"]:
            return None
        if "K" in value:
            return float(value.replace("K", "")) * 1_000
        if "B" in value:
            return float(value.replace("B", "")) * 1_000_000_000
        if "%" in value:
            return float(value.replace("%", "")) / 100
        return float(value)
    except Exception:
        return None


def store_forex_events(events: List[Dict[str, Any]]) -> None:
    if not events:
        return

    offset_hours = detect_mt5_offset()
    date_key: str = events[0]["date"]
    redis_key = f"forex_calendar:{date_key}"

    set_cache(redis_key, json.dumps(events))
    persist_redis_key(redis_key)

    for event in events:
        try:
            event_date, event_time = adjust_event_time(event, offset_hours)
            actual_val = parse_float(event["actual"])
            forecast_val = parse_float(event["forecast"])
            previous_val = parse_float(event["previous"])

            upsert_event(
                event_date,
                event_time,
                event["currency"],
                event["event"],
                event["impact"],
                actual_val,
                forecast_val,
                previous_val,
                "ForexFactory",
                False,
            )
        except Exception as e:
            print(red + f"Error upserting event: {e}" + reset)

    print(green + "Events upserted to PostgreSQL and Redis" + reset)
