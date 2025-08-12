import time
from datetime import datetime, timedelta, timezone, date
from economic_events.scrape_calandar import scrape_forexfactory_day
from economic_events.store_events import store_forex_events
from operations.sql_operations import execute_query
from utils.helpers import green, blue, red, reset


RUN_CALENDAR_FETCH = False  # True = full backfill, False = gentle 3-day fetch


def fetch_events_range(start_date: date, end_date: date) -> None:
    current = start_date
    while current <= end_date:
        events = scrape_forexfactory_day(current)
        if events:
            store_forex_events(events)
        current += timedelta(days=2)
        time.sleep(1)
    print(green + f"Fetch from {start_date} to {end_date} completed!" + reset)


def refetch_pending_actuals() -> None:
    now = datetime.now(timezone.utc)
    query = """
    SELECT DISTINCT event_date FROM economic_events
    WHERE actual IS NULL AND previous IS NOT NULL AND (
        (event_date = %s AND event_time <= %s)
        OR (event_date < %s)
    );
    """
    params = [now.date(), now.time(), now.date()]
    event_dates = execute_query(query, params, fetchall=True)

    if not event_dates:
        return

    for (event_date,) in event_dates:
        if not isinstance(event_date, date):
            continue
        try:
            events = scrape_forexfactory_day(event_date)
            if events:
                store_forex_events(events)
        except Exception as e:
            print(red + f"Error processing {event_date}: {e}" + reset)
        time.sleep(1)


def calendar_fetch_service() -> None:
    today = datetime.now(timezone.utc).date()

    if RUN_CALENDAR_FETCH:
        start_date = datetime.strptime("2025-05-31", "%Y-%m-%d").date()
        end_date = today + timedelta(days=3)
        print(green + "RUN_CALENDAR_FETCH=True - Running full backfill mode!" + reset)
    else:
        start_date = today - timedelta(days=3)
        end_date = today + timedelta(days=3)
    fetch_events_range(start_date, end_date)

    print(blue + "Starting continuous refetch loop for pending actuals..." + reset)


if __name__ == "__main__":
    calendar_fetch_service()
