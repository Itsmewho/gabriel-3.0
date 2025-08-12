from datetime import datetime, timedelta, timezone, date
from time import sleep
from typing import List, Dict, Any

from economic_events.scrape_calandar import scrape_forexfactory_day
from economic_events.store_events import store_forex_events
from utils.helpers import green, blue, reset


def fetch_from_forexfactory() -> None:
    today: date = datetime.now(timezone.utc).date()
    start_date: date = today - timedelta(days=3650)  # 10 years back
    end_date: date = today + timedelta(days=3)

    current: date = start_date
    while current <= end_date:
        print(blue + f"Fetching ForexFactory calendar for: {current}" + reset)
        try:
            events: List[Dict[str, Any]] = scrape_forexfactory_day(current)
            if events:
                store_forex_events(events)
        except Exception as e:
            print(f"Error on {current}: {e}")
        sleep(2)  # Sleep to prevent getting blocked
        current += timedelta(days=1)

    print(green + "All events fetched and stored successfully!" + reset)


if __name__ == "__main__":
    fetch_from_forexfactory()
