from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Any

from economic_events.scrape_calandar import scrape_forexfactory_day
from economic_events.store_events import store_forex_events
from utils.helpers import green, blue, reset


def fetch_from_forexfactory() -> None:
    today: date = datetime.now(timezone.utc).date()
    start_date: date = today - timedelta(days=3)
    end_date: date = today + timedelta(days=3)

    current: date = start_date
    while current <= end_date:
        print(blue + f"Fetching ForexFactory calendar for: {current}" + reset)
        events: List[Dict[str, Any]] = scrape_forexfactory_day(current)
        if events:
            store_forex_events(events)
        current += timedelta(days=1)

    print(green + "All events fetched and stored successfully!" + reset)


if __name__ == "__main__":
    fetch_from_forexfactory()
