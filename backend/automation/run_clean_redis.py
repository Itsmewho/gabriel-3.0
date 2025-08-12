# redis_cleanup.py
import re
from datetime import datetime, timedelta, date, timezone
from typing import Iterable, Optional, List
from utils.helpers import setup_logger
from connections.redis_connect import get_redis_client  # adjust import

_DATE_RE = re.compile(r"^(?P<prefix>.+):(?P<date>\d{4}-\d{2}-\d{2})$")

logger = setup_logger(__name__)


def _server_utc_date(r) -> date:
    """Return Redis server date using its own clock. Day-level only."""
    sec, micro = r.time()  # server clock (seconds, microseconds)
    ts = sec + micro / 1_000_000
    return datetime.fromtimestamp(ts, tz=timezone.utc).date()


def cleanup_old_persisted_date_keys(
    r=None,
    *,
    days: int = 5,
    prefixes: Optional[Iterable[str]] = ("daily_fetch_done", "forex_calendar"),
    dry_run: bool = False,
    scan_count: int = 1000,
) -> List[str]:
    """
    Delete keys like '<prefix>:YYYY-MM-DD' older than N days.
    Only deletes keys without expiry (TTL == -1). Uses Redis server date.
    Returns the list of affected keys.
    """
    client = r
    if client is None:
        if callable(get_redis_client):  # type: ignore
            try:
                client = get_redis_client()  # type: ignore
            except Exception:
                client = None
    if not client:
        logger.error("Redis client unavailable for cleanup")
        return []

    today = _server_utc_date(client)
    cutoff = today - timedelta(days=days)
    pattern = "*:????-??-??"

    selected: set[str] = set()
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=pattern, count=scan_count)  # type: ignore
        for key in keys:  # decode_responses=True -> str
            if prefixes and not any(key.startswith(f"{p}:") for p in prefixes):
                continue
            if client.ttl(key) != -1:  # only persistent
                continue
            m = _DATE_RE.match(key)
            if not m:
                continue
            try:
                kdate = datetime.strptime(m.group("date"), "%Y-%m-%d").date()
            except ValueError:
                continue
            if kdate <= cutoff:
                selected.add(key)
        if cursor == 0:
            break

    if not selected or dry_run:
        return sorted(selected)

    # Delete in pipeline
    with client.pipeline() as p:
        for k in selected:
            p.delete(k)
        p.execute()

    return sorted(selected)


if __name__ == "__main__":
    cleanup_old_persisted_date_keys()
