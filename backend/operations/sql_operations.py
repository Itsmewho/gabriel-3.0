from typing import Any, Iterable, Optional, Sequence, List, Union, Mapping, Tuple, Dict
from psycopg2 import OperationalError, ProgrammingError, DatabaseError
from utils.helpers import red, reset, setup_logger
from connections.postSQL import get_db_connection
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone

logger = setup_logger(__name__)


def execute_query(
    query: str,
    params: Optional[Union[Sequence[Any], Mapping[str, Any]]] = None,
    fetch: bool = False,
    fetchall: bool = False,
) -> Optional[Any]:
    """
    Executes a SQL query.
    - Returns single result if fetch=True.
    - Returns list of results if fetchall=True.
    - Returns None otherwise.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                if fetch:
                    result = cursor.fetchone()
                    # logger.info(green + "Query executed (fetch one)" + reset)
                    return result
                if fetchall:
                    result = cursor.fetchall()
                    # logger.info(
                    #     green + f"Query executed (fetch all) {len(result)} rows" + reset
                    # )
                    return result
                conn.commit()
                # logger.info(yellow + "Query executed and committed." + reset)
    except (OperationalError, ProgrammingError) as db_error:
        logger.error(red + f"DB error during query: {db_error}" + reset)
    except DatabaseError as general_db_error:
        logger.error(red + f"General database error: {general_db_error}" + reset)
    except Exception as e:
        logger.error(red + f"Unexpected error during query: {e}" + reset)
    return None


def insert_record(
    table_name: str, columns: Iterable[str], values: Sequence[Any]
) -> Optional[Any]:
    try:
        placeholders = ",".join(["%s"] * len(values))
        column_names = ",".join(columns)
        query = f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders}) RETURNING id;'
        # logger.info(blue + f"Inserting record into {table_name}..." + reset)
        return execute_query(query, values, fetch=True)
    except Exception as e:
        logger.error(red + f"Insert failed: {e}" + reset)
        return None


def fetch_records(
    table_name: str,
    columns: str = "*",
    where_clause: Optional[str] = None,
    params: Optional[Sequence[Any]] = None,
    limit: Optional[int] = None,
) -> List[Any]:
    try:
        if "." in table_name:
            schema, table = table_name.split(".")
            query = f'SELECT {columns} FROM "{schema}"."{table}"'
        else:
            query = f'SELECT {columns} FROM "{table_name}"'

        if where_clause:
            query += f" WHERE {where_clause}"
        if limit:
            query += f" LIMIT {limit}"

        # logger.info(f"Fetching records from {table_name}...")
        result = execute_query(query, params, fetchall=True)
        return result if result is not None else []
    except Exception as e:
        logger.error(f"Fetch failed for {table_name}: {e}")
        return []


def update_records(
    table_name: str, set_clause: str, where_clause: str, params: Sequence[Any]
) -> bool:
    try:
        query = f'UPDATE "{table_name}" SET {set_clause} WHERE {where_clause}'
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                updated = cursor.rowcount
                # logger.info(
                #     green + f"{updated} row(s) updated in {table_name}." + reset
                # )
                return updated > 0
    except Exception as e:
        logger.error(red + f"Update failed in {table_name}: {e}" + reset)
        return False


def delete_records(
    table_name: str,
    where_clause: Optional[str] = None,
    params: Optional[Sequence[Any]] = None,
) -> bool:
    try:
        query = f'DELETE FROM "{table_name}"'
        if where_clause:
            query += f" WHERE {where_clause}"
        # logger.info(yellow + f"Deleting from {table_name}..." + reset)
        execute_query(query, params)
        return True
    except Exception as e:
        logger.error(red + f"Delete failed in {table_name}: {e}" + reset)
        return False


def upsert_event(
    event_date,
    event_time,
    currency,
    event_name,
    impact,
    actual,
    forecast,
    previous,
    source,
    is_revised,
):
    """
    Inserts a new event or updates only the 'actual' column if event exists and new 'actual' is available.
    """
    query = """
    INSERT INTO economic_events (
        event_date, event_time, currency, event_name, impact, actual, forecast, previous, source, is_revised
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (event_date, event_time, currency, event_name)
    DO UPDATE SET
        actual = EXCLUDED.actual
    WHERE economic_events.actual IS NULL AND EXCLUDED.actual IS NOT NULL;
    """
    return execute_query(
        query,
        [
            event_date,
            event_time,
            currency,
            event_name,
            impact,
            actual,
            forecast,
            previous,
            source,
            is_revised,
        ],
    )


def get_latest_timestamp(symbol: str, timeframe: str):
    table = f"market_data.{symbol.lower()}_{timeframe.lower()}"
    sql = f'SELECT "time" FROM {table} ORDER BY "time" DESC LIMIT 1;'
    # logger.info("Running SQL: %s", sql)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                # logger.info("Row fetched: %s", row)
                if not row:
                    return None

                latest_ts = row["time"]  # type: ignore # works for RealDictRow
                print(f"Latest timestamp for {symbol} {timeframe}: {latest_ts}")
                return latest_ts
    except Exception:
        logger.exception("Query failed for table: %s", table)
        return None


def get_next_event_time() -> Optional[Tuple[datetime, datetime]]:
    """
    Fetches the current server time and the next event time from the database.

    Both times are returned as timezone-aware datetime objects in UTC.
    """
    query = """
        SELECT
            NOW() AT TIME ZONE 'UTC' AS server_time,
            (event_date + event_time)::timestamp AS event_timestamp
        FROM
            public.economic_events
        WHERE
            (event_date + event_time)::timestamp > NOW() AT TIME ZONE 'UTC'
        ORDER BY
            event_timestamp ASC
        LIMIT 1;
    """
    try:
        with get_db_connection() as conn:
            # Using DictCursor to access columns by name
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                result = cur.fetchone()

                if result:
                    server_time: datetime = result["server_time"]
                    event_timestamp: datetime = result["event_timestamp"]

                    if server_time.tzinfo is None:
                        server_time = server_time.replace(tzinfo=timezone.utc)

                    if event_timestamp.tzinfo is None:
                        event_timestamp = event_timestamp.replace(tzinfo=timezone.utc)

                    return server_time, event_timestamp

    except ValueError as e:
        print(f"Database error in get_next_event_time: {e}")


def get_indicator_data_from_db(
    indicator_name: str, symbol: str, timeframe: str, limit: int = 500
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches the latest data for any given indicator from its dedicated database table.
    """
    table_name = (
        f"technical_indicators.{symbol.lower()}_{timeframe.lower()}_{indicator_name}"
    )

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Use a dictionary cursor to get column names automatically
                from psycopg2.extras import RealDictCursor

                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute(
                    f"SELECT * FROM {table_name} ORDER BY time DESC LIMIT %s", (limit,)
                )
                results = cur.fetchall()

                if not results:
                    return None

                # Convert datetime objects to ISO format strings for JSON
                for row in results:
                    row["time"] = row["time"].isoformat()

                return results  # type: ignore
    except Exception as e:
        print(f"Database error fetching indicator '{indicator_name}': {e}")
        return None
