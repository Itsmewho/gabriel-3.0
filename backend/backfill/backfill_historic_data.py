import pandas as pd
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

# --- Main Service Imports ---
from services.mt5_client import connect_mt5
import MetaTrader5 as mt5
from connections.postSQL import get_db_connection
from utils.helpers import green, blue, red, reset, yellow

# --- Configuration ---
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
START_YEAR = 2007  # As per your last run
END_YEAR = datetime.now().year
DB_TABLE_NAME = f"market_data.{SYMBOL.lower()}_1m"


def create_market_data_table(conn):
    """Creates the schema and table if they don't exist."""
    schema_name = DB_TABLE_NAME.split(".")[0]

    create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {schema_name};"
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {DB_TABLE_NAME} (
        time TIMESTAMPTZ PRIMARY KEY,
        open DOUBLE PRECISION,
        high DOUBLE PRECISION,
        low DOUBLE PRECISION,
        close DOUBLE PRECISION,
        volume BIGINT,
        tick_volume BIGINT,
        real_volume BIGINT,
        change DOUBLE PRECISION,
        change_percent DOUBLE PRECISION
    );
    """
    with conn.cursor() as cursor:
        print(blue + f"Ensuring schema '{schema_name}' exists..." + reset)
        cursor.execute(create_schema_query)
        print(blue + f"Ensuring table '{DB_TABLE_NAME}' exists..." + reset)
        cursor.execute(create_table_query)
    conn.commit()


def format_raw_candles(rates_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Takes a raw DataFrame from MT5 and formats it correctly."""
    if rates_df.empty:
        return []
    rates_df["time"] = pd.to_datetime(rates_df["time"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    rates_df["change"] = rates_df["close"] - rates_df["open"]
    rates_df["change_percent"] = (rates_df["change"] / rates_df["open"]) * 100
    rates_df = rates_df.astype(
        {
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "tick_volume": int,
            "spread": int,
            "real_volume": int,
            "change": float,
            "change_percent": float,
        }
    )
    return rates_df.to_dict(orient="records")  # type: ignore


def store_data_in_db(
    market_data: List[Dict[str, Any]], is_full_backfill: bool = False
) -> int:
    """Stores a batch of candle data. If in 'full' mode, it will drop the table first."""
    if not market_data:
        return 0

    try:
        with get_db_connection() as conn:
            if is_full_backfill:
                print(yellow + f"FULL MODE: Dropping table {DB_TABLE_NAME}..." + reset)
                with conn.cursor() as cursor:
                    cursor.execute(f"DROP TABLE IF EXISTS {DB_TABLE_NAME} CASCADE;")
                conn.commit()

            create_market_data_table(conn)

            query = f"""
                INSERT INTO {DB_TABLE_NAME} 
                (time, open, high, low, close, volume, tick_volume, real_volume, change, change_percent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (time) DO NOTHING;
            """
            records = [
                (
                    datetime.fromisoformat(entry["time"].replace("Z", "+00:00")),
                    entry.get("open"),
                    entry.get("high"),
                    entry.get("low"),
                    entry.get("close"),
                    entry.get("tick_volume", 0),
                    entry.get("tick_volume", 0),
                    entry.get("real_volume", 0),
                    entry.get("change", 0),
                    entry.get("change_percent", 0),
                )
                for entry in market_data
            ]

            with conn.cursor() as cursor:
                cursor.executemany(query, records)
                inserted_count = cursor.rowcount
            conn.commit()
            return inserted_count
    except Exception as e:
        print(red + f"Failed to insert data: {e}" + reset)
        return 0


def run_backfill(mode: str):
    """Main function to fetch and store historical data year by year."""
    print(blue + f"--- Starting Historical Data Backfill (Mode: {mode}) ---" + reset)
    total_records_stored = 0
    is_first_run_in_full_mode = True

    for year in range(START_YEAR, END_YEAR + 1):
        start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        print(f"\nFetching data for {year}...")
        try:
            with connect_mt5():
                rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)  # type: ignore
        except Exception as e:
            print(red + f"Could not fetch data for {year}. Error: {e}" + reset)
            continue

        if rates is None or len(rates) == 0:
            print(f"No data found for {year}.")
            continue

        rates_df = pd.DataFrame(rates)
        print(f"Fetched {len(rates_df)} raw candles for {year}.")

        formatted_candles = format_raw_candles(rates_df)

        should_drop_table = mode == "full" and is_first_run_in_full_mode
        count = store_data_in_db(formatted_candles, is_full_backfill=should_drop_table)

        if should_drop_table:
            is_first_run_in_full_mode = False

        print(green + f"Stored {count} new records for {year}." + reset)
        total_records_stored += count

    print(blue + f"\n--- Backfill Complete! ---" + reset)
    print(
        green
        + f"Total new records added to the database: {total_records_stored}"
        + reset
    )
    print(
        yellow
        + "\nTo analyze the data for gaps, run the 'analyze_data_integrity.py' script."
        + reset
    )


if __name__ == "__main__":
    run_mode = "full" if len(sys.argv) > 1 and sys.argv[1] == "full" else "live"
    run_backfill(mode=run_mode)
