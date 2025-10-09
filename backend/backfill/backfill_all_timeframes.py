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
START_YEAR = 2007
END_YEAR = datetime.now().year

# A dictionary to map MT5 constants to the strings we'll use in table names.
TIMEFRAMES_TO_FETCH: Dict[int, str] = {
    mt5.TIMEFRAME_M2: "2m",
    mt5.TIMEFRAME_M3: "3m",
    # mt5.TIMEFRAME_M5: "5m",
    # mt5.TIMEFRAME_M10: "10m",
    # mt5.TIMEFRAME_M15: "15m",  # Note: MT5 does not have M12, M15 is standard
    # mt5.TIMEFRAME_M30: "30m",
    # mt5.TIMEFRAME_H1: "1h",
    # mt5.TIMEFRAME_D1: "1d",
    # mt5.TIMEFRAME_W1: "1w",
    # mt5.TIMEFRAME_MN1: "1mn",
}


# Accepts table_name as an argument to be reusable
def create_market_data_table(conn, table_name: str):
    """Creates the schema and table if they don't exist."""
    schema_name = table_name.split(".")[0]

    create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {schema_name};"
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
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
        print(blue + f"Ensuring table '{table_name}' exists..." + reset)
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
    # Use .get('spread', 0) in case the 'spread' column is not always present
    rates_df["spread"] = rates_df.get("spread", 0)
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


# MODIFIED: Accepts table_name as an argument
def store_data_in_db(
    market_data: List[Dict[str, Any]], table_name: str, is_full_backfill: bool = False
) -> int:
    """Stores a batch of candle data. If in 'full' mode, it will drop the table first."""
    if not market_data:
        return 0

    try:
        with get_db_connection() as conn:
            if is_full_backfill:
                print(yellow + f"FULL MODE: Dropping table {table_name}..." + reset)
                with conn.cursor() as cursor:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
                conn.commit()

            create_market_data_table(conn, table_name)

            query = f"""
                INSERT INTO {table_name}
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
        print(red + f"Failed to insert data into {table_name}: {e}" + reset)
        return 0


# MODIFIED: The main function now loops through all timeframes
def run_backfill(mode: str, timeframes: Dict[int, str]):
    """Main function to fetch and store historical data for multiple timeframes."""
    print(blue + f"--- Starting Historical Data Backfill (Mode: {mode}) ---" + reset)

    for timeframe_const, timeframe_str in timeframes.items():
        print(
            yellow
            + f"\n===== Processing Timeframe: {timeframe_str.upper()} ====="
            + reset
        )

        table_name = f"market_data.{SYMBOL.lower()}_{timeframe_str}"
        total_records_stored = 0
        is_first_run_in_full_mode = True

        for year in range(START_YEAR, END_YEAR + 1):
            start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

            print(f"\nFetching {timeframe_str} data for {year}...")
            try:
                with connect_mt5():
                    rates = mt5.copy_rates_range(
                        SYMBOL, timeframe_const, start_date, end_date
                    )
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
            count = store_data_in_db(
                formatted_candles, table_name, is_full_backfill=should_drop_table
            )

            if should_drop_table:
                is_first_run_in_full_mode = False

            print(
                green
                + f"Stored {count} new records for {year} in {table_name}."
                + reset
            )
            total_records_stored += count

        print(blue + f"\n--- Timeframe {timeframe_str.upper()} Complete! ---" + reset)
        print(
            green
            + f"Total new records for this timeframe: {total_records_stored}"
            + reset
        )

    print(yellow + "\n\n--- Full Backfill For All Timeframes Complete! ---" + reset)
    print(
        yellow
        + "\nTo analyze the data for gaps, run the 'analyze_data_integrity.py' script."
        + reset
    )


if __name__ == "__main__":
    run_mode = "full" if len(sys.argv) > 1 and sys.argv[1] == "full" else "live"
    run_backfill(mode=run_mode, timeframes=TIMEFRAMES_TO_FETCH)
