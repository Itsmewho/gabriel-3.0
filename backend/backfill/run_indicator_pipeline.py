import pandas as pd
import sys
import time
from sqlalchemy import create_engine, text

# Import your configuration
from config.configure import DATABASE_URL

# --- Modular Feature Calculation Imports ---
from indicators.classical_layer import (
    calculate_standard_indicators,
    calculate_bollinger_bands,
    calculate_markov_states,
)
from indicators.multiproces.candlestick import analyze_candlestick_windows


# --- Configuration ---
DATABASE_URI = DATABASE_URL
SOURCE_SCHEMA = "market_data"
SOURCE_TABLE = "eurusd_1m"
INDICATOR_SCHEMA = "technical_indicators"

# These windows are now only used for the 'live' mode fetch limit
LONG_TERM_WINDOW = 10080
SHORT_TERM_WINDOW = 5760
MAX_FETCH_LIMIT = LONG_TERM_WINDOW + 200


# --- Database Interaction (No changes here) ---
def create_indicator_table(
    engine, schema: str, table_name: str, columns_sql: str, is_full_backfill: bool
):
    # ... (this function is correct)
    try:
        with engine.connect() as connection:
            connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
            if is_full_backfill:
                print(
                    f"  - Dropping old table '{schema}.{table_name}' to ensure clean schema..."
                )
                connection.execute(
                    text(f"DROP TABLE IF EXISTS {schema}.{table_name} CASCADE")
                )
            query = f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                id SERIAL PRIMARY KEY,
                time TIMESTAMP WITH TIME ZONE NOT NULL UNIQUE,
                {columns_sql}
            );
            CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {schema}.{table_name}(time);
            """
            connection.execute(text(query))
            connection.commit()
        print(f"Table '{schema}.{table_name}' is ready.")
    except Exception as e:
        print(f"Error creating table {schema}.{table_name}: {e}")
        raise


def store_dataframe(engine, df: pd.DataFrame, schema: str, table_name: str):
    temp_table_name = f"temp_{table_name}"
    df_to_save = df.dropna()
    if df_to_save.empty:
        print(f"  - No new data to store for '{schema}.{table_name}'. Skipping.")
        return

    df_to_save.to_sql(
        temp_table_name,
        engine,
        schema=schema,
        if_exists="replace",
        index=True,
        index_label="time",
    )
    with engine.connect() as connection:
        cols = ", ".join([f'"{c}"' for c in df_to_save.columns])
        update_cols = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in df_to_save.columns])
        merge_query = f"""
        INSERT INTO "{schema}"."{table_name}" (time, {cols})
        SELECT time, {cols} FROM "{schema}"."{temp_table_name}"
        ON CONFLICT (time) DO UPDATE SET {update_cols};
        """
        connection.execute(text(merge_query))
        connection.execute(text(f'DROP TABLE "{schema}"."{temp_table_name}";'))
        connection.commit()


# --- Main Pipeline Runner ---
def run_indicator_pipeline(mode: str = "live", skip_evals: bool = False):
    """
    Main function to run all calculations with optimized data slicing.
    Includes a flag to skip the time-consuming trend evaluation.
    """
    try:
        print(f"Connecting to database for '{mode}' run...")
        engine = create_engine(DATABASE_URI)

        if mode == "live":
            source_query = f"SELECT * FROM {SOURCE_SCHEMA}.{SOURCE_TABLE} ORDER BY time DESC LIMIT {MAX_FETCH_LIMIT}"
        else:
            source_query = f"SELECT * FROM {SOURCE_SCHEMA}.{SOURCE_TABLE} ORDER BY time"

        print(f"Loading raw data from '{SOURCE_SCHEMA}.{SOURCE_TABLE}'...")
        start_time = time.time()
        df = pd.read_sql(source_query, engine, index_col="time", parse_dates=["time"])

        if mode == "live":
            df = df.iloc[::-1]

        # --- Data Cleaning ---
        initial_row_count = len(df)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        dropped_rows = initial_row_count - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to invalid numeric values.")

        if mode == "full":
            df_for_calc = df
            print(
                "Running in FULL mode, all calculations will use the complete dataset."
            )
        else:
            df_for_calc = df.tail(SHORT_TERM_WINDOW)
            print(
                f"Running in LIVE mode, calculations will use the last {SHORT_TERM_WINDOW} data points."
            )

        is_full_backfill = mode == "full"

        # --- Layer 1 Calculations ---
        print("\n--- Processing Layer 1: Classical Indicators ---")
        standard_indicators = calculate_standard_indicators(df_for_calc.copy())
        for name, data in standard_indicators.items():
            table_name_suffix = (
                "_".join(map(str, name)) if isinstance(name, tuple) else str(name)
            )
            table_name = f"{SOURCE_TABLE}_{table_name_suffix}"
            create_indicator_table(
                engine,
                INDICATOR_SCHEMA,
                table_name,
                "value DOUBLE PRECISION",
                is_full_backfill,
            )
            store_dataframe(
                engine, pd.DataFrame({"value": data}), INDICATOR_SCHEMA, table_name
            )

        bb_df = calculate_bollinger_bands(df_for_calc.copy())
        create_indicator_table(
            engine,
            INDICATOR_SCHEMA,
            f"{SOURCE_TABLE}_bb",
            "lower DOUBLE PRECISION, middle DOUBLE PRECISION, upper DOUBLE PRECISION, width DOUBLE PRECISION, percent DOUBLE PRECISION",
            is_full_backfill,
        )
        store_dataframe(engine, bb_df, INDICATOR_SCHEMA, f"{SOURCE_TABLE}_bb")

        markov_df = calculate_markov_states(df_for_calc.copy())
        create_indicator_table(
            engine,
            INDICATOR_SCHEMA,
            f"{SOURCE_TABLE}_markov",
            "state TEXT",
            is_full_backfill,
        )
        store_dataframe(engine, markov_df, INDICATOR_SCHEMA, f"{SOURCE_TABLE}_markov")

        if not skip_evals:
            print("\n--- Processing Candlestick Trend Evaluation ---")
            analysis_windows = [2, 3, 4, 5, 10, 15, 30, 60, 480, 1440, 10080]
            candlestick_results = analyze_candlestick_windows(
                df.copy(), windows=analysis_windows
            )
            for name, analysis_df in candlestick_results.items():
                table_name = f"{SOURCE_TABLE}_{name}"
                columns_sql = "upper_wick_trend TEXT, lower_wick_trend TEXT, candle_body_trend TEXT, price_pressure INTEGER"
                create_indicator_table(
                    engine, INDICATOR_SCHEMA, table_name, columns_sql, is_full_backfill
                )
                store_dataframe(engine, analysis_df, INDICATOR_SCHEMA, table_name)
        else:
            print("\n--- Skipping Candlestick Trend Evaluation as requested ---")

        print(f"\nIndicator pipeline ({mode} mode) completed successfully!")
        return True

    except Exception as e:
        print(f"\nAn error occurred in the indicator pipeline: {e}")
        return False


if __name__ == "__main__":
    run_mode = "full" if "full" in sys.argv else "live"
    should_skip_evals = "--skip-evals" in sys.argv

    print(f"Running in standalone mode for a '{run_mode}' backfill...")
    if should_skip_evals:
        print("Trend evaluations will be SKIPPED.")

    run_indicator_pipeline(mode=run_mode, skip_evals=should_skip_evals)
