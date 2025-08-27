import pandas as pd
from typing import List, Optional, Sequence, Any
from utils.helpers import setup_logger
from operations.sql_operations import execute_query

logger = setup_logger(__name__)


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

        result = execute_query(query, params, fetchall=True)
        return result if result is not None else []
    except Exception as e:
        logger.error(f"Fetch failed for {table_name}: {e}")
        return []


def fetch_sql_market_data(symbol, timeframe, where_clause: Optional[str] = None):
    table_name = f"market_data.{symbol.lower()}_{timeframe}"
    logger.info(f"Fetching base market data from {table_name}...")
    try:
        data = fetch_records(table_name, where_clause=where_clause)
    except Exception as e:
        logger.error(f"Database error fetching market data: {e}")
        return pd.DataFrame()

    if not data:
        logger.error(f"No SQL data found for {symbol} {timeframe}.")
        return pd.DataFrame()

    columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "change",
        "change_percent",
    ]
    return pd.DataFrame(data, columns=columns)


# --- ECONOMIC EVENT FEATURES ---
def fetch_event_features(symbol, candles_df):
    if candles_df.empty:
        return candles_df

    base, quote = symbol[:3], symbol[3:]
    raw_events = fetch_records(
        "public.economic_events",
        where_clause="currency = ANY(%s)",
        params=([base, quote],),
    )

    if not raw_events:
        logger.warning("No relevant economic events found in the database.")
        for col, default in get_event_column_defaults().items():
            candles_df[col] = default
        return candles_df

    df_events = pd.DataFrame(
        raw_events,
        columns=[
            "id",
            "event_date",
            "event_time",
            "currency",
            "event_name",
            "impact",
            "actual",
            "forecast",
            "previous",
            "source",
            "description",
            "is_revised",
            "created_at",
        ],
    )

    df_events["time"] = pd.to_datetime(
        df_events["event_date"].astype(str) + " " + df_events["event_time"].astype(str),
        utc=False,
        errors="coerce",
    )
    df_events.sort_values("time", inplace=True)
    df_events.rename(columns={"time": "event_time_actual"}, inplace=True)
    df_events["impact_score"] = (
        df_events["impact"]
        .map(
            {
                "Low Impact Expected": 1,
                "Medium Impact Expected": 2,
                "High Impact Expected": 3,
            }
        )
        .fillna(0)
        .astype(int)
    )
    df_events["actual"] = pd.to_numeric(df_events["actual"], errors="coerce")
    df_events["forecast"] = pd.to_numeric(df_events["forecast"], errors="coerce")
    df_events["event_surprise"] = df_events["actual"] - df_events["forecast"]

    candles_df["time"] = candles_df["time"].dt.tz_localize(None)
    df_events["event_time_actual"] = df_events["event_time_actual"].dt.tz_localize(None)

    # Perform the asof merge
    pre_event_df = pd.merge_asof(
        left=candles_df.sort_values("time"),
        right=df_events,
        left_on="time",
        right_on="event_time_actual",
        direction="backward",
        tolerance=pd.Timedelta(minutes=30),
    )
    post_event_df = pd.merge_asof(
        left=candles_df.sort_values("time"),
        right=df_events,
        left_on="time",
        right_on="event_time_actual",
        direction="forward",
        tolerance=pd.Timedelta(minutes=30),
    )

    pre_event_df.index = candles_df.index
    post_event_df.index = candles_df.index

    for col, default in get_event_column_defaults().items():
        candles_df[col] = default

    pre_mask = pre_event_df["event_name"].notna()
    candles_df.loc[pre_mask, "event_impact_score"] = pre_event_df.loc[
        pre_mask, "impact_score"
    ]
    candles_df.loc[pre_mask, "event_name"] = pre_event_df.loc[pre_mask, "event_name"]
    candles_df.loc[pre_mask, "event_forecast"] = pre_event_df.loc[pre_mask, "forecast"]
    candles_df.loc[pre_mask, "event_previous"] = pre_event_df.loc[pre_mask, "previous"]
    candles_df.loc[pre_mask, "event_currency"] = pre_event_df.loc[pre_mask, "currency"]
    candles_df.loc[pre_mask, "event_minutes_to_event"] = (
        pre_event_df.loc[pre_mask, "event_time_actual"]
        - pre_event_df.loc[pre_mask, "time"]
    ).dt.total_seconds() / 60.0

    post_mask = post_event_df["event_name"].notna()
    candles_df.loc[post_mask, "event_is_economic"] = 1

    column_mapping = {
        "actual": "event_actual",
        "event_surprise": "event_surprise",
        "currency": "event_currency",
    }
    for source_col, dest_col in column_mapping.items():
        if source_col in post_event_df.columns:
            candles_df.loc[post_mask, dest_col] = post_event_df.loc[
                post_mask, source_col
            ]
        else:
            logger.warning(
                f"Source column '{source_col}' not found in post_event_df during event feature creation."
            )

    candles_df["event_name_id"] = pd.factorize(candles_df["event_name"])[0]
    candles_df["event_currency_id"] = pd.factorize(candles_df["event_currency"])[0]
    candles_df.drop(columns=["event_event_name"], errors="ignore", inplace=True)
    candles_df.fillna(get_event_column_defaults(), inplace=True)

    return candles_df


def get_event_column_defaults():
    return {
        "event_minutes_to_event": 0.0,
        "event_impact_score": 0,
        "event_forecast": 0.0,
        "event_previous": 0.0,
        "event_actual": 0.0,
        "event_surprise": 0.0,
        "event_is_economic": 0,
        "event_name": "none",
        "event_currency": "none",
        "event_name_id": 0,
        "event_currency_id": 0,
    }
