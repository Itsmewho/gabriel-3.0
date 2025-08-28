# Breakouts

import pandas as pd

df = pd.read_csv("eurusd_sma_core.csv", parse_dates=["time"])
df.set_index("time", inplace=True)

breakouts = pd.read_csv("filtered_breakouts.csv", parse_dates=["time", "next_time"])

rows = []
for _, br in breakouts.iterrows():
    start, end = br["time"], br["next_time"]
    side = br["side"]

    period_df = df.loc[start:end]
    if period_df.empty:
        continue

    start_price = period_df["Close"].iloc[0]
    end_price = period_df["Close"].iloc[-1]

    # absolute high/low extremes
    high = period_df["High"].max()
    low = period_df["Low"].min()

    if side == "long":
        price_change = end_price - start_price
        extreme_move = high - start_price  # best case
        adverse_move = low - start_price  # worst case
    else:  # short
        price_change = start_price - end_price
        extreme_move = start_price - low  # best case
        adverse_move = start_price - high  # worst case

    rows.append(
        {
            **br,
            "start_price": start_price,
            "end_price": end_price,
            "price_change": price_change,
            "extreme_move": extreme_move,
            "adverse_move": adverse_move,
        }
    )

out = pd.DataFrame(rows)
out.to_csv("filtered_breakouts_with_moves.csv", index=False)
print(f"Saved -> filtered_breakouts_with_moves.csv ({len(out)} records)")
