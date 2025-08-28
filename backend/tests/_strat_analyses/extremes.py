# extremes_check_v2.py
import pandas as pd

# --- Inputs ---
price_fname = "eurusd_sma_breakout_with_high_low_close.csv"
breakouts_fname = "filtered_breakouts.csv"

# --- Load data ---
px = pd.read_csv(price_fname, parse_dates=["time"]).set_index("time").sort_index()
br = pd.read_csv(breakouts_fname, parse_dates=["time", "next_time"]).sort_values("time")

rows = []
for _, r in br.iterrows():
    start, end, side = r["time"], r["next_time"], r.get("side", None)
    if pd.isna(end):
        continue  # skip last open segment

    win = px.loc[start:end]
    if win.empty:
        continue

    start_px = win["close"].iloc[0]
    end_px = win["close"].iloc[-1]
    hi = float(win["high"].max())
    lo = float(win["low"].min())

    # Net movement (close-to-close)
    if side == "long":
        net = end_px - start_px
    else:
        net = start_px - end_px

    # Extremes vs entry
    if side == "long":
        best = hi - start_px
        worst = lo - start_px
    else:  # short
        best = start_px - lo
        worst = start_px - hi

    rows.append(
        {
            **r.to_dict(),
            "span_hours": (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            / 3600,
            "start_price": float(start_px),
            "end_price": float(end_px),
            "net_move_pips": net * 10000,
            "best_inbar_pips": best * 10000,
            "worst_inbar_pips": worst * 10000,
            "range_pips": (hi - lo) * 10000,
        }
    )

out = pd.DataFrame(rows)
out.to_csv("filtered_breakouts_with_extremes.csv", index=False)
print(f"Saved -> filtered_breakouts_with_extremes.csv ({len(out)} segments)")

# Quick sanity
if not out.empty:
    print("Median net (pips):", out["net_move_pips"].median())
    print("Median best (pips):", out["best_inbar_pips"].median())
    print("Median worst (pips):", out["worst_inbar_pips"].median())
    print("Median full-range (pips):", out["range_pips"].median())
