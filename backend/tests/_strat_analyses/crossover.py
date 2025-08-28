# Crossover


import pandas as pd

# Load SMA-core
core_fname = "eurusd_sma_core.csv"
df = (
    pd.read_csv(core_fname, parse_dates=["time"])
    .sort_values("time")
    .reset_index(drop=True)
)


# --- helpers ---
def crossover(a, b):
    above = a > b
    cross = above.ne(above.shift())
    return cross & above


def crossunder(a, b):
    above = a > b
    cross = above.ne(above.shift())
    return cross & ~above


# 1) 12/24 crosses
df["cross12_24"] = crossover(df.sma_12, df.sma_24) | crossunder(df.sma_12, df.sma_24)

# 2) 12 or 24 vs 50 cross
df["cross12_24_50"] = crossover(df.sma_12, df.sma_50) | crossunder(df.sma_12, df.sma_50)

# 3) Triple alignment vs 130; 'all3' is the aligned direction
all3 = (df.sma_12 > df.sma_130) & (df.sma_24 > df.sma_130) & (df.sma_50 > df.sma_130)
all3_shift = all3.shift(fill_value=False)
df["cross_all3_130"] = all3.ne(all3_shift)

# Events at triple-cross boundaries
events = df.loc[df.cross_all3_130, ["time"]].copy()

# False-breakout filter
events["false_breakout"] = False
min_cross_gap = pd.Timedelta(minutes=20)
min_cross_gap_50 = pd.Timedelta(minutes=10)

for i, row in events.iterrows():
    t0 = row.time
    mask20 = (df.time > t0) & (df.time <= t0 + min_cross_gap) & df.cross12_24
    if mask20.any():
        t1 = df.loc[mask20, "time"].iloc[0]  # type: ignore
        mask10 = (df.time > t1) & (df.time <= t1 + min_cross_gap_50) & df.cross12_24_50
        if mask10.any():
            events.loc[i, "false_breakout"] = True  # type: ignore

valid = events[~events.false_breakout].copy()

# Duration until next valid event
valid["next_time"] = valid.time.shift(-1)
valid["duration"] = (valid.next_time - valid.time).dt.total_seconds() / 60

# Add 'side' at the start of the valid segment
core_idx = df.set_index("time")
side_series = []
for t in valid["time"]:
    if t not in core_idx.index:
        # nearest index if exact stamp not present
        pos = core_idx.index.get_indexer([t], method="nearest")[0]
        t_use = core_idx.index[pos]
    else:
        t_use = t
    row = core_idx.loc[t_use]
    aligned_above = (
        (row.sma_12 > row.sma_130)
        and (row.sma_24 > row.sma_130)
        and (row.sma_50 > row.sma_130)
    )
    side_series.append("long" if aligned_above else "short")  # type: ignore

valid["side"] = side_series

print("Total breakout events:", len(events))
print("False breakouts removed:", int(events.false_breakout.sum()))
print("Valid breakouts left:", len(valid))
print("Average duration (min):", float(valid.duration.mean()))

valid.to_csv("filtered_breakouts.csv", index=False)
print("Saved -> filtered_breakouts.csv")

# Inputs
price_fname = (
    "eurusd_sma_breakout_with_high_low_close.csv"  # has: close, high, low, time
)
core_fname = "eurusd_sma_core.csv"  # for re-deriving side if needed
breakouts_fname = "filtered_breakouts.csv"  # from crossover_fixed.py

# Load
px = pd.read_csv(price_fname, parse_dates=["time"]).set_index("time").sort_index()
core = pd.read_csv(core_fname, parse_dates=["time"]).set_index("time").sort_index()
br = pd.read_csv(breakouts_fname, parse_dates=["time", "next_time"]).sort_values("time")

# Ensure columns exist
for col in ["close", "high", "low"]:
    if col not in px.columns:
        raise ValueError(f"Missing column '{col}' in {price_fname}")

# If 'side' missing, compute from core at start time
if "side" not in br.columns:
    sides = []
    for t in br["time"]:
        if t not in core.index:
            pos = core.index.get_indexer([t], method="nearest")[0]
            t_use = core.index[pos]
        else:
            t_use = t
        r = core.loc[t_use]
        aligned_above = (
            (r.sma_12 > r.sma_130) and (r.sma_24 > r.sma_130) and (r.sma_50 > r.sma_130)
        )
        sides.append("long" if aligned_above else "short")  # type: ignore
    br["side"] = sides

rows = []
for _, r in br.iterrows():
    start, end, side = r["time"], r["next_time"], r["side"]
    if pd.isna(end):
        continue  # last open segment; skip
    # slice prices
    win = px.loc[start:end]
    if win.empty:
        continue
    start_px = win["close"].iloc[0]
    end_px = win["close"].iloc[-1]
    hi = float(win["high"].max())
    lo = float(win["low"].min())

    if side == "long":
        net = end_px - start_px
        best = hi - start_px
        worst = lo - start_px
    else:  # short
        net = start_px - end_px
        best = start_px - lo
        worst = start_px - hi

    rows.append(
        {
            **r.to_dict(),
            "start_price": float(start_px),
            "end_price": float(end_px),
            "highest_price": hi,  # NEW
            "lowest_price": lo,  # NEW
            "net_move_abs": float(net),
            "best_inbar_abs": float(best),
            "worst_inbar_abs": float(worst),
            "net_move_pct": float(net / start_px * 100.0),
            "best_inbar_pct": float(best / start_px * 100.0),
            "worst_inbar_pct": float(worst / start_px * 100.0),
        }
    )

out = pd.DataFrame(rows)
out.to_csv("filtered_breakouts_with_moves.csv", index=False)

# Quick summary
if not out.empty:
    print("Segments:", len(out))
    print("Median net %:", out["net_move_pct"].median())
    print("Mean net %:", out["net_move_pct"].mean())
    print("Median best %:", out["best_inbar_pct"].median())
    print("Median worst %:", out["worst_inbar_pct"].median())
else:
    print("No segments generated.")
