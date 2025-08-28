import pandas as pd
import numpy as np

df = pd.read_csv(
    "eurusd_sma_breakout_with_high_low_close.csv", parse_dates=["time"]
).sort_values("time")
for c in ["high", "low", "close"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 1) Check typical tick size; EURUSD ~0.0001 on 4dp/5dp feeds
step = df["close"].diff().abs()
print("median step:", step.replace(0, np.nan).median())

# 2) Detect rows likely off by 10x (rare but happens in legacy merges)
# Heuristic: price far from rolling median (factor ~10)
med = df["close"].rolling(500, min_periods=100).median()
ratio = (df["close"] / med).abs()
bad_hi = ratio > 5  # e.g., 14.5 vs 1.45
bad_lo = ratio < 0.2  # e.g., 0.145 vs 1.45

print("rows 10x high:", bad_hi.sum(), "rows 10x low:", bad_lo.sum())

# 3) Optional auto-fix: rescale obvious 10x outliers back to local median scale
df.loc[bad_hi, ["high", "low", "close"]] /= 10.0
df.loc[bad_lo, ["high", "low", "close"]] *= 10.0
