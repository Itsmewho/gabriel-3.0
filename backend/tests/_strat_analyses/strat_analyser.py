# CSV Builder

import pandas as pd

df = pd.read_parquet("results/cache/EURUSD_1m_2009-02-01_2010-10-01_features.parquet")
df_out = df[
    ["sma_12", "sma_24", "sma_50", "sma_130", "close", "high", "low", "open"]
].copy()
df_out["time"] = df.index
df_out.to_csv("eurusd_sma_breakout_with_high_low_close.csv", index=False)
