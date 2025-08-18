import pandas as pd
import numpy as np
import mplfinance as mpf
import time


def make_data(n):
    idx = pd.date_range("2020-01-01", periods=n, freq="T")
    price = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame(
        {
            "Open": price,
            "High": price + np.random.rand(n),
            "Low": price - np.random.rand(n),
            "Close": price + np.random.randn(n) * 0.5,
            "Volume": np.random.randint(100, 1000, size=n),
        },
        index=idx,
    )
    return df


for n in [1000, 5000, 10000, 20000, 50000, 100000, 200000]:
    df = make_data(n)
    t0 = time.time()
    try:
        mpf.plot(
            df,
            type="candle",
            style="charles",
            volume=False,
            warn_too_much_data=len(df) + 1,
            savefig=dict(fname=f"test_{n}.png", dpi=600),
        )
        dt = time.time() - t0
        print(f"{n:>6} rows: {dt:.2f}s")
    except Exception as e:
        print(f"{n:>6} rows: failed ({e})")
