# extremes_check.py
import pandas as pd

PX_CSV = "eurusd_sma_breakout_with_high_low_close.csv"  # time, close, high, low
BR_CSV = "filtered_breakouts.csv"  # time, next_time, side, duration
OUT_CSV = "filtered_breakouts_with_extremes.csv"

# Load data
px = pd.read_csv(PX_CSV, parse_dates=["time"]).set_index("time").sort_index()
br = pd.read_csv(BR_CSV, parse_dates=["time", "next_time"]).sort_values("time")

rows = []
for _, r in br.iterrows():
    start, end, side = r["time"], r["next_time"], r.get("side", None)
    if pd.isna(end):  # last open segment -> skip
        continue
    win = px.loc[start:end]
    if win.empty:
        continue

    # Prices at ends
    s_close = float(win["close"].iloc[0])
    e_close = float(win["close"].iloc[-1])

    # Intraperiod extremes + when they occurred
    w_high = float(win["high"].max())
    t_high = win["high"].idxmax()
    w_low = float(win["low"].min())
    t_low = win["low"].idxmin()

    # Directional P/L style moves
    if side == "long":
        net = e_close - s_close
        best = w_high - s_close
        worst = w_low - s_close
    else:  # short
        net = s_close - e_close
        best = s_close - w_low
        worst = s_close - w_high

    rows.append(
        {
            **r.to_dict(),
            "start_price": s_close,
            "end_price": e_close,
            "window_high": w_high,
            "window_low": w_low,
            "t_window_high": t_high,
            "t_window_low": t_low,
            "range_abs": w_high - w_low,
            "range_pct": (w_high - w_low) / s_close * 100.0,
            "net_abs": net,
            "net_pct": net / s_close * 100.0,
            "best_abs": best,
            "best_pct": best / s_close * 100.0,
            "worst_abs": worst,
            "worst_pct": worst / s_close * 100.0,
            "span_hours": (end - start).total_seconds() / 3600.0,
        }
    )

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"Saved -> {OUT_CSV} ({len(out)} segments)")

# Quick diagnostics: long spans with tiny net but big intraperiod moves
if not out.empty:
    tiny_net = out[(out["span_hours"] > 24) & (out["net_pct"].abs() < 0.05)]
    print(f"Long>24h & ~flat net: {len(tiny_net)}")
    if not tiny_net.empty:
        print(
            tiny_net.sort_values(["span_hours", "range_pct"], ascending=[False, False])
            .head(10)[
                [
                    "time",
                    "next_time",
                    "side",
                    "span_hours",
                    "start_price",
                    "end_price",
                    "net_pct",
                    "range_pct",
                    "best_pct",
                    "worst_pct",
                    "t_window_high",
                    "t_window_low",
                ]
            ]
            .to_string(index=False)
        )
