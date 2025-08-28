import glob, pandas as pd

files = glob.glob(r"C:\Users\Itsme\Desktop\Gabriel-3.0\backend\z_signal\*.csv")
print("Found:", len(files), "files")
for f in files[:3]:  # peek at 3 files
    print("\n---", f)
    df = pd.read_csv(f)
    print(df.columns.tolist()[:15])  # show first 15 cols
    print("Rows:", len(df))
    break

import glob, os, pandas as pd
from datetime import timedelta

INPUT_GLOB = r"C:\Users\Itsme\Desktop\Gabriel-3.0\backend\z_signal\*.csv"
TIME_TOL = pd.Timedelta(minutes=20)  # your ±20m window


def pick_col(df, *cands, required=False):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing any of columns: {cands}")
    return None


def load_one(path):
    df = pd.read_csv(path)
    if df.empty:
        return None

    # required time columns
    open_col = pick_col(df, "open_time", "entry_time", required=True)
    close_col = pick_col(df, "close_time", "exit_time", required=True)

    # side & pnl (many names possible)
    side_col = pick_col(df, "side")
    pnl_col = pick_col(df, "net_pnl", "pnl", "gross_pnl", "pnl_abs")

    # strategy identity
    strat_col = pick_col(df, "strategy_id", "strategy", "name")
    magic_col = pick_col(df, "magic_number", "magic")

    out = pd.DataFrame(
        {
            "file": os.path.basename(path),
            "open_time": pd.to_datetime(df[open_col], errors="coerce"),
            "close_time": pd.to_datetime(df[close_col], errors="coerce"),
            "side": df[side_col] if side_col else None,
            "pnl": pd.to_numeric(df[pnl_col], errors="coerce") if pnl_col else None,
            "strategy": (
                df[strat_col]
                if strat_col in df.columns
                else os.path.splitext(os.path.basename(path))[0]
            ),
            "magic": df[magic_col] if magic_col else None,
        }
    )
    out = out.dropna(subset=["open_time", "close_time"])
    return out


# ---- Load all audits
paths = glob.glob(INPUT_GLOB)
all_trades = []
for p in paths:
    try:
        d = load_one(p)
        if d is not None and not d.empty:
            all_trades.append(d)
    except Exception as e:
        print(f"[WARN] {os.path.basename(p)}: {e}")

if not all_trades:
    print("No trades found after parsing.")
    raise SystemExit

df = pd.concat(all_trades, ignore_index=True)
df = df.dropna(subset=["open_time"])  # safety

# ---- Top 100 winners per file (or adjust per strategy if you prefer)
top_winners = (
    df.dropna(subset=["pnl"])
    .sort_values(["file", "pnl"], ascending=[True, False])
    .groupby("file")
    .head(100)
    .reset_index(drop=True)
)

# ---- Pairwise alignment check: do other strategies have a trade within ±20m of this open_time?
# mark whether any OTHER strategy had a same-time signal
top_winners = top_winners.sort_values("open_time")
top_winners["aligned_hits"] = 0
top_winners["aligned_same_dir"] = 0

# quick index by time for speed
df_idx = df.set_index("open_time").sort_index()

for i, r in top_winners.iterrows():
    t0 = r.open_time
    a = df_idx.loc[t0 - TIME_TOL : t0 + TIME_TOL]
    if a.empty:
        continue
    # exclude same-origin record by (file,id) if you added id; here we exclude same file+strategy
    a = a[(a["strategy"] != r["strategy"]) | (a["file"] != r["file"])]
    if a.empty:
        continue
    top_winners.at[i, "aligned_hits"] = len(a)
    if r["side"] is not None and "side" in a.columns:
        same = a[a["side"].str.lower() == str(r["side"]).lower()]
        top_winners.at[i, "aligned_same_dir"] = len(same)

# quick summary
print("Files parsed:", len(paths))
print("Trades parsed:", len(df))
print("Top winners:", len(top_winners))
print("Aligned (any):", int((top_winners["aligned_hits"] > 0).sum()))
print("Aligned (same dir):", int((top_winners["aligned_same_dir"] > 0).sum()))

top_winners.to_csv("aligned_top_winners.csv", index=False)
print("Saved aligned_top_winners.csv")
