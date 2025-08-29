# Strategy Signal Overlap Analyzer
# --------------------------------
# Drop-in script to compare multiple strategy audit CSVs, find the top/bottom/middle trades,
# and measure how often strategies fire signals within a time window, and whether they agree on direction.
#
# Usage:
#   - Put your audit CSVs into a folder.
#   - Adjust INPUT_GLOB below (e.g., "path/to/*.csv").
#   - Run this script. It writes outputs to results/consensus/.
#
# Notes:
#   - Expects audit files with columns like:
#       id, strategy_id (or magic_number), side, lots, entry_time, exit_time, entry_price, exit_price, pnl, ...
#   - If strategy_id is missing, we try to derive it from filename or magic_number.
#   - “Average” bucket = trades with pnl closest to 0 (by absolute value).
#
# Outputs:
#   - results/consensus/all_trades_merged.csv
#   - results/consensus/winners_overlap_summary.csv
#   - results/consensus/losers_overlap_summary.csv
#   - results/consensus/average_overlap_summary.csv
#   - results/consensus/pairwise_agreement_*.csv (agreement rates between strategy pairs)
#
# Strategy Signal Overlap Analyzer (with Buy/Sell breakdowns)
# -----------------------------------------------------------
import os
import re
import glob
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# ---------------- Config ----------------
INPUT_GLOB = r"C:\Users\Itsme\Desktop\Gabriel-3.0\backend\z_signal\*.csv"
TOP_N = 150
WINDOW_MINUTES = 10
TRIPLE_WINDOW_MINUTES = 10

OUT_DIR = Path("results/consensus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIPLE_OUT_DIR = OUT_DIR  # reuse results/consensus


# -------------- Helpers ---------------
def _infer_strategy_name_from_filename(fp: str) -> str:
    base = os.path.basename(fp)
    name = os.path.splitext(base)[0]
    m = re.search(r"(SMA|EMA|RSI|ICHIMOKU|MACD)[-_]?[A-Z]*", name, re.IGNORECASE)
    return m.group(0).upper() if m else name.upper()


def _standardize_columns(df: pd.DataFrame, src: str) -> pd.DataFrame:
    d = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=d)

    if "entry_time" not in df.columns and "open_time" in df.columns:
        df["entry_time"] = df["open_time"]
    if "exit_time" not in df.columns and "close_time" in df.columns:
        df["exit_time"] = df["close_time"]

    for col in ("entry_time", "exit_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if "strategy_id" not in df.columns:
        if "magic_number" in df.columns:
            df["strategy_id"] = df["magic_number"].astype(str)
        else:
            df["strategy_id"] = _infer_strategy_name_from_filename(src)

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.lower()

    if "pnl" not in df.columns:
        for alt in ["net_pnl", "profit", "pl", "p_l"]:
            if alt in df.columns:
                df["pnl"] = pd.to_numeric(df[alt], errors="coerce")
                break
        if "pnl" not in df.columns:
            df["pnl"] = np.nan

    keep = [
        "id",
        "strategy_id",
        "side",
        "lots",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "pnl",
    ]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    out = df[keep].copy()
    out["source_file"] = os.path.basename(src)
    return out


def load_all_audits(glob_pat: str) -> pd.DataFrame:
    files = glob.glob(glob_pat)
    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
        frames.append(_standardize_columns(df, fp))
    if not frames:
        return pd.DataFrame(
            columns=[
                "id",
                "strategy_id",
                "side",
                "lots",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "pnl",
                "source_file",
            ]
        )
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["entry_time"])
    return all_df


def bucketize(df: pd.DataFrame, top_n: int) -> Dict[str, pd.DataFrame]:
    winners = df.sort_values("pnl", ascending=False).groupby("source_file").head(top_n)
    losers = df.sort_values("pnl", ascending=True).groupby("source_file").head(top_n)
    avg_rows = []
    for src, g in df.groupby("source_file"):
        g = g.copy()
        g["abs_pnl"] = g["pnl"].abs()
        avg_rows.append(g.sort_values("abs_pnl", ascending=True).head(top_n))
    average = pd.concat(avg_rows, ignore_index=True) if avg_rows else df.head(0)
    return {"winners": winners, "losers": losers, "average": average}


def calc_overlaps(
    bucket_df: pd.DataFrame, window_min: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if bucket_df.empty:
        return bucket_df.head(0), pd.DataFrame()

    df = bucket_df[
        ["strategy_id", "entry_time", "side", "pnl", "source_file", "id"]
    ].copy()
    df = df.dropna(subset=["entry_time"])
    df["strategy_id"] = df["strategy_id"].astype(str)
    df = df.sort_values("entry_time").reset_index(drop=True)

    groups = {sid: g.reset_index(drop=True) for sid, g in df.groupby("strategy_id")}

    rows = []
    win = pd.Timedelta(minutes=window_min)
    sids = sorted(groups.keys())  # type: ignore

    for i in range(len(sids)):
        sid_a = sids[i]
        A = groups[sid_a]
        for j in range(i + 1, len(sids)):
            sid_b = sids[j]
            B = groups[sid_b]

            pa = 0
            pb = 0
            while pa < len(A) and pb < len(B):
                ta = A.loc[pa, "entry_time"]
                tb = B.loc[pb, "entry_time"]
                dt = tb - ta  # type: ignore
                if abs(dt) <= win:  # type: ignore
                    same_dir = A.loc[pa, "side"] == B.loc[pb, "side"]
                    rows.append(
                        {
                            "sid_a": sid_a,
                            "sid_b": sid_b,
                            "id_a": A.loc[pa, "id"],
                            "id_b": B.loc[pb, "id"],
                            "time_a": ta,
                            "time_b": tb,
                            "delta_minutes": abs(dt).total_seconds() / 60.0,  # type: ignore
                            "side_a": A.loc[pa, "side"],
                            "side_b": B.loc[pb, "side"],
                            "agree": bool(same_dir),
                        }
                    )
                    if ta <= tb:  # type: ignore
                        pa += 1
                    else:
                        pb += 1
                elif ta < tb - win:  # type: ignore
                    pa += 1
                else:
                    pb += 1

    overlaps = pd.DataFrame(rows)
    if overlaps.empty:
        return overlaps, overlaps

    pair = (
        overlaps.groupby(["sid_a", "sid_b"])["agree"]
        .agg(overlaps="count", agreement_rate=lambda s: float(s.mean()))
        .reset_index()
    )

    sid_list = sorted(df["strategy_id"].unique().tolist())
    mat = pd.DataFrame(index=sid_list, columns=sid_list, data=np.nan, dtype=float)
    for _, r in pair.iterrows():
        a, b, rate = r["sid_a"], r["sid_b"], r["agreement_rate"]
        mat.loc[a, b] = rate
        mat.loc[b, a] = rate
        mat.loc[a, a] = 1.0
        mat.loc[b, b] = 1.0

    return overlaps, mat


def write_bucket_outputs(name: str, bucket_df: pd.DataFrame):
    bucket_path = OUT_DIR / f"{name}_trades.csv"
    bucket_df.to_csv(bucket_path, index=False)

    overlaps, pair = calc_overlaps(bucket_df, WINDOW_MINUTES)
    overlaps_path = OUT_DIR / f"{name}_overlaps.csv"
    pair_path = OUT_DIR / f"pairwise_agreement_{name}.csv"
    overlaps.to_csv(overlaps_path, index=False)
    pair.to_csv(pair_path)

    # Pairwise summary
    if not overlaps.empty:
        summary = (
            overlaps.assign(agree_int=overlaps["agree"].astype(int))
            .groupby(["sid_a", "sid_b"])
            .agg(
                overlaps=("agree_int", "size"),
                agreements=("agree_int", "sum"),
                mean_delta_min=("delta_minutes", "mean"),
            )
            .reset_index()
            .sort_values(["sid_a", "sid_b"])
        )
    else:
        summary = pd.DataFrame(
            columns=["sid_a", "sid_b", "overlaps", "agreements", "mean_delta_min"]
        )
    summary_path = OUT_DIR / f"{name}_overlap_summary.csv"
    summary.to_csv(summary_path, index=False)

    # NEW: Pairwise side breakdown (Buy/Sell)
    if not overlaps.empty:
        # use side_a as the canonical side label for the pair row
        side_summary = (
            overlaps.assign(agree_int=overlaps["agree"].astype(int))
            .groupby(["sid_a", "sid_b", "side_a"])
            .agg(
                overlaps=("agree_int", "size"),
                agreements=("agree_int", "sum"),
                mean_delta_min=("delta_minutes", "mean"),
            )
            .reset_index()
            .rename(columns={"side_a": "side"})
        )
    else:
        side_summary = pd.DataFrame(
            columns=[
                "sid_a",
                "sid_b",
                "side",
                "overlaps",
                "agreements",
                "mean_delta_min",
            ]
        )
    side_summary_path = OUT_DIR / f"{name}_pairwise_side_summary.csv"
    side_summary.to_csv(side_summary_path, index=False)

    return {
        "bucket_path": str(bucket_path),
        "overlaps_path": str(overlaps_path),
        "pairwise_path": str(pair_path),
        "summary_path": str(summary_path),
        "pairwise_side_summary_path": str(side_summary_path),
    }


def _canon_strat_tag(s: str) -> str:
    s = str(s).upper()
    if "SMA" in s:
        return "SMA"
    if "EMA" in s:
        return "EMA"
    if "RSI" in s:
        return "RSI"
    return "OTHER"


def _prep_for_triple(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    src_tag = d["strategy_id"].fillna(d.get("source_file", ""))
    d["canon"] = src_tag.apply(_canon_strat_tag)
    d = d[d["canon"].isin(["SMA", "EMA", "RSI"])]
    d = d.dropna(subset=["entry_time", "side", "pnl"])
    d["side"] = d["side"].str.lower()
    return d


def find_triple_consensus(df_in: pd.DataFrame, window_min: int, label: str):
    """Find SMA+EMA+RSI within window_min and same direction; write hits+summary with label suffix."""
    df = _prep_for_triple(df_in)
    if df.empty:
        print(f"[{label}] No SMA/EMA/RSI trades for triple-consensus.")
        return pd.DataFrame(), pd.DataFrame()

    groups = {
        k: v.sort_values("entry_time").reset_index(drop=True)
        for k, v in df.groupby("canon")
    }
    if not all(tag in groups for tag in ("SMA", "EMA", "RSI")):
        print(f"[{label}] Missing at least one of SMA/EMA/RSI.")
        return pd.DataFrame(), pd.DataFrame()

    SMA, EMA, RSI = groups["SMA"], groups["EMA"], groups["RSI"]
    win = pd.Timedelta(minutes=window_min)

    rows = []
    i = j = 0
    while i < len(SMA) and j < len(EMA):
        ta = SMA.loc[i, "entry_time"]
        tb = EMA.loc[j, "entry_time"]
        dt = tb - ta  # type: ignore
        if abs(dt) <= win:  # type: ignore
            ref_t = ta + (dt / 2)  # type: ignore
            k = RSI["entry_time"].searchsorted(ref_t)
            cand = []
            for kk in (k - 2, k - 1, k, k + 1, k + 2):
                if 0 <= kk < len(RSI):
                    tc = RSI.loc[kk, "entry_time"]
                    if abs(tc - ta) <= win and abs(tc - tb) <= win:  # type: ignore
                        cand.append(kk)
            for kk in cand:
                if not (
                    SMA.loc[i, "side"] == EMA.loc[j, "side"] == RSI.loc[kk, "side"]
                ):
                    continue
                side = SMA.loc[i, "side"]
                rows.append(
                    {
                        "time_sma": ta,
                        "time_ema": tb,
                        "time_rsi": RSI.loc[kk, "entry_time"],
                        "delta_min_sma_ema": abs(dt).total_seconds() / 60.0,  # type: ignore
                        "delta_min_sma_rsi": abs(
                            RSI.loc[kk, "entry_time"] - ta  # type: ignore
                        ).total_seconds()  # type: ignore
                        / 60.0,
                        "delta_min_ema_rsi": abs(
                            RSI.loc[kk, "entry_time"] - tb  # type: ignore
                        ).total_seconds()  # type: ignore
                        / 60.0,
                        "side": side,
                        "id_sma": SMA.loc[i, "id"],
                        "id_ema": EMA.loc[j, "id"],
                        "id_rsi": RSI.loc[kk, "id"],
                        "file_sma": SMA.loc[i, "source_file"],
                        "file_ema": EMA.loc[j, "source_file"],
                        "file_rsi": RSI.loc[kk, "source_file"],
                        "pnl_sma": float(
                            SMA.loc[i, "pnl"]  # type: ignore
                        ),  # pyright: ignore[reportArgumentType]
                        "pnl_ema": float(EMA.loc[j, "pnl"]),  # type: ignore
                        "pnl_rsi": float(RSI.loc[kk, "pnl"]),  # type: ignore
                    }
                )
            i += 1 if ta <= tb else 0
            j += 1 if tb < ta else 0
        elif ta < tb - win:  # type: ignore
            i += 1
        else:
            j += 1

    triples = pd.DataFrame(rows)
    if triples.empty:
        print(f"[{label}] No triple-consensus hits.")
        return triples, pd.DataFrame()

    triples["pnl_sum"] = triples[["pnl_sma", "pnl_ema", "pnl_rsi"]].sum(axis=1)
    triples["pnl_mean"] = triples[["pnl_sma", "pnl_ema", "pnl_rsi"]].mean(axis=1)
    triples["all_win"] = (
        (triples["pnl_sma"] > 0) & (triples["pnl_ema"] > 0) & (triples["pnl_rsi"] > 0)
    )
    triples["any_win"] = (
        (triples["pnl_sma"] > 0) | (triples["pnl_ema"] > 0) | (triples["pnl_rsi"] > 0)
    )

    # Triple overall summary
    summary = pd.DataFrame(
        {
            "hits": [len(triples)],
            "mean_pnl_sum": [triples["pnl_sum"].mean()],
            "median_pnl_sum": [triples["pnl_sum"].median()],
            "mean_pnl_mean": [triples["pnl_mean"].mean()],
            "median_pnl_mean": [triples["pnl_mean"].median()],
            "all_win_rate": [triples["all_win"].mean()],
            "any_win_rate": [triples["any_win"].mean()],
            "mean_delta_min_sma_ema": [triples["delta_min_sma_ema"].mean()],
            "mean_delta_min_sma_rsi": [triples["delta_min_sma_rsi"].mean()],
            "mean_delta_min_ema_rsi": [triples["delta_min_ema_rsi"].mean()],
        }
    )

    # NEW: Triple side breakdown (Buy/Sell)
    side_summary = (
        triples.groupby("side")
        .agg(
            hits=("side", "size"),
            total_pnl=("pnl_sum", "sum"),
            mean_pnl=("pnl_sum", "mean"),
            median_pnl=("pnl_sum", "median"),
            all_win_rate=("all_win", "mean"),
            any_win_rate=("any_win", "mean"),
            mean_delta_min_sma_ema=("delta_min_sma_ema", "mean"),
            mean_delta_min_sma_rsi=("delta_min_sma_rsi", "mean"),
            mean_delta_min_ema_rsi=("delta_min_ema_rsi", "mean"),
        )
        .reset_index()
    )

    triples_path = TRIPLE_OUT_DIR / f"triple_consensus_hits_{label}.csv"
    summary_path = TRIPLE_OUT_DIR / f"triple_consensus_summary_{label}.csv"
    side_summary_path = TRIPLE_OUT_DIR / f"triple_consensus_side_summary_{label}.csv"
    triples.to_csv(triples_path, index=False)
    summary.to_csv(summary_path, index=False)
    side_summary.to_csv(side_summary_path, index=False)
    print(f"[{label}] triple-consensus hits: {len(triples)} -> {triples_path}")
    return triples, summary


def run_triple_for_buckets(buckets: Dict[str, pd.DataFrame], window_min: int):
    for name, bdf in buckets.items():
        find_triple_consensus(bdf, window_min, label=name)


# -------------- Main run --------------
all_trades = load_all_audits(INPUT_GLOB)
(OUT_DIR / "all_trades_merged.csv").parent.mkdir(parents=True, exist_ok=True)
all_trades.to_csv(OUT_DIR / "all_trades_merged.csv", index=False)

if all_trades.empty:
    print("No trades found. Check INPUT_GLOB.")
else:
    buckets = bucketize(all_trades, TOP_N)

    # Pairwise outputs (+ side breakdown)
    outputs = {name: write_bucket_outputs(name, bdf) for name, bdf in buckets.items()}

    # Triple-consensus on ALL + each bucket
    find_triple_consensus(all_trades, TRIPLE_WINDOW_MINUTES, label="all")
    run_triple_for_buckets(buckets, TRIPLE_WINDOW_MINUTES)

    print("Merged trades:", str(OUT_DIR / "all_trades_merged.csv"))
    for k, v in outputs.items():
        print(f"[{k}]")
        for label, path in v.items():
            print(f"  {label}: {path}")
