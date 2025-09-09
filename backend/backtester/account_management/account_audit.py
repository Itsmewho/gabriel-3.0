from __future__ import annotations
from pathlib import Path
from typing import Mapping, Optional, Tuple, List
import pandas as pd


def _pause_series_from_equity(
    eq_series: pd.Series,
    dd_threshold: float,
    resume_ratio: float,
) -> Tuple[pd.Series, pd.Series, List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]]]:
    """
    Returns:
      dd_pct (indexed like eq_series)
      paused (bool series)
      spans: list of (pause_start, resume_time) where resume_time may be None if never resumed
    """
    eq = eq_series.ffill().fillna(0.0).astype(float)
    peak = eq.cummax().replace(0, 1e-9)
    dd_pct = 1.0 - (eq / peak)

    paused_vals: List[bool] = []
    spans: List[Tuple[pd.Timestamp, Optional[pd.Timestamp]]] = []
    paused = False
    cur_start: Optional[pd.Timestamp] = None

    for ts, (e, pk, dd) in zip(eq.index, zip(eq.values, peak.values, dd_pct.values)):
        # enter pause
        if not paused and dd >= dd_threshold:
            paused = True
            cur_start = ts
        # exit pause (hysteresis)
        if paused and e >= resume_ratio * pk:
            paused = False
            spans.append((cur_start, ts))  # type: ignore
            cur_start = None
        paused_vals.append(paused)

    # still paused at the end
    if paused and cur_start is not None:
        spans.append((cur_start, None))

    return dd_pct, pd.Series(paused_vals, index=eq.index), spans


def export_account_audit(
    df: pd.DataFrame,
    filename: str,
    dd_thresholds: Optional[
        Mapping[str, float]
    ] = None,  # e.g. {"HEARTBEAT_LONG": 0.30}
    resume_thresholds: Optional[
        Mapping[str, float]
    ] = None,  # e.g. {"HEARTBEAT_LONG": 0.90}
):
    """
    df: output of Ledger.snapshot_df()
    Writes:
      - {filename}: timeline with drawdown_pct and is_paused
      - {filename.replace('.csv','_stats.csv')}: per-strategy stats incl. max DD and pause spans
      - {filename.replace('.csv','_pauses.csv')}: each pause block (start/end/min_eq/max_dd)
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if df is None or df.empty:
        base_cols = [
            "time",
            "strategy_id",
            "kind",
            "amount",
            "equity_before",
            "equity_after",
            "trade_id",
            "drawdown_pct",
            "is_paused",
        ]
        pd.DataFrame(columns=base_cols).to_csv(filename, index=False)
        pd.DataFrame(
            columns=[
                "strategy_id",
                "final_equity",
                "max_drawdown_abs",
                "max_drawdown_pct",
                "n_pauses",
                "events",
            ]
        ).to_csv(filename.replace(".csv", "_stats.csv"), index=False)
        pd.DataFrame(
            columns=[
                "strategy_id",
                "pause_start",
                "resume_time",
                "duration_events",
                "min_equity",
                "max_drawdown_pct_in_pause",
            ]
        ).to_csv(filename.replace(".csv", "_pauses.csv"), index=False)
        print(f"Account audit saved to {filename}")
        return

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce").dt.tz_localize(None)
    out = out.sort_values(["strategy_id", "time"])
    out["drawdown_pct"] = 0.0
    out["is_paused"] = False

    stats_rows = []
    pause_rows = []

    for sid, g in out.groupby("strategy_id", sort=False):
        g = g.sort_values("time").copy()

        # Seed baseline with equity_before of first row just before the first timestamp
        t0 = pd.to_datetime(g.iloc[0]["time"])
        eq0 = float(g.iloc[0]["equity_before"])
        seed_time = t0 - pd.Timedelta(microseconds=1)

        times = pd.to_datetime(g["time"].values)
        after_series = pd.Series(g["equity_after"].astype(float).values, index=times)
        eq_series = pd.concat(
            [pd.Series([eq0], index=[seed_time]), after_series]
        ).sort_index()

        # thresholds (defaults: 30% pause, 90% resume)
        dd_th = (dd_thresholds or {}).get(sid, 0.30)
        res_ratio = (resume_thresholds or {}).get(sid, 0.90)

        dd_pct, paused_ser, spans = _pause_series_from_equity(
            eq_series, dd_th, res_ratio
        )

        # map DD back to event rows (drop the seed point)
        dd_no_seed = dd_pct.drop(index=seed_time, errors="ignore")
        paused_no_seed = paused_ser.drop(index=seed_time, errors="ignore")
        out.loc[g.index, "drawdown_pct"] = dd_no_seed.reindex(
            g["time"].values, method="ffill"
        ).values
        out.loc[g.index, "is_paused"] = (
            paused_no_seed.reindex(g["time"].values, method="ffill").astype(bool).values
        )

        peak = eq_series.cummax().replace(0, 1e-9)
        dd_abs = peak - eq_series

        stats_rows.append(
            {
                "strategy_id": sid,
                "final_equity": float(eq_series.iloc[-1]),
                "max_drawdown_abs": float(dd_abs.max()),
                "max_drawdown_pct": float(dd_pct.max()),
                "n_pauses": int(len(spans)),
                "events": int(len(g)),
            }
        )

        # Pause blocks table
        for start, resume in spans:
            span_slice = eq_series.loc[
                start : (resume if resume is not None else eq_series.index[-1])
            ]
            dd_slice = dd_pct.loc[span_slice.index]
            pause_rows.append(
                {
                    "strategy_id": sid,
                    "pause_start": start,
                    "resume_time": resume,
                    "duration_events": int(
                        (
                            out[
                                (out["strategy_id"] == sid)
                                & (
                                    out["time"].between(
                                        start, resume if resume else out["time"].max()
                                    )
                                )
                            ]
                        ).shape[0]
                    ),
                    "min_equity": float(span_slice.min()),
                    "max_drawdown_pct_in_pause": float(dd_slice.max()),
                }
            )

    out.to_csv(filename, index=False)
    pd.DataFrame(stats_rows).to_csv(filename.replace(".csv", "_stats.csv"), index=False)
    pd.DataFrame(pause_rows).to_csv(
        filename.replace(".csv", "_pauses.csv"), index=False
    )
    print(f"Account audit saved to {filename}")
