# backtester/main_analyzer.py
# A complete pipeline from raw M1 data to a confluence report, aligned to Pepperstone server time.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Imports from pipeline (resampling + VWAP, DST-aware server time)
# --------------------------------------------------------------------------
# This file expects backtester/pipeline/resample_and_vwap.py to exist (our Phase 1 module).

from backtester.performance.resample_and_vwap import (
    add_server_time,
    resample_from_m1,
    add_multi_vwap,
)

# --------------------------------------------------------------------------
# PART 1: FEATURE GENERATION (VWAP, Sessions, Cross-timeframe merge)
# --------------------------------------------------------------------------


class FeatureGenerator:
    """Prepare features with DST-aware sessions and multi-period VWAPs.
    Keeps everything in server time. No UTC conversions.
    """

    # --- London / New York DST helpers (mirror pipeline logic) ---
    @staticmethod
    def _nth_weekday_of_month(
        year: int, month: int, weekday: int, n: int
    ) -> pd.Timestamp:
        d = pd.Timestamp(year=year, month=month, day=1)
        add = (weekday - d.weekday()) % 7
        return (d + pd.Timedelta(days=add) + pd.Timedelta(weeks=n - 1)).normalize()

    @classmethod
    def _us_dst_bounds(cls, year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return cls._nth_weekday_of_month(year, 3, 6, 2), cls._nth_weekday_of_month(
            year, 11, 6, 1
        )

    @staticmethod
    def _eu_dst_bounds(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
        march = pd.Timestamp(year=year, month=3, day=31)
        m_last_sun = march - pd.Timedelta(days=(march.weekday() + 1) % 7)
        octo = pd.Timestamp(year=year, month=10, day=31)
        o_last_sun = octo - pd.Timedelta(days=(octo.weekday() + 1) % 7)
        return m_last_sun.normalize(), o_last_sun.normalize()

    @classmethod
    def _is_us_dst(cls, day: pd.Timestamp) -> bool:
        day = pd.Timestamp(day).normalize()
        start, end = cls._us_dst_bounds(day.year)
        return start <= day < end

    @classmethod
    def _is_eu_dst(cls, day: pd.Timestamp) -> bool:
        day = pd.Timestamp(day).normalize()
        start, end = cls._eu_dst_bounds(day.year)
        return start <= day < end

    @classmethod
    def server_offset_hours(cls, day: pd.Timestamp) -> int:
        return 3 if cls._is_us_dst(day) else 2

    @classmethod
    def london_offset_hours(cls, day: pd.Timestamp) -> int:
        return 1 if cls._is_eu_dst(day) else 0

    @classmethod
    def newyork_offset_hours(cls, day: pd.Timestamp) -> int:
        return -4 if cls._is_us_dst(day) else -5

    # --- Basic hygiene ---
    @staticmethod
    def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index)
        out = out.sort_index()
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        return out

    @staticmethod
    def _ensure_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        df = df.copy()
        auto = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
        df.rename(
            columns={k: v for k, v in auto.items() if k in df.columns}, inplace=True
        )
        need = {"Open", "High", "Low", "Close"}
        if not need.issubset(df.columns):
            raise KeyError(f"Missing OHLC columns: {sorted(need - set(df.columns))}")
        for c in ("tick_volume", "Volume", "volume", "Vol", "VOL"):
            if c in df.columns:
                return df, c
        raise KeyError("No volume/tick_volume column found")

    # --- Sessions ---
    @classmethod
    def _local_to_server_hour(
        cls, local_hour: int, day: pd.Timestamp, local_offset_hours: int
    ) -> int:
        return int(
            (local_hour - local_offset_hours + cls.server_offset_hours(day)) % 24
        )

    @classmethod
    def _day_session_windows_server(
        cls, day: pd.Timestamp
    ) -> Dict[str, Tuple[int, int]]:
        lon_off = cls.london_offset_hours(day)
        ny_off = cls.newyork_offset_hours(day)
        ldn = (
            cls._local_to_server_hour(8, day, lon_off),
            cls._local_to_server_hour(17, day, lon_off),
        )
        ny = (
            cls._local_to_server_hour(8, day, ny_off),
            cls._local_to_server_hour(17, day, ny_off),
        )
        return {"LDN": ldn, "NY": ny}

    @classmethod
    def session_labels(cls, df: pd.DataFrame) -> pd.Series:
        d = add_server_time(df)
        ts = pd.to_datetime(d["server_ts"])  # server time
        hrs = ts.dt.hour
        days = ts.dt.normalize()
        labels = pd.Series("ASIA/OUTER", index=d.index, dtype=object)
        for day in sorted(days.unique()):
            (ldn0, ldn1) = cls._day_session_windows_server(day)["LDN"]
            (ny0, ny1) = cls._day_session_windows_server(day)["NY"]
            dm = days == day
            labels[dm & (hrs >= ldn0) & (hrs < ldn1)] = "LDN"
            labels[dm & (hrs >= ny0) & (hrs < ny1)] = "NY"
        return labels

    # --- Pipeline steps ---
    def run_all(self, df_m1: pd.DataFrame) -> pd.DataFrame:
        """Resample M1, add VWAPs, add sessions, merge D1 context onto H1.
        Returns an H1 feature dataframe with D1/W1/MN1 context.
        """
        # Resample to TFs and keep M1 server stamps
        res = resample_from_m1(df_m1)
        d_m1 = res.frames["M1"]  # noqa: F841
        df_h1 = res.frames["H1"].copy()
        df_d1 = res.frames["D1"].copy()

        # Add VWAPs to H1 and D1
        df_h1 = add_multi_vwap(df_h1)
        df_d1 = add_multi_vwap(df_d1)

        # Session labels and session VWAP on H1
        df_h1["server_date"] = df_h1.index.normalize()
        df_h1["session"] = self.session_labels(df_h1)
        df_h1 = self._add_session_vwap(df_h1)

        # D1 context columns to carry onto H1 (prior day values)
        d1_ctx = df_d1[[c for c in df_d1.columns if c.startswith("VWAP_")]].copy()
        d1_ctx["server_date"] = d1_ctx.index.normalize()
        d1_ctx = d1_ctx.add_prefix("D1_")
        d1_ctx.rename(columns={"D1_server_date": "server_date"}, inplace=True)
        d1_ctx = d1_ctx.groupby("server_date").last()

        # Merge D1 context onto H1 by yesterday's date
        merged = df_h1.join(d1_ctx.shift(1), on="server_date")

        # Keep essential OHLCV
        return merged.dropna(how="any")

    @staticmethod
    def _add_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Ensure volume
        _, vol_col = FeatureGenerator._ensure_ohlcv(df)
        key = ["server_date", "session"]
        pv = df["Close"] * pd.to_numeric(  # noqa: F841
            df[vol_col], errors="coerce"
        ).fillna(  # noqa: F841
            0
        )  # noqa: F841
        df["_cum_vol"] = df.groupby(key)[vol_col].cumsum()
        df["_cum_pv"] = (
            df.groupby(key)
            .apply(
                lambda x: (
                    x["Close"] * pd.to_numeric(x[vol_col], errors="coerce")
                ).cumsum()
            )
            .reset_index(level=0, drop=True)
        )
        df["VWAP_Session"] = (df["_cum_pv"] / df["_cum_vol"]).astype(float)
        return df.drop(columns=["_cum_vol", "_cum_pv"])  # keep VWAP_Session


# --------------------------------------------------------------------------
# PART 2: ANALYSIS & REPORTING (Breakouts + Confluence tables)
# --------------------------------------------------------------------------


class ReportGenerator:
    """Analyze N-week breakouts with VWAP confluence. Output Markdown."""

    @staticmethod
    def _weekly_high_low_from_d1(
        df_d1: pd.DataFrame, lookback_weeks: int
    ) -> pd.DataFrame:
        d = df_d1.copy()
        d["prev_high"] = (
            d["High"]
            .shift(1)
            .rolling(window=lookback_weeks * 5, min_periods=lookback_weeks * 5)
            .max()
        )
        d["prev_low"] = (
            d["Low"]
            .shift(1)
            .rolling(window=lookback_weeks * 5, min_periods=lookback_weeks * 5)
            .min()
        )
        return d

    def analyze_breakouts_with_confluence(
        self, df_features: pd.DataFrame, lookback_weeks: int = 4
    ) -> pd.DataFrame:
        # Daily bars from feature set
        df_d1 = (
            df_features.resample("D")
            .agg({"High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
        d = self._weekly_high_low_from_d1(df_d1, lookback_weeks)

        results: List[Dict[str, Any]] = []
        for date, row in d.iterrows():
            if not np.isfinite(row.get("prev_high", np.nan)) or not np.isfinite(
                row.get("prev_low", np.nan)
            ):
                continue
            up = bool(row["High"] > row["prev_high"])
            dn = bool(row["Low"] < row["prev_low"])
            if not (up or dn):
                continue

            # Confluence snapshot from H1 features at breakout day open
            # Choose first H1 bar of that server_date
            try:
                day_mask = df_features.index.normalize() == date.normalize()
                snap = df_features.loc[day_mask].iloc[0]
            except Exception:
                continue

            above_vwap_d1 = (
                bool(snap.get("VWAP_D", np.nan) and snap["Close"] > snap["VWAP_D"])
                if "VWAP_D" in df_features.columns
                else np.nan
            )
            above_vwap_w = (
                bool(snap.get("VWAP_W", np.nan) and snap["Close"] > snap["VWAP_W"])
                if "VWAP_W" in df_features.columns
                else np.nan
            )
            above_vwap_m = (
                bool(snap.get("VWAP_M", np.nan) and snap["Close"] > snap["VWAP_M"])
                if "VWAP_M" in df_features.columns
                else np.nan
            )

            # Simple next-day outcome
            idx = df_d1.index.get_indexer([date])[0]
            outcome = "End_of_Data"
            if idx + 1 < len(df_d1):
                nxt = df_d1.iloc[idx + 1]
                outcome = (
                    "Success"
                    if (up and nxt["High"] > row["High"])
                    or (dn and nxt["Low"] < row["Low"])
                    else "Failure"
                )

            results.append(
                {
                    "date": date,
                    "type": "Up" if up else "Down",
                    "outcome": outcome,
                    "above_vwap_d": above_vwap_d1,
                    "above_vwap_w": above_vwap_w,
                    "above_vwap_m": above_vwap_m,
                }
            )
        return pd.DataFrame(results)

    def generate_report(
        self, df_analysis: pd.DataFrame, out_dir: str, symbol: str
    ) -> Path:
        if df_analysis.empty:
            report = f"# {symbol} Confluence Analysis\n\n_No breakouts found._"
        else:
            lines: List[str] = [f"# {symbol} Confluence Analysis"]
            success = (
                df_analysis["outcome"].value_counts(normalize=True).get("Success", 0.0)
                * 100
            )
            lines.append(
                f"\n## Overall Performance\n- **Overall Breakout Success:** {success:.1f}%"
            )

            # Confluence tables
            def _pivot(col: str) -> str:
                if col not in df_analysis.columns:
                    return "_N/A_"
                tab = (
                    pd.crosstab(
                        df_analysis[col], df_analysis["outcome"], normalize="index"
                    )
                    * 100
                )
                return tab.to_markdown()

            lines.append("\n## Confluence: Daily VWAP")
            lines.append(_pivot("above_vwap_d"))

            lines.append("\n## Confluence: Weekly VWAP")
            lines.append(_pivot("above_vwap_w"))

            lines.append("\n## Confluence: Monthly VWAP")
            lines.append(_pivot("above_vwap_m"))

            report = "\n".join(lines)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        md_path = Path(out_dir) / f"{symbol}_confluence_report.md"
        md_path.write_text(report, encoding="utf-8")
        return md_path


# --------------------------------------------------------------------------
# Example CLI usage
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy M1 data for structure test
    idx = pd.date_range(start="2024-01-01", end="2024-03-31", freq="1T")
    rng = np.random.default_rng(0)
    base = 1.1 + np.cumsum(rng.normal(0, 0.00005, size=len(idx)))
    df_m1 = pd.DataFrame(
        {
            "Open": base,
            "High": base + 0.0001,
            "Low": base - 0.0001,
            "Close": base + rng.normal(0, 0.00003, size=len(idx)),
            "tick_volume": rng.integers(10, 100, size=len(idx)),
        },
        index=idx,
    )

    fg = FeatureGenerator()
    feats = fg.run_all(df_m1)

    rep = ReportGenerator()
    analysis = rep.analyze_breakouts_with_confluence(feats, lookback_weeks=4)
    path = rep.generate_report(analysis, out_dir=".", symbol="EURUSD")
    print(f"Report saved: {path}")
