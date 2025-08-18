import pandas as pd
import mplfinance as mpf
from pathlib import Path
from typing import Optional


# --- MODIFIED: Function signature now accepts fill_between_config ---
def generate_feature_plot(
    df: pd.DataFrame,
    plots_to_add: list,
    chart_title: str,
    filename: Path,
    hlines_config: Optional[dict] = None,
    fill_between_config: Optional[dict] = None,  # Added for Ichimoku
):
    """
    A generic function to generate and save a feature chart.
    """
    s = mpf.make_mpf_style(base_mpf_style="yahoo", gridstyle="-.")

    num_panels = 1 + max([p.get("panel", 0) for p in plots_to_add], default=0)
    panel_ratios = [6] + [2] * (num_panels - 1)

    plot_kwargs = {
        "type": "candle",
        "style": s,
        "title": chart_title,
        "ylabel": "Price ($)",
        "addplot": plots_to_add,
        "panel_ratios": panel_ratios,
        "figscale": 1.5,
        "savefig": str(filename),
        "warn_too_much_data": 10000,
    }

    if hlines_config:
        plot_kwargs["hlines"] = hlines_config

    if fill_between_config:
        plot_kwargs["fill_between"] = fill_between_config

    mpf.plot(df, **plot_kwargs)
    print(f"Chart saved to {filename}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    CACHE_FILE_PATH = "results/cache/EURUSD_1m_2023-08-01_2025-08-01_features.parquet"
    CHART_START_DATE = "2023-09-05 00:00:00"
    CHART_END_DATE = "2023-09-05 02:00:00"

    INDICATOR_PLOTS = {
        "EMA_Crossover": {"overlays": ["ema_fast", "ema_slow"]},
        "SMA_Cluster": {"overlays": ["sma_fast", "sma_slow", "sma_trend"]},
        "MACD": {
            "overlays": ["macd_line", "macd_signal"],
            "panel_1": {"histogram": "macd_hist"},
        },
        "RSI_Bands": {
            "panel_1": {
                "lines": ["rsi", "rsi_upper_band", "rsi_middle_band", "rsi_lower_band"]
            }
        },
        "Ichimoku_Cloud": {
            "overlays": [
                "tenkan_sen",
                "kijun_sen",
                "chikou_span",
                "senkou_span_a_now",
                "senkou_span_b_now",
            ],
            "fill": {
                "y1": "senkou_span_a_now",
                "y2": "senkou_span_b_now",
                "alpha": 0.2,
            },
        },
        "Trendlines": {"overlays": ["tl_highs_value", "tl_lows_value"]},
        "Keltner_Channels": {"overlays": ["kc_upper", "kc_mid", "kc_lower"]},
        "ICT_Events": {
            "event_markers": {
                # Plot the 'high' value where 'pivot_high' is 1
                "pivot_high": {"price_source": "high", "marker": "v", "color": "red"},
                # Plot the 'low' value where 'pivot_low' is 1
                "pivot_low": {"price_source": "low", "marker": "^", "color": "green"},
                # Plot the 'fvg_up_mid' value where it exists
                "fvg_up_mid": {
                    "price_source": "self",
                    "marker": "_",
                    "color": "blue",
                    "markersize": 100,
                },
                # Plot the 'fvg_dn_mid' value where it exists
                "fvg_dn_mid": {
                    "price_source": "self",
                    "marker": "_",
                    "color": "black",
                    "markersize": 100,
                },
            }
        },
    }

    print(f"Loading data from {CACHE_FILE_PATH}...")
    try:
        features_df = pd.read_parquet(CACHE_FILE_PATH)
        features_df["time"] = pd.to_datetime(features_df["time"])
        plot_df = features_df.set_index("time").loc[CHART_START_DATE:CHART_END_DATE]

        if plot_df.empty:
            raise ValueError(f"No data found for the specified date range.")

        results_path = Path("results/inspector_charts")
        results_path.mkdir(exist_ok=True, parents=True)

        # --- MODIFIED: Main loop now handles "event_markers" ---
        for name, config in INDICATOR_PLOTS.items():
            plots_to_add = []
            hlines_config = None
            fill_between_config = None

            if "overlays" in config:
                cols = [c for c in config["overlays"] if c in plot_df.columns]
                if cols:
                    plots_to_add.append(
                        mpf.make_addplot(
                            plot_df[cols],
                            width=0.7 if name == "Ichimoku_Cloud" else 1.0,
                            alpha=0.4 if name == "Ichimoku_Cloud" else 1.0,
                        )
                    )

            # --- NEW: Logic for plotting discrete event markers ---
            if "event_markers" in config:
                for event_col, settings in config["event_markers"].items():
                    if event_col in plot_df.columns:
                        price_source = settings.get("price_source")

                        if price_source == "self":
                            # For columns that are already price levels (like fvg_up_mid)
                            plot_series = plot_df[event_col]
                        else:

                            plot_series = plot_df[price_source].where(
                                plot_df[event_col] == 1
                            )

                        plots_to_add.append(
                            mpf.make_addplot(
                                plot_series,
                                type="scatter",
                                marker=settings.get("marker", "o"),
                                color=settings.get("color", "b"),
                                markersize=settings.get("markersize", 60),
                            )
                        )

            if name == "Ichimoku_Cloud" and "fill" in config:
                fill_cfg = config["fill"]
                y1 = plot_df[fill_cfg["y1"]]
                y2 = plot_df[fill_cfg["y2"]]
                fill_between_config = [
                    dict(
                        y1=y1.values,
                        y2=y2.values,
                        where=(y1 >= y2),
                        alpha=fill_cfg["alpha"],
                        color="lightgreen",
                    ),
                    dict(
                        y1=y1.values,
                        y2=y2.values,
                        where=(y1 < y2),
                        alpha=fill_cfg["alpha"],
                        color="lightcoral",
                    ),
                ]

            if "panel_1" in config:
                if "lines" in config["panel_1"]:
                    cols = [
                        c for c in config["panel_1"]["lines"] if c in plot_df.columns
                    ]
                    if cols:
                        plots_to_add.append(
                            mpf.make_addplot(plot_df[cols], panel=1, title=name)
                        )
                if "histogram" in config["panel_1"]:
                    col = config["panel_1"]["histogram"]
                    if col in plot_df:
                        colors = ["g" if v >= 0 else "r" for v in plot_df[col]]
                        plots_to_add.append(
                            mpf.make_addplot(
                                plot_df[col],
                                type="bar",
                                panel=1,
                                color=colors,
                                title="MACD Histogram",
                            )
                        )

            if "hlines" in config:
                hlines_config = dict(
                    hlines=config["hlines"]["levels"],
                    colors=config["hlines"]["colors"],
                    linestyle=config["hlines"]["linestyle"],
                )

            filename = (
                results_path / f"inspect_{name}_{CHART_START_DATE.split(' ')[0]}.png"
            )
            chart_title = f"{name} Features ({CHART_START_DATE} to {CHART_END_DATE})"
            generate_feature_plot(
                plot_df,
                plots_to_add,
                chart_title,
                filename,
                hlines_config,
                fill_between_config,  # type: ignore
            )

    except FileNotFoundError:
        print(f"Error: Cache file not found at '{CACHE_FILE_PATH}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
