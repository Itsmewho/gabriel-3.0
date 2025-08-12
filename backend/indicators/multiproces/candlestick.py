import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

# Import the new worker function
from .candlestick_worker import process_single_window


def analyze_candlestick_windows(df: pd.DataFrame, windows: list) -> dict:
    """
    Calculates and stores candlestick analysis for multiple windows IN PARALLEL.
    """

    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_size"] = (df["open"] - df["close"]).abs()

    # Use a multiprocessing Pool to distribute the work across available CPU cores.
    # This will significantly speed up the calculation for multiple windows.
    num_processes = min(len(windows), cpu_count())

    with Pool(processes=num_processes) as pool:
        # Create a list of arguments to pass to the worker function
        args = [(df.copy(), window) for window in windows]

        # pool.map will apply the `process_single_window` function to each item in `args`
        # across the available processes.
        results_list = pool.map(process_single_window, args)

    # Convert the list of (name, df) tuples back into a dictionary
    return dict(results_list)
