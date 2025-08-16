import pandas as pd
import numpy as np
from typing import Optional

# NOTE: No backfilling of warm-up. No future-aligned shifts in features.
# Dropping warm-up rows eliminates look-ahead.


# --- RSI (Wilder) ---
def _rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / length, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# --- Renko builder (sequential, no future info) ---
def _build_renko_columns(
    df: pd.DataFrame,
    *,
    brick_size_pips: float = 10.0,
    pip_size: float = 0.0001,
    use_close: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df
    brick = brick_size_pips * pip_size
    price_series = df["close"] if use_close else df[["high", "low"]].mean(axis=1)
    renko_close = np.empty(len(df))
    renko_dir = np.zeros(len(df), dtype=int)
    renko_flip = np.zeros(len(df), dtype=int)
    renko_bricks_since_flip = np.zeros(len(df), dtype=int)
    p0 = float(price_series.iloc[0])
    r_close = round(p0 / brick) * brick
    regime = 0
    bricks_since_flip = 0
    for i, p in enumerate(price_series.astype(float)):
        delta = p - r_close
        n = int(abs(delta) // brick)
        if n > 0:
            step = brick * (1 if delta > 0 else -1)
            r_close += step * n
            new_regime = 1 if step > 0 else -1
            if new_regime != regime:
                renko_flip[i] = 1
                regime = new_regime
                bricks_since_flip = n
            else:
                bricks_since_flip += n
        renko_close[i] = r_close
        renko_dir[i] = regime
        renko_bricks_since_flip[i] = bricks_since_flip
    out = df.copy()
    out["renko_close"] = renko_close
    out["renko_dir"] = renko_dir
    out["renko_flip"] = renko_flip
    out["renko_bricks_since_flip"] = renko_bricks_since_flip
    return out


# --- Gaussian Channel 3.0 ---
def build_gaussian_channel(
    df: pd.DataFrame,
    *,
    gauss_len: int = 20,
    gauss_std: float = 5.0,
    atr_len: int = 14,
    band_mult: float = 2.0,
) -> pd.DataFrame:
    """
    FIXED: Replaced np.convolve with a causal (non-lookahead) rolling apply method.
    The original `np.convolve` with `mode="same"` used future data, creating look-ahead bias.
    This version ensures that the calculation for each bar only uses data from the past.
    """
    if gauss_len <= 1:
        raise ValueError("gauss_len must be > 1")

    out = df.copy()

    lags = np.arange(gauss_len, dtype=float)
    w = np.exp(-0.5 * (lags / float(gauss_std)) ** 2)
    w /= w.sum()
    # The rolling window is applied in chronological order, so we reverse the weights.
    w_for_apply = w[::-1]

    def _gauss_apply(x: np.ndarray) -> float:
        return float(np.dot(x, w_for_apply))

    # Apply the causal filter using a rolling window.
    close = out["close"].astype(float)
    gauss_mid = close.rolling(window=gauss_len, min_periods=gauss_len).apply(
        _gauss_apply, raw=True
    )
    out["gauss_mid"] = gauss_mid

    # ATR for bands (this part was already correct and causal)
    prev_close = out["close"].shift(1)
    tr1 = (out["high"] - out["low"]).abs()
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_len, adjust=False).mean()

    # Calculate final channel bands and signals
    out["gauss_upper"] = out["gauss_mid"] + band_mult * atr
    out["gauss_lower"] = out["gauss_mid"] - band_mult * atr
    out["gauss_slope"] = (
        np.sign(pd.Series(out["gauss_mid"]).diff()).fillna(0).astype(int)  # type: ignore
    )
    out["gauss_break_long"] = (
        (out["close"] > out["gauss_upper"]) & (out["gauss_slope"] > 0)
    ).astype(int)
    out["gauss_break_short"] = (
        (out["close"] < out["gauss_lower"]) & (out["gauss_slope"] < 0)
    ).astype(int)
    return out


# --- Core features builder ---
def build_features(
    df: pd.DataFrame,
    *,
    ema_fast: int = 14,
    ema_slow: int = 30,
    sma_fast: int = 5,
    sma_slow: int = 50,
    sma_trend: int = 14,
    rsi_len: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_sig: int = 9,
    tenkan_len: int = 9,
    kijun_len: int = 26,
    span_b_len: int = 52,
    renko_brick_pips: float = 10.0,
    pip_size: float = 0.0001,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    # --- Moving averages ---
    out["ema_fast"] = out["close"].ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=ema_slow, adjust=False).mean()
    out["sma_fast"] = out["close"].rolling(window=sma_fast, min_periods=sma_fast).mean()
    out["sma_slow"] = out["close"].rolling(window=sma_slow, min_periods=sma_slow).mean()
    out["sma_trend"] = (
        out["close"].rolling(window=sma_trend, min_periods=sma_trend).mean()
    )
    # --- RSI ---
    out["rsi"] = _rsi_wilder(out["close"], rsi_len)
    # --- MACD ---
    ema_fast_series = out["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow_series = out["close"].ewm(span=macd_slow, adjust=False).mean()
    out["macd_line"] = ema_fast_series - ema_slow_series
    out["macd_signal"] = out["macd_line"].ewm(span=macd_sig, adjust=False).mean()
    out["macd_hist"] = out["macd_line"] - out["macd_signal"]
    # --- Ichimoku ---
    high = out["high"]
    low = out["low"]
    out["tenkan_sen"] = (
        high.rolling(tenkan_len, min_periods=tenkan_len).max()
        + low.rolling(tenkan_len, min_periods=tenkan_len).min()
    ) / 2
    out["kijun_sen"] = (
        high.rolling(kijun_len, min_periods=kijun_len).max()
        + low.rolling(kijun_len, min_periods=kijun_len).min()
    ) / 2
    out["senkou_span_a_now"] = (out["tenkan_sen"] + out["kijun_sen"]) / 2
    out["senkou_span_b_now"] = (
        high.rolling(span_b_len, min_periods=span_b_len).max()
        + low.rolling(span_b_len, min_periods=span_b_len).min()
    ) / 2
    out["senkou_span_a_plot"] = out["senkou_span_a_now"].shift(kijun_len)
    out["senkou_span_b_plot"] = out["senkou_span_b_now"].shift(kijun_len)
    out["chikou_span"] = out["close"].shift(kijun_len)

    # FIX: Removed unnecessary and problematic `.values` calls.
    # np.maximum/minimum work directly on pandas Series.
    span_top = np.maximum(out["senkou_span_a_now"], out["senkou_span_b_now"])
    span_bot = np.minimum(out["senkou_span_a_now"], out["senkou_span_b_now"])

    # FIX: Removed `.values` here as well for consistency and type safety.
    out["ichi_price_above_cloud"] = (out["close"] > span_top).astype(int)
    out["ichi_price_below_cloud"] = (out["close"] < span_bot).astype(int)
    out["ichi_tenkan_above_kijun"] = (out["tenkan_sen"] > out["kijun_sen"]).astype(int)

    # This part is correct. The .fillna(0) handles NaNs from the rolling calculations.
    # The type checker was likely confused by the previous `.values` calls.
    out["ichi_cloud_color"] = (
        np.sign(out["senkou_span_a_now"] - out["senkou_span_b_now"])
        .fillna(0)  # type: ignore
        .astype(int)
    )

    out["ichi_kijun_slope"] = np.sign(out["kijun_sen"].diff()).fillna(0).astype(int)  # type: ignore
    # --- Volume ---
    out["avg_volume"] = out["tick_volume"].rolling(window=50, min_periods=50).mean()
    # --- Renko ---
    out = _build_renko_columns(
        out, brick_size_pips=renko_brick_pips, pip_size=pip_size, use_close=True
    )
    out = out.ffill()
    if drop_warmup:
        needed = [
            "ema_fast",
            "ema_slow",
            "sma_fast",
            "sma_slow",
            "sma_trend",
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a_now",
            "senkou_span_b_now",
            "chikou_span",
            "avg_volume",
        ]
        out = out.dropna(subset=needed)
    return out


# --- Modular add-ons ---
def add_keltner_features(
    df: pd.DataFrame,
    *,
    kc_len: int = 20,
    atr_len: int = 14,
    kc_mult: float = 1.5,
    drop_warmup: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_len, adjust=False).mean()
    kc_mid = out["close"].ewm(span=kc_len, adjust=False).mean()
    out["kc_mid"] = kc_mid
    out["kc_upper"] = kc_mid + kc_mult * atr
    out["kc_lower"] = kc_mid - kc_mult * atr
    out["kc_slope"] = np.sign(out["kc_mid"].diff()).fillna(0).astype(int)  # type: ignore
    out["kc_width"] = out["kc_upper"] - out["kc_lower"]
    out["kc_percent_b"] = ((out["close"] - out["kc_lower"]) / out["kc_width"]).clip(
        lower=-5, upper=5
    )
    if drop_warmup:
        out = out.dropna(subset=["kc_mid", "kc_upper", "kc_lower"])
    return out


def add_keltner_renko_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["renko_close", "renko_dir", "kc_upper", "kc_lower"]:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")
    out["renko_above_kc"] = (out["renko_close"] > out["kc_upper"]).astype(int)
    out["renko_below_kc"] = (out["renko_close"] < out["kc_lower"]).astype(int)
    out["renko_break_long"] = (
        (out["renko_dir"] == 1) & (out["renko_close"] > out["kc_upper"])
    ).astype(int)
    out["renko_break_short"] = (
        (out["renko_dir"] == -1) & (out["renko_close"] < out["kc_lower"])
    ).astype(int)
    return out


def add_gaussian_channel(
    df: pd.DataFrame,
    *,
    length: int = 20,
    std: float = 5.0,
    atr_len: int = 14,
    band_mult: float = 2.0,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    if length <= 1:
        raise ValueError("length must be > 1")
    out = df.copy()
    lags = np.arange(length, dtype=float)
    w = np.exp(-0.5 * (lags / float(std)) ** 2)
    w /= w.sum()
    w_for_apply = w[::-1]

    def _gauss_apply(x: np.ndarray) -> float:
        return float(np.dot(x, w_for_apply))

    close = out["close"].astype(float)
    gauss_mid = close.rolling(window=length, min_periods=length).apply(
        _gauss_apply, raw=True
    )
    out["gauss_mid"] = gauss_mid
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            (out["high"] - out["low"]).abs(),
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_len, adjust=False).mean()
    out["gauss_upper"] = out["gauss_mid"] + band_mult * atr
    out["gauss_lower"] = out["gauss_mid"] - band_mult * atr
    out["gauss_slope"] = np.sign(out["gauss_mid"].diff()).fillna(0).astype(int)  # type: ignore
    out["gauss_break_long"] = (
        (out["close"] > out["gauss_upper"]) & (out["gauss_slope"] > 0)
    ).astype(int)
    out["gauss_break_short"] = (
        (out["close"] < out["gauss_lower"]) & (out["gauss_slope"] < 0)
    ).astype(int)
    if drop_warmup:
        out = out.dropna(subset=["gauss_mid", "gauss_upper", "gauss_lower"])
    return out
