"""
Rich feature engineering utilities.

- Computes many technical indicators for price-like series and macro series.
- Can operate on multiple series (brent_close, wti_close, dxy_close, vix_close, usdinr_close).
- Produces a wide 2D feature DataFrame (rows = dates, cols = engineered features).
- Leaves 'date' and 'close' in the frame where present.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional


def _pct_change(x: pd.Series, periods: int):
    return x.pct_change(periods)


def _log_return(x: pd.Series, periods: int = 1):
    return np.log(x).diff(periods)


def ma(x: pd.Series, window: int):
    return x.rolling(window, min_periods=1).mean()


def ema(x: pd.Series, span: int):
    return x.ewm(span=span, adjust=False).mean()


def rolling_std(x: pd.Series, window: int):
    return x.pct_change().rolling(window, min_periods=1).std(ddof=0)


def rolling_skew(x: pd.Series, window: int):
    return x.rolling(window, min_periods=1).skew()


def rolling_kurt(x: pd.Series, window: int):
    return x.rolling(window, min_periods=1).kurt()


def rsi(x: pd.Series, window: int = 14):
    delta = x.diff()
    up = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    down = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(x: pd.Series):
    ema12 = x.ewm(span=12, adjust=False).mean()
    ema26 = x.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def bollinger_bands(x: pd.Series, window: int = 20):
    rol_mean = x.rolling(window, min_periods=1).mean()
    rol_std = x.rolling(window, min_periods=1).std(ddof=0)
    upper = rol_mean + 2 * rol_std
    lower = rol_mean - 2 * rol_std
    return upper, lower


def zscore(x: pd.Series, window: int = 20):
    mu = x.rolling(window, min_periods=1).mean()
    sigma = x.rolling(window, min_periods=1).std(ddof=0).replace(0, np.nan)
    return (x - mu) / sigma


def add_series_indicators(df: pd.DataFrame, col: str, prefix: Optional[str] = None) -> pd.DataFrame:
    """
    Add a broad set of indicators for a single series column.
    Returns DataFrame with new columns.
    """
    p = prefix or col
    s = df[col].astype(float)

    out = pd.DataFrame(index=df.index)
    # Returns & log returns
    out[f"{p}_ret_1d"] = _pct_change(s, 1)
    out[f"{p}_ret_5d"] = _pct_change(s, 5)
    out[f"{p}_ret_10d"] = _pct_change(s, 10)
    out[f"{p}_logret_1d"] = _log_return(s, 1)
    out[f"{p}_logret_5d"] = _log_return(s, 5)
    out[f"{p}_logret_10d"] = _log_return(s, 10)

    # Moving averages & EMAs
    for w in (5, 10, 20, 50, 100):
        out[f"{p}_ma_{w}"] = ma(s, w)
        out[f"{p}_ema_{w}"] = ema(s, w)

    # Volatility
    for w in (5, 10, 20, 50):
        out[f"{p}_vol_{w}"] = rolling_std(s, w)

    # Momentum / trend
    out[f"{p}_mom_10"] = s.diff(10)
    out[f"{p}_trend_spread_10_50"] = out.get(f"{p}_ma_10", ma(s, 10)) - out.get(f"{p}_ma_50", ma(s, 50))

    # RSI, MACD, Bollinger
    out[f"{p}_rsi_14"] = rsi(s, 14)
    macd_line, macd_sig, macd_hist = macd(s)
    out[f"{p}_macd"] = macd_line
    out[f"{p}_macd_sig"] = macd_sig
    out[f"{p}_macd_hist"] = macd_hist
    bb_h, bb_l = bollinger_bands(s, 20)
    out[f"{p}_bb_h_20"] = bb_h
    out[f"{p}_bb_l_20"] = bb_l

    # statistical moments
    for w in (10, 20, 50):
        out[f"{p}_skew_{w}"] = rolling_skew(s, w)
        out[f"{p}_kurt_{w}"] = rolling_kurt(s, w)

    # z-scores and lags
    out[f"{p}_z_20"] = zscore(s, 20)
    out[f"{p}_z_50"] = zscore(s, 50)
    for lag in (1, 2, 3, 5, 10, 20):
        out[f"{p}_lag_{lag}"] = s.shift(lag)

    # last observed value (useful for non-price exogenous features)
    out[f"{p}_last"] = s

    return out


def add_cross_features(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    """
    Add cross-series features like spreads, ratios between price-like series.
    base_cols: list of column names which are price-like (e.g. 'brent_close','wti_close')
    """
    out = pd.DataFrame(index=df.index)
    # pairwise spreads and ratios
    for i in range(len(base_cols)):
        for j in range(i + 1, len(base_cols)):
            a = base_cols[i]
            b = base_cols[j]
            name = f"{a.replace('_close','')}_minus_{b.replace('_close','')}"
            out[name] = df[a] - df[b]
            out[f"{a.replace('_close','')}_div_{b.replace('_close','')}"] = df[a] / (df[b].replace(0, np.nan))

            # 1/5/20 day spread returns
            out[f"{name}_ret_1d"] = out[name].pct_change()
            out[f"{name}_ret_5d"] = out[name].pct_change(5)
    return out


def build_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Input:
      df: must contain 'date' column and at least one price column like 'close' or 'brent_close'
    Behavior:
      - Identify price-like macro columns (ending with '_close')
      - Compute indicators for each of them
      - Compute cross-series features
      - Keep 'date' and 'close' (brent close) as final columns
    """
    df = df.copy()
    # normalize column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Ensure date column exists
    if "date" not in df.columns:
        raise KeyError("Input df must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Identify price-like columns
    price_cols = [c for c in df.columns if c.endswith("_close")]
    # If only generic 'close' exists, map it to 'brent_close' for naming consistency
    if "close" in df.columns and "brent_close" not in df.columns:
        df = df.rename(columns={"close": "brent_close"})
        if price_col == "close":
            price_col = "brent_close"
    # refresh price_cols after rename
    price_cols = [c for c in df.columns if c.endswith("_close")]

    # If user provided a price_col that matches a column, keep it, else fallback
    if price_col not in df.columns and "brent_close" in df.columns:
        price_col = "brent_close"

    # Initialize feature container
    all_feats = [df[["date"]].copy()]

    # Add per-series indicators
    for col in price_cols:
        feats = add_series_indicators(df, col, prefix=col.replace("_", ""))
        # prefix in add_series_indicators already includes col name; keep unique
        # align index
        feats.index = df.index
        all_feats.append(feats)

    # Also compute technical indicators for 'brent_close' under simple names for models expecting 'ma_10' etc.
    if "brent_close" in df.columns:
        b = df["brent_close"].astype(float)
        simple = pd.DataFrame(index=df.index)
        simple["close"] = b
        # a few legacy columns expected by some code paths
        simple["return_1d"] = b.pct_change()
        simple["return_5d"] = b.pct_change(5)
        for w in (5, 10, 20, 50):
            simple[f"ma_{w}"] = b.rolling(w, min_periods=1).mean()
            simple[f"ema_{w}"] = b.ewm(span=w, adjust=False).mean()
        simple["vol_10"] = b.pct_change().rolling(10, min_periods=1).std(ddof=0)
        simple["vol_20"] = b.pct_change().rolling(20, min_periods=1).std(ddof=0)
        # lags
        for lag in (1, 2, 3, 5, 10, 20):
            simple[f"lag_{lag}"] = b.shift(lag)
        all_feats.append(simple)

    # Cross-series features (spreads / ratios)
    if len(price_cols) >= 2:
        cross = add_cross_features(df, price_cols)
        all_feats.append(cross)

    # Merge all features horizontally
    feat_df = pd.concat(all_feats, axis=1)

    # Remove duplicate columns (if any) while preserving order
    _, idx = np.unique(feat_df.columns, return_index=True)
    feat_df = feat_df.iloc[:, np.sort(idx)]

    # Drop rows with NA in date or main price
    if price_col in df.columns:
        keep_mask = pd.notna(df[price_col])
        feat_df = feat_df.loc[keep_mask].reset_index(drop=True)
    else:
        feat_df = feat_df.reset_index(drop=True)

    # Final housekeeping: fill forward/backfill for any remaining holes
    feat_df = feat_df.fillna(method="ffill").fillna(method="bfill")

    return feat_df
