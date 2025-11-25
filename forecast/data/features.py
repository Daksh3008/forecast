"""
Create lag features, rolling features, returns, technical indicators.
This generates a 2D DataFrame where rows = days, cols = engineered features.

Later, sequence building converts this into 3D for DL models.
"""

import pandas as pd
import numpy as np


def add_returns(df: pd.DataFrame, price_col="close") -> pd.DataFrame:
    df = df.copy()
    df["return_1d"] = df[price_col].pct_change()
    df["return_5d"] = df[price_col].pct_change(5)
    df["return_10d"] = df[price_col].pct_change(10)
    return df


def add_moving_averages(df: pd.DataFrame, price_col="close") -> pd.DataFrame:
    df = df.copy()
    for w in [5, 10, 20, 50]:
        df[f"ma_{w}"] = df[price_col].rolling(w).mean()
        df[f"ema_{w}"] = df[price_col].ewm(span=w).mean()
    return df


def add_volatility(df: pd.DataFrame, price_col="close") -> pd.DataFrame:
    df = df.copy()
    df["vol_10"] = df[price_col].pct_change().rolling(10).std()
    df["vol_20"] = df[price_col].pct_change().rolling(20).std()
    return df


def add_lags(df: pd.DataFrame, price_col="close") -> pd.DataFrame:
    df = df.copy()
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"lag_{lag}"] = df[price_col].shift(lag)
    return df


def build_features(df: pd.DataFrame, price_col="close") -> pd.DataFrame:
    df = add_returns(df, price_col)
    df = add_moving_averages(df, price_col)
    df = add_volatility(df, price_col)
    df = add_lags(df, price_col)

    df = df.dropna().reset_index(drop=True)
    return df


# --------------------------
# Convert features → sequences for DL models
# --------------------------

def build_sequences(
    df: pd.DataFrame,
    target_col="close",
    seq_len=60,
):
    """
    Convert a 2D feature matrix into:
        X: (samples, seq_len, features)
        y: (samples,)
    """
    values = df.drop(columns=["date"]).values
    target = df[target_col].values

    X_seq = []
    y_seq = []

    for i in range(seq_len, len(df)):
        X_seq.append(values[i - seq_len:i])
        y_seq.append(target[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    return X_seq, y_seq
