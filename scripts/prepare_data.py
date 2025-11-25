"""
Prepare data from the provided feature matrix (deepakntr_bo_features.csv).

Assumptions:
 - Input file: data/processed/deepakntr_bo_features.csv
 - Column 'Date' exists and is parseable as datetime
 - Column 'Close' exists (we predict next-day Close)

Outputs:
 - data/processed/X_seq.npy        (3D: samples, seq_len, features)
 - data/processed/y_seq.npy        (1D: samples -> next-day Close)
 - data/processed/X_tab.npy        (2D: rows aligned with original rows, features only)
 - models/*/scalers/feature_scaler.joblib
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.data.scalers import FeatureScaler

SEQ_LEN = 120
INPUT_PATH = Path("data/processed/deepakntr_bo_features.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Feature matrix not found: {INPUT_PATH}")

    print(f"Loading feature matrix from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    # ensure Date is datetime and sorted
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    # Ensure Close exists
    if "Close" not in df.columns:
        raise KeyError("Column 'Close' not found in feature matrix.")

    # Build target = next-day Close
    df = df.copy()
    df["target_close"] = df["Close"].shift(-1)

    # Drop final row which has no target
    df = df.dropna(subset=["target_close"]).reset_index(drop=True)

    # Keep feature columns (exclude Date and any target columns)
    exclude_cols = {"Date", "target_close"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print(f"Using {len(feature_cols)} feature columns.")
    X_tab = df[feature_cols].values.astype(float)  # shape (N, F)
    # --- LightGBM y target (no sequence shifting needed because we already aligned target_return or Close.shift(-1))
    y_tab = df["target_return"].values.astype(np.float32)

    np.save("data/processed/y_tab.npy", y_tab)
    print("Saved y_tab.npy for LightGBM")


    y = df["target_close"].values.astype(float)    # shape (N,)

    # Build sequences for DL models: X_seq shape (samples, seq_len, features)
    N = len(df)
    if N <= SEQ_LEN:
        raise ValueError(f"Not enough rows ({N}) to build sequences with seq_len={SEQ_LEN}.")

    X_seq = []
    y_seq = []
    # sequence window: for sample i we take rows [i : i+seq_len), predict y at i+seq_len-1? 
    # We want X that ends at t and predict Close at t+1. To do that we iterate up to:
    for start in range(0, N - SEQ_LEN):
        end = start + SEQ_LEN
        X_seq.append(X_tab[start:end])
        y_seq.append(y[end - 1])  # y at row end-1 corresponds to next-day close relative to last row? 
        # Explanation: since y was constructed as Close.shift(-1), y[row] = Close[row+1].
        # For sequence covering rows start..end-1, we want target Close at row end (which is y[end-1]).
    X_seq = np.asarray(X_seq, dtype=np.float32)  # (samples, seq_len, features)
    y_seq = np.asarray(y_seq, dtype=np.float32)  # (samples,)

    print(f"Built sequences: X_seq.shape={X_seq.shape}, y_seq.shape={y_seq.shape}")

    # Fit scalers for LSTM/TCN (3D) and LightGBM (2D)
    os.makedirs("models/lstm_attention/scalers", exist_ok=True)
    os.makedirs("models/tcn/scalers", exist_ok=True)
    os.makedirs("models/lightgbm/scalers", exist_ok=True)

    lstm_scaler = FeatureScaler()
    tcn_scaler = FeatureScaler()
    lgbm_scaler = FeatureScaler()

    X_lstm = lstm_scaler.fit_transform(X_seq.copy())
    X_tcn = tcn_scaler.fit_transform(X_seq.copy())
    X_lgbm = lgbm_scaler.fit_transform(X_tab.copy())

    lstm_scaler.save("models/lstm_attention/scalers/feature_scaler.joblib")
    tcn_scaler.save("models/tcn/scalers/feature_scaler.joblib")
    lgbm_scaler.save("models/lightgbm/scalers/feature_scaler.joblib")

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_seq.npy", X_lstm)   # we'll use a single scaler file for both DL models
    np.save("data/processed/y_seq.npy", y_seq)
    np.save("data/processed/X_tab.npy", X_lgbm)
    np.save("data/processed/feature_cols.npy", np.array(feature_cols, dtype=object))

    print("Saved processed arrays and scalers.")
    print("Done.")


if __name__ == "__main__":
    main()
