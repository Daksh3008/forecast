"""
Prepare data pipeline (upgraded):

1. Loads raw brent + merged_macro
2. Builds rich features via forecast.data.features.build_features
3. Saves canonical features CSV: data/processed/brent_features.csv
4. Builds sequences (X_seq, y_seq) where target = log(close_next)
5. Builds tabular features (X_tab, y_tab)
6. Fits FeatureScaler instances and saves them
7. Saves feature_cols.npy and processed arrays used by training
"""

import pandas as pd
import numpy as np
from pathlib import Path

from forecast.data.load_raw import load_raw_csv
from forecast.data.preprocess import preprocess
from forecast.data.features import build_features
from forecast.data.scalers import FeatureScaler

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CSV_OUT = PROCESSED_DIR / "brent_features.csv"
FEATURE_COLS_PATH = PROCESSED_DIR / "feature_cols.npy"

SEQ_LEN = 60  # keep consistent with config/base.yaml


def main():
    print("Loading raw data...")
    price = load_raw_csv("data/raw/brent.csv")
    macro = load_raw_csv("data/raw/merged_macro.csv")

    # lowercase columns
    price.columns = [c.lower() for c in price.columns]
    macro.columns = [c.lower() for c in macro.columns]

    price = preprocess(price)
    macro = preprocess(macro)

    print("Merging price + macro...")
    df = price.merge(macro, on="date", how="left").sort_values("date").reset_index(drop=True)

    print("Building rich feature matrix (this may take a few seconds)...")
    df_features = build_features(df, price_col="brent_close" if "brent_close" in df.columns else "close")

    # Save canonical features CSV so inference & audit use the identical matrix
    df_features.to_csv(FEATURE_CSV_OUT, index=False)
    print(f"Saved feature CSV: {FEATURE_CSV_OUT}")

    # Ensure log-close and next-day log target exist
    if "brent_close" in df_features.columns:
        df_features["close"] = df_features["brent_close"]
    if "close" not in df_features.columns:
        raise KeyError("Feature matrix missing 'close' price column for brent.")

    df_features["log_close"] = np.log(df_features["close"].astype(float))
    df_features["log_close_next"] = df_features["log_close"].shift(-1)
    # drop last row where next is NaN
    df_features = df_features.dropna(subset=["log_close_next"]).reset_index(drop=True)


    # ----------------------------------------
    # SANITIZE FEATURE MATRIX (CRITICAL)
    # ----------------------------------------

    # Convert INF → NaN
    df_features = df_features.replace([np.inf, -np.inf], np.nan)

    # Forward fill and backfill remaining holes
    df_features = df_features.fillna(method="ffill").fillna(method="bfill")

    # Replace still remaining NaN (rare) with 0
    df_features = df_features.fillna(0)

    # Clip extreme values to prevent float32 overflow
    # Clip only numeric columns (avoid date column)
    num_cols = df_features.select_dtypes(include=[np.number]).columns
    df_features[num_cols] = df_features[num_cols].clip(lower=-1e6, upper=1e6)





    # determine feature columns (exclude date, close, log_close, log_close_next)
    exclude = {"date", "close", "log_close", "log_close_next"}
    feature_cols = [c for c in df_features.columns if c not in exclude]
    print(f"Number of feature columns: {len(feature_cols)}")
    np.save(FEATURE_COLS_PATH, np.array(feature_cols, dtype=object))
    print(f"Saved feature order to {FEATURE_COLS_PATH}")

    # Build tabular arrays (LightGBM)
    X_tab = df_features[feature_cols].values.astype(np.float32)
    y_tab = df_features["log_close_next"].values.astype(np.float32)

    # Build sequential arrays for DL (X_seq aligned to predict log_close_next)
    N = len(df_features)
    if N <= SEQ_LEN:
        raise ValueError(f"Not enough rows ({N}) for seq_len={SEQ_LEN}")

    X_seq = []
    y_seq = []
    for i in range(SEQ_LEN, N):
        X_seq.append(df_features[feature_cols].iloc[i - SEQ_LEN:i].values.astype(np.float32))
        y_seq.append(df_features["log_close_next"].iloc[i].astype(np.float32))

    X_seq = np.stack(X_seq).astype(np.float32)
    y_seq = np.stack(y_seq).astype(np.float32)

    print(f"Built X_seq: {X_seq.shape}, y_seq: {y_seq.shape}, X_tab: {X_tab.shape}")

    # Fit scalers
    seq_scaler = FeatureScaler()
    tab_scaler = FeatureScaler()

    X_seq_scaled = seq_scaler.fit_transform(X_seq.copy())
    X_tab_scaled = tab_scaler.fit_transform(X_tab.copy())

    # Save scalers (ensure dirs)
    Path("models/lstm_attention/scalers").mkdir(parents=True, exist_ok=True)
    Path("models/tcn/scalers").mkdir(parents=True, exist_ok=True)
    Path("models/lightgbm/scalers").mkdir(parents=True, exist_ok=True)

    seq_scaler.save("models/lstm_attention/scalers/feature_scaler.joblib")
    seq_scaler.save("models/tcn/scalers/feature_scaler.joblib")
    tab_scaler.save("models/lightgbm/scalers/feature_scaler.joblib")

    # Save arrays
    np.save(PROCESSED_DIR / "X_seq.npy", X_seq_scaled)
    np.save(PROCESSED_DIR / "y_seq.npy", y_seq)
    np.save(PROCESSED_DIR / "X_tab.npy", X_tab_scaled)
    np.save(PROCESSED_DIR / "y_tab.npy", y_tab)

    print("Saved processed arrays and scalers.")
    print("Done.")


if __name__ == "__main__":
    main()
