"""
Train all models on the prepared sequences / feature matrix.

Behavior:
 - Uses almost all available data for training.
 - Reserves a tiny validation slice for early stopping & checkpoint selection.
 - seq_len = 120 (ensured by prepare_data.py)
"""

import os
import numpy as np

from forecast.training.train_lstm import train_lstm
from forecast.training.train_tcn import train_tcn
from forecast.training.train_lightgbm import train_lightgbm

from forecast.utils.optuna_loader import load_optuna_params


# -----------------------------
# Validation split logic
# -----------------------------
def get_train_val_slices(n_samples):
    val_size = max(30, int(0.02 * n_samples))   # 2% or minimum 30 samples
    if val_size >= n_samples:
        val_size = max(1, n_samples // 10)
    split = n_samples - val_size
    return split


# -----------------------------
# MAIN TRAINING ROUTINE
# -----------------------------
def main():

    # ----------------------------------------
    # Load prepared data
    # ----------------------------------------
    X_seq = np.load("data/processed/X_seq.npy")      # (samples, seq_len, features)
    y_seq = np.load("data/processed/y_seq.npy")      # (samples,)
    X_tab = np.load("data/processed/X_tab.npy")      # (rows, features)
    y_tab = np.load("data/processed/y_tab.npy")      # *** tabular target for LightGBM ***

    n_samples = X_seq.shape[0]
    split = get_train_val_slices(n_samples)

    print(f"Total samples: {n_samples}, train={split}, val={n_samples - split}")

    # Train/Val split for sequence models
    X_train_seq = X_seq[:split]
    X_val_seq = X_seq[split:]
    y_train = y_seq[:split]
    y_val = y_seq[split:]

    # ----------------------------------------
    # Load Optuna params (if exist)
    # ----------------------------------------
    lstm_opt = load_optuna_params("models/lstm_attention/best_optuna_params.json")
    tcn_opt  = load_optuna_params("models/tcn/best_optuna_params.json")
    lgb_opt  = load_optuna_params("models/lightgbm/best_optuna_params.json")

    # =====================================================================
    # LSTM CONFIG
    # =====================================================================
    lstm_config = {
        "input_size": X_seq.shape[-1],
        "hidden_size": 128,
        "n_layers": 2,
        "n_heads": 4,
        "dropout": 0.1,
        "bidirectional": False,
        "fc_hidden": 64,
        "batch_size": 32,
        "epochs": 50,
        "lr": 1e-3,
    }

    # Override with Optuna values
    if lstm_opt:
        print("Applying Optuna params to LSTM...")
        lstm_config.update(lstm_opt)

    os.makedirs("models/lstm_attention/checkpoints", exist_ok=True)

    train_lstm(
        X_train_seq, y_train,
        X_val_seq, y_val,
        lstm_config,
        save_path="models/lstm_attention/checkpoints/best.pt"
    )

    # =====================================================================
    # TCN CONFIG
    # =====================================================================
    tcn_config = {
        "input_size": X_seq.shape[-1],
        "num_channels": [64, 128, 128],
        "kernel_size": 3,
        "dropout": 0.1,
        "batch_size": 32,
        "epochs": 50,
        "lr": 1e-3,
    }

    if tcn_opt:
        print("Applying Optuna params to TCN...")
        tcn_config.update(tcn_opt)

    os.makedirs("models/tcn/checkpoints", exist_ok=True)

    train_tcn(
        X_train_seq, y_train,
        X_val_seq, y_val,
        tcn_config,
        save_path="models/tcn/checkpoints/best.pt"
    )

    # =====================================================================
    # LIGHTGBM CONFIG
    # =====================================================================

    # Align LightGBM features to same number of rows as sequence labels
    X_tab_for_model = X_tab[:n_samples]
    y_tab_for_model = y_tab[:n_samples]

    X_lgbm_train = X_tab_for_model[:split]
    X_lgbm_val = X_tab_for_model[split:]
    y_lgbm_train = y_tab_for_model[:split]
    y_lgbm_val = y_tab_for_model[split:]

    lgbm_config = {
        "params": {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "device": "gpu",  # auto GPU if available
        },
        "num_boost_round": 2000,
        "early_stopping": 50,
        "seed": 42,
    }

    if lgb_opt:
        print("Applying Optuna params to LightGBM...")
        lgbm_config["params"].update(lgb_opt)

    os.makedirs("models/lightgbm/checkpoints", exist_ok=True)

    train_lightgbm(
        X_lgbm_train, y_lgbm_train,
        X_lgbm_val, y_lgbm_val,
        lgbm_config,
        save_path="models/lightgbm/checkpoints/best.pkl"
    )

    print("\n===============================")
    print("All models trained and saved.")
    print("===============================")


if __name__ == "__main__":
    main()
