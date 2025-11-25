"""
Optuna optimization for TCN (Option B: last 20% val)
Usage:
    python -m scripts.optuna.optimize_tcn --trials 50
"""
import os
import json
import argparse
import numpy as np
import optuna
import torch
import torch.nn as nn
from forecast.models.tcn import TCNModel

DATA_X = "data/processed/X_seq.npy"
DATA_Y = "data/processed/y_seq.npy"
OUT_PATH = "models/tcn/best_optuna_params.json"

def load_data():
    X = np.load(DATA_X)
    y = np.load(DATA_Y)
    n = len(y)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]

def objective(trial):
    X_train, y_train, X_val, y_val = load_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # hyperparams
    kernel_size = trial.suggest_categorical("kernel_size", [2,3,5])
    base_channel = trial.suggest_categorical("base_channel", [32,64,128])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    num_channels = [base_channel * (2**i) for i in range(num_layers)]
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-5, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [32,64,128])
    epochs = 20

    model = TCNModel(input_size=X_train.shape[-1], num_channels=num_channels, kernel_size=kernel_size, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    from torch.utils.data import DataLoader, TensorDataset
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xvl = torch.tensor(X_val, dtype=torch.float32)
    yvl = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xvl, yvl), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        for Xb, yb in train_loader:
            Xb = Xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0; cnt = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device); yb = yb.to(device)
                l = loss_fn(model(Xb), yb).item()
                val_loss += l; cnt+=1
        val_loss = val_loss / max(1, cnt)
        if val_loss < best_val:
            best_val = val_loss
        trial.report(best_val, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val

def run(trials:int):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    print("Best trial:", study.best_trial.params)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"Saved best params to {OUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    run(args.trials)
