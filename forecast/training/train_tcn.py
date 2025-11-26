"""
Train TCN on log-price targets.
Reports RMSE and saves best checkpoint.
"""
import yaml
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from forecast.models.tcn import TCNModel

CFG_PATH = Path("config/train_tcn.yaml")
BASE_CFG = Path("config/base.yaml")

def load_cfg():
    try:
        cfg = yaml.safe_load(CFG_PATH.read_text())
    except Exception:
        cfg = {}
    try:
        base = yaml.safe_load(BASE_CFG.read_text())
    except Exception:
        base = {}
    seq_len = int(cfg.get("seq_len", base.get("seq_len", 60)))
    return cfg, seq_len

def train_tcn(save_path="models/tcn/checkpoints/best.pt"):
    cfg, seq_len = load_cfg()
    X = np.load("data/processed/X_seq.npy")
    y = np.load("data/processed/y_seq.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_size = X.shape[-1]

    model = TCNModel(
        input_size=input_size,
        num_channels=cfg.get("num_channels", [32, 64, 128]),
        kernel_size=int(cfg.get("kernel_size", 3)),
        dropout=float(cfg.get("dropout", 0.1))
    ).to(device)

    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 50))
    lr = float(cfg.get("lr", 4.4e-3))

    n = len(y)
    split = int(n * 0.8)
    Xtr, Xvl = X[:split], X[split:]
    ytr, yvl = y[:split], y[split:]

    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(Xvl, dtype=torch.float32), torch.tensor(yvl, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_rmse = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * Xb.size(0)
        tr_rmse = (tr_loss / len(Xtr)) ** 0.5

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device); yb = yb.to(device)
                mse = loss_fn(model(Xb), yb).item()
                val_mse += mse * Xb.size(0)
        val_rmse = (val_mse / len(Xvl)) ** 0.5

        print(f"[TCN] Epoch {epoch}/{epochs} train_rmse={tr_rmse:.6f} val_rmse={val_rmse:.6f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            model.save(save_path, extra={"val_rmse": float(best_val_rmse), "epoch": best_epoch})
            print(f"  Saved best model at epoch {epoch} val_rmse={best_val_rmse:.6f}")

    return {"best_val_rmse": best_val_rmse, "best_epoch": best_epoch}

if __name__ == "__main__":
    train_tcn()
