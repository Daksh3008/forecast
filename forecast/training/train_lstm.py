import os
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict
from forecast.training.trainer_utils import create_dataloader, move_batch_to_device, compute_loss
from forecast.models.lstm_attention import LSTMAttentionModel
import config
from forecast.utils.optuna_loader import load_optuna_params





def train_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    config: Dict,
    save_path: str,
):
    optuna_path = "models/lstm_attention/best_optuna_params.json"
    opt_params = load_optuna_params(optuna_path)

    if opt_params is not None:
        config.update(opt_params)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LSTMAttentionModel(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
        fc_hidden=config["fc_hidden"],
        device=device,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    train_loader = create_dataloader(X_train, y_train, batch_size=config["batch_size"])
    val_loader = create_dataloader(X_val, y_val, batch_size=config["batch_size"], shuffle=False)

    best_loss = float("inf")
    epochs = config["epochs"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            Xb, yb = move_batch_to_device(batch, device)
            optimizer.zero_grad()

            preds = model(Xb)
            loss = compute_loss(preds, yb, loss_fn)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                Xb, yb = move_batch_to_device(batch, device)
                preds = model(Xb)
                loss = compute_loss(preds, yb, loss_fn)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch}/{epochs}  Train Loss: {train_loss:.5f}  Val Loss: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            model.save(save_path, extra={"val_loss": val_loss})
            print(f"  ✔ Saved best model (loss={val_loss:.5f})")

    return best_loss
