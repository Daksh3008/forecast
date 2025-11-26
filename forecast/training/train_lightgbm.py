"""
Train LightGBM on log-price targets (y_tab).
Compatible with LightGBM versions that do NOT accept early_stopping_rounds in lgb.train().
"""

import os
import json
import yaml
from pathlib import Path
import numpy as np
import lightgbm as lgb

from forecast.models.lightgbm_model import LightGBMWrapper

CFG_PATH = Path("config/train_lightgbm.yaml")
OPTUNA_PATH = Path("models/lightgbm/best_optuna_params.json")


def load_cfg():
    try:
        cfg = yaml.safe_load(CFG_PATH.read_text())
    except Exception:
        cfg = {}
    return cfg


def  train_lightgbm(save_path="models/lightgbm/checkpoints/best.pkl"):
    cfg = load_cfg()

    X = np.load("data/processed/X_tab.npy")
    y = np.load("data/processed/y_tab.npy")  # log-price next-day

    n = len(y)
    split = int(n * 0.8)
    Xtr, Xvl = X[:split], X[split:]
    ytr, yvl = y[:split], y[split:]

    params = cfg.get("params", {})

    # Load optuna tuned parameters if exists
    if OPTUNA_PATH.exists():
        try:
            with open(OPTUNA_PATH, "r") as f:
                opt = json.load(f)
            params.update(opt)
            print("Loaded optuna params for LightGBM.")
        except Exception:
            pass

    params.setdefault("objective", "regression")
    params.setdefault("metric", "rmse")
    params["min_gain_to_split"] = 0.0
    params["verbose"] = -1
    num_boost_round = int(cfg.get("num_boost_round", 1000))
    early_stop = int(cfg.get("early_stopping", 50))

    dtrain = lgb.Dataset(Xtr, label=ytr)
    dvalid = lgb.Dataset(Xvl, label=yvl, reference=dtrain)

    # ---------------------------
    # FIX: use callback for early stopping
    # ---------------------------
    callbacks = [lgb.early_stopping(early_stop, verbose=True)]

    print("Training LightGBM with callbacks for early stopping...")

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks  # FIXED
    )

    wrapper = LightGBMWrapper(
        params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stop
    )
    wrapper.bst = bst
    wrapper.save(save_path)
    print(f"Saved LightGBM model to {save_path}")

    # return validation RMSE
    preds = bst.predict(Xvl, num_iteration=bst.best_iteration)
    rmse = float(np.sqrt(((preds - yvl) ** 2).mean()))

    print(f"Validation RMSE: {rmse:.6f}")

    return {
        "val_rmse": rmse,
        "best_iteration": int(bst.best_iteration)
    }


if __name__ == "__main__":
    train_lightgbm()
