"""
Optuna optimization for LightGBM (Option B: last 20% val)
Usage:
    python -m scripts.optuna.optimize_lightgbm --trials 50
"""
import os
import json
import argparse
import numpy as np
import optuna
import lightgbm as lgb
from forecast.models.lightgbm_model import LightGBMWrapper
import torch
import torch.nn

DATA_X = "data/processed/X_tab.npy"
DATA_Y = "data/processed/y_tab.npy"
OUT_PATH = "models/lightgbm/best_optuna_params.json"

def load_data():
    X = np.load(DATA_X)
    y = np.load(DATA_Y)
    n = len(y)
    split = int(n * 0.8)
    return X[:split], y[:split], X[split:], y[split:]

def objective(trial):
    X_train, y_train, X_val, y_val = load_data()
    # param search space
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.2),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        "device": "gpu" if torch.cuda.is_available() else "cpu"
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    bst = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dtrain, dvalid], valid_names=["train","valid"], callbacks=callbacks)
    preds = bst.predict(X_val, num_iteration=bst.best_iteration)
    rmse = np.sqrt(np.mean((preds - y_val)**2))
    # store booster in trial user attrs for later optional use
    trial.set_user_attr("best_bst", bst.model_to_string())
    return rmse

def run(trials:int):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    best = study.best_trial.params
    print("Best trial:", best)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved best params to {OUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    run(args.trials)
