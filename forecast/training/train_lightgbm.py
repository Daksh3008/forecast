from typing import Dict
from forecast.models.lightgbm_model import LightGBMWrapper
import numpy as np
import os
import config
from forecast.utils.optuna_loader import load_optuna_params



def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict,
    save_path: str,
):
    optuna_path = "models/lightgbm/best_optuna_params.json"
    opt_params = load_optuna_params(optuna_path)

    if opt_params is not None:
        config.update(opt_params)
    params = config["params"]
    num_boost_round = config["num_boost_round"]
    early_stop = config["early_stopping"]

    model = LightGBMWrapper(
        params=params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stop,
        seed=config["seed"],
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model.fit(X_train, y_train, X_val, y_val, verbose=50)
    model.save(save_path)

    print(f"✔ Saved LightGBM model at {save_path}")
    return model
