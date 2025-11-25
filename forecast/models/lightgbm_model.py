"""
LightGBM wrapper for regression forecasting.
Uses lightgbm's sklearn API for easy integration.

Provides:
- fit(X, y, X_val=None, y_val=None)
- predict(X)
- save(path)
- load(path)

Note: this wrapper does not perform GPU-specific operations inside LightGBM itself.
To enable LightGBM GPU training, pass params with "device": "gpu" or "boosting_type":"gbdt" and "device": "gpu"
(e.g. params['device'] = 'gpu').
"""
import os
import joblib
import numpy as np
from typing import Optional, Dict
import lightgbm as lgb


class LightGBMWrapper:
    def __init__(self, params: Optional[Dict] = None, num_boost_round: int = 1000, early_stopping_rounds: int = 50, seed: int = 42):
        """
        params: LightGBM parameters dict. Example:
            {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'device': 'gpu'   # optional: 'cpu' or 'gpu' (if compiled with GPU)
            }
        """
        self.params = params or {"objective": "regression", "metric": "rmse"}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.model: Optional[lgb.Booster] = None

    def fit(self, X, Y, X_val=None, y_val=None, verbose=100):
        lgb_train = lgb.Dataset(X, label=Y)

        valid_sets = [lgb_train]
        valid_names = ["train"]

        callbacks = []

        # If validation data passed → enable early stopping callback
        if X_val is not None and y_val is not None:
            lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            valid_sets.append(lgb_valid)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(stopping_rounds=self.early_stopping_rounds))

        # Log training metrics every N iterations
        callbacks.append(lgb.log_evaluation(period=verbose))

        # Train the model
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks  # <── FIXED
        )


    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first or load() a model.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path: str):
        # Save Booster object + params
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"params": self.params, "num_boost_round": self.num_boost_round, "model": self.model}, path)

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        wrapper = cls(params=obj.get("params"), num_boost_round=obj.get("num_boost_round", 1000))
        wrapper.model = obj.get("model")
        return wrapper
