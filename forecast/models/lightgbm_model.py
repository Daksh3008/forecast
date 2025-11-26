"""
LightGBM wrapper that saves and loads Booster with joblib.
"""

from pathlib import Path
import json
import joblib
import lightgbm as lgb
import numpy as np


class LightGBMWrapper:
    def __init__(self, params=None, num_boost_round: int = 1000, early_stopping_rounds: int = 50):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.bst = None

    def fit(self, X, y, X_val=None, y_val=None):
        dtrain = lgb.Dataset(X, label=y)
        valid_sets = None
        callbacks = []
        if X_val is not None and y_val is not None:
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets = [dtrain, dvalid]
            callbacks = [lgb.early_stopping(stopping_rounds=self.early_stopping_rounds), lgb.log_evaluation(period=100)]
        self.bst = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round, valid_sets=valid_sets, valid_names=["train", "valid"] if valid_sets else None, callbacks=callbacks)
        return self

    def predict(self, X):
        if self.bst is None:
            raise ValueError("Model is not trained.")
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Save booster model string + metadata
        joblib.dump({"params": self.params, "num_boost_round": self.num_boost_round, "model_str": self.bst.model_to_string()}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        obj = cls(params=data.get("params", {}), num_boost_round=data.get("num_boost_round", 1000))
        model_str = data.get("model_str")
        if model_str:
            obj.bst = lgb.Booster(model_str=model_str)
        return obj
