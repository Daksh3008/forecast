"""
StandardScaler wrapper that handles saving/loading
and works for both:
 - 2D LightGBM input
 - 3D Deep Learning input (flatten + reshape)
"""

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class FeatureScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        """
        X can be:
          - (samples, features)
          - (samples, seq_len, features)
        """
        if X.ndim == 3:
            s, t, f = X.shape
            X2 = X.reshape(s, t * f)
            X2 = self.scaler.fit_transform(X2)
            return X2.reshape(s, t, f)
        return self.scaler.fit_transform(X)

    def transform(self, X):
        if X.ndim == 3:
            s, t, f = X.shape
            X2 = X.reshape(s, t * f)
            X2 = self.scaler.transform(X2)
            return X2.reshape(s, t, f)
        return self.scaler.transform(X)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    def load(self, path):
        self.scaler = joblib.load(path)
        return self
