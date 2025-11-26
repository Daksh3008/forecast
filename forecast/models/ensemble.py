"""
Ensemble helper. Supports equal-weight average (default) or weighted ensemble
by providing weights dict {"lstm": w1, "tcn": w2, "lightgbm": w3}
"""

from typing import Dict


def ensemble_average(preds: Dict[str, float], weights: Dict[str, float] = None) -> float:
    if weights is None:
        vals = list(preds.values())
        return sum(vals) / len(vals)
    # normalize weights
    keys = list(preds.keys())
    wsum = sum(weights.get(k, 1.0) for k in keys)
    return sum(preds[k] * (weights.get(k, 1.0) / wsum) for k in keys)
