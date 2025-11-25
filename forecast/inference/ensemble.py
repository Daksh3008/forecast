"""
Simple ensemble averaging.
Takes outputs from LSTM, TCN, LightGBM and returns final price.
"""

from typing import Dict


def ensemble_average(preds: Dict[str, float]) -> float:
    """
    preds = {
        "lstm": <float>,
        "tcn": <float>,
        "lightgbm": <float>
    }
    """
    vals = list(preds.values())
    return sum(vals) / len(vals)
