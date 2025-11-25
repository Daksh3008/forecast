"""
Centralized path resolver.

Example:
    from forecast.utils.paths import get_path
    full_path = get_path("models/lstm_attention/checkpoints/best.pt")
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # predictor/


def get_path(relative: str | Path) -> Path:
    """Convert relative project path → absolute path."""
    return PROJECT_ROOT / relative
