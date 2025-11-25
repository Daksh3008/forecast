"""
Logging utility to load logging.yaml configuration.
"""

import logging
import logging.config
import yaml
from pathlib import Path


def setup_logging(path="config/logging.yaml"):
    path = Path(path)
    if not path.exists():
        print("⚠ logging.yaml not found, using default logging.")
        logging.basicConfig(level=logging.INFO)
        return

    with open(path, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)

    logger = logging.getLogger("predictor")
    return logger
