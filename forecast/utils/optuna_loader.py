import json
from pathlib import Path

def load_optuna_params(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARNING] Optuna params not found: {path}, using default config.")
        return None
    with open(path, "r") as f:
        return json.load(f)
