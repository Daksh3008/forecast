"""
Backtest the forecasting pipeline over the last N days.

This script calls forecast.inference.forecast_to_date for each date in the evaluation window.
It collects predictions and computes RMSE and MAPE vs. actual 'target_close' in the processed CSV.

Usage:
    python -m scripts.backtest
"""

from datetime import timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import math
import sys

from forecast.inference.forecast_to_date import forecast_to_date

# read configs to get feature csv
FORECAST_CFG_PATH = Path("config/forecast.yaml")
try:
    fcfg = yaml.safe_load(FORECAST_CFG_PATH.read_text())
except Exception:
    fcfg = {}
FEATURE_CSV = Path(fcfg.get("paths", {}).get("feature_csv", "data/processed/deepakntr_bo_features.csv"))

def evaluate_last_n_days(n_days:int = 200):
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURE_CSV}")

    df = pd.read_csv(FEATURE_CSV)
    if "Date" in df.columns:
        df = df.rename(columns={c: c.lower() for c in df.columns if c in ["Date", "Close", "Adj Close"]})
    # build canonical features and drop rows without target_close
    from forecast.data.features import build_features
    df_proc = build_features(df, price_col="close" if "close" in df.columns else "Close")
    df_proc = df_proc.dropna(subset=["target_close"]).reset_index(drop=True)
    dates = pd.to_datetime(df_proc["date"]).dt.date

    results = []
    last_date = dates.iloc[-1]
    start_date = last_date - timedelta(days=n_days)
    cand_dates = [d for d in dates if d > start_date]

    for d in cand_dates:
        try:
            out = forecast_to_date(str(d))
        except Exception as e:
            print(f"Skipping {d}: {e}")
            continue
        # lookup actual target_close for the same date
        actual_row = df_proc[pd.to_datetime(df_proc["date"]).dt.date == d]
        if actual_row.empty:
            continue
        actual = float(actual_row["target_close"].iloc[0])
        pred = float(out["ensemble"])
        results.append({"date": str(d), "actual": actual, "pred": pred})

    results_df = pd.DataFrame(results)
    results_df["error"] = results_df["pred"] - results_df["actual"]
    results_df["abs_error"] = results_df["error"].abs()
    results_df["pct_error"] = results_df["abs_error"] / results_df["actual"].abs() * 100.0

    rmse = np.sqrt((results_df["error"] ** 2).mean()) if not results_df.empty else float("nan")
    mape = results_df["pct_error"].mean() if not results_df.empty else float("nan")

    print(f"Backtest results on {len(results_df)} points: RMSE={rmse:.4f}, MAPE={mape:.2f}%")
    results_df.to_csv("forecasts/results/backtest_results.csv", index=False)
    print("Saved forecasts/results/backtest_results.csv")

if __name__ == "__main__":
    evaluate_last_n_days(200)
