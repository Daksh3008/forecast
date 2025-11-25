"""
Simple rolling backtest.

Usage:
    python scripts/backtest.py
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from forecast.inference.forecast_to_date import forecast_to_date


def main():
    df = pd.read_csv("data/raw/prices.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    dates = df["date"].iloc[-200:]  # last 200 days
    actual = df["close"].iloc[-200:].values

    preds = []

    for d in tqdm(dates, desc="Backtesting"):
        try:
            result = forecast_to_date(str(d.date()))
            preds.append(result["ensemble"])
        except Exception:
            preds.append(np.nan)

    # store output
    out = pd.DataFrame({
        "date": dates.values,
        "actual": actual,
        "predicted": preds,
    })

    out.to_csv("forecasts/results/backtest.csv", index=False)
    print("✔ Backtest saved to forecasts/results/backtest.csv")


if __name__ == "__main__":
    main()
