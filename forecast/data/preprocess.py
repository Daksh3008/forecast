"""
Clean, forward-fill, fix datatypes, handle missing values.

Output: clean DataFrame ready for feature engineering.
"""

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure correct dtype for date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

    # Forward fill numerical data
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(method="ffill")
    df[num_cols] = df[num_cols].fillna(method="bfill")

    # Drop rows with missing date
    df = df.dropna(subset=["date"])

    df = df.reset_index(drop=True)
    return df
