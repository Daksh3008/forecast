"""
Load raw CSV files from data/raw/.
User will supply these files manually.

Your files may be:
 - prices.csv
 - macro.csv
 - sentiment.csv
Any new files can be added easily.
"""

import pandas as pd
from pathlib import Path


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    return pd.read_csv(path)


def load_main_price_file(root="data/raw/prices.csv") -> pd.DataFrame:
    return load_raw_csv(root)


def load_sentiment_file(root="data/raw/sentiment.csv") -> pd.DataFrame:
    return load_raw_csv(root)


def load_macro_file(root="data/raw/macro.csv") -> pd.DataFrame:
    return load_raw_csv(root)


def merge_raw_sources(price_df: pd.DataFrame,
                      sentiment_df: pd.DataFrame | None = None,
                      macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = price_df.copy()

    if sentiment_df is not None:
        df = df.merge(sentiment_df, on="date", how="left")

    if macro_df is not None:
        df = df.merge(macro_df, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    return df
