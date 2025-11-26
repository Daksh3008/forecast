import pandas as pd
from pathlib import Path

def load_raw_csv(path):
    return pd.read_csv(path)

def load_main_price_file():
    return load_raw_csv("data/raw/brent.csv")

def load_macro_file():
    return load_raw_csv("data/raw/merged_macro.csv")

def merge_raw_sources(price_df, macro_df=None, sentiment_df=None):
    df = price_df.copy()
    if macro_df is not None:
        df = df.merge(macro_df, on="date", how="left")
    if sentiment_df is not None:
        df = df.merge(sentiment_df, on="date", how="left")
    return df.sort_values("date").reset_index(drop=True)
