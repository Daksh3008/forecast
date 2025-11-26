"""
Fetch daily data for Brent, DXY, USDINR, VIX, and WTI
from 2007-01-01 until today.

Features:
 - Downloads using yfinance
 - Cleans MultiIndex columns
 - Normalizes names
 - Creates continuous daily data (forward-fill)
 - Saves each asset individually
 - Saves merged macro indicators into merged_macro.csv
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

OUT = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = "2007-01-01"

ASSETS = {
    "brent": "BZ=F",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "vix": "^VIX",
    "wti": "CL=F"
}


def normalize_columns(df: pd.DataFrame):
    """Flatten MultiIndex (if any) + lowercase clean names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c) for c in col if c not in ("", None)]).lower()
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return df


def clean_df(df: pd.DataFrame):
    """Normalize OHLC names, ensure continuous daily data."""
    df = df.reset_index()
    df = normalize_columns(df)

    # Map OHLC columns
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "volume": "volume"
    }

    for col in list(df.columns):
        base_col = col.split("_")[0]
        if base_col in rename_map:
            df = df.rename(columns={col: rename_map[base_col]})

    # Keep essential columns
    keep = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    df = df[[c for c in keep if c in df.columns]]

    # Create continuous daily data
    df["date"] = pd.to_datetime(df["date"])
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
    df = df.reset_index()

    # Fill missing data
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")

    return df


def main():
    all_frames = []

    for name, ticker in ASSETS.items():
        print(f"\nDownloading {name} ({ticker}) from {START_DATE} to today...")

        df = yf.download(
            ticker,
            start=START_DATE,
            end=None,
            auto_adjust=True,
            interval="1d"
        )

        if df.empty:
            print(f"⚠ WARNING: No data for {ticker}, skipping.")
            continue

        df = clean_df(df)
        df.to_csv(OUT / f"{name}.csv", index=False)

        # rename close column for merging
        if "close" in df.columns:
            df_renamed = df.rename(columns={"close": f"{name}_close"})
            all_frames.append(df_renamed[["date", f"{name}_close"]])

    # ---------- MERGE MACRO INDICATORS ----------
    if not all_frames:
        print("❌ ERROR: No data downloaded. Check tickers or internet.")
        return

    merged = all_frames[0]
    for df in all_frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.fillna(method="ffill").fillna(method="bfill")

    merged.to_csv(OUT / "merged_macro.csv", index=False)

    print("\n✔ Saved:")
    print(" data/raw/brent.csv")
    print(" data/raw/dxy.csv")
    print(" data/raw/usdinr.csv")
    print(" data/raw/vix.csv")
    print(" data/raw/wti.csv")
    print(" data/raw/merged_macro.csv")
    print("✔ Historical data (2007 → today) fetched successfully.")


if __name__ == "__main__":
    main()
