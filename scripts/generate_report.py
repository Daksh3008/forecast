#!/usr/bin/env python3
"""
Generate a dynamic forecast report for Deepak Nitrite without sentiment analysis.
Uses:
 - Fuzzy relevance matching on news headlines
 - Recency-weighted ranking
 - Technical snapshot
 - Macro snapshot
 - Correlation summary
 - Plain-English narrative
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from datetime import datetime, timedelta

try:
    from forecast.inference.forecast_to_date import forecast_to_date
except Exception:
    forecast_to_date = None

FEATURE_CSV = Path("data/processed/deepakntr_bo_features.csv")
NEWS_CSV = Path("data/raw/news_real.csv")
FORECAST_DIR = Path("forecasts/results")
REPORT_DIR = Path("forecasts/reports")


# -----------------------------
# 1) Forecast loader
# -----------------------------
def ensure_forecast_exists(target_date: str):
    csv_path = FORECAST_DIR / f"forecast_{target_date}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df.iloc[0].to_dict()
    else:
        if forecast_to_date is None:
            raise FileNotFoundError("No forecast found and forecast_to_date() unavailable.")
        print("Forecast not found — generating...")
        res = forecast_to_date(target_date)
        csv_path = Path(res["csv"])
        df = pd.read_csv(csv_path)
        return df.iloc[0].to_dict()


# -----------------------------
# 2) Feature matrix + correlations
# -----------------------------
def load_feature_matrix():
    return pd.read_csv(FEATURE_CSV, parse_dates=["Date"])


def compute_correlations(df):
    numeric = df.select_dtypes(include=[np.number]).copy()
    if "Close" not in numeric.columns:
        raise KeyError("Close column missing in feature matrix.")
    corrs = numeric.corr()["Close"].drop("Close").abs().sort_values(ascending=False)
    return list(corrs.items())  # list of (feature, abs_corr)


# -----------------------------
# 3) Fuzzy news relevance (NO sentiment)
# -----------------------------
def fuzzy_relevance_score(text, query=("Deepak Nitrite", "Deepak Nitrite Limited", "DEEPAKNTR")):
    txt = (text or "").lower()
    best = 0
    for q in query:
        ql = q.lower()
        pr = fuzz.partial_ratio(ql, txt)
        ts = fuzz.token_sort_ratio(ql, txt)
        best = max(best, pr, ts)
    if "deepak" in txt:
        best += 20  # small bonus
    return min(100.0, best)


def select_top_news(news_df, top_k=10, days_window=60):
    df = news_df.copy()

    # Identify date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "published" in df.columns:
        df["date"] = pd.to_datetime(df["published"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # Recency score
    now = pd.Timestamp.now().normalize()
    df["days_since"] = (now - df["date"]).dt.days.fillna(9999)
    half_life = 30
    df["recency_w"] = np.exp(-np.log(2) * (df["days_since"] / half_life))

    # Pick a text field
    def extract_text(row):
        for c in ["headline", "title", "summary", "text", "body", "content"]:
            if c in row and pd.notna(row[c]):
                return str(row[c])
        return ""

    df["raw_text"] = df.apply(extract_text, axis=1)
    df["fuzzy"] = df["raw_text"].apply(fuzzy_relevance_score)

    # Combined relevance
    df["relevance"] = df["fuzzy"] * df["recency_w"]

    # Sort and pick top_k
    ranked = df.sort_values("relevance", ascending=False).head(top_k)

    # Convert to simple dict list
    results = []
    for _, r in ranked.iterrows():
        results.append({
            "title": r.get("headline") or r.get("title") or r.get("raw_text"),
            "date": r.get("date"),
            "source": r.get("source") or r.get("publisher") or r.get("url") or ""
        })
    return results


# -----------------------------
# 4) Snapshots for report
# -----------------------------
def technical_snapshot(last_row):
    rsi = last_row.get("ind_rsi_14", np.nan)
    ma10 = last_row.get("ind_ma10", np.nan)
    ma50 = last_row.get("ind_ma50", np.nan)
    ma200 = last_row.get("ind_ma200", np.nan)

    condition = "neutral"
    if rsi <= 30:
        condition = "oversold"
    elif rsi >= 70:
        condition = "overbought"

    return (
        f"RSI ({rsi:.1f}) indicates {condition}. "
        f"IND_MA10 = {ma10:.2f}, IND_MA50 = {ma50:.2f}, IND_MA200 = {ma200:.2f}"
    )


def macro_snapshot(last_row):
    brent = last_row.get("brent_close", np.nan)
    fx = last_row.get("usd_inr", np.nan)
    return (
        f"Crude oil is around ${brent:.2f}/bbl. "
        f"USD/INR at {fx:.2f}. "
        f"Both impact input costs for Deepak Nitrite."
    )


# -----------------------------
# 5) Main report generator
# -----------------------------
def main(target_date):
    # Ensure forecast exists
    row = ensure_forecast_exists(target_date)
    pred_price = float(row["ensemble"])

    # Build naive CI from model disagreement
    try:
        preds = np.array([
            float(row.get("model_lstm", np.nan)),
            float(row.get("model_tcn", np.nan)),
            float(row.get("model_lightgbm", np.nan))
        ])
        sd = np.nanstd(preds)
        conf_low = pred_price - 1.96 * sd
        conf_high = pred_price + 1.96 * sd
    except:
        conf_low = pred_price * 0.98
        conf_high = pred_price * 1.02

    # Load feature matrix
    df_feat = load_feature_matrix()
    last_row = df_feat.sort_values("Date").iloc[-1]

    # Correlations
    corr_list = compute_correlations(df_feat)

    # Top news
    if not NEWS_CSV.exists():
        raise FileNotFoundError(f"Missing news_real.csv at {NEWS_CSV}")
    news_df = pd.read_csv(NEWS_CSV)
    top_news = select_top_news(news_df, top_k=10, days_window=60)

    # Generate report structure
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"report_{target_date}.md"

    lines = []
    lines.append("=" * 80)
    lines.append("🔮 Predicting Deepak Nitrite price (LSTM + Attention)")
    lines.append("=" * 80 + "\n")

    # Executive section
    lines.append("🧾 Executive Market Overview")
    lines.append("-" * 60)
    lines.append(
        f"Our model forecasts Deepak Nitrite’s price at approximately ₹{pred_price:,.2f}.\n"
        f"95% confidence range: ₹{conf_low:,.2f} — ₹{conf_high:,.2f}\n"
    )

    # --- Model Price Table ---
    lines.append("\n📊 Model Predictions")
    lines.append("-" * 60)
    lines.append("| Model          | Predicted Price |")
    lines.append("|----------------|------------------|")
    lines.append(f"| LSTM+Attention | ₹{float(row.get('model_lstm', 0)):,.2f} |")
    lines.append(f"| TCN            | ₹{float(row.get('model_tcn', 0)):,.2f} |")
    lines.append(f"| LightGBM       | ₹{float(row.get('model_lightgbm', 0)):,.2f} |")
    lines.append(f"| Ensemble    | ₹{pred_price:,.2f} |")





    # Macro
    lines.append("\n🌍 Macro Context")
    lines.append("-" * 60)
    lines.append(macro_snapshot(last_row) + "\n")

    # Technical
    lines.append("\n📈 Technical / Micro Context")
    lines.append("-" * 60)
    lines.append(technical_snapshot(last_row) + "\n")

    # News
    lines.append("\n📰 News Summary (Top 10 most relevant)")
    lines.append("-" * 60)
    for i, n in enumerate(top_news, start=1):
        title = n["title"] or "(no title)"
        source = n["source"] or ""
        lines.append(f"{i}. {title} - {source}")

    # Correlations
    lines.append("\n\n🔎 Top correlated features with price")
    lines.append("-" * 60)
    for feat, corr in corr_list[:10]:
        lines.append(f"• {feat} → correlation (abs) = {corr:.2f}")

    # Plain English summary
    lines.append("\n\n📝 Plain English Summary")
    lines.append("-" * 60)
    lines.append(
        "• Crude oil and USD/INR movements influence Deepak Nitrite’s cost structure.\n"
        "• Recent news headlines highlight market sensitivity to earnings, margins, and sector trends.\n"
        "• Technical indicators show the current momentum and trend context.\n"
        "• Correlation table highlights which features historically moved most strongly with price.\n"
    )

    lines.append("=" * 80)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    main(args.target)
