"""
Recursive multi-step forecasting to a user-specified target date.

Key behaviour:
 - If target_date <= last available date: forecast for that date using available history.
 - If target_date > last available date: perform recursive forecasting:
     repeatedly predict next-day Close, append synthetic row, recompute indicators,
     and continue until target_date is reached.
 - All technical indicators are recomputed at each synthetic step (MA, EMA, RSI, MACD, BB, vol, zscore, lags).
 - Exogenous series (brent_close, usd_inr, news_compound) are kept constant equal to their last observed value.
 - Saves CSV and Markdown report named using the requested target date.

Usage:
    python -m forecast.inference.forecast_to_date --target 2026-01-01
"""
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import math
from datetime import timedelta

from forecast.data.scalers import FeatureScaler
from forecast.models.lstm_attention import LSTMAttentionModel
from forecast.models.tcn import TCNModel
from forecast.models.lightgbm_model import LightGBMWrapper
from forecast.inference.ensemble import ensemble_average
from forecast.inference.report_builder import build_markdown_report

SEQ_LEN = 120
FEATURE_CSV = Path("data/processed/deepakntr_bo_features.csv")


# -------------------------
# Technical indicator functions
# -------------------------
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with at least 'Close' (and possibly 'Open','High','Low','Volume'),
    compute/overwrite the following columns used by your feature matrix:
      - pct_change, log_return
      - ind_rsi_14
      - ind_ma10, ind_ma50, ind_ma200
      - ind_ema20
      - ind_bb_h, ind_bb_l (Bollinger bands: mean +/- 2*std over 20)
      - ind_mom_10
      - ind_macd, ind_macd_sig, macd_hist
      - log_return_lag1/2/5/10/20
      - vol_5, vol_10, vol_20 (rolling std of pct_change)
      - zscore_20, zscore_50 (z-score of Close)
      - trend_spread (ma10 - ma50)
      - ret_scaled (z-score of log_return over 20)
      - target_close (Close.shift(-1))
    """
    df = df.copy()

    # Basic returns
    df["pct_change"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()

    # MA / EMA
    df["ind_ma10"] = df["Close"].rolling(10, min_periods=1).mean()
    df["ind_ma50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["ind_ma200"] = df["Close"].rolling(200, min_periods=1).mean()
    df["ind_ema20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # Bollinger bands (20)
    roll20 = df["Close"].rolling(20, min_periods=1)
    roll20_mean = roll20.mean()
    roll20_std = roll20.std(ddof=0)
    df["ind_bb_h"] = roll20_mean + 2 * roll20_std
    df["ind_bb_l"] = roll20_mean - 2 * roll20_std

    # Momentum
    df["ind_mom_10"] = df["Close"].diff(10)

    # MACD: EMA12 - EMA26, signal = EMA9 of MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["ind_macd"] = ema12 - ema26
    df["ind_macd_sig"] = df["ind_macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["ind_macd"] - df["ind_macd_sig"]

    # Lagged log returns
    df["log_return_lag1"] = df["log_return"].shift(1)
    df["log_return_lag2"] = df["log_return"].shift(2)
    df["log_return_lag5"] = df["log_return"].shift(5)
    df["log_return_lag10"] = df["log_return"].shift(10)
    df["log_return_lag20"] = df["log_return"].shift(20)

    # Volatility
    df["vol_5"] = df["pct_change"].rolling(5, min_periods=1).std(ddof=0)
    df["vol_10"] = df["pct_change"].rolling(10, min_periods=1).std(ddof=0)
    df["vol_20"] = df["pct_change"].rolling(20, min_periods=1).std(ddof=0)

    # Z-scores for Close
    roll20_mean_close = df["Close"].rolling(20, min_periods=1).mean()
    roll20_std_close = df["Close"].rolling(20, min_periods=1).std(ddof=0)
    df["zscore_20"] = (df["Close"] - roll20_mean_close) / (roll20_std_close.replace(0, np.nan))

    roll50_mean_close = df["Close"].rolling(50, min_periods=1).mean()
    roll50_std_close = df["Close"].rolling(50, min_periods=1).std(ddof=0)
    df["zscore_50"] = (df["Close"] - roll50_mean_close) / (roll50_std_close.replace(0, np.nan))

    # Trend spread
    df["trend_spread"] = df["ind_ma10"] - df["ind_ma50"]

    # ret_scaled: z-score of log_return over 20
    lr_roll_mean = df["log_return"].rolling(20, min_periods=1).mean()
    lr_roll_std = df["log_return"].rolling(20, min_periods=1).std(ddof=0)
    df["ret_scaled"] = (df["log_return"] - lr_roll_mean) / (lr_roll_std.replace(0, np.nan))

    # target_close (next day's Close)
    df["target_close"] = df["Close"].shift(-1)

    # Keep columns in expected order if they exist; do not fail if some are missing.
    return df


# -------------------------
# Helper: build features for a single sequence row (1, seq_len, features)
# -------------------------
def make_sequence_from_df(df_proc: pd.DataFrame, seq_end_idx: int, feature_cols):
    """
    Build a sequence (1, seq_len, features) that ends at seq_end_idx (inclusive).
    """
    start_idx = seq_end_idx - (SEQ_LEN - 1)
    if start_idx < 0:
        raise ValueError(f"Not enough history to build a sequence of length {SEQ_LEN}. Need at least {SEQ_LEN} rows.")
    X_rows = df_proc.iloc[start_idx: seq_end_idx + 1][feature_cols].values.astype(np.float32)
    return X_rows.reshape(1, SEQ_LEN, -1)


# -------------------------
# Main forecasting function
# -------------------------
def forecast_to_date(target_date: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature CSV not found: {FEATURE_CSV}")

    df_raw = pd.read_csv(FEATURE_CSV)
    if "Date" not in df_raw.columns:
        raise KeyError("Feature CSV must contain a 'Date' column.")
    if "Close" not in df_raw.columns:
        raise KeyError("Feature CSV must contain a 'Close' column.")

    df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
    df_raw = df_raw.sort_values("Date").reset_index(drop=True)

    # We'll work on a copy and compute technicals consistently
    df = df_raw.copy()
    df = compute_technical_indicators(df)

    # drop last row where target_close is NaN for training alignment; but keep it for forecasting inputs
    df_proc = df.dropna(subset=["Close"]).reset_index(drop=True)

    # Determine last available date (date for which we have Close)
    last_available_date = df_proc["Date"].iloc[-1].date()

    tdate = pd.to_datetime(target_date).date()

    # Feature columns: pick columns from your earlier list present in df_proc, excluding Date & target_close
    exclude = {"Date", "target_close"}
    feature_cols = [c for c in df_proc.columns if c not in exclude]

    # Load scalers
    lstm_scaler = FeatureScaler().load("models/lstm_attention/scalers/feature_scaler.joblib")
    tcn_scaler = FeatureScaler().load("models/tcn/scalers/feature_scaler.joblib")
    lgbm_scaler = FeatureScaler().load("models/lightgbm/scalers/feature_scaler.joblib")

    # Load models
    lstm_ckpt = Path("models/lstm_attention/checkpoints/best.pt")
    tcn_ckpt = Path("models/tcn/checkpoints/best.pt")
    lgbm_ckpt = Path("models/lightgbm/checkpoints/best.pkl")

    if not (lstm_ckpt.exists() and tcn_ckpt.exists() and lgbm_ckpt.exists()):
        raise FileNotFoundError("One or more model checkpoints are missing. Train models first.")

    lstm_model = LSTMAttentionModel.load(str(lstm_ckpt), map_location=device,
                                        input_size=len(feature_cols), hidden_size=128, n_layers=2, n_heads=4,
                                        dropout=0.1, bidirectional=False, fc_hidden=64).to(device)
    tcn_model = TCNModel.load(str(tcn_ckpt), map_location=device,
                              input_size=len(feature_cols), num_channels=[64, 128, 128],
                              kernel_size=3, dropout=0.1).to(device)
    lgbm_model = LightGBMWrapper.load(str(lgbm_ckpt))

    # If target_date is <= last_available_date: just forecast for that date using existing historical data
    if tdate <= last_available_date:
        seq_end_date = tdate - timedelta(days=1)
        # find last index <= seq_end_date
        mask = df_proc["Date"].dt.date <= seq_end_date
        if mask.sum() < SEQ_LEN:
            raise ValueError(f"Not enough history before {tdate} to build a sequence of length {SEQ_LEN}.")
        seq_end_idx = mask[mask].index[-1]
        X_seq = make_sequence_from_df(df_proc, seq_end_idx, feature_cols)
        X_lstm = lstm_scaler.transform(X_seq.copy())
        X_tcn = tcn_scaler.transform(X_seq.copy())
        X_lgbm_row = df_proc.iloc[seq_end_idx][feature_cols].values.reshape(1, -1)
        X_lgbm = lgbm_scaler.transform(X_lgbm_row.copy())

        lstm_model.eval(); tcn_model.eval()
        with torch.no_grad():
            lstm_pred = lstm_model(torch.tensor(X_lstm, dtype=torch.float32).to(device)).cpu().numpy()[0]
            tcn_pred = tcn_model(torch.tensor(X_tcn, dtype=torch.float32).to(device)).cpu().numpy()[0]
        lgb_pred = lgbm_model.predict(X_lgbm)[0]

        preds = {"lstm": float(lstm_pred), "tcn": float(tcn_pred), "lightgbm": float(lgb_pred)}
        ensemble_pred = float(ensemble_average(preds))

        forecast_for = tdate
    else:
        # Recursive multi-step forecasting until target_date
        days_ahead = (tdate - last_available_date).days
        print(f"Target date {tdate} is {days_ahead} days ahead of last available date {last_available_date}.")
        # start df_sim as a copy of df_proc (we'll append synthetic rows)
        df_sim = df_proc.copy().reset_index(drop=True)

        # Keep track of prediction for final target
        final_pred = None
        forecast_for = tdate

        for step in range(days_ahead):
            # Build sequence that ends at last row of df_sim
            seq_end_idx = len(df_sim) - 1
            if seq_end_idx - (SEQ_LEN - 1) < 0:
                raise ValueError(f"Not enough history to build a sequence of length {SEQ_LEN} for recursive forecasting.")

            X_seq = make_sequence_from_df(df_sim, seq_end_idx, feature_cols)
            X_lstm = lstm_scaler.transform(X_seq.copy())
            X_tcn = tcn_scaler.transform(X_seq.copy())

            # For LGBM, use last row features
            X_lgbm_row = df_sim.iloc[seq_end_idx][feature_cols].values.reshape(1, -1)
            X_lgbm = lgbm_scaler.transform(X_lgbm_row.copy())

            lstm_model.eval(); tcn_model.eval()
            with torch.no_grad():
                lstm_pred = lstm_model(torch.tensor(X_lstm, dtype=torch.float32).to(device)).cpu().numpy()[0]
                tcn_pred = tcn_model(torch.tensor(X_tcn, dtype=torch.float32).to(device)).cpu().numpy()[0]
            lgb_pred = lgbm_model.predict(X_lgbm)[0]

            preds = {"lstm": float(lstm_pred), "tcn": float(tcn_pred), "lightgbm": float(lgb_pred)}
            ensemble_pred_step = float(ensemble_average(preds))

            # Build synthetic next-day row:
            last_row = df_sim.iloc[-1].to_dict()

            # Determine new date
            last_date = pd.to_datetime(last_row["Date"]).date()
            new_date = last_date + timedelta(days=1)

            # Exogenous values (brent_close, usd_inr, news_compound) kept constant at last observed
            synthetic = {}
            for col in df_sim.columns:
                synthetic[col] = last_row.get(col, np.nan)

            # Update Date and price columns using predicted ensemble price
            synthetic["Date"] = pd.Timestamp(new_date)
            # We'll set Close & Adj Close to ensemble_pred_step
            synthetic["Close"] = ensemble_pred_step
            synthetic["Adj Close"] = ensemble_pred_step
            # Volume: keep last known (could be modified later)
            # For any numeric exogenous field (brent_close, usd_inr, news_compound) we keep last value
            # Now append synthetic row to df_sim
            df_sim = pd.concat([df_sim, pd.DataFrame([synthetic])], ignore_index=True)

            # Recompute indicators for the whole df_sim (so newly appended row has correct indicators)
            df_sim = compute_technical_indicators(df_sim)

            # Keep only the columns we require (avoid proliferation)
            # continue until final step
            final_pred = ensemble_pred_step

            # Print progress occasionally
            if (step + 1) % 10 == 0 or step == days_ahead - 1:
                print(f"  Step {step+1}/{days_ahead} done — synthetic date {new_date} predicted: {ensemble_pred_step:.4f}")

        # After loop, final_pred is prediction for target_date
        preds = {"lstm": float(lstm_pred), "tcn": float(tcn_pred), "lightgbm": float(lgb_pred)}
        ensemble_pred = float(final_pred)

    # -------------------------
    # Save CSV & report (use the user-requested target_date in filename)
    # -------------------------
    os.makedirs("forecasts/results", exist_ok=True)
    csv_path = Path(f"forecasts/results/forecast_{tdate}.csv")
    out_df = pd.DataFrame([{
        "forecast_date": str(tdate),
        "model_lstm": preds["lstm"],
        "model_tcn": preds["tcn"],
        "model_lightgbm": preds["lightgbm"],
        "ensemble": ensemble_pred
    }])
    out_df.to_csv(csv_path, index=False)
    print(f"Saved forecast CSV: {csv_path}")

    os.makedirs("forecasts/reports", exist_ok=True)
    md = build_markdown_report(str(tdate), preds, ensemble_pred)
    md_path = Path(f"forecasts/reports/forecast_{tdate}.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved report: {md_path}")

    return {
        "forecast_date": str(tdate),
        "preds": preds,
        "ensemble": ensemble_pred,
        "csv": str(csv_path),
        "report": str(md_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="target date YYYY-MM-DD (we forecast Close on this date)")
    args = parser.parse_args()
    forecast_to_date(args.target)
