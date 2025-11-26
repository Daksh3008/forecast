"""
Forecast to a target date using models that predict log-price (log(close_t+1)).

Behavior:
 - Loads raw inputs (data/raw/brent.csv and data/raw/merged_macro.csv), merges them,
   and runs the canonical build_features() to construct the exact same feature matrix
   used in training.
 - Loads feature_cols.npy to obtain the exact column ordering used in training.
 - Uses saved FeatureScaler objects to transform sequences/tabular rows.
 - Models predict log-price; converts predictions with exp() to price.
 - For recursive multi-step forecasting, appends synthetic *raw* rows (brent_close set
   to predicted price, macro columns frozen), recomputes features, and continues.
"""

import argparse
import os
from pathlib import Path
from datetime import timedelta
import yaml
import numpy as np
import pandas as pd
import torch

from forecast.data.load_raw import load_raw_csv
from forecast.data.preprocess import preprocess
from forecast.data.features import build_features
from forecast.data.scalers import FeatureScaler
from forecast.models.lstm_attention import LSTMAttentionModel
from forecast.models.tcn import TCNModel
from forecast.models.lightgbm_model import LightGBMWrapper
from forecast.inference.ensemble import ensemble_average
from forecast.inference.report_builder import build_markdown_report

BASE_CFG_PATH = Path("config/base.yaml")
FORECAST_CFG_PATH = Path("config/forecast.yaml")

try:
    base_cfg = yaml.safe_load(BASE_CFG_PATH.read_text())
except Exception:
    base_cfg = {}
SEQ_LEN = int(base_cfg.get("seq_len", 60))

try:
    fcfg = yaml.safe_load(FORECAST_CFG_PATH.read_text())
except Exception:
    fcfg = {}

# Raw inputs (we prefer raw + merged macro as source of truth)
RAW_BRENT = Path("data/raw/brent.csv")
RAW_MERGED = Path("data/raw/merged_macro.csv")
FEATURE_COLS_PATH = Path("data/processed/feature_cols.npy")

SCALER_LSTM = Path(fcfg.get("paths", {}).get("scaler_lstm", "models/lstm_attention/scalers/feature_scaler.joblib"))
SCALER_TCN = Path(fcfg.get("paths", {}).get("scaler_tcn", "models/tcn/scalers/feature_scaler.joblib"))
SCALER_LGBM = Path(fcfg.get("paths", {}).get("scaler_lgbm", "models/lightgbm/scalers/feature_scaler.joblib"))

LSTM_CKPT = Path(fcfg.get("paths", {}).get("lstm_checkpoint", "models/lstm_attention/checkpoints/best.pt"))
TCN_CKPT = Path(fcfg.get("paths", {}).get("tcn_checkpoint", "models/tcn/checkpoints/best.pt"))
LGBM_CKPT = Path(fcfg.get("paths", {}).get("lightgbm_checkpoint", "models/lightgbm/checkpoints/best.pkl"))


def load_feature_cols():
    if FEATURE_COLS_PATH.exists():
        return list(np.load(FEATURE_COLS_PATH, allow_pickle=True))
    return None


def make_sequence_from_df(df_proc: pd.DataFrame, seq_end_idx: int, feature_cols, seq_len=SEQ_LEN):
    start_idx = seq_end_idx - (seq_len - 1)
    if start_idx < 0:
        raise ValueError(f"Not enough history to build a sequence of length {seq_len}. Need at least {seq_len} rows.")
    X_rows = df_proc.iloc[start_idx: seq_end_idx + 1][feature_cols].values.astype(np.float32)
    return X_rows.reshape(1, seq_len, -1)


def sanitize_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infs, fill, clip numeric columns — same logic as prepare_data.py"""
    df = df.replace([np.inf, -np.inf], np.nan)
    # prefer explicit ffills per pandas future warning
    df = df.ffill().bfill()
    df = df.fillna(0)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].clip(lower=-1e6, upper=1e6)
    return df


def load_and_build_features():
    """Load raw brent + merged macro, merge, preprocess, and build features."""
    if not RAW_BRENT.exists():
        raise FileNotFoundError(f"Missing raw brent data: {RAW_BRENT}")
    if not RAW_MERGED.exists():
        raise FileNotFoundError(f"Missing merged macro data: {RAW_MERGED}")

    price = load_raw_csv(str(RAW_BRENT))
    macro = load_raw_csv(str(RAW_MERGED))

    # normalize column names
    price.columns = [c.lower().strip().replace(" ", "_") for c in price.columns]
    macro.columns = [c.lower().strip().replace(" ", "_") for c in macro.columns]

    price = preprocess(price)
    macro = preprocess(macro)

    df = price.merge(macro, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)

    # ensure raw date and raw price column naming matches build_features expectations
    # build_features expects 'brent_close' ideally; if price file has 'close' rename
    if "close" in df.columns and "brent_close" not in df.columns:
        df = df.rename(columns={"close": "brent_close"})

    # build engineered features
    df_feat = build_features(df, price_col="brent_close" if "brent_close" in df.columns else "close")

    # sanitize numeric columns (same as during prepare_data)
    df_feat = sanitize_features_df(df_feat)

    return df, df_feat


def forecast_to_date(target_date: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load feature columns (MUST exist)
    feature_cols = load_feature_cols()
    if feature_cols is None:
        raise FileNotFoundError("feature_cols.npy not found. Run `python -m scripts.prepare_data` first.")

    # Build fresh features from raw files (keeps exact pipeline parity with training)
    df_raw, df_proc = load_and_build_features()

    # Validate feature columns exist in the built feature DF
    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        raise KeyError(f"Missing feature columns in processed features: {missing}")

    # Determine last available date
    last_available_date = pd.to_datetime(df_proc["date"]).dt.date.iloc[-1]
    tdate = pd.to_datetime(target_date).date()

    # Load scalers and models
    lstm_scaler = FeatureScaler().load(str(SCALER_LSTM))
    tcn_scaler = FeatureScaler().load(str(SCALER_TCN))
    lgbm_scaler = FeatureScaler().load(str(SCALER_LGBM))

    if not (LSTM_CKPT.exists() and TCN_CKPT.exists() and LGBM_CKPT.exists()):
        raise FileNotFoundError("One or more model checkpoints are missing. Train models first.")

    try:
        train_lstm_cfg = yaml.safe_load(Path("config/train_lstm.yaml").read_text())
    except Exception:
        train_lstm_cfg = {}
    try:
        train_tcn_cfg = yaml.safe_load(Path("config/train_tcn.yaml").read_text())
    except Exception:
        train_tcn_cfg = {}

    lstm_model = LSTMAttentionModel.load(str(LSTM_CKPT), map_location=device,
                                        input_size=len(feature_cols),
                                        hidden_size=int(train_lstm_cfg.get("hidden_size", 256)),
                                        n_layers=int(train_lstm_cfg.get("n_layers", 2)),
                                        n_heads=int(train_lstm_cfg.get("n_heads", 4)),
                                        dropout=float(train_lstm_cfg.get("dropout", 0.1)),
                                        bidirectional=bool(train_lstm_cfg.get("bidirectional", False)),
                                        fc_hidden=int(train_lstm_cfg.get("fc_hidden", 128)),
                                        seq_len=int(train_lstm_cfg.get("seq_len", SEQ_LEN))
                                        ).to(device)

    tcn_model = TCNModel.load(str(TCN_CKPT), map_location=device,
                              input_size=len(feature_cols),
                              num_channels=train_tcn_cfg.get("num_channels", [32, 64, 128]),
                              kernel_size=int(train_tcn_cfg.get("kernel_size", 3)),
                              dropout=float(train_tcn_cfg.get("dropout", 0.1))
                              ).to(device)

    lgbm_model = LightGBMWrapper.load(str(LGBM_CKPT))

    # ---------- forecasting logic ----------
    if tdate <= last_available_date:
        # in-sample / backtest-like forecast for an existing date
        seq_end_date = tdate - timedelta(days=1)
        mask = pd.to_datetime(df_proc["date"]).dt.date <= seq_end_date
        if mask.sum() < SEQ_LEN:
            raise ValueError(f"Not enough history before {tdate} to build a sequence of length {SEQ_LEN}.")
        seq_end_idx = mask[mask].index[-1]

        X_seq = make_sequence_from_df(df_proc, seq_end_idx, feature_cols, seq_len=SEQ_LEN)
        X_lstm = lstm_scaler.transform(X_seq.copy())
        X_tcn = tcn_scaler.transform(X_seq.copy())
        X_lgbm_row = df_proc.iloc[seq_end_idx][feature_cols].values.reshape(1, -1)
        X_lgbm = lgbm_scaler.transform(X_lgbm_row.copy())

        lstm_model.eval(); tcn_model.eval()
        with torch.no_grad():
            pred_log_lstm = lstm_model(torch.tensor(X_lstm, dtype=torch.float32).to(device)).cpu().numpy()[0]
            pred_log_tcn = tcn_model(torch.tensor(X_tcn, dtype=torch.float32).to(device)).cpu().numpy()[0]
        pred_log_lgb = lgbm_model.predict(X_lgbm)[0]

        # convert log-price -> price
        lstm_price = float(np.exp(pred_log_lstm))
        tcn_price = float(np.exp(pred_log_tcn))
        lgb_price = float(np.exp(pred_log_lgb))

        preds = {"lstm": lstm_price, "tcn": tcn_price, "lightgbm": lgb_price}
        ensemble_pred = float(ensemble_average(preds))
        forecast_for = tdate

    else:
        # recursive forecasting: append synthetic *raw* rows and recompute features each step
        days_ahead = (tdate - last_available_date).days
        print(f"Target date {tdate} is {days_ahead} days ahead of last available date {last_available_date}.")

        # df_raw is the raw merged (containing date, brent_close, wti_close, dxy_close, etc.)
        df_sim_raw = df_raw.copy().reset_index(drop=True)
        final_pred = None

        # identify macro/exogenous columns to freeze (all numeric raw columns except brent_close & date)
        raw_price_col = "brent_close" if "brent_close" in df_sim_raw.columns else ("close" if "close" in df_sim_raw.columns else None)
        macro_cols = [c for c in df_sim_raw.columns if c not in ("date", raw_price_col)]

        for step in range(days_ahead):
            # Build engineered features from the current raw sim
            df_sim_feat = build_features(df_sim_raw.copy(), price_col=raw_price_col)
            df_sim_feat = sanitize_features_df(df_sim_feat)

            seq_end_idx = len(df_sim_feat) - 1
            if seq_end_idx - (SEQ_LEN - 1) < 0:
                raise ValueError(f"Not enough history to build a sequence of length {SEQ_LEN} for recursive forecasting.")

            # Prepare model inputs
            X_seq = make_sequence_from_df(df_sim_feat, seq_end_idx, feature_cols, seq_len=SEQ_LEN)
            X_lstm = lstm_scaler.transform(X_seq.copy())
            X_tcn = tcn_scaler.transform(X_seq.copy())

            X_lgbm_row = df_sim_feat.iloc[seq_end_idx][feature_cols].values.reshape(1, -1)
            X_lgbm = lgbm_scaler.transform(X_lgbm_row.copy())

            lstm_model.eval(); tcn_model.eval()
            with torch.no_grad():
                pred_log_lstm = lstm_model(torch.tensor(X_lstm, dtype=torch.float32).to(device)).cpu().numpy()[0]
                pred_log_tcn = tcn_model(torch.tensor(X_tcn, dtype=torch.float32).to(device)).cpu().numpy()[0]
            pred_log_lgb = lgbm_model.predict(X_lgbm)[0]

            # convert to price
            lstm_price = float(np.exp(pred_log_lstm))
            tcn_price = float(np.exp(pred_log_tcn))
            lgb_price = float(np.exp(pred_log_lgb))

            preds_step = {"lstm": lstm_price, "tcn": tcn_price, "lightgbm": lgb_price}
            ensemble_price_step = float(ensemble_average(preds_step))
            ensemble_log_step = float(np.log(ensemble_price_step))

            # Append synthetic RAW row: base on last raw row, override date & brent_close
            last_raw = df_sim_raw.iloc[-1].to_dict()
            last_date = pd.to_datetime(last_raw["date"]).date()
            new_date = last_date + timedelta(days=1)

            synthetic_raw = {col: last_raw.get(col, np.nan) for col in df_sim_raw.columns}
            synthetic_raw["date"] = pd.Timestamp(new_date)
            # set brent raw price to ensemble price
            synthetic_raw[raw_price_col] = ensemble_price_step

            # freeze macro columns to last observed values
            for mc in macro_cols:
                synthetic_raw[mc] = last_raw.get(mc, synthetic_raw.get(mc, np.nan))

            df_sim_raw = pd.concat([df_sim_raw, pd.DataFrame([synthetic_raw])], ignore_index=True)

            final_pred = ensemble_price_step
            if (step + 1) % 10 == 0 or step == days_ahead - 1:
                print(f"  Step {step+1}/{days_ahead} done — synthetic date {new_date} predicted: {ensemble_price_step:.4f}")

        # after loop, last prediction is final_pred; use last model outputs for reporting
        preds = {"lstm": float(lstm_price), "tcn": float(tcn_price), "lightgbm": float(lgb_price)}
        ensemble_pred = float(final_pred)
        forecast_for = tdate


    # -------------------------
    # Prepare additional report items: macro snapshot, tech snapshot, correlations, news
    # -------------------------
    # Macro snapshot: recent raw numeric columns (try to pick common names)
    macro_snapshot = {}
    # look for common macro columns in df_proc or df_raw
    for k in ["brent_close", "wti_close", "dxy_close", "usd_inr", "usdinr_close", "vix_close"]:
        if k in df_proc.columns:
            val = df_proc[k].iloc[-1]
            macro_snapshot[k] = f"{val:.4f}" if pd.notna(val) else "N/A"
    # fallback: if raw df (df_raw) exists, prefer that
    try:
        if "df_raw" in locals():
            for k in ["brent_close", "wti_close", "dxy_close", "usd_inr", "usdinr_close", "vix_close"]:
                if k in df_raw.columns:
                    macro_snapshot[k] = f"{df_raw[k].iloc[-1]:.4f}"
    except Exception:
        pass

    # Technical snapshot: extract some canonical indicators (if present)
    tech_snapshot = {}
    # use feature names commonly present from build_features
    for tk in ["brentclose_rsi_14", "brentclose_ma_10", "brentclose_ma_50", "brentclose_ma_200",
               "close", "ind_ma10", "ind_ma50", "ind_ma200"]:
        if tk in df_proc.columns:
            tech_snapshot[tk] = f"{df_proc[tk].iloc[-1]:.4f}"

    # If 'close' exists, compute RSI/MA using last few rows if the canonical names don't exist
    if "close" in df_proc.columns and not any(k in tech_snapshot for k in ["ind_ma10", "ind_ma50", "ind_ma200"]):
        try:
            close_ser = df_proc["close"].astype(float)
            tech_snapshot["ind_ma10"] = f"{close_ser.rolling(10, min_periods=1).mean().iloc[-1]:.4f}"
            tech_snapshot["ind_ma50"] = f"{close_ser.rolling(50, min_periods=1).mean().iloc[-1]:.4f}"
            tech_snapshot["ind_ma200"] = f"{close_ser.rolling(200, min_periods=1).mean().iloc[-1]:.4f}"
        except Exception:
            pass

    # Correlations: compute pearson between each numeric feature and close (or log_close)
    correlations = []
    target_col_for_corr = "log_close" if "log_close" in df_proc.columns else ("close" if "close" in df_proc.columns else None)
    if target_col_for_corr:
        numeric_cols = df_proc.select_dtypes(include=["number"]).columns.tolist()
        # exclude the target itself
        if target_col_for_corr in numeric_cols:
            numeric_cols.remove(target_col_for_corr)
        # compute correlations
        for col in numeric_cols:
            try:
                corr = df_proc[col].corr(df_proc[target_col_for_corr])
                correlations.append((col, abs(float(corr) if not pd.isna(corr) else 0.0)))
            except Exception:
                continue
        correlations = sorted(correlations, key=lambda x: x[1], reverse=True)

    # News: load pre-fetched news file if available
    news_file = Path("data/processed/news_top100.json")
    news_top = []
    if news_file.exists():
        try:
            import json
            with open(news_file, "r", encoding="utf8") as f:
                news_top = json.load(f)
        except Exception:
            news_top = []

    # Compute ensemble sigma (approx): use RMSE across model preds as a proxy
    # We can use the sample std of model predictions (in log-price space if available)
    try:
        # preds dict might be prices — convert back to log-space for sigma estimate
        model_logs = []
        for v in preds.values():
            if v is None:
                continue
            try:
                model_logs.append(np.log(float(v)))
            except Exception:
                pass
        sigma = float(np.std(model_logs)) if len(model_logs) > 1 else None
    except Exception:
        sigma = None

    # Build final report using report_builder
    from forecast.inference.report_builder import build_markdown_report as build_report
    additional_notes = [
        "• Models trained on log-price targets; predictions converted to price via exp().",
        "• News items were scored by fuzzy matching + recency; top items included below.",
        "• Correlations are absolute Pearson correlations computed on the processed feature matrix."
    ]
    additional_notes = "\n".join(additional_notes)

    report_out = build_report(
        target_date=str(forecast_for),
        preds=preds,
        ensemble_pred=ensemble_pred,
        sigma=sigma,
        macro_snapshot=macro_snapshot,
        tech_snapshot=tech_snapshot,
        correlations=correlations,
        news_top=news_top,
        additional_notes=additional_notes,
        title=f"Brent Crude Forecast (Ensemble)"
    )

    # report_out contains {'md': path, 'pdf': path or None}
    md_path = report_out.get("md")
    pdf_path = report_out.get("pdf")

    print(f"Report saved: {md_path}  (pdf: {pdf_path})")



    # -------------------------
    # Save CSV & report (use config paths)
    # -------------------------
    csv_dir = Path(fcfg.get("forecast_output", {}).get("csv_dir", "forecasts/results"))
    report_dir = Path(fcfg.get("forecast_output", {}).get("report_dir", "forecasts/reports"))
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    csv_path = csv_dir / f"forecast_{forecast_for}.csv"
    out_df = pd.DataFrame([{
        "forecast_date": str(forecast_for),
        "model_lstm": preds["lstm"],
        "model_tcn": preds["tcn"],
        "model_lightgbm": preds["lightgbm"],
        "ensemble": ensemble_pred
    }])
    out_df.to_csv(csv_path, index=False)
    print(f"Saved forecast CSV: {csv_path}")

    # build_markdown_report now returns {'md': '<path>', 'pdf': '<path or None>'}
    report_out = build_markdown_report(
        target_date=str(forecast_for),
        preds=preds,
        ensemble_pred=ensemble_pred,
        sigma=sigma if 'sigma' in locals() else None,
        macro_snapshot=macro_snapshot if 'macro_snapshot' in locals() else None,
        tech_snapshot=tech_snapshot if 'tech_snapshot' in locals() else None,
        correlations=correlations if 'correlations' in locals() else None,
        news_top=news_top if 'news_top' in locals() else None,
        additional_notes=additional_notes if 'additional_notes' in locals() else None,
        title=f"Brent Crude Forecast (Ensemble)"
    )

    md_path = report_out.get("md")
    pdf_path = report_out.get("pdf")

    print(f"Saved markdown report: {md_path}")
    if pdf_path:
        print(f"Saved PDF report: {pdf_path}")
    else:
        print("PDF report not created (weasyprint/pdfkit not available).")


    return {
        "forecast_date": str(forecast_for),
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
