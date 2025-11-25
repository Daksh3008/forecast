python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt

data/raw/
    prices.csv
    sentiment.csv        (optional)
    macro.csv            (optional)


python scripts/prepare_data.py
output saved in :
data/processed/
models/*/scalers/



train all models
python scripts/train_all.py

forecast price for a target date
python scripts/forecast_date.py --target 2026-01-01



🤖 Models Included
1. LSTM + Attention (PyTorch)

Multi-head attention

Positional encoding

Bidirectional optional

Fully GPU-supported

2. TCN (PyTorch)

Dilated causal convolutions

Residual blocks

Global pooling

3. LightGBM (GPU)

Fast boosting

Great for tabular features

Optional GPU acceleration

4. Ensemble Model
ensemble = (lstm + tcn + lgbm) / 3

run commands

python -m scripts.prepare_data

this generates:
data/processed/X_seq.npy
data/processed/y_seq.npy
data/processed/X_tab.npy
data/processed/y_tab.npy
data/processed/feature_cols.npy


run optuna hyperparameter optimization
python -m scripts.optuna.optimize_lstm --trials 50
python -m scripts.optuna.optimize_tcn --trials 50
python -m scripts.optuna.optimize_lightgbm --trials 50

models/lstm_attention/best_optuna_params.json
models/tcn/best_optuna_params.json
models/lightgbm/best_optuna_params.json

train_all.py will automatically load these

python -m scripts.train_all

checkpoints saved to:
models/lstm_attention/checkpoints/best.pt
models/tcn/checkpoints/best.pt
models/lightgbm/checkpoints/best.pkl



forecast a future price for any date with report:
python -m scripts.generate_report --target 2026-01-01
change date above to whatever date you want price prediction for


output saved to forecasts/reports/report_2026-01-01.md





