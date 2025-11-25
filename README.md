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

run command 
python -m scripts.generate_report --target 2026-01-01
change date above to whatever date you want price prediction for
, 




