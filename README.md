python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt

data/raw/
    prices.csv
    sentiment.csv        (optional)
    macro.csv            (optional)


python -m scripts.prepare_data
output saved in :
data/processed/
models/*/scalers/

Technical Term	Executive-Friendly Explanation
--LSTM + Attention	A model that “remembers” past trends and highlights the most important signals

--TCN	A model that spots repeating patterns in sequences

--LightGBM	A fast model that handles structured business data

--Optuna Optimization	Automated fine-tuning to maximize accuracy

--Ensemble	Combining multiple expert opinions for a stronger forecast

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



fetch multi-asset data (fetches data from 2007 till current date for brent crude, dxy, usdinr, vix, wti)
python -m scripts.fetch_data

fetch news 
python -m scripts.fetch_news


prepare feature matrix
python -m scripts.prepare_data
outputs in data/processed/

train all models
python -m scripts.train_all

forecast price for a target date
python -m scripts.forecast_date --target 2026-01-01


train all models (trains LightGBM, LSTM+Attention, TCN)
python -m scripts.train_all
checkpoints saved to models/

forecast brent crude price  
python -m scripts.forecast_date --target 2026-01-01
outputs: forecasts/reports/  and forecasts/results/


generate report 
python -m scripts.generate_report --target YYYY-MM-DD

sample report structure
================================================================================
🔮 Brent Crude Forecast (Ensemble)
================================================================================
 
**Generated:** 2025-11-26  
**Target Forecast Date:** 2026-01-01
 
---
 
🧾 Executive Market Overview
------------------------------------------------------------
Our ensemble model forecasts the price at **$62.42**.  
95% confidence range: **$62.14 — $62.69**
 
---
 
📊 Model Predictions
------------------------------------------------------------
| Model | Predicted Price |
|---|---:|
| lstm | $59.52 |
| tcn | $74.53 |
| lightgbm | $53.20 |
| **Ensemble** | **$62.42** |
 
 
---
 
🌍 Macro Context
------------------------------------------------------------
- **brent_close**: 62.4200
- **wti_close**: 57.9600
- **dxy_close**: 99.8530
- **usdinr_close**: 89.2320
- **vix_close**: 18.3500
 
 
---
 
📈 Technical / Micro Context
------------------------------------------------------------
- **brentclose_rsi_14**: 47.8195
- **brentclose_ma_10**: 63.2820
- **brentclose_ma_50**: 63.6828
- **close**: 62.4200
- **ind_ma10**: 63.2820
- **ind_ma50**: 63.6828
- **ind_ma200**: 66.8273
 
 
---
 
📰 News Summary (Top 10 most relevant)
------------------------------------------------------------
- OPEC+ Policies Reshape Global Oil Markets, Impacting Gas Prices and Consumer Budgets Across the United States - Energy Reporters
- 1.7M BPD by 2030: How Guyana Supply is Reshaping Global Oil Markets - Energy Capital & Power
- JPMorgan Warns Brent Crude Could Plunge To $30s By 2027 On Global Oversupply - Benzinga
- Crude Oil Prices Stabilize as Market Weighs Oversupply Outlook, Russia-Ukraine Peace Talks
- SPE Leaders to Deliver Keynotes at MSGBC Oil, Gas & Power 2025 - Energy Capital & Power
- Oil Holds Near Month-Low as Trump Hails Progress on Ukraine Deal
- Hungary says it will help Serbia after oil supplies to Russian-owned refinery stopped - Reuters
- Colombia's Petro Claims Trump's Venezuela Push Is Really About Oil - Crude Oil Prices Today | OilPrice.com
- Oil Stalls as Market Awaits Clarity on Russia-Ukraine Negotiations
- Oil Stalls as Market Awaits Clarity on Russia-Ukraine Negotiations - Barron's
 
 
---
 
### 🔎 Brent Oil Summary
---
 
• Geopolitical tensions are adding volatility to crude markets.
• RSI is neutral, indicating balanced market momentum.
 
 
---
 
🔎 Top correlated features with price
------------------------------------------------------------
- brentclose_last → correlation (abs) = 1.0000
- brentclose_ema_5 → correlation (abs) = 0.9987
- ema_5 → correlation (abs) = 0.9987
- brentclose_lag_1 → correlation (abs) = 0.9984
- lag_1 → correlation (abs) = 0.9984
- brentclose_ma_5 → correlation (abs) = 0.9981
- ma_5 → correlation (abs) = 0.9981
- brentclose_ema_10 → correlation (abs) = 0.9969
- ema_10 → correlation (abs) = 0.9969
- brentclose_lag_2 → correlation (abs) = 0.9968
- lag_2 → correlation (abs) = 0.9968
- brentclose_ma_10 → correlation (abs) = 0.9956
- ma_10 → correlation (abs) = 0.9956
- brentclose_lag_3 → correlation (abs) = 0.9953
- lag_3 → correlation (abs) = 0.9953
 
 
---
 
📝 Plain English Summary
------------------------------------------------------------
• The model projects Brent crude at around $62.42. This reflects the balance between global supply-demand conditions, macro risk sentiment, and USD-driven pricing pressure.
• Stable-to-lower inventory levels indicate tighter supply conditions.
• OPEC+ policy discussions in the news continue to shape supply expectations.
• Overall, the forecast reflects a combination of macro headwinds, supply-side adjustments, and mixed sentiment across global energy markets.
 
================================================================================
 

95% confidence interval:  forecast_+ 1.96 *sigma (sigma = RMSE of validation residuals)




Ignore from this line:
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





