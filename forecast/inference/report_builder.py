"""
Generate Markdown report for forecast results.
"""

import datetime


def build_markdown_report(target_date: str, preds: dict, ensemble_pred: float) -> str:
    today = datetime.date.today().isoformat()

    md = f"""
# Forecast Report
**Generated:** {today}  
**Target Forecast Date:** {target_date}

---

## Model Predictions
| Model      | Predicted Price |
|------------|-----------------|
| LSTM+Attention | {preds.get("lstm"):.4f} |
| TCN           | {preds.get("tcn"):.4f} |
| LightGBM      | {preds.get("lightgbm"):.4f} |

---

## Ensemble Forecast
**Final Ensemble Price:** **{ensemble_pred:.4f}**

---
"""
    return md
