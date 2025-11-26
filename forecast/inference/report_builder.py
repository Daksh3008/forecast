"""
Build a rich markdown + (optional) PDF report.

Enhancements added:
 - Only clean news headlines (no links)
 - Brent-aware macro + technical summary generator
 - Clean fuzzy-matched news formatting
 - Correlation insights
 - Plain-English explanation
"""

from datetime import datetime
from pathlib import Path
import math
import re

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except Exception:
    PDFKIT_AVAILABLE = False


OUT_DIR = Path("forecasts/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility formatting
# ============================================================

def _fmt_price(x, currency="$", decimals=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{currency}{x:,.{decimals}f}"


def clean_headline(text: str) -> str:
    """Remove URLs, brackets, extra spaces — keep only readable headline."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)     # remove URLs
    text = re.sub(r"\[.*?\]", "", text)              # remove bracketed junk
    text = " ".join(text.split())                    # normalize spacing
    return text.strip()


# ============================================================
# Brent-aware summary generator
# ============================================================

def generate_brent_summary(news_list, macro, tech):
    """
    Creates a domain-specific summary for Brent crude based on:
    - macro drivers (USD, inventories, OPEC)
    - news themes (geopolitics, supply/demand, recession)
    - technical indicators (RSI, MA structure)
    """

    summary = []

    # ---------------- MACRO ----------------
    usd = macro.get("usd_inr")
    usd_ma50 = macro.get("usd_inr_ma50")
    vix = macro.get("vix")
    oil = macro.get("brent_close")

    if usd and usd_ma50:
        if usd > usd_ma50:
            summary.append("A stronger USD is putting downward pressure on crude prices.")
        else:
            summary.append("A softer USD is helping support crude prices.")

    if macro.get("inventories_rising"):
        summary.append("Recent builds in US crude inventories indicate weakening demand expectations.")

    if macro.get("opec_cuts_expected"):
        summary.append("OPEC+ supply cuts continue to provide medium-term support for Brent prices.")

    # ---------------- NEWS THEMES ----------------
    joined = " ".join(news_list).lower()

    if any(w in joined for w in ["geopolitic", "conflict", "war", "attack"]):
        summary.append("Geopolitical tensions are adding volatility to crude markets.")

    if any(w in joined for w in ["recession", "slowdown", "weak demand"]):
        summary.append("Demand concerns due to global economic slowdown are mentioned repeatedly.")

    if any(w in joined for w in ["inventory", "stockpile", "build"]):
        summary.append("Multiple reports highlight inventory builds, implying supply overhang.")

    # ---------------- TECHNICAL ----------------
    rsi = tech.get("rsi", 50)
    ma10 = tech.get("ma10")
    ma50 = tech.get("ma50")

    if rsi > 70:
        summary.append("RSI suggests overbought conditions and potential price correction.")
    elif rsi < 30:
        summary.append("RSI indicates oversold conditions, increasing rebound probability.")
    else:
        summary.append("RSI is neutral, indicating balanced market momentum.")

    if ma10 and ma50:
        if ma10 > ma50:
            summary.append("Short-term trend remains bullish (MA10 > MA50).")
        else:
            summary.append("Short-term trend is weakening (MA10 < MA50).")

    return "### 🔎 Brent Oil Summary\n---\n" + "\n• " + "\n• ".join(summary) + "\n"


def generate_plain_english_summary(ensemble_pred, macro, tech, news_list):
    """
    Produces a narrative-style human-readable market summary for Brent crude
    derived from macro conditions, news topics, and technical indicators.
    """

    summary = []

    # ===========================
    # 1. Price Interpretation
    # ===========================
    summary.append(
        f"The model projects Brent crude at around {_fmt_price(ensemble_pred)}. "
        "This reflects the balance between global supply-demand conditions, macro risk sentiment, "
        "and USD-driven pricing pressure."
    )

    # ===========================
    # 2. Macro Drivers
    # ===========================
    usd = macro.get("usd_inr")
    usd_ma50 = macro.get("usd_inr_ma50")
    vix = macro.get("vix")

    if usd and usd_ma50:
        if usd > usd_ma50:
            summary.append(
                "A stronger U.S. dollar is putting downward pressure on oil prices, "
                "making commodities relatively expensive for international buyers."
            )
        else:
            summary.append(
                "The softer USD environment is mildly supportive for crude pricing."
            )

    if macro.get("inventories_rising"):
        summary.append(
            "Rising U.S. crude inventories suggest weaker short-term demand and may limit upside momentum."
        )
    else:
        summary.append(
            "Stable-to-lower inventory levels indicate tighter supply conditions."
        )

    if macro.get("opec_cuts_expected"):
        summary.append(
            "Expectations of OPEC+ production cuts continue to lend structural support."
        )

    if vix and vix > 18:
        summary.append(
            "Elevated market volatility is contributing to price fluctuations, driven by risk-off sentiment."
        )
    elif vix and vix < 14:
        summary.append(
            "Low volatility conditions indicate calmer market sentiment, reducing extreme price swings."
        )

    # ===========================
    # 3. News Themes
    # ===========================
    joined = " ".join(news_list).lower()

    if any(w in joined for w in ["attack", "strike", "geopolitic", "conflict"]):
        summary.append(
            "Geopolitical tensions mentioned in recent headlines are adding short-term upside risk."
        )

    if any(w in joined for w in ["recession", "slowdown", "weak demand"]):
        summary.append(
            "News flow points to concerns about global economic slowdown, weighing on crude demand expectations."
        )

    if any(w in joined for w in ["inventory", "stockpile", "build"]):
        summary.append(
            "Reports of inventory builds in key regions suggest oversupply risks."
        )

    if any(w in joined for w in ["opec", "production", "cut"]):
        summary.append(
            "OPEC+ policy discussions in the news continue to shape supply expectations."
        )

    # ===========================
    # 4. Technical Indicators
    # ===========================
    rsi = tech.get("rsi")
    ma10 = tech.get("ma10")
    ma50 = tech.get("ma50")

    if rsi:
        if rsi > 70:
            summary.append("RSI readings indicate overbought conditions, hinting at a potential pullback.")
        elif rsi < 30:
            summary.append("RSI signals oversold conditions, suggesting room for a rebound.")
        else:
            summary.append("RSI levels remain neutral, implying balanced momentum.")

    if ma10 and ma50:
        if ma10 > ma50:
            summary.append("Short-term momentum remains positive, with MA10 trending above MA50.")
        else:
            summary.append("Short-term trend is weakening as MA10 has slipped below MA50.")

    # ===========================
    # 5. Overall Narrative
    # ===========================
    summary.append(
        "Overall, the forecast reflects a combination of macro headwinds, supply-side adjustments, "
        "and mixed sentiment across global energy markets."
    )

    return "• " + "\n• ".join(summary)



# ============================================================
# Main report builder
# ============================================================

def build_markdown_report(target_date: str,
                          preds: dict,
                          ensemble_pred: float,
                          sigma: float | None = None,
                          macro_snapshot: dict | None = None,
                          tech_snapshot: dict | None = None,
                          correlations: list | None = None,
                          news_top: list | None = None,
                          additional_notes: str | None = None,
                          title: str | None = None) -> dict:

    now = datetime.utcnow().date().isoformat()
    title = title or "Brent Crude Forecast Report"

    # ---------------- Confidence interval ----------------
    if sigma is not None:
        lo = ensemble_pred - 1.96 * sigma
        hi = ensemble_pred + 1.96 * sigma
    else:
        lo = hi = ensemble_pred

    # ---------------- Model table ----------------
    models_md = "| Model | Predicted Price |\n|---|---:|\n"
    for m, v in preds.items():
        models_md += f"| {m} | {_fmt_price(v)} |\n"
    models_md += f"| **Ensemble** | **{_fmt_price(ensemble_pred)}** |\n"

    # ---------------- Macro snapshot ----------------
    macro_md = ""
    if macro_snapshot:
        for k, v in macro_snapshot.items():
            macro_md += f"- **{k}**: {v}\n"

    # ---------------- Technical snapshot ----------------
    tech_md = ""
    if tech_snapshot:
        for k, v in tech_snapshot.items():
            tech_md += f"- **{k}**: {v}\n"

    # ---------------- NEWS (cleaned, no links) ----------------
    news_md = ""
    clean_list = []
    if news_top:
        for item in news_top[:10]:
            title_raw = item.get("title", "")
            cleaned = clean_headline(title_raw)
            if cleaned:
                clean_list.append(cleaned)
                news_md += f"- {cleaned}\n"

    # ---------------- Correlations ----------------
    corr_md = ""
    if correlations:
        for feat, corr in correlations[:15]:
            corr_md += f"- {feat} → correlation (abs) = {corr:.4f}\n"

    # ---------------- Brent Summary ----------------
    brent_summary_text = ""
    if macro_snapshot and tech_snapshot and clean_list:
        brent_summary_text = generate_brent_summary(clean_list, macro_snapshot, tech_snapshot)

    # ---------------- Full Markdown ----------------
    md = f"""
================================================================================
🔮 {title}
================================================================================

**Generated:** {now}  
**Target Forecast Date:** {target_date}

---

🧾 Executive Market Overview
------------------------------------------------------------
Our ensemble model forecasts the price at **{_fmt_price(ensemble_pred)}**.  
95% confidence range: **{_fmt_price(lo)} — {_fmt_price(hi)}**

---

📊 Model Predictions
------------------------------------------------------------
{models_md}

---

🌍 Macro Context
------------------------------------------------------------
{macro_md or 'No macro snapshot available.'}

---

📈 Technical / Micro Context
------------------------------------------------------------
{tech_md or 'No technical snapshot available.'}

---

📰 News Summary (Top 10 most relevant)
------------------------------------------------------------
{news_md or 'No news available.'}

---

{brent_summary_text}

---

🔎 Top correlated features with price
------------------------------------------------------------
{corr_md or 'No correlations computed.'}

---

📝 Plain English Summary
------------------------------------------------------------
{generate_plain_english_summary(ensemble_pred, macro_snapshot or {}, tech_snapshot or {}, clean_list)}

================================================================================
"""

    # ---------------- Save Markdown ----------------
    fname = f"forecast_{target_date}"
    md_path = OUT_DIR / f"{fname}.md"
    md_path.write_text(md, encoding="utf8")

    # ---------------- Try generating PDF ----------------
    pdf_path = None
    try:
        import markdown
        html = markdown.markdown(md)
        html = f"<html><body>{html}</body></html>"

        if WEASYPRINT_AVAILABLE:
            pdf_path = OUT_DIR / f"{fname}.pdf"
            HTML(string=html).write_pdf(str(pdf_path))
        elif PDFKIT_AVAILABLE:
            pdf_path = OUT_DIR / f"{fname}.pdf"
            pdfkit.from_string(html, str(pdf_path))

    except Exception as e:
        print("PDF generation failed:", e)
        pdf_path = None

    print(f"Saved markdown report: {md_path}")
    if pdf_path:
        print(f"Saved PDF report: {pdf_path}")
    else:
        print("PDF report not created (weasyprint/pdfkit not available).")

    return {"md": str(md_path), "pdf": str(pdf_path) if pdf_path else None}
