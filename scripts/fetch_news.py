"""
Fetch many RSS feeds and store scored, deduplicated news.

Saves:
 - data/raw/news.json           (raw collected items)
 - data/processed/news_top100.json (top-100 by hybrid score)

Hybrid scoring = fuzzy(title+summary vs keywords) + keyword count + recency boost
"""
import json
from pathlib import Path
from datetime import datetime, timezone
import time
import feedparser
import requests
from rapidfuzz import fuzz, process

OUT_RAW = Path("data/raw")
OUT_PROCESSED = Path("data/processed")
OUT_RAW.mkdir(parents=True, exist_ok=True)
OUT_PROCESSED.mkdir(parents=True, exist_ok=True)

# A broad list of RSS sources covering news, finance, oil, geopolitics
RSS_FEEDS = [
    # global news
    "https://news.google.com/rss/search?q=oil+OR+brent+OR+crude+OR+opec&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=usd+OR+dxy+OR+usd+inr&hl=en-US&gl=US&ceid=US:en",
    "http://feeds.reuters.com/reuters/businessNews",
    "http://feeds.reuters.com/reuters/UKTopNews",
    "https://www.reuters.com/markets/commodities/feed",  # commodities
    "https://www.reuters.com/finance/markets/us", 
    # finance sites
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    # oil-focused
    "https://oilprice.com/feeds/main",
    "https://www.spglobal.com/marketintelligence/en/news-insights/latest-news.rss?q=oil",
    # market watchers
    "https://www.marketwatch.com/rss/topstories",
    "https://www.ft.com/?format=rss",
    # local / region / geopolitics
    "https://www.aljazeera.com/xml/rss/all.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    # specialized
    "https://www.argusmedia.com/rss",
    "https://www.offshore-technology.com/feed/",
    # Yahoo Finance (search)
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=CL=F&region=US&lang=en-US",
]

# relevance keywords (expanded)
KEYWORDS = [
    "brent", "crude", "oil", "opec", "opec+", "saudi", "russia", "iran", "venezuela",
    "pipeline", "refinery", "outage", "hurricane", "storm", "weather", "attack",
    "strike", "sanctions", "supply", "demand", "inventories", "api", "eia",
    "us crude", "wti", "dxy", "usd", "dollar", "usd/inr", "forex", "inflation",
    "fed", "interest rate", "central bank", "geopolitical", "war", "houthi",
    "shipping", "suez", "black sea", "gas", "lng"
]

def fetch_feed(url):
    try:
        parsed = feedparser.parse(url)
        items = []
        for e in parsed.entries:
            title = e.get("title", "")
            summary = e.get("summary", "") or e.get("description", "")
            link = e.get("link", "")
            # prefer published_parsed, then updated
            published = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                published = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
            elif hasattr(e, "updated_parsed") and e.updated_parsed:
                published = datetime.fromtimestamp(time.mktime(e.updated_parsed), tz=timezone.utc)
            else:
                published = datetime.now(timezone.utc)
            source = parsed.feed.get("title", url)
            items.append({
                "title": title,
                "summary": summary,
                "link": link,
                "published": published.isoformat(),
                "source": source,
                "fetched_from": url
            })
        return items
    except Exception as e:
        print("Feed error", url, e)
        return []

def score_item(item, keywords=KEYWORDS):
    # hybrid score: fuzzy title+summary vs keywords + keyword count + recency boost
    text = (item.get("title", "") + " " + item.get("summary", ""))[:1000]
    # fuzzy best match against keywords
    best = 0
    for kw in keywords:
        score = fuzz.token_set_ratio(text.lower(), kw.lower())
        if score > best:
            best = score
    # keyword count (simple)
    kcount = sum(text.lower().count(kw) for kw in keywords)
    # recency: hours since published
    try:
        pub = datetime.fromisoformat(item.get("published"))
        age_hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600.0
    except Exception:
        age_hours = 9999.0
    recency_score = max(0.0, 1.0 - min(age_hours / (24 * 14), 1.0))  # boost articles within 14 days
    # combine
    score = 0.6 * (best / 100.0) + 0.3 * (min(kcount, 5) / 5.0) + 0.1 * recency_score
    # normalize to 0-1
    return float(score)

def dedupe(items):
    seen = {}
    out = []
    for it in items:
        key = (it.get("title","").strip().lower(), it.get("link","").strip().lower())
        if key in seen:
            continue
        seen[key] = True
        out.append(it)
    return out

def main():
    print("Fetching RSS feeds...")
    all_items = []
    for url in RSS_FEEDS:
        items = fetch_feed(url)
        print(f"  {url} → {len(items)} items")
        all_items.extend(items)

    print(f"Total raw items: {len(all_items)}")
    all_items = dedupe(all_items)
    print(f"After dedupe: {len(all_items)}")

    # compute scores
    for it in all_items:
        it["score"] = score_item(it)

    # sort by score desc, then recency
    def sort_key(it):
        try:
            pub = datetime.fromisoformat(it.get("published"))
            age_hours = (datetime.now(timezone.utc) - pub).total_seconds() / 3600.0
        except Exception:
            age_hours = 99999.0
        return (it.get("score", 0.0), -min(age_hours, 99999.0))
    all_items_sorted = sorted(all_items, key=sort_key, reverse=True)

    # save raw & top-N
    raw_path = OUT_RAW / "news.json"
    processed_path = OUT_PROCESSED / "news_top100.json"
    with open(raw_path, "w", encoding="utf8") as f:
        json.dump(all_items_sorted, f, default=str, indent=2)
    with open(processed_path, "w", encoding="utf8") as f:
        json.dump(all_items_sorted[:200], f, default=str, indent=2)

    print(f"Saved raw news → {raw_path}")
    print(f"Saved top news → {processed_path} (top 200)")

if __name__ == "__main__":
    main()
