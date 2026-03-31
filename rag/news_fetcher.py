"""
FinSight AI — News Fetcher (US + Indian stocks)
"""

import logging
import os
import time
import psycopg2
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("finsight.news_fetcher")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "finsight", "user": "finsight_user", "password": "finsight_pass",
}

# US + Indian stock queries
SYMBOL_QUERIES = {
    # US Stocks
    "AAPL":  "Apple stock AAPL",
    "MSFT":  "Microsoft stock MSFT",
    "GOOGL": "Google Alphabet stock",
    "AMZN":  "Amazon stock AMZN",
    "NVDA":  "NVIDIA stock GPU AI",
    "TSLA":  "Tesla stock TSLA",
    "JPM":   "JPMorgan stock banking",
    "META":  "Meta Facebook stock",
    "NFLX":  "Netflix stock NFLX",
    "SPY":   "S&P 500 market index",
    # Indian NSE Stocks
    "RELIANCE.NS":   "Reliance Industries NSE India",
    "TCS.NS":        "TCS Tata Consultancy Services NSE",
    "INFY.NS":       "Infosys NSE India IT",
    "HDFCBANK.NS":   "HDFC Bank NSE India",
    "ICICIBANK.NS":  "ICICI Bank NSE India",
    "WIPRO.NS":      "Wipro NSE India IT",
    "SBIN.NS":       "State Bank of India SBI NSE",
    "BAJFINANCE.NS": "Bajaj Finance NSE India",
    "ADANIENT.NS":   "Adani Enterprises NSE India",
    "^NSEI":         "Nifty 50 NSE India market",
}


def fetch_news_for_symbol(symbol: str, days_back: int = 7) -> list:
    if not NEWS_API_KEY:
        logger.warning("⚠️ NEWS_API_KEY not set")
        return []

    query     = SYMBOL_QUERIES.get(symbol, symbol)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    try:
        r = requests.get(NEWS_API_URL, params={
            "q": query, "from": from_date, "sortBy": "publishedAt",
            "language": "en", "pageSize": 20, "apiKey": NEWS_API_KEY,
        }, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        logger.info(f"📰 Fetched {len(articles)} articles for {symbol}")
        return articles
    except Exception as e:
        logger.error(f"❌ News error for {symbol}: {e}")
        return []


def store_articles(conn, symbol: str, articles: list) -> int:
    stored = 0
    for article in articles:
        title   = article.get("title", "")
        url     = article.get("url", "")
        source  = article.get("source", {}).get("name", "")
        content = article.get("description") or article.get("content") or title
        pub_at  = article.get("publishedAt", datetime.now(timezone.utc).isoformat())

        if not title or title == "[Removed]":
            continue

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO news_articles (published_at, symbol, title, url, source, summary)
                    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING id
                """, (pub_at, symbol, title, url, source, content))
                if cur.fetchone():
                    stored += 1
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Store error: {e}")
            conn.rollback()

    logger.info(f"💾 Stored {stored} articles for {symbol}")
    return stored


def run_news_fetcher():
    logger.info("📰 Starting News Fetcher (US + Indian stocks)")
    conn  = psycopg2.connect(**DB_CONFIG)
    total = 0

    for symbol in SYMBOL_QUERIES.keys():
        articles = fetch_news_for_symbol(symbol)
        total   += store_articles(conn, symbol, articles)
        time.sleep(1)

    conn.close()
    logger.info(f"✅ News fetch complete — {total} new articles stored")
    return total


if __name__ == "__main__":
    run_news_fetcher()
