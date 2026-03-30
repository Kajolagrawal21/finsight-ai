"""
FinSight AI — News Fetcher
Fetches financial news from NewsAPI for watchlist symbols.
Stores raw articles in TimescaleDB and queues them for embedding.
"""

import logging
import os
import time
import psycopg2
import psycopg2.extras
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.news_fetcher")

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsight",
    "user":     "finsight_user",
    "password": "finsight_pass",
}

# Map symbol to company name for better search results
SYMBOL_QUERIES = {
    "AAPL":  "Apple stock",
    "MSFT":  "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN":  "Amazon stock",
    "NVDA":  "NVIDIA stock",
    "TSLA":  "Tesla stock",
    "JPM":   "JPMorgan stock",
    "META":  "Meta Facebook stock",
    "NFLX":  "Netflix stock",
    "SPY":   "S&P 500 market",
}


def fetch_news_for_symbol(symbol: str, days_back: int = 7) -> list:
    """Fetch news articles for a symbol from NewsAPI."""
    if not NEWS_API_KEY:
        logger.warning("⚠️ NEWS_API_KEY not set — skipping news fetch")
        return []

    query = SYMBOL_QUERIES.get(symbol, symbol)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 20,
        "apiKey":   NEWS_API_KEY,
    }

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = data.get("articles", [])
        logger.info(f"📰 Fetched {len(articles)} articles for {symbol}")
        return articles

    except Exception as e:
        logger.error(f"❌ Error fetching news for {symbol}: {e}")
        return []


def store_articles(conn, symbol: str, articles: list) -> int:
    """Store news articles in TimescaleDB."""
    if not articles:
        return 0

    stored = 0
    for article in articles:
        try:
            title   = article.get("title", "")
            url     = article.get("url", "")
            source  = article.get("source", {}).get("name", "")
            content = article.get("description") or article.get("content") or title
            pub_at  = article.get("publishedAt", datetime.now(timezone.utc).isoformat())

            if not title or title == "[Removed]":
                continue

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO news_articles
                        (published_at, symbol, title, url, source, summary)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """, (pub_at, symbol, title, url, source, content))

                result = cur.fetchone()
                if result:
                    stored += 1

            conn.commit()

        except Exception as e:
            logger.error(f"❌ Error storing article: {e}")
            conn.rollback()

    logger.info(f"💾 Stored {stored} new articles for {symbol}")
    return stored


def run_news_fetcher():
    """Fetch and store news for all watchlist symbols."""
    logger.info("📰 Starting News Fetcher")
    conn = psycopg2.connect(**DB_CONFIG)

    total = 0
    for symbol in SYMBOL_QUERIES.keys():
        articles = fetch_news_for_symbol(symbol)
        stored   = store_articles(conn, symbol, articles)
        total   += stored
        time.sleep(1)  # polite rate limiting

    conn.close()
    logger.info(f"✅ News fetch complete — {total} new articles stored")
    return total


if __name__ == "__main__":
    run_news_fetcher()
