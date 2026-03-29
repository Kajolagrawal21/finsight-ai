"""
FinSight AI — Stock Data Kafka Producer
Fetches OHLCV data from yfinance and publishes to Kafka topic.
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import List

import yfinance as yf
from kafka import KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.producer")


# ─── Config ──────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC_OHLCV             = "stock.ohlcv.raw"
TOPIC_TICKER            = "stock.ticker.live"
POLL_INTERVAL_SEC       = 60       # fetch every 60s during market hours
HISTORY_PERIOD          = "5d"     # initial backfill period
HISTORY_INTERVAL        = "1m"     # 1-minute granularity

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "META", "NFLX", "SPY"
]


# ─── Serializer ──────────────────────────────────────────────────────────────

def json_serializer(data: dict) -> bytes:
    return json.dumps(data, default=str).encode("utf-8")


# ─── Producer Factory ─────────────────────────────────────────────────────────

def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",                    # wait for all replicas
        retries=3,
        linger_ms=10,                  # small batch delay for throughput
        compression_type="gzip",
    )


# ─── Fetch & Publish OHLCV ───────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, period: str = "1d", interval: str = "1m") -> List[dict]:
    """Fetch OHLCV bars from yfinance and return as list of dicts."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data for {symbol}")
            return []

        df.reset_index(inplace=True)
        records = []

        for _, row in df.iterrows():
            records.append({
                "symbol":    symbol,
                "time":      row["Datetime"].isoformat() if hasattr(row["Datetime"], "isoformat") else str(row["Datetime"]),
                "open":      round(float(row["Open"]),   4),
                "high":      round(float(row["High"]),   4),
                "low":       round(float(row["Low"]),    4),
                "close":     round(float(row["Close"]),  4),
                "volume":    int(row["Volume"]),
                "source":    "yfinance",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })

        logger.info(f"✅ Fetched {len(records)} bars for {symbol}")
        return records

    except Exception as e:
        logger.error(f"❌ Error fetching {symbol}: {e}")
        return []


def fetch_live_quote(symbol: str) -> dict | None:
    """Fetch latest quote (price, change, volume) for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.fast_info

        return {
            "symbol":        symbol,
            "last_price":    round(float(info.last_price), 4),
            "prev_close":    round(float(info.previous_close), 4),
            "change_pct":    round((info.last_price - info.previous_close) / info.previous_close * 100, 4),
            "volume":        int(info.three_month_average_volume or 0),
            "market_cap":    getattr(info, "market_cap", None),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error fetching quote for {symbol}: {e}")
        return None


# ─── Publish Helpers ─────────────────────────────────────────────────────────

def publish_ohlcv_batch(producer: KafkaProducer, symbols: List[str], period: str = "1d"):
    """Fetch and publish OHLCV data for all symbols."""
    total = 0
    for symbol in symbols:
        records = fetch_ohlcv(symbol, period=period)
        for record in records:
            future = producer.send(
                topic=TOPIC_OHLCV,
                key=symbol,
                value=record
            )
            future.add_errback(lambda e: logger.error(f"Kafka send error: {e}"))
            total += 1

        time.sleep(0.5)   # polite rate-limiting for yfinance

    producer.flush()
    logger.info(f"📤 Published {total} OHLCV records to Kafka topic '{TOPIC_OHLCV}'")


def publish_live_quotes(producer: KafkaProducer, symbols: List[str]):
    """Publish live quotes for all symbols."""
    for symbol in symbols:
        quote = fetch_live_quote(symbol)
        if quote:
            producer.send(
                topic=TOPIC_TICKER,
                key=symbol,
                value=quote
            )
            logger.info(f"📡 Live quote {symbol}: ${quote['last_price']} ({quote['change_pct']:+.2f}%)")

    producer.flush()


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run_producer():
    logger.info("🚀 Starting FinSight AI Kafka Producer")
    producer = create_producer()

    # Initial backfill — publish last 5 days of 1-min bars
    logger.info("📦 Running initial historical backfill...")
    publish_ohlcv_batch(producer, WATCHLIST, period=HISTORY_PERIOD)

    # Continuous live loop
    logger.info(f"🔄 Starting live loop — polling every {POLL_INTERVAL_SEC}s")
    while True:
        try:
            publish_ohlcv_batch(producer, WATCHLIST, period="1d")
            publish_live_quotes(producer, WATCHLIST)
            logger.info(f"💤 Sleeping {POLL_INTERVAL_SEC}s...")
            time.sleep(POLL_INTERVAL_SEC)

        except KafkaError as e:
            logger.error(f"Kafka error: {e} — retrying in 10s")
            time.sleep(10)

        except KeyboardInterrupt:
            logger.info("🛑 Producer stopped by user")
            break

    producer.close()
    logger.info("✅ Producer closed cleanly")


if __name__ == "__main__":
    run_producer()
