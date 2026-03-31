"""
FinSight AI — Stock Data Kafka Producer
Fetches OHLCV data for US + Indian NSE stocks and publishes to Kafka.
Indian NSE stocks use .NS suffix. Nifty 50 index = ^NSEI
NSE market hours: 9:15 AM - 3:30 PM IST (3:45 AM - 10:00 AM UTC)
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

KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC_OHLCV             = "stock.ohlcv.raw"
TOPIC_TICKER            = "stock.ticker.live"
POLL_INTERVAL_SEC       = 60
HISTORY_PERIOD          = "5d"

US_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "META", "NFLX", "SPY"
]

INDIA_WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "WIPRO.NS", "SBIN.NS",
    "BAJFINANCE.NS", "ADANIENT.NS", "^NSEI",
]

WATCHLIST = US_WATCHLIST + INDIA_WATCHLIST


def json_serializer(data: dict) -> bytes:
    return json.dumps(data, default=str).encode("utf-8")


def create_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all", retries=3, linger_ms=10, compression_type="gzip",
    )


def is_indian(symbol: str) -> bool:
    return ".NS" in symbol or symbol == "^NSEI"


def fetch_ohlcv(symbol: str, period: str = "1d", interval: str = "1m") -> List[dict]:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return []

        df.reset_index(inplace=True)
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol":      symbol,
                "time":        row[time_col].isoformat() if hasattr(row[time_col], "isoformat") else str(row[time_col]),
                "open":        round(float(row["Open"]),  4),
                "high":        round(float(row["High"]),  4),
                "low":         round(float(row["Low"]),   4),
                "close":       round(float(row["Close"]), 4),
                "volume":      int(row["Volume"]),
                "source":      "yfinance",
                "market":      "NSE" if is_indian(symbol) else "US",
                "currency":    "INR" if is_indian(symbol) else "USD",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })
        logger.info(f"✅ Fetched {len(records)} bars for {symbol}")
        return records
    except Exception as e:
        logger.error(f"❌ Error fetching {symbol}: {e}")
        return []


def fetch_live_quote(symbol: str) -> dict | None:
    try:
        info = yf.Ticker(symbol).fast_info
        return {
            "symbol":     symbol,
            "last_price": round(float(info.last_price), 4),
            "prev_close": round(float(info.previous_close), 4),
            "change_pct": round((info.last_price - info.previous_close) / info.previous_close * 100, 4),
            "volume":     int(info.three_month_average_volume or 0),
            "market":     "NSE" if is_indian(symbol) else "US",
            "currency":   "INR" if is_indian(symbol) else "USD",
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Quote error {symbol}: {e}")
        return None


def publish_ohlcv_batch(producer, symbols, period="1d"):
    total = 0
    for symbol in symbols:
        for record in fetch_ohlcv(symbol, period=period):
            producer.send(TOPIC_OHLCV, key=symbol, value=record)
            total += 1
        time.sleep(0.5)
    producer.flush()
    logger.info(f"📤 Published {total} OHLCV records")


def publish_live_quotes(producer, symbols):
    for symbol in symbols:
        q = fetch_live_quote(symbol)
        if q:
            producer.send(TOPIC_TICKER, key=symbol, value=q)
            cur = q.get("currency", "USD")
            logger.info(f"📡 {symbol}: {cur} {q['last_price']} ({q['change_pct']:+.2f}%)")
    producer.flush()


def run_producer():
    logger.info("🚀 Starting FinSight AI Producer (US + NSE India)")
    producer = create_producer()
    logger.info("📦 Initial backfill...")
    publish_ohlcv_batch(producer, WATCHLIST, period=HISTORY_PERIOD)

    while True:
        try:
            publish_ohlcv_batch(producer, WATCHLIST, period="1d")
            publish_live_quotes(producer, WATCHLIST)
            logger.info(f"💤 Sleeping {POLL_INTERVAL_SEC}s...")
            time.sleep(POLL_INTERVAL_SEC)
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("🛑 Stopped")
            break

    producer.close()


if __name__ == "__main__":
    run_producer()
