"""
FinSight AI — Stock Data Kafka Consumer
Reads OHLCV records from Kafka and persists them to TimescaleDB.
"""

import json
import logging
from datetime import datetime

import psycopg2
import psycopg2.extras
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.consumer")


# ─── Config ──────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
KAFKA_GROUP_ID          = "finsight-timescale-writer"
TOPIC_OHLCV             = "stock.ohlcv.raw"
BATCH_SIZE              = 100       # insert rows in batches for efficiency

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsight",
    "user":     "finsight_user",
    "password": "finsight_pass",
}


# ─── DB Connection ────────────────────────────────────────────────────────────

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


# ─── Batch Insert ─────────────────────────────────────────────────────────────

INSERT_OHLCV = """
    INSERT INTO stock_prices (time, symbol, open, high, low, close, volume, source)
    VALUES %s
    ON CONFLICT DO NOTHING
"""

def flush_batch(conn, batch: list):
    """Batch insert OHLCV records into TimescaleDB."""
    if not batch:
        return

    rows = [
        (
            record["time"],
            record["symbol"],
            record["open"],
            record["high"],
            record["low"],
            record["close"],
            record["volume"],
            record.get("source", "yfinance"),
        )
        for record in batch
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_OHLCV, rows, page_size=BATCH_SIZE)

    conn.commit()
    logger.info(f"💾 Inserted batch of {len(rows)} records into TimescaleDB")


# ─── Consumer Loop ────────────────────────────────────────────────────────────

def run_consumer():
    logger.info("🚀 Starting FinSight AI Kafka Consumer")

    consumer = KafkaConsumer(
        TOPIC_OHLCV,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",           # replay from start if no offset
        enable_auto_commit=False,               # manual commit for reliability
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        max_poll_records=BATCH_SIZE,
        session_timeout_ms=30000,
    )

    conn = get_db_connection()
    batch = []

    logger.info(f"👂 Listening on topic '{TOPIC_OHLCV}'...")

    try:
        for message in consumer:
            record = message.value
            batch.append(record)

            logger.debug(
                f"📨 Received: {record['symbol']} @ {record['time']} "
                f"close=${record['close']}"
            )

            # Flush batch when it reaches threshold
            if len(batch) >= BATCH_SIZE:
                flush_batch(conn, batch)
                consumer.commit()
                batch.clear()

    except KafkaError as e:
        logger.error(f"Kafka error: {e}")

    except KeyboardInterrupt:
        logger.info("🛑 Consumer stopped by user")

    finally:
        # Flush remaining
        if batch:
            flush_batch(conn, batch)
            consumer.commit()

        conn.close()
        consumer.close()
        logger.info("✅ Consumer closed cleanly")


if __name__ == "__main__":
    run_consumer()
