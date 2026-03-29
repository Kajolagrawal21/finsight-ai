"""
FinSight AI — Feature Engineering Pipeline
Computes technical indicators from raw OHLCV data and stores in TimescaleDB.
"""

import logging
import pandas as pd
import psycopg2
import psycopg2.extras
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.features")

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsight",
    "user":     "finsight_user",
    "password": "finsight_pass",
}


def get_raw_ohlcv(conn, symbol: str, limit: int = 500) -> pd.DataFrame:
    query = """
        SELECT time, open, high, low, close, volume
        FROM stock_prices
        WHERE symbol = %s
        ORDER BY time ASC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(symbol, limit), parse_dates=["time"])
    df.set_index("time", inplace=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()
    macd_ind = MACD(close=df["close"])
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_14"] = atr.average_true_range()
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    df["obv"] = obv.on_balance_volume()
    return df.dropna(subset=["rsi_14", "macd"])


INSERT_INDICATORS = """
    INSERT INTO technical_indicators
        (time, symbol, rsi_14, macd, macd_signal, macd_hist,
         bb_upper, bb_middle, bb_lower, ema_20, ema_50, sma_200, atr_14, obv)
    VALUES %s
    ON CONFLICT DO NOTHING
"""


def store_indicators(conn, symbol: str, df: pd.DataFrame):
    rows = []
    for ts, row in df.iterrows():
        rows.append((
            ts, symbol,
            float(row["rsi_14"]) if pd.notna(row.get("rsi_14")) else None,
            float(row["macd"]) if pd.notna(row.get("macd")) else None,
            float(row["macd_signal"]) if pd.notna(row.get("macd_signal")) else None,
            float(row["macd_hist"]) if pd.notna(row.get("macd_hist")) else None,
            float(row["bb_upper"]) if pd.notna(row.get("bb_upper")) else None,
            float(row["bb_middle"]) if pd.notna(row.get("bb_middle")) else None,
            float(row["bb_lower"]) if pd.notna(row.get("bb_lower")) else None,
            float(row["ema_20"]) if pd.notna(row.get("ema_20")) else None,
            float(row["ema_50"]) if pd.notna(row.get("ema_50")) else None,
            float(row["sma_200"]) if pd.notna(row.get("sma_200")) else None,
            float(row["atr_14"]) if pd.notna(row.get("atr_14")) else None,
            float(row["obv"]) if pd.notna(row.get("obv")) else None,
        ))

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_INDICATORS, rows, page_size=200)
    conn.commit()
    logger.info(f"💾 Stored {len(rows)} indicator rows for {symbol}")


def get_watchlist(conn) -> list:
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM watchlist WHERE active = TRUE")
        return [row[0] for row in cur.fetchall()]


def run_feature_pipeline():
    logger.info("🔧 Starting Feature Engineering Pipeline")
    conn = psycopg2.connect(**DB_CONFIG)
    symbols = get_watchlist(conn)
    logger.info(f"📋 Processing {len(symbols)} symbols: {symbols}")

    for symbol in symbols:
        try:
            df = get_raw_ohlcv(conn, symbol)
            if len(df) < 20:
                logger.warning(f"⚠️ Not enough data for {symbol}, skipping")
                continue
            df_with_indicators = compute_indicators(df)
            store_indicators(conn, symbol, df_with_indicators)
            logger.info(f"✅ {symbol}: computed {len(df_with_indicators)} indicator rows")
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            conn.rollback()

    conn.close()
    logger.info("🏁 Feature Engineering Pipeline complete!")


if __name__ == "__main__":
    run_feature_pipeline()