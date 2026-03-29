"""
FinSight AI — Feature Engineering Pipeline
Computes technical indicators from raw OHLCV data and stores in TimescaleDB.
"""

import logging
import pandas as pd
import psycopg2
import psycopg2.extras
from ta import add_all_ta_features
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
    """Load OHLCV data from TimescaleDB for a symbol."""
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
    """
    Compute all technical indicators.
    Returns a DataFrame with indicator columns aligned to original index.
    """
    df = df.copy()

    # RSI (14-period)
    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    # MACD
    macd_ind = MACD(close=df["close"])
    df["macd"]        = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"]   = macd_ind.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"]  = bb.bollinger_lband()

    # EMA 20 & 50
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()

    # SMA 200
    df["sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()

    # ATR 14
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_14"] = atr.average_true_range()

    # OBV
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    df["obv"] = obv.on_balance_volume()

    return df.dropna(subset=["rsi_14", "macd"])   # drop rows where indicators aren't ready yet


INSERT_INDICATORS = """
    INSERT INTO technical_indicators
        (time, symbol, rsi_14, macd, macd_signal, macd_hist,
         bb_upper, bb_middle, bb_lower, ema_20, ema_50, sma_200, atr_14, obv)
    VALUES %s
    ON CONFLICT DO NOTHING
"""

def store_indicators(conn, symbol: str, df: pd.DataFrame):
    """Batch insert computed indicators into TimescaleDB."""
    rows = []
    for ts, row in df.iterrows():
        rows.append((
            ts, symbol,
            row.get("rsi_14"),  row.get("macd"),     row.get("macd_signal"),
            row.get("macd_hist"), row.get("bb_upper"), row.get("bb_middle"),
            row.get("bb_lower"), row.get("ema_20"),   row.get("ema_50"),
            row.get("sma_200"),  row.get("atr_14"),   row.get("obv"),
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
    """Run the full feature engineering pipeline for all watchlist symbols."""
    logger.info("🔧 Starting Feature Engineering Pipeline")
    conn = psycopg2.connect(**DB_CONFIG)

    symbols = get_watchlist(conn)
    logger.info(f"📋 Processing {len(symbols)} symbols: {symbols}")

    for symbol in symbols:
        try:
            df = get_raw_ohlcv(conn, symbol)

            if len(df) < 20:
                logger.warning(f"⚠️ Not enough data for {symbol} ({len(df)} rows), skipping")
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
