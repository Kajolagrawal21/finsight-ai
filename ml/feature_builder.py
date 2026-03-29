"""
FinSight AI — ML Feature Builder
Loads OHLCV + technical indicators from TimescaleDB
and prepares feature matrices for model training.
"""

import logging
import pandas as pd
import numpy as np
import psycopg2
from typing import Tuple

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


def load_features(symbol: str, limit: int = 2000) -> pd.DataFrame:
    """
    Load OHLCV + technical indicators joined together for a symbol.
    Returns a clean DataFrame ready for ML.
    """
    query = """
        SELECT
            s.time, s.open, s.high, s.low, s.close, s.volume,
            i.rsi_14, i.macd, i.macd_signal, i.macd_hist,
            i.bb_upper, i.bb_middle, i.bb_lower,
            i.ema_20, i.ema_50, i.atr_14, i.obv
        FROM stock_prices s
        JOIN technical_indicators i
          ON s.time = i.time AND s.symbol = i.symbol
        WHERE s.symbol = %s
        ORDER BY s.time ASC
        LIMIT %s
    """
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(query, conn, params=(symbol, limit), parse_dates=["time"])
    conn.close()

    df.set_index("time", inplace=True)
    df.dropna(inplace=True)

    logger.info(f"✅ Loaded {len(df)} rows for {symbol}")
    return df


def add_target_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Add target columns for ML:
    - future_return: % price change after `horizon` bars
    - direction: 1 if price goes UP, 0 if DOWN (for classification)
    """
    df = df.copy()
    df["future_close"]  = df["close"].shift(-horizon)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    df["direction"]     = (df["future_return"] > 0).astype(int)  # 1=UP, 0=DOWN
    df.dropna(inplace=True)
    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for training."""
    return [
        "open", "high", "low", "close", "volume",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "ema_20", "ema_50", "atr_14", "obv"
    ]


def prepare_train_test(
    symbol: str,
    horizon: int = 1,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
           pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full pipeline: load → add labels → split train/test.
    Returns X_train, X_test, y_train, y_test
    plus raw_train and raw_test DataFrames for Prophet.
    """
    df = load_features(symbol)
    df = add_target_labels(df, horizon=horizon)

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["direction"]

    # Time-series split — NO random shuffle (would leak future data!)
    split = int(len(df) * (1 - test_size))

    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    # Raw splits for Prophet (needs time index)
    raw_train = df.iloc[:split]
    raw_test  = df.iloc[split:]

    logger.info(
        f"📊 {symbol} split: {len(X_train)} train rows, "
        f"{len(X_test)} test rows"
    )

    return X_train, X_test, y_train, y_test, raw_train, raw_test
