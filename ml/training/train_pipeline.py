"""
FinSight AI — Master Training Pipeline
Trains XGBoost, Prophet, and LSTM for all watchlist symbols.
Logs everything to MLflow. Run this to train all models at once.
"""

import logging
import psycopg2
from ml.feature_builder import prepare_train_test
from ml.models.xgboost_model import train_xgboost
from ml.models.prophet_model import train_prophet
from ml.models.lstm_model import train_lstm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.training")

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "finsight",
    "user":     "finsight_user",
    "password": "finsight_pass",
}

# Train on these symbols (start with 3, expand later)
TRAINING_SYMBOLS = None


def get_symbols_with_enough_data(min_rows: int = 200) -> list:
    """Only train on symbols that have enough data."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()
    cur.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM technical_indicators
        GROUP BY symbol
        HAVING COUNT(*) >= %s
        ORDER BY cnt DESC
    """, (min_rows,))
    symbols = [row[0] for row in cur.fetchall()]
    conn.close()
    logger.info(f"📋 Symbols with enough data: {symbols}")
    return symbols


def run_training_pipeline():
    logger.info("🚀 Starting FinSight AI Training Pipeline")

    symbols = get_symbols_with_enough_data()
    # Filter to our training set
    if TRAINING_SYMBOLS:
        symbols = [s for s in TRAINING_SYMBOLS if s in symbols]
        
    

    if not symbols:
        logger.error("❌ No symbols with enough data! Run feature engineering first.")
        return

    results = {}

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 Training models for {symbol}")
        logger.info(f"{'='*50}")

        try:
            # Prepare features
            X_train, X_test, y_train, y_test, raw_train, raw_test = \
                prepare_train_test(symbol)

            symbol_results = {}

            # ─── XGBoost ─────────────────────────────────────
            logger.info(f"\n📦 Training XGBoost for {symbol}...")
            _, xgb_metrics = train_xgboost(
                X_train, X_test, y_train, y_test, symbol
            )
            symbol_results["xgboost"] = xgb_metrics
            logger.info(f"✅ XGBoost done | AUC: {xgb_metrics['roc_auc']:.4f}")

            # ─── Prophet ─────────────────────────────────────
            logger.info(f"\n📦 Training Prophet for {symbol}...")
            _, _, prophet_metrics = train_prophet(
                raw_train, raw_test, symbol
            )
            symbol_results["prophet"] = prophet_metrics
            logger.info(f"✅ Prophet done | MAPE: {prophet_metrics['mape']:.2f}%")

            # ─── LSTM ─────────────────────────────────────────
            logger.info(f"\n📦 Training LSTM for {symbol}...")
            _, _, _, lstm_metrics = train_lstm(
                X_train, X_test,
                raw_train["close"], raw_test["close"],
                symbol
            )
            symbol_results["lstm"] = lstm_metrics
            logger.info(f"✅ LSTM done | RMSE: {lstm_metrics['test_rmse']:.4f}")

            results[symbol] = symbol_results

        except Exception as e:
            logger.error(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ─── Summary ─────────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info("🏁 Training Pipeline Complete!")
    logger.info(f"{'='*50}")

    for symbol, res in results.items():
        logger.info(f"\n📊 {symbol}:")
        if "xgboost" in res:
            logger.info(f"   XGBoost  → AUC: {res['xgboost']['roc_auc']:.4f} | F1: {res['xgboost']['f1_score']:.4f}")
        if "prophet" in res:
            logger.info(f"   Prophet  → MAPE: {res['prophet']['mape']:.2f}%")
        if "lstm" in res:
            logger.info(f"   LSTM     → RMSE: {res['lstm']['test_rmse']:.4f}")

    logger.info(f"\n🌐 View all experiments at: http://localhost:5002")


if __name__ == "__main__":
    run_training_pipeline()
