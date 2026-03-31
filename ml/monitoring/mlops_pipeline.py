"""
FinSight AI — MLOps Pipeline
Orchestrates all monitoring, quality checks, drift detection,
and automated retraining. Run this daily via cron or GitHub Actions.
"""

import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.mlops")

SYMBOLS = ["AAPL", "MSFT", "NVDA"]


def run_mlops_pipeline():
    """
    Full MLOps pipeline:
    1. Data quality checks
    2. Drift detection
    3. Auto-retrain if needed
    4. Performance report
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 FinSight AI MLOps Pipeline — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"{'='*60}")

    # ─── Step 1: Data Quality ─────────────────────────────────
    logger.info("\n📋 Step 1: Data Quality Checks")
    from ml.monitoring.data_quality import run_quality_checks
    quality_results = run_quality_checks(SYMBOLS)

    quality_passed = sum(1 for r in quality_results.values() if r.passed)
    logger.info(f"Quality: {quality_passed}/{len(SYMBOLS)} symbols passed")

    # ─── Step 2: Drift Detection ──────────────────────────────
    logger.info("\n🔍 Step 2: Model Drift Detection")
    from ml.monitoring.drift_detector import run_drift_detection
    drift_results = run_drift_detection(SYMBOLS)
    needs_retrain = drift_results["needs_retrain"]

    # ─── Step 3: Auto-Retrain if needed ──────────────────────
    if needs_retrain:
        logger.info(f"\n🔄 Step 3: Auto-Retraining {len(needs_retrain)} models...")
        for item in needs_retrain:
            symbol, model = item.split("/")
            logger.info(f"  Retraining {model} for {symbol}...")
            try:
                from ml.feature_builder import prepare_train_test
                X_train, X_test, y_train, y_test, raw_train, raw_test = \
                    prepare_train_test(symbol)

                if model == "xgboost":
                    from ml.models.xgboost_model import train_xgboost
                    train_xgboost(X_train, X_test, y_train, y_test, symbol)
                elif model == "prophet":
                    from ml.models.prophet_model import train_prophet
                    train_prophet(raw_train, raw_test, symbol)
                elif model == "lstm":
                    from ml.models.lstm_model import train_lstm
                    train_lstm(X_train, X_test,
                               raw_train["close"], raw_test["close"], symbol)

                logger.info(f"  ✅ {symbol}/{model} retrained!")
            except Exception as e:
                logger.error(f"  ❌ Retrain failed for {symbol}/{model}: {e}")
    else:
        logger.info("\n✅ Step 3: No retraining needed — all models healthy!")

    # ─── Step 4: Performance Report ──────────────────────────
    logger.info("\n📊 Step 4: Generating Performance Report")
    from ml.evaluation.evaluator import generate_performance_report
    report = generate_performance_report(SYMBOLS)

    if not report.empty:
        logger.info(f"\n{report.to_string(index=False)}")

    # ─── Summary ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("🏁 MLOps Pipeline Complete!")
    logger.info(f"  Data quality: {quality_passed}/{len(SYMBOLS)} passed")
    logger.info(f"  Models retrained: {len(needs_retrain)}")
    logger.info(f"  View results at: http://localhost:5002")
    logger.info(f"{'='*60}")

    return {
        "quality":       quality_results,
        "drift":         drift_results,
        "needs_retrain": needs_retrain,
        "report":        report,
    }


if __name__ == "__main__":
    run_mlops_pipeline()
