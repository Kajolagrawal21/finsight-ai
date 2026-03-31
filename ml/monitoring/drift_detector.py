"""
FinSight AI — Model Drift Detector
Monitors model performance over time and detects drift.
Triggers retraining when performance degrades beyond threshold.
"""

import logging
import mlflow
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.drift")

MLFLOW_URI = "http://localhost:5002"

DB_CONFIG = {
    "host": "localhost", "port": 5432,
    "dbname": "finsight", "user": "finsight_user", "password": "finsight_pass",
}

# Thresholds — retrain if performance drops below these
DRIFT_THRESHOLDS = {
    "xgboost": {"roc_auc": 0.45},    # retrain if AUC drops below 0.45
    "prophet":  {"mape": 10.0},       # retrain if MAPE exceeds 10%
    "lstm":     {"test_rmse": 10.0},  # retrain if RMSE exceeds 10
}


@dataclass
class DriftReport:
    symbol:          str
    model_name:      str
    current_metric:  float
    baseline_metric: float
    drift_detected:  bool
    retrain_needed:  bool
    metric_name:     str


def get_latest_run_metrics(experiment_name: str, metric_name: str) -> Optional[float]:
    """Get the most recent metric value from MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = mlflow.tracking.MlflowClient()

        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            return None

        return runs[0].data.metrics.get(metric_name)

    except Exception as e:
        logger.error(f"MLflow error: {e}")
        return None


def check_prediction_accuracy(symbol: str, days_back: int = 1) -> dict:
    """
    Check how accurate recent predictions were vs actual prices.
    Compares predicted direction with actual price movement.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT p.time, p.direction, p.confidence,
                       s1.close as pred_close,
                       s2.close as actual_close
                FROM predictions p
                JOIN stock_prices s1 ON DATE_TRUNC('minute', p.time) = DATE_TRUNC('minute', s1.time)
                    AND p.symbol = s1.symbol
                LEFT JOIN stock_prices s2 ON s2.time = p.time + INTERVAL '1 day'
                    AND p.symbol = s2.symbol
                WHERE p.symbol = %s
                AND p.time > NOW() - INTERVAL '%s days'
                ORDER BY p.time DESC
                LIMIT 100
            """, (symbol, days_back))
            rows = cur.fetchall()
        conn.close()

        if not rows:
            return {"accuracy": None, "n_predictions": 0}

        correct = 0
        total   = 0
        for row in rows:
            if row["actual_close"] and row["pred_close"]:
                actual_up    = row["actual_close"] > row["pred_close"]
                predicted_up = row["direction"] == "UP"
                if actual_up == predicted_up:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else None
        return {"accuracy": accuracy, "n_predictions": total}

    except Exception as e:
        logger.error(f"Error checking accuracy: {e}")
        return {"accuracy": None, "n_predictions": 0}


def detect_drift(symbol: str) -> list:
    """
    Run drift detection for all models for a symbol.
    Returns list of DriftReports.
    """
    reports = []

    # ─── XGBoost drift ───────────────────────────────────────
    exp_name  = f"finsight-xgboost-{symbol}"
    roc_auc   = get_latest_run_metrics(exp_name, "roc_auc")
    threshold = DRIFT_THRESHOLDS["xgboost"]["roc_auc"]

    if roc_auc is not None:
        drift    = roc_auc < threshold
        reports.append(DriftReport(
            symbol=symbol, model_name="xgboost",
            current_metric=roc_auc, baseline_metric=threshold,
            drift_detected=drift, retrain_needed=drift,
            metric_name="roc_auc"
        ))
        status = "⚠️ DRIFT" if drift else "✅ OK"
        logger.info(f"{symbol} XGBoost AUC={roc_auc:.4f} {status}")

    # ─── Prophet drift ───────────────────────────────────────
    exp_name  = f"finsight-prophet-{symbol}"
    mape      = get_latest_run_metrics(exp_name, "mape")
    threshold = DRIFT_THRESHOLDS["prophet"]["mape"]

    if mape is not None:
        drift    = mape > threshold
        reports.append(DriftReport(
            symbol=symbol, model_name="prophet",
            current_metric=mape, baseline_metric=threshold,
            drift_detected=drift, retrain_needed=drift,
            metric_name="mape"
        ))
        status = "⚠️ DRIFT" if drift else "✅ OK"
        logger.info(f"{symbol} Prophet MAPE={mape:.2f}% {status}")

    # ─── LSTM drift ──────────────────────────────────────────
    exp_name  = f"finsight-lstm-{symbol}"
    rmse      = get_latest_run_metrics(exp_name, "test_rmse")
    threshold = DRIFT_THRESHOLDS["lstm"]["test_rmse"]

    if rmse is not None:
        drift    = rmse > threshold
        reports.append(DriftReport(
            symbol=symbol, model_name="lstm",
            current_metric=rmse, baseline_metric=threshold,
            drift_detected=drift, retrain_needed=drift,
            metric_name="test_rmse"
        ))
        status = "⚠️ DRIFT" if drift else "✅ OK"
        logger.info(f"{symbol} LSTM RMSE={rmse:.4f} {status}")

    return reports


def run_drift_detection(symbols: list = None) -> dict:
    """Run drift detection for all symbols."""
    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA"]

    all_reports  = {}
    needs_retrain = []

    for symbol in symbols:
        logger.info(f"\n🔍 Checking drift for {symbol}...")
        reports = detect_drift(symbol)
        all_reports[symbol] = reports

        for r in reports:
            if r.retrain_needed:
                needs_retrain.append(f"{symbol}/{r.model_name}")

    if needs_retrain:
        logger.warning(f"\n⚠️ Models needing retraining: {needs_retrain}")
    else:
        logger.info(f"\n✅ All models within acceptable performance range!")

    return {"reports": all_reports, "needs_retrain": needs_retrain}


if __name__ == "__main__":
    result = run_drift_detection()
    print(f"\nModels needing retraining: {result['needs_retrain']}")
