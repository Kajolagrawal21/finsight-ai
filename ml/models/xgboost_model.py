"""
FinSight AI — XGBoost Price Direction Classifier
Predicts whether price will go UP or DOWN in next N bars.
All experiments tracked in MLflow.
"""

import logging
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.xgboost")


def train_xgboost(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    symbol: str,
    mlflow_uri: str = "http://localhost:5002"
) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier and log everything to MLflow.
    Returns the trained model.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"finsight-xgboost-{symbol}")

    # ─── Hyperparameters ─────────────────────────────────────
    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }

    with mlflow.start_run(run_name=f"xgboost-{symbol}"):

        # Log params
        mlflow.log_params(params)
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # ─── Train ───────────────────────────────────────────
        logger.info(f"🏋️ Training XGBoost for {symbol}...")
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # ─── Evaluate ────────────────────────────────────────
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1_score":  f1_score(y_test, y_pred, zero_division=0),
            "roc_auc":   roc_auc_score(y_test, y_pred_prob),
        }

        mlflow.log_metrics(metrics)

        logger.info(f"📊 XGBoost Results for {symbol}:")
        for k, v in metrics.items():
            logger.info(f"   {k}: {v:.4f}")

        # ─── Feature Importance ──────────────────────────────
        importance_df = pd.DataFrame({
            "feature":    X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_df.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")

        logger.info(f"\n🔍 Top 5 Features:\n{importance_df.head()}")

        # ─── Log Model ───────────────────────────────────────
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
            registered_model_name=f"finsight-xgboost-{symbol}"
        )

        logger.info(f"✅ XGBoost model logged to MLflow!")

    return model, metrics
