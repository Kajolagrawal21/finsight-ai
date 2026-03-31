"""
FinSight AI — Model Evaluator
Backtests models on historical data and generates performance reports.
"""

import logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.evaluator")

MLFLOW_URI = "http://localhost:5002"


def walk_forward_validation(symbol: str, n_splits: int = 5) -> dict:
    """
    Walk-forward validation — the correct way to backtest time series models.
    
    Instead of one train/test split, we do multiple:
    Split 1: Train on weeks 1-4, test on week 5
    Split 2: Train on weeks 1-5, test on week 6
    Split 3: Train on weeks 1-6, test on week 7
    ... and so on
    
    This gives a much more realistic performance estimate.
    """
    from ml.feature_builder import load_features, add_target_labels, get_feature_columns
    from ml.models.xgboost_model import train_xgboost
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    logger.info(f"📊 Walk-forward validation for {symbol} ({n_splits} splits)")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(f"finsight-backtest-{symbol}")

    df = load_features(symbol, limit=2000)
    df = add_target_labels(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df["direction"].values

    n       = len(X)
    min_train = int(n * 0.5)   # minimum 50% for initial training
    step      = int((n - min_train) / n_splits)

    scores = []

    with mlflow.start_run(run_name=f"walkforward-{symbol}"):
        for i in range(n_splits):
            train_end = min_train + i * step
            test_end  = min(train_end + step, n)

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_test  = X[train_end:test_end]
            y_test  = y[train_end:test_end]

            if len(X_test) == 0:
                break

            # Train XGBoost
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42,
                eval_metric="logloss"
            )
            model.fit(X_train, y_train, verbose=False)

            y_prob = model.predict_proba(X_test)[:, 1]
            auc    = roc_auc_score(y_test, y_prob)
            scores.append(auc)

            logger.info(f"  Split {i+1}: train={train_end}, test={len(X_test)}, AUC={auc:.4f}")

        mean_auc = np.mean(scores)
        std_auc  = np.std(scores)

        mlflow.log_metrics({
            "mean_auc":   mean_auc,
            "std_auc":    std_auc,
            "min_auc":    min(scores),
            "max_auc":    max(scores),
            "n_splits":   n_splits,
        })

        logger.info(f"✅ Walk-forward AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    return {
        "symbol":   symbol,
        "mean_auc": mean_auc,
        "std_auc":  std_auc,
        "scores":   scores,
    }


def generate_performance_report(symbols: list = None) -> pd.DataFrame:
    """Generate a performance comparison report across all symbols and models."""
    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA"]

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    rows = []
    for symbol in symbols:
        for model_type in ["xgboost", "prophet", "lstm"]:
            exp_name = f"finsight-{model_type}-{symbol}"
            try:
                exp = client.get_experiment_by_name(exp_name)
                if not exp:
                    continue

                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=1
                )
                if not runs:
                    continue

                metrics = runs[0].data.metrics
                row = {"symbol": symbol, "model": model_type}
                row.update(metrics)
                rows.append(row)

            except Exception as e:
                logger.error(f"Error for {symbol}/{model_type}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"\n📊 Performance Report:\n{df.to_string()}")
    return df


if __name__ == "__main__":
    # Generate report
    report = generate_performance_report()
    if not report.empty:
        print("\n📊 Model Performance Report:")
        print(report.to_string(index=False))

    # Run walk-forward for AAPL
    result = walk_forward_validation("AAPL", n_splits=3)
    print(f"\n✅ Walk-forward AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
