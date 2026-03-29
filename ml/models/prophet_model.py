"""
FinSight AI — Prophet Price Forecaster
Forecasts stock price trend for next N days.
Handles seasonality, holidays, and trend changepoints.
"""

import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.prophet")


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet requires columns named 'ds' (date) and 'y' (value).
    """
    prophet_df = df[["close"]].copy()
    prophet_df = prophet_df.reset_index()
    prophet_df.columns = ["ds", "y"]

    # Prophet needs timezone-naive datetimes
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)
    return prophet_df


def train_prophet(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    symbol: str,
    forecast_horizon: int = 7,
    mlflow_uri: str = "http://localhost:5002"
) -> Prophet:
    """
    Train Prophet model and log to MLflow.
    Returns trained model + forecast DataFrame.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"finsight-prophet-{symbol}")

    train_df = prepare_prophet_df(raw_train)
    test_df  = prepare_prophet_df(raw_test)

    with mlflow.start_run(run_name=f"prophet-{symbol}"):

        # ─── Model Config ────────────────────────────────────
        params = {
            "changepoint_prior_scale":  0.05,   # flexibility of trend
            "seasonality_prior_scale":  10.0,   # strength of seasonality
            "holidays_prior_scale":     10.0,
            "seasonality_mode":         "multiplicative",
            "daily_seasonality":        True,
            "weekly_seasonality":       True,
            "yearly_seasonality":       True,
        }

        mlflow.log_params(params)
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("forecast_horizon_days", forecast_horizon)
        mlflow.log_param("train_rows", len(train_df))

        # ─── Train ───────────────────────────────────────────
        logger.info(f"🏋️ Training Prophet for {symbol}...")
        model = Prophet(**params)
        model.fit(train_df)

        # ─── Forecast on test period ─────────────────────────
        future = model.make_future_dataframe(
            periods=len(test_df) + forecast_horizon,
            freq="1min"
        )
        forecast = model.predict(future)

        # Align forecast with test actuals
        forecast_test = forecast.tail(len(test_df))
        y_pred = forecast_test["yhat"].values
        y_true = test_df["y"].values[:len(y_pred)]

        # ─── Metrics ─────────────────────────────────────────
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        metrics = {"mae": mae, "rmse": rmse, "mape": mape}
        mlflow.log_metrics(metrics)

        logger.info(f"📊 Prophet Results for {symbol}:")
        logger.info(f"   MAE:  {mae:.4f}")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   MAPE: {mape:.2f}%")

        # ─── Save forecast CSV ───────────────────────────────
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
            "/tmp/prophet_forecast.csv", index=False
        )
        mlflow.log_artifact("/tmp/prophet_forecast.csv")

        logger.info(f"✅ Prophet model logged to MLflow!")

    return model, forecast, metrics
