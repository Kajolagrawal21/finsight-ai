"""
FinSight AI — LSTM Price Predictor
Uses sequence of past N bars to predict next price movement.
Built with PyTorch, tracked in MLflow.
"""

import logging
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("finsight.lstm")

# Use MPS (Apple Silicon GPU) if available, else CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"🖥️ Using device: {DEVICE}")


# ─── Dataset ─────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """
    Sliding window dataset.
    Each sample = sequence of `seq_len` bars → predict next close price.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return x_seq, y_val


# ─── LSTM Model ──────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden  = lstm_out[:, -1, :]   # take last timestep
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


# ─── Training ────────────────────────────────────────────────────────────────

def train_lstm(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    symbol: str,
    seq_len: int = 60,
    epochs: int = 30,
    batch_size: int = 32,
    mlflow_uri: str = "http://localhost:5002"
):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(f"finsight-lstm-{symbol}")

    # ─── Scale features ──────────────────────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_sc  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # ─── Datasets ────────────────────────────────────────────
    train_dataset = StockSequenceDataset(X_train_sc, y_train_sc, seq_len)
    test_dataset  = StockSequenceDataset(X_test_sc,  y_test_sc,  seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    params = {
        "seq_len":    seq_len,
        "epochs":     epochs,
        "batch_size": batch_size,
        "hidden_size": 128,
        "num_layers":  2,
        "dropout":     0.2,
        "optimizer":  "Adam",
        "lr":          0.001,
    }

    with mlflow.start_run(run_name=f"lstm-{symbol}"):
        mlflow.log_params(params)
        mlflow.log_param("symbol", symbol)
        mlflow.log_param("device", str(DEVICE))

        # ─── Model, Loss, Optimizer ──────────────────────────
        model     = LSTMModel(
            input_size=X_train.shape[1],
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"]
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        # ─── Training Loop ───────────────────────────────────
        logger.info(f"🏋️ Training LSTM for {symbol} on {DEVICE}...")
        best_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss   = criterion(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    y_pred  = model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(test_loader)
            scheduler.step(avg_val)

            mlflow.log_metrics({
                "train_loss": avg_train,
                "val_loss":   avg_val
            }, step=epoch)

            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}"
                )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), "/tmp/best_lstm.pt")

        # ─── Load best model & evaluate ──────────────────────
        model.load_state_dict(torch.load("/tmp/best_lstm.pt"))
        model.eval()

        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = model(X_batch.to(DEVICE)).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y_batch.numpy())

        # Inverse scale
        preds_orig = scaler_y.inverse_transform(
            np.array(all_preds).reshape(-1, 1)
        ).flatten()
        true_orig  = scaler_y.inverse_transform(
            np.array(all_true).reshape(-1, 1)
        ).flatten()

        mae  = mean_absolute_error(true_orig, preds_orig)
        rmse = np.sqrt(mean_squared_error(true_orig, preds_orig))
        mape = np.mean(np.abs((true_orig - preds_orig) / (true_orig + 1e-8))) * 100

        metrics = {
            "test_mae":  mae,
            "test_rmse": rmse,
            "test_mape": mape,
            "best_val_loss": best_val_loss
        }
        mlflow.log_metrics(metrics)

        logger.info(f"📊 LSTM Results for {symbol}:")
        logger.info(f"   MAE:  {mae:.4f}")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   MAPE: {mape:.2f}%")

        # Log model
        mlflow.pytorch.log_model(
            model,
            artifact_path="lstm_model",
            registered_model_name=f"finsight-lstm-{symbol}"
        )

        logger.info("✅ LSTM model logged to MLflow!")

    return model, scaler_X, scaler_y, metrics
