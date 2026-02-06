"""strategy_lstm.py (STABLE)

PyTorch LSTM-based trading strategy with stability improvements:
1) Fixed random seed (reproducible results)
2) Gradient clipping (prevents exploding gradients)
3) Learning-rate scheduler (stable convergence)
4) Early stopping (reduces overfitting)
5) Safer weight initialization

Output DataFrame columns:
  price, prediction, signal, trade, confidence

Signal rule:
  predicted_return = (prediction - price) / price
  BUY  if predicted_return > threshold
  SELL if predicted_return < -threshold
  else HOLD

Notes:
- This is still a simple educational model. Real trading requires more careful validation.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[SEED] Random seed set to {seed}")


class LSTMModel(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        # Better initialization for stability
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.zero_()

        nn.init.xavier_uniform_(self.fc.weight.data)
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def create_sequences(data: np.ndarray, window_size: int = 30):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_lstm(
    price_series: pd.Series,
    window_size: int = 30,
    epochs: int = 80,
    batch_size: int = 32,
    verbose: int = 0,
    seed: int = 42,
    learning_rate: float = 0.00005,
    patience: int = 15,
    min_delta: float = 1e-6,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()

    X, y = create_sequences(scaled, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/val split 80/20
    train_size = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:train_size]).to(device)
    y_train = torch.FloatTensor(y[:train_size]).reshape(-1, 1).to(device)
    X_val = torch.FloatTensor(X[train_size:]).to(device)
    y_val = torch.FloatTensor(y[train_size:]).reshape(-1, 1).to(device)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-7,
)

    print(f"Training LSTM... epochs={epochs}, lr={learning_rate:.6f}, window={window_size}, seed={seed}")
    print("Stability: grad_clip=1.0, ReduceLROnPlateau, early_stopping")

    best_val = float("inf")
    best_state = None
    wait = 0

    n_train = X_train.shape[0]

    for epoch in range(epochs):
        model.train()

        idx = torch.randperm(n_train, device=device)
        X_train_shuf = X_train[idx]
        y_train_shuf = y_train[idx]

        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            xb = X_train_shuf[start:end]
            yb = y_train_shuf[start:end]

            pred = model(xb)
            loss = criterion(pred, yb)

            if torch.isnan(loss):
                print(f"[WARN] NaN loss at epoch {epoch+1}. Stop training.")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * (end - start)

        epoch_loss /= max(n_train, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:>3}/{epochs} | train={epoch_loss:.6f} | val={val_loss:.6f} | lr={lr:.7f} | wait={wait}/{patience}")

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[EARLY STOP] epoch={epoch+1}, best_val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, train_size, device


def predict_with_lstm(model, scaler, price_series: pd.Series, window_size: int = 30, device=None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    scaled = scaler.transform(price_series.values.reshape(-1, 1)).flatten()

    preds = [np.nan] * window_size
    with torch.no_grad():
        for i in range(window_size, len(scaled)):
            window = scaled[i - window_size:i].reshape(1, window_size, 1)
            x = torch.FloatTensor(window).to(device)
            pred_scaled = model(x).cpu().numpy()[0][0]
            pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
            preds.append(pred_price)

    return np.array(preds)


def calculate_confidence(predicted_prices: np.ndarray, actual_prices: np.ndarray) -> np.ndarray:
    predicted_returns = (predicted_prices - actual_prices) / actual_prices
    valid = predicted_returns[~np.isnan(predicted_returns)]
    if len(valid) == 0:
        return np.zeros_like(predicted_returns)

    max_ret = np.percentile(np.abs(valid), 95)
    if max_ret <= 0:
        return np.zeros_like(predicted_returns)

    conf = np.abs(predicted_returns) / max_ret
    return np.clip(conf, 0, 1)


def generate_signals(price: pd.Series, predictions: np.ndarray, threshold: float = 0.02):
    actual = price.values
    pred_ret = (predictions - actual) / actual

    signal = np.zeros(len(price))
    signal[pred_ret > threshold] = 1
    signal[pred_ret < -threshold] = -1

    trade = np.zeros(len(price))
    trade[1:] = np.diff(signal)

    confidence = calculate_confidence(predictions, actual)
    return signal, trade, confidence


def lstm_strategy(
    price: pd.Series,
    window_size: int = 30,
    epochs: int = 80,
    threshold: float = 0.02,
    verbose: int = 0,
    seed: int = 42,
    learning_rate: float = 0.00005,
    patience: int = 15,
) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print("Running LSTM Strategy (STABLE)")
    print(f"Data points: {len(price)} | window={window_size} | epochs={epochs} | threshold={threshold:.2%}")
    print(f"seed={seed} | lr={learning_rate:.6f} | patience={patience}")
    print(f"{'='*70}")

    model, scaler, train_size, device = train_lstm(
        price_series=price,
        window_size=window_size,
        epochs=epochs,
        verbose=verbose,
        seed=seed,
        learning_rate=learning_rate,
        patience=patience,
    )

    predictions = predict_with_lstm(model, scaler, price, window_size, device)
    signal, trade, confidence = generate_signals(price, predictions, threshold=threshold)

    df = pd.DataFrame(
        {
            "price": price.values,
            "prediction": predictions,
            "signal": signal,
            "trade": trade,
            "confidence": confidence,
        },
        index=price.index,
    )

    n_buy = int((df["trade"] == 1.0).sum())
    n_sell = int((df["trade"] == -1.0).sum())
    print(f"Signals: BUY={n_buy} | SELL={n_sell}")
    print(f"{'='*70}\n")

    return df
