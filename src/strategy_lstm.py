"""
File: strategy_lstm.py (PyTorch version)
Purpose:
  LSTM-based trading strategy for stock price prediction using PyTorch.
  Generates buy/sell signals with confidence scores.

Key Features:
  - Uses sliding window approach for time series
  - Predicts next-day price
  - Generates signals with confidence (0-1 scale)
  - Confidence = predicted_return / max_historical_return

Output:
  DataFrame with columns: price, prediction, signal, trade, confidence
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for stock price prediction
    
    Architecture:
      - LSTM Layer 1 (50 units) + Dropout(0.2)
      - LSTM Layer 2 (50 units) + Dropout(0.2)
      - Fully Connected Layer (1 output)
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def create_sequences(data, window_size=30):
    """
    Create sliding window sequences for LSTM
    
    Args:
        data: 1D array of prices
        window_size: look-back window
    
    Returns:
        X: (n_samples, window_size, 1)
        y: (n_samples, 1)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


def train_lstm(price_series: pd.Series, window_size=30, epochs=100, batch_size=32, verbose=0):
    """
    Train LSTM model on price data using PyTorch
    
    Returns:
        model: trained PyTorch model
        scaler: MinMaxScaler for inverse transform
        train_size: number of training samples
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose > 0:
        print(f"Using device: {device}")
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(scaled_data.flatten(), window_size)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Train/test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train = torch.FloatTensor(X[:train_size]).to(device)
    y_train = torch.FloatTensor(y[:train_size]).reshape(-1, 1).to(device)  # FIX: Add .reshape(-1, 1)
    X_test = torch.FloatTensor(X[train_size:]).to(device)
    y_test = torch.FloatTensor(y[train_size:]).reshape(-1, 1).to(device)  # FIX: Add .reshape(-1, 1)
    
    # Build model
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate: 0.001 → 0.0001
    
    print(f"Training LSTM model... (epochs={epochs}, lr=0.0001)")
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print every 10 epochs to monitor training
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
        train_loss = criterion(train_pred, y_train).item()
        test_loss = criterion(test_pred, y_test).item()
    
    print(f"Training MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
    
    return model, scaler, train_size, device


def predict_with_lstm(model, scaler, price_series: pd.Series, window_size=30, device='cpu'):
    """
    Generate predictions for entire series
    
    Returns:
        predictions: array of predicted prices (same length as input)
    """
    model.eval()
    scaled_data = scaler.transform(price_series.values.reshape(-1, 1)).flatten()
    
    predictions = []
    
    # First window_size predictions are NaN
    predictions.extend([np.nan] * window_size)
    
    # Predict for each time step
    with torch.no_grad():
        for i in range(window_size, len(scaled_data)):
            window = scaled_data[i-window_size:i].reshape(1, window_size, 1)
            window_tensor = torch.FloatTensor(window).to(device)
            pred_scaled = model(window_tensor).cpu().numpy()[0][0]
            pred_price = scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred_price)
    
    return np.array(predictions)


def calculate_confidence(predicted_prices, actual_prices, method='return_based'):
    """
    Calculate confidence score for each prediction
    
    Args:
        predicted_prices: LSTM predictions
        actual_prices: actual price series
        method: 'return_based' or 'error_based'
    
    Returns:
        confidence: array of confidence scores (0-1)
    """
    if method == 'return_based':
        # Confidence based on predicted return magnitude
        predicted_returns = (predicted_prices - actual_prices) / actual_prices
        
        # Normalize to 0-1 range
        valid_returns = predicted_returns[~np.isnan(predicted_returns)]
        if len(valid_returns) > 0:
            max_return = np.percentile(np.abs(valid_returns), 95)
            confidence = np.abs(predicted_returns) / max_return
            confidence = np.clip(confidence, 0, 1)
        else:
            confidence = np.zeros_like(predicted_returns)
        
        return confidence
    
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_signals_with_confidence(
    price: pd.Series,
    predictions: np.ndarray,
    confidence: np.ndarray,
    threshold=0.02
):
    """
    Generate buy/sell signals with confidence scores
    
    Logic:
        BUY: predicted_return > threshold
        SELL: predicted_return < -threshold
        HOLD: otherwise
    
    Returns:
        signal: 1 (long), 0 (flat), -1 (should sell)
        trade: +1 (buy), -1 (sell), 0 (hold)
        confidence: 0-1 score
    """
    predicted_return = (predictions - price.values) / price.values
    
    signal = np.zeros(len(price))
    signal[predicted_return > threshold] = 1
    signal[predicted_return < -threshold] = -1
    
    # Generate trades
    trade = np.zeros(len(price))
    trade[1:] = np.diff(signal)
    
    return signal, trade, confidence


def lstm_strategy(
    price: pd.Series,
    window_size=30,
    epochs=50,
    threshold=0.02,
    verbose=0
) -> pd.DataFrame:
    """
    Complete LSTM trading strategy pipeline (PyTorch version)
    
    Args:
        price: price series with datetime index
        window_size: LSTM look-back window
        epochs: training epochs
        threshold: return threshold for signal generation
        verbose: training verbosity
    
    Returns:
        DataFrame with columns: price, prediction, signal, trade, confidence
    """
    print(f"\n{'='*60}")
    print(f"Running LSTM Strategy (PyTorch)")
    print(f"Data points: {len(price)}")
    print(f"Window size: {window_size} | Epochs: {epochs} | Threshold: {threshold:.1%}")
    print(f"{'='*60}")
    
    # Train model
    model, scaler, train_size, device = train_lstm(price, window_size, epochs, verbose=verbose)
    
    # Generate predictions
    predictions = predict_with_lstm(model, scaler, price, window_size, device)
    
    # Calculate confidence
    confidence = calculate_confidence(predictions, price.values, method='return_based')
    
    # Generate signals
    signal, trade, confidence = generate_signals_with_confidence(
        price, predictions, confidence, threshold
    )
    
    # Build output DataFrame
    df = pd.DataFrame({
        'price': price.values,
        'prediction': predictions,
        'signal': signal,
        'trade': trade,
        'confidence': confidence
    }, index=price.index)
    
    # Count signals
    n_buy = (df['trade'] == 1).sum()
    n_sell = (df['trade'] == -1).sum()
    print(f"\nSignals generated: {n_buy} BUY | {n_sell} SELL")
    print(f"{'='*60}\n")
    
    return df


# Example usage
if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(500) * 2),
        index=dates
    )
    
    signals_df = lstm_strategy(prices, window_size=30, epochs=20, verbose=1)
    
    print("\nSample output:")
    print(signals_df.tail(10))
    
    print("\nPyTorch LSTM strategy ready!")