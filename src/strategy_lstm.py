"""
File: strategy_lstm.py
Purpose:
  LSTM-based trading strategy for stock price prediction.
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
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def create_sequences(data, window_size=30):
    """
    Create sliding window sequences for LSTM
    
    Args:
        data: 1D array of prices
        window_size: look-back window
    
    Returns:
        X: (n_samples, window_size, 1)
        y: (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)


def build_lstm_model(window_size=30):
    """
    Build a simple 2-layer LSTM model
    
    Architecture:
      - LSTM(50 units) + Dropout(0.2)
      - LSTM(50 units) + Dropout(0.2)
      - Dense(1) - output layer
    """
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        layers.Dropout(0.2),
        layers.LSTM(50, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm(price_series: pd.Series, window_size=30, epochs=50, verbose=0):
    """
    Train LSTM model on price data
    
    Returns:
        model: trained Keras model
        scaler: MinMaxScaler for inverse transform
        train_size: number of training samples
    """
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(scaled_data.flatten(), window_size)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Train/test split (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_lstm_model(window_size)
    
    print(f"Training LSTM model... (epochs={epochs})")
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=verbose
    )
    
    # Evaluate
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Training MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    
    return model, scaler, train_size


def predict_with_lstm(model, scaler, price_series: pd.Series, window_size=30):
    """
    Generate predictions for entire series
    
    Returns:
        predictions: array of predicted prices (same length as input)
    """
    scaled_data = scaler.transform(price_series.values.reshape(-1, 1)).flatten()
    
    predictions = []
    
    # First window_size predictions are NaN
    predictions.extend([np.nan] * window_size)
    
    # Predict for each time step
    for i in range(window_size, len(scaled_data)):
        window = scaled_data[i-window_size:i].reshape(1, window_size, 1)
        pred_scaled = model.predict(window, verbose=0)[0][0]
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
        # Higher predicted return = higher confidence
        predicted_returns = (predicted_prices - actual_prices) / actual_prices
        
        # Normalize to 0-1 range
        # Use historical max return as reference
        valid_returns = predicted_returns[~np.isnan(predicted_returns)]
        if len(valid_returns) > 0:
            max_return = np.percentile(np.abs(valid_returns), 95)  # 95th percentile
            confidence = np.abs(predicted_returns) / max_return
            confidence = np.clip(confidence, 0, 1)  # cap at 1.0
        else:
            confidence = np.zeros_like(predicted_returns)
        
        return confidence
    
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_signals_with_confidence(
    price: pd.Series,
    predictions: np.ndarray,
    confidence: np.ndarray,
    threshold=0.02  # 2% predicted return to trigger signal
):
    """
    Generate buy/sell signals with confidence scores
    
    Logic:
        BUY: predicted_return > threshold
        SELL: predicted_return < -threshold
        HOLD: otherwise
    
    Returns:
        signal: 1 (long), 0 (flat), -1 (should sell but we're long-only)
        trade: +1 (buy), -1 (sell), 0 (hold)
        confidence: 0-1 score
    """
    predicted_return = (predictions - price.values) / price.values
    
    signal = np.zeros(len(price))
    signal[predicted_return > threshold] = 1  # Buy signal
    signal[predicted_return < -threshold] = -1  # Sell signal (mean revert or stop loss)
    
    # Generate trades (changes in signal)
    trade = np.zeros(len(price))
    trade[1:] = np.diff(signal)
    
    # For long-only: convert -1 signal to sell action if we're holding
    # This will be handled by backtest
    
    return signal, trade, confidence


def lstm_strategy(
    price: pd.Series,
    window_size=30,
    epochs=50,
    threshold=0.02,
    verbose=0
) -> pd.DataFrame:
    """
    Complete LSTM trading strategy pipeline
    
    Args:
        price: price series with datetime index
        window_size: LSTM look-back window
        epochs: training epochs
        threshold: return threshold for signal generation
        verbose: training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        DataFrame with columns: price, prediction, signal, trade, confidence
    """
    print(f"\n{'='*60}")
    print(f"Running LSTM Strategy")
    print(f"Data points: {len(price)}")
    print(f"Window size: {window_size} | Epochs: {epochs} | Threshold: {threshold:.1%}")
    print(f"{'='*60}")
    
    # Train model
    model, scaler, train_size = train_lstm(price, window_size, epochs, verbose)
    
    # Generate predictions
    predictions = predict_with_lstm(model, scaler, price, window_size)
    
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
