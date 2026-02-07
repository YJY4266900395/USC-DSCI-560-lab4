import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import warnings
import random
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """
    Args:
        seed: default 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[SEED] Random seed set to {seed} for reproducibility")


class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for stock price prediction
    
    Architecture:
      - LSTM Layer 1 (50 units) + Dropout(0.2)
      - LSTM Layer 2 (50 units) + Dropout(0.2)
      - Fully Connected Layer (1 output)
    
    Improvements:
      - Xavier/He initialization for weights
      - Gradient clipping ready
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
        
        # weights initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Xavier/He initialization for LSTM weights for stability
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # FC layer initialization
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        
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
    Predicts percentage change instead of absolute price
    
    Args:
        data: 1D array of prices
        window_size: look-back window
    
    Returns:
        X: (n_samples, window_size, 1) - price sequences
        y: (n_samples, 1) - next-day percentage change
    """
    # X, y = [], []
    # for i in range(len(data) - window_size):
    #     X.append(data[i:i+window_size])
    #     # y = percentage change from last price in window to next price
    #     pct_change = (data[i+window_size] - data[i+window_size-1]) / data[i+window_size-1]
    #     y.append(pct_change)
    # return np.array(X), np.array(y)
    # MOVE TO TRAINING FUNCTION FOR BETTER CONTROL
    pass


def train_lstm(
    price_series: pd.Series, 
    window_size=30, 
    epochs=100, 
    batch_size=32, 
    verbose=0,
    seed=42,
    learning_rate=0.00005,
    patience=15,
    min_delta=1e-6
):
    """
    Train LSTM model on price data
    
    Args:
        price_series: pandas Series of prices
        window_size: look-back window
        epochs: training epochs
        batch_size: batch size
        verbose: verbosity level
        seed: random seed
        learning_rate: initial learning rate
        patience: early stopping patience
        min_delta: minimum change to qualify as improvement
    
    Returns:
        model: trained model
        scaler: MinMaxScaler for inverse transform
        train_size: number of training samples
        device: torch device
    """
    # set seed
    # set_seed(seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose > 0:
        print(f"Using device: {device}")
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(price_series.values.reshape(-1, 1)).flatten()
    
    # Create sequences - moved to here!
    X, y = [], []
    original_prices = price_series.values
    
    for i in range(len(scaled_prices) - window_size):
        # X: normalized price window
        X.append(scaled_prices[i:i+window_size])
        y.append(scaled_prices[i+window_size])
        
        # y: percentage change on ORIGINAL prices (not normalized)
        # pct_change = (original_prices[i+window_size] - original_prices[i+window_size-1]) / original_prices[i+window_size-1]
        # y.append(pct_change)

    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Use all data to train (no train/test split - backtest is the real test)
    X_train = torch.FloatTensor(X).to(device)
    y_train = torch.FloatTensor(y).reshape(-1, 1).to(device)
    
    # Build model
    model = LSTMModel(
        input_size=1, 
        hidden_size=50, 
        num_layers=2, 
        dropout=0.2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # ReduceLROnPlateau for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10,
        min_lr=1e-7
    )
    
    print(f"Training LSTM model... (epochs={epochs}, lr={learning_rate:.6f}, seed={seed})")
    print(f"Predicting: Next-day percentage change")
    
    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"[WARNING] NaN loss detected at epoch {epoch+1}. Stopping training.")
            break
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent gradient explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping check
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1}/{epochs}] | Loss: {loss.item():.6f} | "
                  f"LR: {current_lr:.7f} | Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch+1}.")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[LOADED] Best model from validation")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_pred = model(X_train)
        final_loss = criterion(final_pred, y_train).item()
    
    print(f"Final Training Loss: {final_loss:.6f}")
    
    return model, scaler, len(X), device


def predict_with_lstm(model, scaler, price_series: pd.Series, window_size=30, device='cpu'):
    """
    Generate predictions for entire series
    Model predicts percentage change, convert to price
    
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
            
            # Model predicts percentage change
            pred_pct_change = model(window_tensor).cpu().numpy()[0][0]
            
            # Convert to predicted price: current_price * (1 + predicted_change)
            current_price = price_series.iloc[i]
            pred_price = current_price * (1 + pred_pct_change)
            
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
    epochs=100,
    threshold=0.02,
    verbose=0,
    seed=42,
    learning_rate=0.00005,
    patience=15
) -> pd.DataFrame:
    """
    Complete LSTM trading strategy pipeline
    
    Args:
        price: price series with datetime index
        window_size: LSTM look-back window
        epochs: training epochs
        threshold: return threshold for signal generation
        verbose: training verbosity
        seed: random seed
        learning_rate: initial learning rate
        patience: early stopping patience
    
    Returns:
        DataFrame with columns: price, prediction, signal, trade, confidence
    """
    print(f"\n{'='*70}")
    print(f"Running LSTM Strategy (PyTorch - STABLE)")
    print(f"Data points: {len(price)}")
    print(f"Window size: {window_size} | Epochs: {epochs} | Threshold: {threshold:.1%}")
    print(f"Seed: {seed} | Learning Rate: {learning_rate:.6f} | Patience: {patience}")
    print(f"{'='*70}")
    
    # Train model
    model, scaler, train_size, device = train_lstm(
        price, 
        window_size, 
        epochs, 
        verbose=verbose,
        seed=seed,
        learning_rate=learning_rate,
        patience=patience
    )
    
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
    print(f"{'='*70}\n")
    
    return df


# Example usage
if __name__ == "__main__":
    # Test with sample data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    
    # Set seed for test data
    np.random.seed(42)
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(500) * 2),
        index=dates
    )
    
    print("Running test with seed=42...")
    signals_df_1 = lstm_strategy(prices, window_size=30, epochs=20, verbose=1, seed=42)
    
    print("\nRunning test again with same seed=42...")
    signals_df_2 = lstm_strategy(prices, window_size=30, epochs=20, verbose=1, seed=42)
    
    # Check reproducibility
    print("\n" + "="*70)
    print("REPRODUCIBILITY CHECK")
    print("="*70)
    predictions_match = np.allclose(
        signals_df_1['prediction'].dropna(), 
        signals_df_2['prediction'].dropna(),
        rtol=1e-5
    )
    trades_match = (signals_df_1['trade'] == signals_df_2['trade']).all()
    
    print(f"Predictions match: {predictions_match}")
    print(f"Trades match: {trades_match}")
    
    if predictions_match and trades_match:
        print("STABLE: Results are reproducible with same seed!")
    else:
        print("WARNING: Results differ (check CUDNN settings)")
    
    print("\nSample output:")
    print(signals_df_1.tail(10))