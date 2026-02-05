"""
DSCI-560 Lab 4 - Simple Moving Average (Baseline)
Purpose: Kick-start implementation with basic MA crossover strategy

This is our initial approach before moving to LSTM.
"""

import os
import pandas as pd
from strategy_ma import moving_average_signals
from backtest_portfolio import backtest_portfolio
from metrics import compute_metrics, print_metrics
from plot import plot_portfolio_value
import argparse

def load_price_data(ticker, data_dir="data"):
    """Load single stock price data"""
    csv_path = os.path.join(data_dir, f"{ticker}_prices.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Find date column
    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
    close_col = next((col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()), None)
    
    if close_col is None:
        close_col = next((col for col in df.columns if 'close' in col.lower()), None)
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()
    
    return df[close_col].astype(float)


def main():
    print("\n" + "="*70)
    print(" " * 15 + "DSCI-560 LAB 4: MOVING AVERAGE BASELINE")
    print("="*70)
    print("Strategy: Simple MA Crossover (5-day / 20-day)")
    print("Purpose: Kick-start implementation before advanced algorithms")
    print("="*70 + "\n")

    parser = argparse.ArgumentParser(
        description="DSCI-560 Lab 4: Moving Average Trading System"
    )
    
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Stock ticker to trade (default: AAPL)"
    )

    args = parser.parse_args()
    
    # Configuration
    ticker = args.ticker.upper()
    initial_cash = 10000.0
    short_window = 5
    long_window = 20
    
    # Create output directory
    os.makedirs("outputs_ma", exist_ok=True)
    
    # Step 1: Load data
    print(f"Loading {ticker} data...")
    price = load_price_data(ticker, data_dir="data")
    print(f"[OK] Loaded {len(price)} days of data\n")
    
    # Step 2: Generate MA signals
    print(f"Generating MA signals (short={short_window}, long={long_window})...")
    signals = moving_average_signals(price, short=short_window, long=long_window)
    
    # Add confidence (set to 0.5 for all MA signals - no confidence scoring)
    signals['confidence'] = 0.5
    
    n_buy = (signals['trade'] == 1).sum()
    n_sell = (signals['trade'] == -1).sum()
    print(f"[OK] Generated {n_buy} BUY signals, {n_sell} SELL signals\n")
    
    # Step 3: Backtest (single stock)
    print("Running backtest...")
    signals_dict = {ticker: signals}
    
    # Simple allocation: 80% of cash (no confidence bonus for MA baseline)
    portfolio_df, trades_df = backtest_portfolio(
        signals_dict=signals_dict,
        initial_cash=initial_cash,
        base_allocation=0.80,  # Use 80% for single stock
        confidence_bonus=0.0   # No bonus for MA
    )
    
    # Step 4: Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = compute_metrics(portfolio_df, trading_days=252, rf_annual=0.0)
    print_metrics(metrics, title="MA Strategy Performance")
    
    # Step 5: Save results
    portfolio_df.to_csv("outputs_ma/portfolio_value.csv")
    if len(trades_df) > 0:
        trades_df.to_csv("outputs_ma/trades.csv", index=False)
    
    plot_portfolio_value(portfolio_df, "outputs_ma/portfolio_value.png")
    
    print(f"\n[SAVED] Results saved to outputs_ma/")
    
    # Summary
    print("\n" + "="*70)
    print(" " * 25 + "MA BASELINE COMPLETE")
    print("="*70)
    print(f"Total Return:     {metrics['total_return']:>10.2%}")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print("="*70)
    
    print("\n")


if __name__ == "__main__":
    main()
