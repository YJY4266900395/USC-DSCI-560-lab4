"""
DSCI-560 Lab 4 - Multi-Stock LSTM Trading System
Main entry point for the complete pipeline.

Usage:
    python main.py [options]

Pipeline:
    1. Load stock price data (or fetch if not available)
    2. Train LSTM models for each stock
    3. Generate trading signals with confidence scores
    4. Backtest portfolio with shared cash pool
    5. Calculate performance metrics
    6. Generate visualizations
"""

import os
import argparse
import pandas as pd
from datetime import datetime

from fetch_data import fetch_multiple_stocks
from strategy_lstm import lstm_strategy
from backtest_portfolio import backtest_portfolio
from metrics import compute_metrics, print_metrics
from plot import (
    plot_portfolio_value,
    plot_cash_vs_holdings,
    plot_stock_holdings,
    plot_all_stocks_signals
)


def load_or_fetch_data(tickers, data_dir="data", start="2022-01-01", end="2024-12-31"):
    """
    Load stock data from CSV or fetch from Yahoo Finance if not available
    
    Returns:
        dict: {ticker: price_series}
    """
    price_data = {}
    missing_tickers = []
    
    # Try to load existing data
    for ticker in tickers:
        csv_path = os.path.join(data_dir, f"{ticker}_prices.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                price_data[ticker] = df['Close'].astype(float)
                print(f"[LOADED] {ticker} from {csv_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load {ticker}: {e}")
                missing_tickers.append(ticker)
        else:
            missing_tickers.append(ticker)
    
    # Fetch missing data
    if missing_tickers:
        print(f"\n[FETCHING] Missing tickers: {', '.join(missing_tickers)}")
        fetch_multiple_stocks(missing_tickers, start, end, out_dir=data_dir)
        
        # Load newly fetched data
        for ticker in missing_tickers:
            csv_path = os.path.join(data_dir, f"{ticker}_prices.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                price_data[ticker] = df['Close'].astype(float)
    
    return price_data


def main():
    parser = argparse.ArgumentParser(
        description="DSCI-560 Lab 4: Multi-Stock LSTM Trading System"
    )
    
    # Data parameters
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "GOOGL"],
        help="Stock tickers to trade (default: AAPL MSFT GOOGL)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing stock data CSVs"
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Start date for data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        default="2024-12-31",
        help="End date for data (YYYY-MM-DD)"
    )
    
    # LSTM parameters
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="LSTM look-back window (default: 30)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="LSTM training epochs (default: 50)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Signal threshold for predicted return (default: 0.02 = 2%%)"
    )
    
    # Portfolio parameters
    parser.add_argument(
        "--cash",
        type=float,
        default=10000.0,
        help="Initial cash (default: 10000)"
    )
    parser.add_argument(
        "--base_alloc",
        type=float,
        default=0.20,
        help="Base allocation per trade (default: 0.20 = 20%%)"
    )
    parser.add_argument(
        "--conf_bonus",
        type=float,
        default=0.10,
        help="Max confidence bonus (default: 0.10 = 10%%)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="LSTM training verbosity (0=silent, 1=progress, 2=detailed)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" " * 15 + "DSCI-560 LAB 4: LSTM TRADING SYSTEM")
    print("="*70)
    print(f"Tickers:     {', '.join(args.tickers)}")
    print(f"Date Range:  {args.start} to {args.end}")
    print(f"Initial Cash: ${args.cash:,.2f}")
    print(f"LSTM Window: {args.window} days | Epochs: {args.epochs}")
    print(f"Allocation:  {args.base_alloc:.0%} base + up to {args.conf_bonus:.0%} bonus")
    print("="*70 + "\n")
    
    # Step 1: Load or fetch data
    print("STEP 1: Loading stock data...")
    price_data = load_or_fetch_data(
        tickers=args.tickers,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end
    )
    
    if len(price_data) == 0:
        print("[ERROR] No stock data available. Exiting.")
        return
    
    print(f"[OK] Loaded {len(price_data)} stocks\n")
    
    # Step 2: Generate LSTM signals for each stock
    print("STEP 2: Generating LSTM trading signals...")
    signals_dict = {}
    
    for ticker, price_series in price_data.items():
        print(f"\nProcessing {ticker}...")
        signals_df = lstm_strategy(
            price=price_series,
            window_size=args.window,
            epochs=args.epochs,
            threshold=args.threshold,
            verbose=args.verbose
        )
        signals_dict[ticker] = signals_df
    
    print("\n[OK] All signals generated\n")
    
    # Step 3: Backtest portfolio
    print("STEP 3: Running portfolio backtest...")
    portfolio_df, trades_df = backtest_portfolio(
        signals_dict=signals_dict,
        initial_cash=args.cash,
        base_allocation=args.base_alloc,
        confidence_bonus=args.conf_bonus
    )
    
    # Step 4: Calculate metrics
    print("STEP 4: Calculating performance metrics...")
    metrics = compute_metrics(portfolio_df, trading_days=252, rf_annual=0.0)
    print_metrics(metrics, title="Portfolio Performance Metrics")
    
    # Step 5: Generate plots
    print("STEP 5: Generating visualizations...")
    plot_portfolio_value(portfolio_df, f"{args.output_dir}/portfolio_value.png")
    plot_cash_vs_holdings(portfolio_df, f"{args.output_dir}/cash_vs_holdings.png")
    plot_stock_holdings(portfolio_df, args.tickers, f"{args.output_dir}/stock_holdings.png")
    plot_all_stocks_signals(signals_dict, args.output_dir)
    
    # Step 6: Save results
    print("\nSTEP 6: Saving results...")
    portfolio_df.to_csv(f"{args.output_dir}/portfolio_value.csv")
    print(f"[SAVED] {args.output_dir}/portfolio_value.csv")
    
    if len(trades_df) > 0:
        trades_df.to_csv(f"{args.output_dir}/trades.csv", index=False)
        print(f"[SAVED] {args.output_dir}/trades.csv")
    
    # Save metrics to text file
    with open(f"{args.output_dir}/metrics_summary.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("DSCI-560 LAB 4: PORTFOLIO PERFORMANCE METRICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Strategy: LSTM with Confidence-Based Allocation\n")
        f.write(f"Tickers: {', '.join(args.tickers)}\n")
        f.write(f"Date Range: {args.start} to {args.end}\n")
        f.write(f"Initial Cash: ${args.cash:,.2f}\n\n")
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'return' in key.lower() or 'volatility' in key.lower() or 'drawdown' in key.lower():
                    f.write(f"{key.replace('_', ' ').title():<25} {value:>15.2%}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title():<25} {value:>15.2f}\n")
            else:
                f.write(f"{key.replace('_', ' ').title():<25} {value:>15}\n")
        f.write("="*70 + "\n")
    
    print(f"[SAVED] {args.output_dir}/metrics_summary.txt")
    
    # Final summary
    print("\n" + "="*70)
    print(" " * 25 + "EXECUTION COMPLETE")
    print("="*70)
    print(f"Total Return:     {metrics['total_return']:>10.2%}")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
