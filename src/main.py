"""main.py

DSCI-560 Lab 4 - Multi-Stock Trading System

This version focuses on *stability + improved risk-adjusted returns*:
- Uses the STABLE LSTM implementation (fixed seed, grad clipping, LR scheduler, early stopping)
- Adds portfolio risk controls in the backtester (cooldown, stop-loss/take-profit, position cap, vol filter)
- Less aggressive default allocations (more stable, usually higher Sharpe)

Usage (examples):
  # LSTM portfolio (stable defaults)
  python src/main.py

  # Tune LSTM signal sensitivity
  python src/main.py --threshold 0.025 --epochs 60

  # If you only want MA baseline, run src/main_ma.py
"""

import os
import argparse
import pandas as pd
import time

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
    """Load stock data from CSV or fetch from Yahoo Finance if not available."""
    price_data = {}
    missing_tickers = []

    for ticker in tickers:
        csv_path = os.path.join(data_dir, f"{ticker}_prices.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)

                # Flatten multi-index columns if any
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]

                # Remove ticker prefixes if present
                df.columns = [str(col).replace(f"{ticker}_", "") for col in df.columns]

                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                close_col = next((col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()), None)
                if close_col is None:
                    close_col = next((col for col in df.columns if 'close' in col.lower()), None)

                if date_col is None or close_col is None:
                    raise ValueError(f"Bad columns in {csv_path}: {list(df.columns)}")

                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

                price_data[ticker] = df[close_col].astype(float)
                print(f"[LOADED] {ticker} from {csv_path} ({len(price_data[ticker])} rows)")
            except Exception as e:
                print(f"[ERROR] Failed to load {ticker}: {e}")
                missing_tickers.append(ticker)
        else:
            missing_tickers.append(ticker)

    if missing_tickers:
        print(f"\n[FETCHING] Missing tickers: {', '.join(missing_tickers)}")
        fetch_multiple_stocks(missing_tickers, start, end, out_dir=data_dir)

        for ticker in missing_tickers:
            csv_path = os.path.join(data_dir, f"{ticker}_prices.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
                df.columns = [str(col).replace(f"{ticker}_", "") for col in df.columns]

                date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                close_col = next((col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()), None)
                if close_col is None:
                    close_col = next((col for col in df.columns if 'close' in col.lower()), None)

                if date_col is None or close_col is None:
                    raise ValueError(f"Bad columns in {csv_path}: {list(df.columns)}")

                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

                price_data[ticker] = df[close_col].astype(float)
                print(f"[LOADED] {ticker} (newly fetched, {len(price_data[ticker])} rows)")
            except Exception as e:
                print(f"[ERROR] Failed to load newly fetched {ticker}: {e}")

    return price_data


def main():
    parser = argparse.ArgumentParser(description="DSCI-560 Lab 4: Multi-Stock Trading System (Stable)")

    # Data
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL", "NVDA"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")

    # LSTM signals
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--verbose", type=int, default=0)

    # Portfolio / backtest
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--base_alloc", type=float, default=0.15)
    parser.add_argument("--conf_bonus", type=float, default=0.10)
    parser.add_argument("--max_position_pct", type=float, default=0.35)
    parser.add_argument("--cooldown_days", type=int, default=5)
    parser.add_argument("--stop_loss", type=float, default=0.08)
    parser.add_argument("--take_profit", type=float, default=0.18)
    parser.add_argument("--vol_threshold", type=float, default=0.04)  # set 0 to disable
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--transaction_cost", type=float, default=0.0005)

    # Output
    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vol_thr = None if args.vol_threshold <= 0 else args.vol_threshold

    print("\n" + "="*70)
    print(" " * 14 + "DSCI-560 LAB 4: STABLE LSTM TRADING SYSTEM")
    print("="*70)
    print(f"Tickers:      {', '.join(args.tickers)}")
    print(f"Date Range:   {args.start} to {args.end}")
    print(f"Initial Cash: ${args.cash:,.2f}")
    print(f"LSTM: window={args.window}, epochs={args.epochs}, threshold={args.threshold:.2%}")
    print(f"Seed={args.seed}, lr={args.learning_rate:.6f}, patience={args.patience}")
    print(f"Allocation:   base={args.base_alloc:.0%} + bonus up to {args.conf_bonus:.0%}")
    print(f"Risk: max_pos={args.max_position_pct:.0%}, cooldown={args.cooldown_days}d, SL={args.stop_loss:.0%}, TP={args.take_profit:.0%}")
    print(f"Vol filter:   {'disabled' if vol_thr is None else f'{vol_thr:.2%} (lookback {args.vol_lookback}d)'}")
    print(f"Tx cost:      {args.transaction_cost:.2%}")
    print("="*70 + "\n")

    # Step 1: Data
    print("STEP 1: Loading stock data...")
    price_data = load_or_fetch_data(args.tickers, args.data_dir, args.start, args.end)
    if not price_data:
        print("[ERROR] No stock data available. Exiting.")
        return
    print(f"[OK] Loaded {len(price_data)} stocks\n")

    # Step 2: Signals (stable + reproducible)
    print("STEP 2: Generating LSTM trading signals...")
    signals_dict = {}
    for ticker, price_series in price_data.items():
        print(f"\nProcessing {ticker}...")
        signals_df = lstm_strategy(
            price=price_series,
            window_size=args.window,
            epochs=args.epochs,
            threshold=args.threshold,
            verbose=args.verbose,
            seed=args.seed,
            learning_rate=args.learning_rate,
            patience=args.patience
        )
        signals_dict[ticker] = signals_df
    print("\n[OK] All signals generated\n")

    # Step 3: Backtest with risk controls
    print("STEP 3: Running portfolio backtest...")
    portfolio_df, trades_df = backtest_portfolio(
        signals_dict=signals_dict,
        initial_cash=args.cash,
        base_allocation=args.base_alloc,
        confidence_bonus=args.conf_bonus,
        max_position_pct=args.max_position_pct,
        cooldown_days=args.cooldown_days,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        vol_lookback=args.vol_lookback,
        vol_threshold=vol_thr,
        transaction_cost=args.transaction_cost
    )

    # Step 4: Metrics
    print("STEP 4: Calculating performance metrics...")
    metrics = compute_metrics(portfolio_df, trading_days=252, rf_annual=0.0)
    print_metrics(metrics, title="Portfolio Performance Metrics")

    # Step 5: Plots
    print("STEP 5: Generating visualizations...")
    plot_portfolio_value(portfolio_df, f"{args.output_dir}/portfolio_value.png")
    plot_cash_vs_holdings(portfolio_df, f"{args.output_dir}/cash_vs_holdings.png")
    plot_stock_holdings(portfolio_df, args.tickers, f"{args.output_dir}/stock_holdings.png")
    plot_all_stocks_signals(signals_dict, args.output_dir)

    # Step 6: Save
    print("\nSTEP 6: Saving results...")
    portfolio_df.to_csv(f"{args.output_dir}/portfolio_value.csv")
    print(f"[SAVED] {args.output_dir}/portfolio_value.csv")

    if len(trades_df) > 0:
        trades_df.to_csv(f"{args.output_dir}/trades.csv", index=False)
        print(f"[SAVED] {args.output_dir}/trades.csv")

    with open(f"{args.output_dir}/metrics_summary.txt", "w") as f:
        f.write("="*70 + "\n")
        f.write("DSCI-560 LAB 4: PORTFOLIO PERFORMANCE METRICS\n")
        f.write("="*70 + "\n\n")
        f.write("Strategy: STABLE LSTM + Risk Controls\n")
        f.write(f"Tickers: {', '.join(args.tickers)}\n")
        f.write(f"Date Range: {args.start} to {args.end}\n")
        f.write(f"Threshold: {args.threshold:.2%}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Initial Cash: ${args.cash:,.2f}\n\n")
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'return' in key.lower() or 'volatility' in key.lower() or 'drawdown' in key.lower():
                    f.write(f"{key.replace('_',' ').title():<25} {value:>15.2%}\n")
                else:
                    f.write(f"{key.replace('_',' ').title():<25} {value:>15.2f}\n")
            else:
                f.write(f"{key.replace('_',' ').title():<25} {value:>15}\n")
        f.write("="*70 + "\n")

    print(f"[SAVED] {args.output_dir}/metrics_summary.txt")

    print("\n" + "="*70)
    print(" " * 25 + "EXECUTION COMPLETE")
    print("="*70)
    print(f"Total Return:     {metrics['total_return']:>10.2%}")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total Execution Time: {time.time() - start:.2f} seconds")
