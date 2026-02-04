"""
DSCI-560 Lab 4 - Main Entry
You can run directly: python src/main.py
"""

import os
import argparse
import pandas as pd

from strategy_ma import moving_average_signals
from backtest import backtest_long_only
from metrics import compute_metrics
from plot import plot_portfolio_value, plot_price_and_signals


def load_price_series_csv(
    path: str,
    date_col: str = "Date",
    price_col: str = "Close"
) -> pd.Series:
    """
    Robust CSV loader (guaranteed to remove junk rows like:
    Date = NaN and price columns = 'AAPL').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    # ---- date column detection ----
    if date_col not in df.columns:
        for candidate in ["date", "Datetime", "datetime", "timestamp", "Time", "time"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col not in df.columns:
            raise ValueError(f"Date column not found. Columns: {list(df.columns)}")

    # convert date FIRST, then drop invalid rows FIRST (this removes the 'AAPL' junk row)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # remove duplicate dates if any
    df = df.drop_duplicates(subset=[date_col])

    # sort & index
    df = df.sort_values(date_col).set_index(date_col)

    # price column detection
    if price_col not in df.columns:
        for candidate in ["Close", "close", "Adj Close", "AdjClose", "adj_close", "price"]:
            if candidate in df.columns:
                price_col = candidate
                break
        if price_col not in df.columns:
            raise ValueError(f"Price column not found. Columns: {list(df.columns)}")

    # debug AFTER cleaning (so first row will be valid)
    print("[DEBUG] CSV columns:", list(df.columns))
    print("[DEBUG] first row:", df.iloc[0].to_dict())

    # ---- numeric conversion ----
    price = pd.to_numeric(df[price_col], errors="coerce").dropna()

    if len(price) < 5:
        raise ValueError(
            f"Not enough valid numeric prices after cleaning. "
            f"price_col={price_col}, rows={len(price)}"
        )

    return price.astype(float)



def main():
    parser = argparse.ArgumentParser(description="DSCI-560 Lab 4 Trading System")
    parser.add_argument(
        "--csv",
        default="data/AAPL_prices.csv",
        help="Path to stock price CSV (default: data/AAPL_prices.csv)"
    )

    parser.add_argument("--date_col", default="Date")
    parser.add_argument("--price_col", default="Close")
    parser.add_argument("--short", type=int, default=5)
    parser.add_argument("--long", type=int, default=20)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--rf", type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # 1. Load data
    price = load_price_series_csv(
        args.csv,
        date_col=args.date_col,
        price_col=args.price_col
    )

    # 2. Strategy
    signals = moving_average_signals(price, short=args.short, long=args.long)

    # 3. Backtest
    portfolio_df, trades_df = backtest_long_only(
        signals, initial_cash=args.cash
    )

    # 4. Metrics
    metrics = compute_metrics(portfolio_df, trading_days=252, rf_annual=args.rf)

    # 5. Save outputs
    portfolio_df.to_csv("outputs/portfolio_value.csv")
    trades_df.to_csv("outputs/trades.csv", index=False)

    plot_portfolio_value(portfolio_df, "outputs/portfolio_value.png")
    plot_price_and_signals(signals, "outputs/price_ma_signals.png")

    # 6. Print results
    print("\n===== Lab 4 Results =====")
    print(f"CSV used: {args.csv}")
    print(f"MA windows: short={args.short}, long={args.long}")
    print(f"Initial cash: {args.cash}")
    print("-------------------------")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("-------------------------")
    print("Outputs saved in ./outputs/")


if __name__ == "__main__":
    main()
