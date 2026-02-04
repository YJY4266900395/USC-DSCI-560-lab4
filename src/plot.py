"""
File: plot.py
Purpose:
  Create plots for demo/video:
    1) Portfolio total value over time
    2) Price with short/long moving averages and buy/sell markers

Outputs:
  outputs/portfolio_value.png
  outputs/price_ma_signals.png
"""

import os
import matplotlib.pyplot as plt

def plot_portfolio_value(portfolio_df, out_path="outputs/portfolio_value.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(portfolio_df.index, portfolio_df["total_value"])
    plt.title("Portfolio Total Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_price_and_signals(signals_df, out_path="outputs/price_ma_signals.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    buy_points = signals_df[signals_df["trade"] == 1.0]
    sell_points = signals_df[signals_df["trade"] == -1.0]

    plt.figure()
    plt.plot(signals_df.index, signals_df["price"], label="Price")
    plt.plot(signals_df.index, signals_df["ma_short"], label="MA Short")
    plt.plot(signals_df.index, signals_df["ma_long"], label="MA Long")

    plt.scatter(buy_points.index, buy_points["price"], marker="^", label="Buy")
    plt.scatter(sell_points.index, sell_points["price"], marker="v", label="Sell")

    plt.title("Price + Moving Averages + Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
