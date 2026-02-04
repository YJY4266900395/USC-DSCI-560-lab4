"""
File: strategy_ma.py
Purpose:
  Implement Moving Average (MA) crossover strategy to generate buy/sell signals.

Logic:
  - Compute short MA and long MA
  - signal = 1 (long) if short MA > long MA else 0 (flat)
  - trade = signal change:
      +1 => BUY signal (0 -> 1)
      -1 => SELL signal (1 -> 0)

Output:
  DataFrame columns: price, ma_short, ma_long, signal, trade
"""

import pandas as pd

def moving_average_signals(price: pd.Series, short: int = 5, long: int = 20) -> pd.DataFrame:
    if short >= long:
        raise ValueError("short window must be smaller than long window")

    df = pd.DataFrame({"price": price}).copy()

    df["ma_short"] = df["price"].rolling(short).mean()
    df["ma_long"] = df["price"].rolling(long).mean()

    df["signal"] = (df["ma_short"] > df["ma_long"]).astype(int)
    df["trade"] = df["signal"].diff().fillna(0)  # +1 buy, -1 sell

    return df
