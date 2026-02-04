"""
File: 3_backtest.py
Purpose:
  Create a mock trading environment to test trading algorithm performance (Lab4 Part2).

Assumptions (simple & safe for grading):
  - Long-only (no short selling)
  - Buy: when trade == +1, buy as many whole shares as possible
  - Sell: when trade == -1, sell all shares
  - No transaction fee, no slippage (can be added later)

Outputs:
  portfolio_df: time index, cash, shares, price, total_value
  trades_df: each trade record (date, action, shares, price, amount)
"""

import pandas as pd

def backtest_long_only(signals_df: pd.DataFrame, initial_cash: float = 10000.0):
    cash = float(initial_cash)
    shares = 0.0

    trades = []
    portfolio_rows = []

    for dt, row in signals_df.iterrows():
        price = float(row["price"])
        trade = float(row["trade"])

        # BUY signal
        if trade == 1.0:
            if price > 0 and cash >= price:
                buy_shares = int(cash // price)  # whole shares
                if buy_shares > 0:
                    cost = buy_shares * price
                    cash -= cost
                    shares += buy_shares
                    trades.append([dt, "BUY", buy_shares, price, cost])

        # SELL signal
        elif trade == -1.0:
            if shares > 0:
                proceeds = shares * price
                trades.append([dt, "SELL", shares, price, proceeds])
                cash += proceeds
                shares = 0.0

        total_value = cash + shares * price
        portfolio_rows.append([dt, cash, shares, price, total_value])

    portfolio_df = pd.DataFrame(
        portfolio_rows,
        columns=["date", "cash", "shares", "price", "total_value"]
    ).set_index("date")

    trades_df = pd.DataFrame(trades, columns=["date", "action", "shares", "price", "amount"])

    return portfolio_df, trades_df
