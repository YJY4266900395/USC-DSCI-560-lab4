"""
File: metrics.py
Purpose:
  Compute performance metrics for trading strategy:
    - Final Portfolio Value
    - Total Return
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio (requested by Lab4)

Notes:
  - Use daily returns from portfolio total value
  - Risk-free rate set to 0 by default for simplicity
"""

import numpy as np
import pandas as pd

def compute_metrics(portfolio_df: pd.DataFrame, trading_days: int = 252, rf_annual: float = 0.0) -> dict:
    values = portfolio_df["total_value"].astype(float)

    if len(values) < 3:
        raise ValueError("Not enough data points to compute metrics.")

    daily_returns = values.pct_change().dropna()

    final_value = float(values.iloc[-1])
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)

    # annualized return (approx)
    n = len(values) - 1
    annualized_return = float((values.iloc[-1] / values.iloc[0]) ** (trading_days / max(n, 1)) - 1.0)

    annualized_vol = float(daily_returns.std(ddof=1) * np.sqrt(trading_days))

    # Sharpe ratio
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = daily_returns - rf_daily
    if excess.std(ddof=1) == 0:
        sharpe = float("nan")
    else:
        sharpe = float((excess.mean() / excess.std(ddof=1)) * np.sqrt(trading_days))

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe
    }
