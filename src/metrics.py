"""
File: metrics.py
Purpose:
  Compute performance metrics for trading strategy:
    - Final Portfolio Value
    - Total Return
    - Annualized Return
    - Annualized Volatility
    - Sharpe Ratio
    - Maximum Drawdown (NEW)

Notes:
  - Use daily returns from portfolio total value
  - Risk-free rate set to 0 by default for simplicity
"""

import numpy as np
import pandas as pd


def compute_metrics(portfolio_df: pd.DataFrame, trading_days: int = 252, rf_annual: float = 0.0) -> dict:
    """
    Calculate portfolio performance metrics
    
    Args:
        portfolio_df: DataFrame with 'total_value' column
        trading_days: number of trading days per year (default 252)
        rf_annual: annual risk-free rate (default 0.0)
    
    Returns:
        dict of metrics
    """
    values = portfolio_df["total_value"].astype(float)

    if len(values) < 3:
        raise ValueError("Not enough data points to compute metrics.")

    daily_returns = values.pct_change().dropna()

    # Basic metrics
    initial_value = float(values.iloc[0])
    final_value = float(values.iloc[-1])
    total_return = float(final_value / initial_value - 1.0)

    # Annualized return
    n = len(values) - 1
    annualized_return = float((final_value / initial_value) ** (trading_days / max(n, 1)) - 1.0)

    # Annualized volatility
    annualized_vol = float(daily_returns.std(ddof=1) * np.sqrt(trading_days))

    # Sharpe ratio
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = daily_returns - rf_daily
    if excess.std(ddof=1) == 0:
        sharpe = float("nan")
    else:
        sharpe = float((excess.mean() / excess.std(ddof=1)) * np.sqrt(trading_days))

    # Maximum Drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    return {
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown
    }


def print_metrics(metrics: dict, title="Portfolio Performance Metrics"):
    """Pretty print metrics"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    print(f"Initial Value:        ${metrics['initial_value']:>15,.2f}")
    print(f"Final Value:          ${metrics['final_value']:>15,.2f}")
    print(f"Total Return:          {metrics['total_return']:>15.2%}")
    print(f"Annualized Return:     {metrics['annualized_return']:>15.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:>15.2%}")
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>15.2f}")
    print(f"Maximum Drawdown:      {metrics['max_drawdown']:>15.2%}")
    print(f"{'='*70}\n")


# Example usage
if __name__ == "__main__":
    # Test with mock data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    portfolio_df = pd.DataFrame({
        'total_value': 10000 + np.cumsum(np.random.randn(500) * 50)
    }, index=dates)
    
    metrics = compute_metrics(portfolio_df)
    print_metrics(metrics)
