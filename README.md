1. Overview
In this lab, we analyze stock price data and build an algorithmic trading strategy that generates buy/sell signals.
We then test the strategy in a mock trading environment and evaluate performance using standard portfolio metrics.

2. Data Source
We download historical stock price data from Yahoo Finance using the yfinance Python library.
The dataset includes Date and OHLCV fields (Open, High, Low, Close, Adj Close, Volume).
During loading, we clean the data by:
- coercing the Date column to datetime and dropping invalid rows
- converting price columns to numeric values and removing non-numeric entries

3. Trading Algorithm (Moving Average Crossover)
We implement a Moving Average (MA) crossover strategy:
- Compute short MA (default: 5-day)
- Compute long MA (default: 20-day)
Signals:
- BUY when short MA crosses above long MA (signal changes 0 → 1)
- SELL when short MA crosses below long MA (signal changes 1 → 0)

Rationale:
MA crossover is a standard technical analysis baseline that is interpretable and easy to validate.

4. Mock Trading Environment
We simulate trading with:
- Initial cash: $10,000
- Long-only trading (no short selling)
- On BUY: purchase the maximum number of whole shares possible
- On SELL: sell all shares (close the position)
Portfolio value is tracked over time:
total_value = cash + shares × price

Trade history is saved to outputs/trades.csv.

5. Performance Metrics
We evaluate the strategy using:
- Final portfolio value
- Total return
- Annualized return
- Annualized volatility
- Sharpe ratio (risk-adjusted return)

Example run (AAPL, short=5, long=20, initial_cash=10000):
- final_value: 14750.01
- total_return: 0.4750
- annualized_return: 0.1393
- annualized_volatility: 0.1788
- sharpe_ratio: 0.8183

