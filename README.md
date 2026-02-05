# DSCI-560 Lab 4: Multi-Stock LSTM Trading System

**Team Name:** Trojan Trio 
**Team Members:** Jinyao Yang (4266900395), Sen Pang (8598139533), Qianshu Peng (5063709968)

---

## Project Overview

This project implements an **LSTM-based algorithmic trading system** for multi-stock portfolio management. The system uses deep learning to predict stock prices and generates buy/sell signals with confidence scores. It simulates a realistic trading environment with a shared cash pool across multiple stocks.

### Key Features

1. **LSTM Price Prediction**: 2-layer LSTM network with dropout for time series forecasting
2. **Confidence-Based Allocation**: "Base + Bonus" strategy (20% base + up to 10% bonus based on confidence)
3. **Multi-Stock Portfolio**: Shared cash pool across 3+ stocks (AAPL, MSFT, GOOGL, NVDA)
4. **Comprehensive Metrics**: Total return, Sharpe ratio, max drawdown, annualized volatility
5. **Professional Visualizations**: Portfolio value, cash vs holdings, individual stock performance

---

## Installation

### Requirements

- Python 3.8+
- PyTorch
- Required packages:

```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib
```

### Quick Setup

```bash
# Clone or extract the project
cd USC-DSCI-560-LAB4

# Install dependencies
pip install -r src/requirements.txt

# Run the system
python src/main.py
```

---

## File Structure

```
lab4_project/
├── src/                    # Stock price data (CSV files)
│   ├── main.py                    # Main entry point
│   ├── fetch_data.py              # Stock data downloader
│   ├── strategy_lstm.py           # LSTM trading strategy
|   ├── backtest_portfolio.py      # Multi-stock backtesting engine
|   ├── metrics.py                 # Performance metrics calculator
|   ├── plot.py                    # Visualization module
├── data/                      # Stock price data (CSV files)
│   ├── AAPL_prices.csv
│   ├── MSFT_prices.csv
│   └── GOOGL_prices.csv
└── outputs/                   # Results and plots
    ├── portfolio_value.csv
    ├── trades.csv
    ├── metrics_summary.txt
    └── *.png (various plots)
```

---

## Usage

### Basic Usage

Run with default settings (AAPL, MSFT, GOOGL, NVDA from 2022-2024):

```bash
python main.py
```

### Advanced Usage

Customize parameters:

```bash
python main.py \
  --tickers AAPL MSFT GOOGL NVDA TSLA \
  --cash 20000 \
  --epochs 100 \
  --window 60 \
  --base_alloc 0.15 \
  --conf_bonus 0.15
```

### Parameter Explanations

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tickers` | Stock symbols to trade | AAPL MSFT GOOGL NVDA |
| `--cash` | Initial investment ($) | 10000 |
| `--window` | LSTM look-back days | 30 |
| `--epochs` | Training epochs | 50 |
| `--threshold` | Signal threshold (%) | 0.02 (2%) |
| `--base_alloc` | Base allocation (%) | 0.20 (20%) |
| `--conf_bonus` | Max confidence bonus (%) | 0.10 (10%) |

---

## Algorithm Explanation

### LSTM Architecture

```
Input Layer (30 days of prices)
    ↓
LSTM Layer 1 (50 units) + Dropout(0.2)
    ↓
LSTM Layer 2 (50 units) + Dropout(0.2)
    ↓
Dense Layer (1 output - next day price)
```

### Trading Strategy

**Signal Generation:**
- **BUY**: If predicted return > 2%
- **SELL**: If predicted return < -2%
- **HOLD**: Otherwise

**Allocation Strategy (Your Innovation!):**
```python
allocation = base_allocation + (confidence_score * bonus)
allocation = 20% + (confidence * 10%)

Example:
- Low confidence (0.3): 20% + (0.3 * 10%) = 23% of cash
- High confidence (0.8): 20% + (0.8 * 10%) = 28% of cash
```

This "base + bonus" approach:
- Ensures diversification (base allocation)
- Rewards strong predictions (confidence bonus)
- Limits risk (capped at 50% max)

### Portfolio Management

- **Shared Cash Pool**: All stocks compete for the same capital
- **Long-Only**: No short selling
- **No Transaction Costs**: Simplified for educational purposes
- **Whole Shares Only**: Realistic constraint

---

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | (Final Value - Initial Value) / Initial Value |
| **Annualized Return** | Compound annual growth rate |
| **Sharpe Ratio** | Risk-adjusted return (higher is better) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Annualized Volatility** | Standard deviation of returns |

---

## Expected Results

With default settings (2022-2024 data):

- **Total Return**: 40-60% (varies by market conditions)
- **Sharpe Ratio**: 1.0-1.5
- **Max Drawdown**: -15% to -25%

*Note: Results will vary based on the time period and market conditions.*

---

## Outputs

After running, check the `outputs/` directory for:

1. **portfolio_value.csv** - Time series of portfolio value
2. **trades.csv** - All buy/sell transactions
3. **metrics_summary.txt** - Performance metrics report
4. **portfolio_value.png** - Total value over time
5. **cash_vs_holdings.png** - Portfolio composition
6. **stock_holdings.png** - Individual stock shares
7. **[TICKER]_signals.png** - Price predictions and signals for each stock

---

## Troubleshooting

### "No data available for ticker"
- Check ticker symbol is correct
- Ensure internet connection for Yahoo Finance API
- Try different date range

### LSTM training is slow
- Reduce `--epochs` to 30-40
- Reduce `--window` to 20
- Use a GPU if available

### Out of memory error
- Reduce number of stocks
- Reduce `--window` size
- Use a machine with more RAM

---

## References

1. **LSTM Architecture**: Hochreiter & Schmidhuber (1997), "Long Short-Term Memory"
2. **Technical Analysis**: Murphy, J. (1999), "Technical Analysis of Financial Markets"
3. **Portfolio Theory**: Markowitz, H. (1952), "Portfolio Selection"
4. **Sharpe Ratio**: Sharpe, W. (1966), "Mutual Fund Performance"

---

## Future Improvements

1. Add transaction costs and slippage
2. Implement short selling capability
3. Use hybrid models (LSTM + ARIMA)
4. Add risk management (stop-loss, position sizing)
5. Real-time trading with live data feeds

---

## License

Educational project for DSCI-560 course.

---

**Last Updated:** February 2026
