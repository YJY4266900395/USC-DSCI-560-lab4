"""
File: backtest_portfolio.py
Purpose:
  Multi-stock portfolio backtesting with shared cash pool.
  Implements "base allocation + confidence bonus" strategy.

Key Features:
  - Shared cash pool across all stocks
  - Base allocation: 20% of available cash per buy signal
  - Confidence bonus: up to +10% based on LSTM confidence
  - Long-only strategy (no short selling)

Trading Logic:
  - When BUY signal: allocate (20% + confidence*10%) of available cash
  - When SELL signal: sell all shares of that stock
  - Multiple stocks can be held simultaneously
  
Output:
  portfolio_df: time series of portfolio value and holdings
  trades_df: record of all trades
"""

import pandas as pd
import numpy as np


def backtest_portfolio(
    signals_dict: dict,
    initial_cash: float = 10000.0,
    base_allocation: float = 0.20,  # 20% base
    confidence_bonus: float = 0.10   # up to 10% bonus
):
    """
    Backtest multi-stock portfolio with confidence-based allocation
    
    Args:
        signals_dict: {ticker: DataFrame with ['price', 'trade', 'confidence']}
        initial_cash: starting capital
        base_allocation: base % of cash to allocate on BUY (e.g., 0.20 = 20%)
        confidence_bonus: max bonus % based on confidence (e.g., 0.10 = 10%)
    
    Returns:
        portfolio_df: DataFrame with portfolio value over time
        trades_df: DataFrame with all trade records
    """
    
    # Initialize
    cash = float(initial_cash)
    holdings = {ticker: 0.0 for ticker in signals_dict.keys()}
    
    # Get all unique dates (union of all stocks' dates)
    all_dates = sorted(set().union(*[set(df.index) for df in signals_dict.values()]))
    
    trades = []
    portfolio_rows = []
    
    print(f"\n{'='*70}")
    print(f"Starting Portfolio Backtest")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print(f"Stocks: {', '.join(signals_dict.keys())}")
    print(f"Base allocation: {base_allocation:.0%} | Confidence bonus: up to {confidence_bonus:.0%}")
    print(f"{'='*70}\n")
    
    # Iterate through each date
    for date in all_dates:
        # Collect all signals for this date
        day_signals = {}
        for ticker, df in signals_dict.items():
            if date in df.index:
                day_signals[ticker] = {
                    'price': float(df.loc[date, 'price']),
                    'trade': float(df.loc[date, 'trade']),
                    'confidence': float(df.loc[date, 'confidence']) if not np.isnan(df.loc[date, 'confidence']) else 0.0
                }
        
        # Process SELL signals first (to free up cash)
        for ticker, signal in day_signals.items():
            if signal['trade'] == -1.0 and holdings[ticker] > 0:
                # SELL all shares
                shares = holdings[ticker]
                price = signal['price']
                proceeds = shares * price
                
                cash += proceeds
                holdings[ticker] = 0.0
                
                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'amount': proceeds,
                    'confidence': signal['confidence']
                })
        
        # Process BUY signals
        buy_signals = [(ticker, signal) for ticker, signal in day_signals.items() 
                       if signal['trade'] == 1.0]
        
        if buy_signals:
            # Calculate allocation for each buy signal
            for ticker, signal in buy_signals:
                price = signal['price']
                confidence = signal['confidence']
                
                # Calculate allocation: base + confidence bonus
                # allocation_pct = base_allocation + (confidence * confidence_bonus)
                allocation_pct = base_allocation + (confidence * confidence_bonus)
                allocation_pct = min(allocation_pct, 0.50)  # cap at 50% to avoid over-concentration
                
                allocation_amount = cash * allocation_pct
                
                if price > 0 and allocation_amount >= price:
                    shares_to_buy = int(allocation_amount // price)
                    
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        
                        cash -= cost
                        holdings[ticker] += shares_to_buy
                        
                        trades.append({
                            'date': date,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': price,
                            'amount': cost,
                            'confidence': confidence
                        })
        
        # Calculate portfolio value for this date
        current_prices = {ticker: signal['price'] 
                         for ticker, signal in day_signals.items()}
        
        # For stocks not trading today, use last known price
        for ticker in holdings.keys():
            if ticker not in current_prices and ticker in signals_dict:
                # Find last available price
                df = signals_dict[ticker]
                past_prices = df[df.index <= date]['price']
                if len(past_prices) > 0:
                    current_prices[ticker] = float(past_prices.iloc[-1])
                else:
                    current_prices[ticker] = 0.0
        
        holdings_value = sum(holdings[ticker] * current_prices.get(ticker, 0) 
                            for ticker in holdings.keys())
        total_value = cash + holdings_value
        
        # Record portfolio state
        portfolio_rows.append({
            'date': date,
            'cash': cash,
            'holdings_value': holdings_value,
            'total_value': total_value,
            **{f'{ticker}_shares': holdings[ticker] for ticker in holdings.keys()}
        })
    
    # Create DataFrames
    portfolio_df = pd.DataFrame(portfolio_rows).set_index('date')
    trades_df = pd.DataFrame(trades)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Backtest Complete")
    print(f"Final Portfolio Value: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(portfolio_df['total_value'].iloc[-1] / initial_cash - 1):.2%}")
    print(f"Total Trades: {len(trades_df)}")
    if len(trades_df) > 0:
        print(f"  BUY:  {(trades_df['action'] == 'BUY').sum()}")
        print(f"  SELL: {(trades_df['action'] == 'SELL').sum()}")
    print(f"{'='*70}\n")
    
    return portfolio_df, trades_df


def backtest_portfolio_simple(
    signals_dict: dict,
    initial_cash: float = 10000.0,
    allocation_per_trade: float = 0.25  # 25% of cash per trade
):
    """
    Simplified version: fixed allocation without confidence bonus
    Use this if confidence-based allocation seems too complex
    """
    return backtest_portfolio(
        signals_dict,
        initial_cash,
        base_allocation=allocation_per_trade,
        confidence_bonus=0.0  # no bonus
    )


# Example usage
if __name__ == "__main__":
    # Test with mock data
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    
    # Mock signals for 3 stocks
    signals_dict = {}
    for ticker in ['AAPL', 'MSFT', 'GOOGL', 'NVDA']:
        signals_dict[ticker] = pd.DataFrame({
            'price': 100 + np.cumsum(np.random.randn(100)),
            'trade': np.random.choice([0, 1, -1], size=100, p=[0.90, 0.05, 0.05]),
            'confidence': np.random.rand(100)
        }, index=dates)
    
    portfolio_df, trades_df = backtest_portfolio(signals_dict, initial_cash=10000)
    
    print("\nPortfolio Summary:")
    print(portfolio_df.tail())
    print("\nRecent Trades:")
    print(trades_df.tail())
