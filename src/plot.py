"""
File: plot.py
Purpose:
  Visualization for multi-stock portfolio performance.

Plots:
  1. Portfolio total value over time
  2. Cash vs Holdings value over time
  3. Individual stock holdings over time
  4. Price predictions with buy/sell signals (per stock)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def plot_portfolio_value(portfolio_df, out_path="outputs/portfolio_value.png"):
    """Plot portfolio total value over time"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(portfolio_df.index, portfolio_df['total_value'], 
            linewidth=2, label='Total Value', color='#2E86AB')
    ax.fill_between(portfolio_df.index, portfolio_df['total_value'], 
                     alpha=0.3, color='#2E86AB')
    
    # Add initial value line
    initial_value = portfolio_df['total_value'].iloc[0]
    ax.axhline(y=initial_value, color='gray', linestyle='--', 
               linewidth=1, alpha=0.7, label=f'Initial: ${initial_value:,.0f}')
    
    ax.set_title('Portfolio Total Value Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {out_path}")


def plot_cash_vs_holdings(portfolio_df, out_path="outputs/cash_vs_holdings.png"):
    """Plot cash vs holdings value over time (stacked area)"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.fill_between(portfolio_df.index, 0, portfolio_df['cash'], 
                     label='Cash', alpha=0.7, color='#06A77D')
    ax.fill_between(portfolio_df.index, portfolio_df['cash'], 
                     portfolio_df['total_value'],
                     label='Holdings Value', alpha=0.7, color='#F77F00')
    
    ax.set_title('Portfolio Composition: Cash vs Holdings', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Value ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {out_path}")


def plot_stock_holdings(portfolio_df, tickers, out_path="outputs/stock_holdings.png"):
    """Plot individual stock holdings over time"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#E63946', '#2A9D8F', '#F4A261', '#264653', '#E76F51']
    
    for i, ticker in enumerate(tickers):
        col_name = f'{ticker}_shares'
        if col_name in portfolio_df.columns:
            ax.plot(portfolio_df.index, portfolio_df[col_name], 
                   label=ticker, linewidth=2, color=colors[i % len(colors)])
    
    ax.set_title('Stock Holdings Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Shares', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {out_path}")


def plot_price_with_signals(
    ticker: str,
    signals_df,
    out_path=None
):
    """
    Plot stock price with LSTM predictions and buy/sell signals
    
    Args:
        ticker: stock ticker symbol
        signals_df: DataFrame with ['price', 'prediction', 'trade', 'confidence']
        out_path: output file path
    """
    if out_path is None:
        out_path = f"outputs/{ticker}_signals.png"
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Top panel: Price and predictions
    ax1.plot(signals_df.index, signals_df['price'], 
            label='Actual Price', linewidth=2, color='#023047', alpha=0.8)
    ax1.plot(signals_df.index, signals_df['prediction'], 
            label='LSTM Prediction', linewidth=1.5, color='#FB8500', 
            linestyle='--', alpha=0.7)
    
    # Mark buy/sell signals
    buy_signals = signals_df[signals_df['trade'] == 1.0]
    sell_signals = signals_df[signals_df['trade'] == -1.0]
    
    if len(buy_signals) > 0:
        ax1.scatter(buy_signals.index, buy_signals['price'], 
                   marker='^', s=100, color='#06A77D', 
                   label='BUY Signal', zorder=5, edgecolors='black', linewidths=0.5)
    
    if len(sell_signals) > 0:
        ax1.scatter(sell_signals.index, sell_signals['price'], 
                   marker='v', s=100, color='#E63946', 
                   label='SELL Signal', zorder=5, edgecolors='black', linewidths=0.5)
    
    ax1.set_title(f'{ticker} - Price, Predictions & Trading Signals', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Confidence scores
    ax2.fill_between(signals_df.index, 0, signals_df['confidence'], 
                     alpha=0.6, color='#8338EC', label='Confidence')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_title('LSTM Confidence Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Confidence', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {out_path}")


def plot_all_stocks_signals(signals_dict, out_dir="outputs"):
    """Plot signals for all stocks in the portfolio"""
    for ticker, signals_df in signals_dict.items():
        plot_price_with_signals(
            ticker=ticker,
            signals_df=signals_df,
            out_path=f"{out_dir}/{ticker}_signals.png"
        )


# Example usage
if __name__ == "__main__":
    import pandas as pd
    
    # Mock portfolio data
    dates = pd.date_range('2022-01-01', periods=100, freq='D')
    portfolio_df = pd.DataFrame({
        'cash': 10000 - np.cumsum(np.random.rand(100) * 50),
        'holdings_value': np.cumsum(np.random.rand(100) * 100),
        'total_value': 10000 + np.cumsum(np.random.randn(100) * 50),
        'AAPL_shares': np.cumsum(np.random.choice([0, 1, -1], 100)),
        'MSFT_shares': np.cumsum(np.random.choice([0, 1, -1], 100)),
    }, index=dates)
    
    plot_portfolio_value(portfolio_df)
    plot_cash_vs_holdings(portfolio_df)
    plot_stock_holdings(portfolio_df, ['AAPL', 'MSFT'])
    
    print("\nTest plots generated in outputs/")
