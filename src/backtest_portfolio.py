"""backtest_portfolio.py

Multi-stock portfolio backtesting with a shared cash pool.

This version is an improved / more stable backtester:
- Position cap per ticker (avoid over-concentration)
- Cooldown (avoid churn when signals flip frequently)
- Stop-loss / Take-profit (reduce drawdowns, stabilize returns)
- Optional volatility filter (avoid opening new positions in extreme volatility)
- Optional transaction cost (penalize over-trading slightly)

Expected input:
  signals_dict: {ticker: DataFrame indexed by date with columns:
      - price (float)
      - trade (+1 buy, -1 sell, 0 hold)
      - confidence (0~1, optional; if missing, treated as 0)

Output:
  portfolio_df: time series of portfolio value and holdings
  trades_df: record of all trades
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def _safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def backtest_portfolio(
    signals_dict: dict,
    initial_cash: float = 10000.0,
    base_allocation: float = 0.15,     # ✅ less aggressive default for stability
    confidence_bonus: float = 0.10,    # ✅ up to +10% bonus based on confidence
    max_position_pct: float = 0.35,    # ✅ cap single ticker exposure (of total equity)
    cooldown_days: int = 5,            # ✅ no new trade for N days after any trade
    stop_loss: float = 0.08,           # ✅ 8% stop-loss
    take_profit: float = 0.18,         # ✅ 18% take-profit
    vol_lookback: int = 20,            # ✅ volatility lookback (days)
    vol_threshold: float | None = 0.04,# ✅ daily vol threshold; set None to disable
    transaction_cost: float = 0.0005   # ✅ 5 bps per trade (set 0 to disable)
):
    """Backtest a multi-stock long-only portfolio with risk controls."""

    if not signals_dict:
        raise ValueError("signals_dict is empty.")

    cash = float(initial_cash)
    holdings = {ticker: 0.0 for ticker in signals_dict.keys()}

    # Track entry price and last trade date for cooldown
    entry_price = {ticker: None for ticker in signals_dict.keys()}
    last_trade_date = {ticker: None for ticker in signals_dict.keys()}

    # Union of all dates
    all_dates = sorted(set().union(*[set(df.index) for df in signals_dict.values()]))

    trades = []
    portfolio_rows = []

    print(f"\n{'='*70}")
    print("Starting Portfolio Backtest (Improved Stability)")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print(f"Stocks: {', '.join(signals_dict.keys())}")
    print(f"Allocation: base={base_allocation:.0%} | bonus up to={confidence_bonus:.0%}")
    print(f"Risk controls: max_pos={max_position_pct:.0%}, cooldown={cooldown_days}d, "
          f"SL={stop_loss:.0%}, TP={take_profit:.0%}, tx_cost={transaction_cost:.2%}")
    if vol_threshold is not None:
        print(f"Vol filter: lookback={vol_lookback}d, threshold={vol_threshold:.2%} (daily std)")
    else:
        print("Vol filter: disabled")
    print(f"{'='*70}\n")

    # Precompute rolling volatility per ticker if enabled
    rolling_vol = {}
    if vol_threshold is not None:
        for ticker, df in signals_dict.items():
            if 'price' in df.columns:
                r = df['price'].astype(float).pct_change()
                rolling_vol[ticker] = r.rolling(vol_lookback).std()

    # Helper to get last known price up to date
    def last_price(ticker: str, date) -> float:
        df = signals_dict[ticker]
        past = df[df.index <= date]['price']
        return float(past.iloc[-1]) if len(past) else 0.0

    for date in all_dates:
        # Collect day signals
        day_signals = {}
        for ticker, df in signals_dict.items():
            if date in df.index:
                row = df.loc[date]
                day_signals[ticker] = {
                    'price': _safe_float(row.get('price', np.nan), 0.0),
                    'trade': _safe_float(row.get('trade', 0.0), 0.0),
                    'confidence': _safe_float(row.get('confidence', 0.0), 0.0)
                }

        # Build current prices map (use last known for missing)
        current_prices = {}
        for ticker in holdings.keys():
            if ticker in day_signals:
                current_prices[ticker] = float(day_signals[ticker]['price'])
            else:
                current_prices[ticker] = last_price(ticker, date)

        # Current total equity
        holdings_value = sum(holdings[t] * current_prices.get(t, 0.0) for t in holdings.keys())
        total_equity = cash + holdings_value

        # 1) Risk controls: stop-loss / take-profit (forced sell)
        for ticker in list(holdings.keys()):
            if holdings[ticker] <= 0:
                continue
            ep = entry_price[ticker]
            if ep is None or ep <= 0:
                continue
            price = current_prices.get(ticker, 0.0)
            if price <= 0:
                continue

            pnl = (price - ep) / ep
            if pnl <= -stop_loss or pnl >= take_profit:
                shares = holdings[ticker]
                proceeds = shares * price
                fee = proceeds * transaction_cost
                cash += (proceeds - fee)
                holdings[ticker] = 0.0
                entry_price[ticker] = None
                last_trade_date[ticker] = date

                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'amount': proceeds,
                    'fee': fee,
                    'confidence': np.nan,
                    'reason': 'STOP_LOSS' if pnl <= -stop_loss else 'TAKE_PROFIT'
                })

        # 2) Process explicit SELL signals (after forced sells)
        for ticker, signal in day_signals.items():
            if signal['trade'] == -1.0 and holdings[ticker] > 0:
                # cooldown check for churn
                ltd = last_trade_date[ticker]
                if ltd is not None and (date - ltd).days < cooldown_days:
                    continue

                shares = holdings[ticker]
                price = signal['price']
                proceeds = shares * price
                fee = proceeds * transaction_cost

                cash += (proceeds - fee)
                holdings[ticker] = 0.0
                entry_price[ticker] = None
                last_trade_date[ticker] = date

                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'amount': proceeds,
                    'fee': fee,
                    'confidence': signal['confidence'],
                    'reason': 'SIGNAL'
                })

        # Recompute equity after sells
        holdings_value = sum(holdings[t] * current_prices.get(t, 0.0) for t in holdings.keys())
        total_equity = cash + holdings_value

        # 3) Process BUY signals
        buy_candidates = [(t, s) for t, s in day_signals.items() if s['trade'] == 1.0]
        if buy_candidates and cash > 0:
            # Equal-split cash across BUY signals for stability
            cash_per_signal = cash / max(len(buy_candidates), 1)

            for ticker, signal in buy_candidates:
                # cooldown check
                ltd = last_trade_date[ticker]
                if ltd is not None and (date - ltd).days < cooldown_days:
                    continue

                # volatility filter
                if vol_threshold is not None and ticker in rolling_vol:
                    v = rolling_vol[ticker].reindex([date]).iloc[0] if date in rolling_vol[ticker].index else np.nan
                    if not np.isnan(v) and v > vol_threshold:
                        continue

                price = signal['price']
                if price <= 0:
                    continue

                confidence = float(signal['confidence'])
                alloc_pct = base_allocation + (confidence * confidence_bonus)
                alloc_pct = min(max(alloc_pct, 0.0), 0.50)  # keep a hard cap

                # desired allocation for this signal, but using the per-signal cash bucket
                desired = cash_per_signal * alloc_pct

                # position cap based on total equity
                max_pos_value = total_equity * max_position_pct
                current_pos_value = holdings[ticker] * price
                remaining_capacity = max(0.0, max_pos_value - current_pos_value)

                alloc_amount = min(desired, remaining_capacity, cash)

                if alloc_amount < price:
                    continue

                shares_to_buy = int(alloc_amount // price)
                if shares_to_buy <= 0:
                    continue

                cost = shares_to_buy * price
                fee = cost * transaction_cost
                total_cost = cost + fee

                if total_cost > cash:
                    # adjust shares down if needed
                    shares_to_buy = int(cash // (price * (1 + transaction_cost)))
                    if shares_to_buy <= 0:
                        continue
                    cost = shares_to_buy * price
                    fee = cost * transaction_cost
                    total_cost = cost + fee

                cash -= total_cost
                holdings[ticker] += shares_to_buy

                # record entry price (if opening a new position)
                if entry_price[ticker] is None and holdings[ticker] > 0:
                    entry_price[ticker] = price
                last_trade_date[ticker] = date

                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'amount': cost,
                    'fee': fee,
                    'confidence': confidence,
                    'reason': 'SIGNAL'
                })

        # Record portfolio state
        holdings_value = sum(holdings[t] * current_prices.get(t, 0.0) for t in holdings.keys())
        total_value = cash + holdings_value

        portfolio_rows.append({
            'date': date,
            'cash': cash,
            'holdings_value': holdings_value,
            'total_value': total_value,
            **{f'{t}_shares': holdings[t] for t in holdings.keys()}
        })

    portfolio_df = pd.DataFrame(portfolio_rows).set_index('date')
    trades_df = pd.DataFrame(trades)

    print(f"\n{'='*70}")
    print("Backtest Complete")
    print(f"Final Portfolio Value: ${portfolio_df['total_value'].iloc[-1]:,.2f}")
    print(f"Total Return: {(portfolio_df['total_value'].iloc[-1] / initial_cash - 1):.2%}")
    print(f"Total Trades: {len(trades_df)}")
    if len(trades_df) > 0:
        print(f"  BUY:  {(trades_df['action'] == 'BUY').sum()}")
        print(f"  SELL: {(trades_df['action'] == 'SELL').sum()}")
    print(f"{'='*70}\n")

    return portfolio_df, trades_df
