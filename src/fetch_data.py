"""
File: fetch_data.py
Purpose:
  Download multiple stock price data from Yahoo Finance and save to CSV files.
  
Usage:
  python fetch_data.py
  
Output:
  data/AAPL_prices.csv
  data/MSFT_prices.csv
  data/GOOGL_prices.csv
  data/NVDA_prices.csv
"""
import os
import yfinance as yf
import pandas as pd

def fetch_and_save(ticker: str, start: str, end: str, interval: str = "1d", out_dir: str = "data") -> str:
    """Download stock data and save to CSV"""
    os.makedirs(out_dir, exist_ok=True)

    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker symbol or date range.")

    df = df.reset_index()  # Date becomes a column
    out_path = os.path.join(out_dir, f"{ticker}_prices.csv")
    df.to_csv(out_path, index=False)

    print(f"[OK] Saved: {out_path}  rows={len(df)}  interval={interval}")
    return out_path


def fetch_multiple_stocks(tickers: list, start: str, end: str, interval: str = "1d", out_dir: str = "data") -> dict:
    """
    Download multiple stocks at once
    
    Returns:
        dict: {ticker: csv_path}
    """
    paths = {}
    for ticker in tickers:
        try:
            path = fetch_and_save(ticker, start, end, interval, out_dir)
            paths[ticker] = path
        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker}: {e}")
    
    return paths


if __name__ == "__main__":
    # Download 3 tech stocks for portfolio
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    
    print("=" * 60)
    print("Downloading stock data for portfolio...")
    print(f"Tickers: {', '.join(tickers)}")
    print("=" * 60)
    
    paths = fetch_multiple_stocks(
        tickers=tickers,
        start="2022-01-01",
        end="2024-12-31",
        interval="1d"
    )
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Total stocks: {len(paths)}")
    print("=" * 60)
