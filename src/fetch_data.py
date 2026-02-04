"""
File: fetch_data.py
Purpose:
  Download stock price data from Yahoo Finance (via yfinance) and save to CSV.
Why needed:
  Lab4 requires stock price data. If you didn't collect data in Lab3, you can generate it here.

Output:
  data/<TICKER>_prices.csv  (contains Date, Open, High, Low, Close, Adj Close, Volume)
"""
import os
import yfinance as yf

def fetch_and_save(ticker: str, start: str, end: str, interval: str = "1d", out_dir: str = "data") -> str:
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

if __name__ == "__main__":
    fetch_and_save("AAPL", start="2022-01-01", end="2024-12-31", interval="1d")