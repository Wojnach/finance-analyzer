"""Prepare OHLCV training data for Kronos fine-tuning.

Collects max historical data for all tickers and writes CSVs in the
format Kronos expects: timestamps, open, high, low, close, volume, amount
"""

import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
try:
    os.nice(19)
except (OSError, AttributeError):
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "kronos_training"

# Tickers grouped by source
BINANCE_SPOT = ["BTCUSDT", "ETHUSDT"]
BINANCE_FAPI = ["XAUUSDT", "XAGUSDT"]
ALPACA_TICKERS = [
    "PLTR", "NVDA", "AMD", "GOOGL", "AMZN", "AAPL", "AVGO", "META",
    "MU", "SOUN", "SMCI", "TSM", "TTWO", "VRT", "LMT", "MSTR",
]

def fetch_binance_full(symbol, interval="1h", fapi=False):
    """Fetch max Binance history by paginating backwards."""
    import requests

    base = "https://fapi.binance.com" if fapi else "https://api.binance.com"
    endpoint = f"{base}/fapi/v1/klines" if fapi else f"{base}/api/v3/klines"

    all_candles = []
    end_time = None

    for batch in range(50):  # Max 50 pages = 50,000 candles
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end_time:
            params["endTime"] = end_time

        resp = requests.get(endpoint, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"    API error {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        if not data:
            break

        all_candles = data + all_candles
        end_time = data[0][0] - 1  # Before first candle of this batch

        if len(data) < 1000:
            break  # No more data

        time.sleep(0.1)  # Rate limit

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamps"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["amount"] = df["quote_volume"].astype(float)

    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].drop_duplicates("timestamps")


def fetch_yfinance(ticker, interval="1h"):
    """Fetch max history via yfinance."""
    try:
        import yfinance as yf

        # yfinance max history for 1h is ~730 days
        tk = yf.Ticker(ticker)
        df = tk.history(period="max", interval=interval)

        if df.empty:
            return None

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamps", "Datetime": "timestamps",
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })

        if "timestamps" not in df.columns:
            df["timestamps"] = df.index

        df["amount"] = df["volume"] * df["close"]  # Approximate

        return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
    except Exception as e:
        print(f"    yfinance error: {e}")
        return None


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_files = []
    total_bars = 0

    print("=" * 60)
    print("Preparing Kronos Training Data")
    print("=" * 60)

    # Binance spot
    for symbol in BINANCE_SPOT:
        ticker_name = symbol.replace("USDT", "-USD")
        print(f"\n{ticker_name} (Binance spot 1h)...", flush=True)
        df = fetch_binance_full(symbol, "1h", fapi=False)
        if df is not None and len(df) > 100:
            path = DATA_DIR / f"{ticker_name}_1h.csv"
            df.to_csv(path, index=False)
            print(f"  {len(df)} bars, {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
            all_files.append(path)
            total_bars += len(df)

    # Binance FAPI
    for symbol in BINANCE_FAPI:
        ticker_name = symbol.replace("USDT", "-USD")
        print(f"\n{ticker_name} (Binance FAPI 1h)...", flush=True)
        df = fetch_binance_full(symbol, "1h", fapi=True)
        if df is not None and len(df) > 100:
            path = DATA_DIR / f"{ticker_name}_1h.csv"
            df.to_csv(path, index=False)
            print(f"  {len(df)} bars, {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
            all_files.append(path)
            total_bars += len(df)
        else:
            print(f"  Insufficient data, trying daily...")
            df = fetch_binance_full(symbol, "1d", fapi=True)
            if df is not None and len(df) > 100:
                path = DATA_DIR / f"{ticker_name}_1d.csv"
                df.to_csv(path, index=False)
                print(f"  {len(df)} daily bars")
                all_files.append(path)
                total_bars += len(df)

    # Stocks via yfinance (more history than Alpaca)
    for ticker in ALPACA_TICKERS:
        print(f"\n{ticker} (yfinance 1h)...", flush=True)
        df = fetch_yfinance(ticker, "1h")
        if df is not None and len(df) > 100:
            path = DATA_DIR / f"{ticker}_1h.csv"
            df.to_csv(path, index=False)
            print(f"  {len(df)} bars, {df['timestamps'].iloc[0]} to {df['timestamps'].iloc[-1]}")
            all_files.append(path)
            total_bars += len(df)
        else:
            print(f"  1h insufficient, trying daily...")
            df = fetch_yfinance(ticker, "1d")
            if df is not None and len(df) > 100:
                path = DATA_DIR / f"{ticker}_1d.csv"
                df.to_csv(path, index=False)
                print(f"  {len(df)} daily bars")
                all_files.append(path)
                total_bars += len(df)

    # Combine all into one mega CSV for training
    print(f"\n{'='*60}")
    print(f"Combining {len(all_files)} files, {total_bars} total bars...")

    combined = []
    for f in all_files:
        df = pd.read_csv(f)
        df["_source"] = f.stem
        combined.append(df)

    if combined:
        mega = pd.concat(combined, ignore_index=True)
        mega = mega.sort_values("timestamps")
        mega_path = DATA_DIR / "combined_all.csv"
        mega[["timestamps", "open", "high", "low", "close", "volume", "amount"]].to_csv(mega_path, index=False)
        print(f"Combined: {len(mega)} bars -> {mega_path}")

        # Also write per-asset-class combined files
        crypto_files = [f for f in all_files if "BTC" in str(f) or "ETH" in str(f)]
        if crypto_files:
            crypto = pd.concat([pd.read_csv(f) for f in crypto_files])
            crypto.sort_values("timestamps").to_csv(DATA_DIR / "combined_crypto.csv", index=False)
            print(f"Crypto combined: {len(crypto)} bars")

    print(f"\nData ready in {DATA_DIR}")
    print(f"Total: {total_bars} bars across {len(all_files)} tickers")


if __name__ == "__main__":
    main()
