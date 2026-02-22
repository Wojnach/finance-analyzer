import time
import pandas as pd
from pathlib import Path

from portfolio.http_retry import fetch_with_retry

BINANCE_BASE = "https://api.binance.com/api/v3"
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "user_data"
    / "data"
    / "binance"
    / "futures"
)

PAIRS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
}


def download_klines(symbol, interval="1h", days=365):
    all_data = []
    end_time = int(time.time() * 1000)
    ms_per_candle = {"1h": 3600000, "4h": 14400000, "1d": 86400000}[interval]
    start_time = end_time - (days * 86400000)

    while start_time < end_time:
        r = fetch_with_retry(
            f"{BINANCE_BASE}/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "limit": 1000,
            },
            timeout=30,
        )
        if r is None:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_data.extend(batch)
        start_time = batch[-1][0] + ms_per_candle
        time.sleep(0.2)

    df = pd.DataFrame(
        all_data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_vol",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = (
        df.drop_duplicates(subset=["open_time"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def refresh_all(days=365):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, symbol in PAIRS.items():
        fname = f"{name}_USDT_USDT-1h-futures.feather"
        path = DATA_DIR / fname
        print(f"Downloading {symbol} 1h ({days}d)...", end=" ", flush=True)
        df = download_klines(symbol, interval="1h", days=days)
        df.to_feather(path)
        print(f"{len(df)} candles -> {path.name}")


if __name__ == "__main__":
    refresh_all()
