import json
import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_BASE = "https://api.binance.com/api/v3"
ALPACA_BASE = "https://data.alpaca.markets/v2"
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
TICKER_MAP = {
    "BTC-USD": ("binance", "BTCUSDT"),
    "ETH-USD": ("binance", "ETHUSDT"),
    "MSTR": ("alpaca", "MSTR"),
    "PLTR": ("alpaca", "PLTR"),
    "NVDA": ("alpaca", "NVDA"),
}
_alpaca_headers = None

_cache = {}
DXY_TTL = 3600
VOLUME_TTL = 300


def get_dxy():
    now = time.time()
    cached = _cache.get("dxy")
    if cached and now - cached["time"] < DXY_TTL:
        return cached["data"]

    import yfinance as yf

    t = yf.Ticker("DX-Y.NYB")
    h = t.history(period="30d")
    if h.empty:
        return None

    close = h["Close"]
    current = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    pct_5d = (
        float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
    )

    if current > sma20:
        trend = "strong"
    else:
        trend = "weak"

    result = {
        "value": round(current, 2),
        "sma20": round(sma20, 2),
        "trend": trend,
        "change_5d_pct": round(pct_5d, 2),
    }
    _cache["dxy"] = {"data": result, "time": now}
    return result


def _fetch_klines(ticker):
    source_type, symbol = TICKER_MAP.get(ticker, (None, None))
    if source_type == "binance":
        r = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": "15m", "limit": 100},
            timeout=10,
        )
        r.raise_for_status()
        raw = r.json()
        df = pd.DataFrame(
            raw,
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
                "tb",
                "tq",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    elif source_type == "alpaca":
        global _alpaca_headers
        if _alpaca_headers is None:
            cfg = json.loads(CONFIG_FILE.read_text())
            acfg = cfg.get("alpaca", {})
            _alpaca_headers = {
                "APCA-API-KEY-ID": acfg.get("api_key", ""),
                "APCA-API-SECRET-KEY": acfg.get("secret_key", ""),
            }
        r = requests.get(
            f"{ALPACA_BASE}/stocks/{symbol}/bars",
            params={"timeframe": "15Min", "limit": 100, "feed": "iex"},
            headers=_alpaca_headers,
            timeout=10,
        )
        r.raise_for_status()
        bars = r.json().get("bars", [])
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "t": "time",
            }
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    return None


def get_volume_signal(ticker):
    now = time.time()
    cached = _cache.get(f"vol_{ticker}")
    if cached and now - cached["time"] < VOLUME_TTL:
        return cached["data"]

    klines_df = _fetch_klines(ticker)
    if klines_df is None or klines_df.empty:
        return None
    vol = klines_df["volume"].astype(float)
    close = klines_df["close"].astype(float)
    current_vol = float(vol.iloc[-1])
    avg20 = (
        float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    )
    ratio = current_vol / avg20 if avg20 > 0 else 1.0

    # Price direction over last 3 candles
    if len(close) >= 4:
        price_change = float(close.iloc[-1] / close.iloc[-4] - 1)
    else:
        price_change = 0.0

    # Volume spike (>1.5x avg) confirms direction
    # No spike = abstain (HOLD)
    if ratio > 1.5:
        if price_change > 0:
            action = "BUY"
        elif price_change < 0:
            action = "SELL"
        else:
            action = "HOLD"
    else:
        action = "HOLD"

    result = {
        "ratio": round(ratio, 2),
        "spike": ratio > 1.5,
        "price_change_3": round(price_change * 100, 2),
        "action": action,
    }
    _cache[f"vol_{ticker}"] = {"data": result, "time": now}
    return result


if __name__ == "__main__":
    dxy = get_dxy()
    print(f"DXY: {dxy}")
    for t in ["BTC-USD", "ETH-USD", "MSTR", "PLTR", "NVDA"]:
        print(f"{t}: {get_volume_signal(t)}")
