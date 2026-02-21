import json
import threading
import time
from pathlib import Path

import pandas as pd
import requests

from portfolio.api_utils import get_alpaca_headers, load_config
from portfolio.http_retry import fetch_with_retry

BINANCE_BASE = "https://api.binance.com/api/v3"
ALPACA_BASE = "https://data.alpaca.markets/v2"
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"

from portfolio.tickers import TICKER_SOURCE_MAP as TICKER_MAP


# --- Rate limiter for yfinance calls (no official limit, be polite — 30/min) ---
class _RateLimiter:
    """Token-bucket rate limiter. Sleeps when calls exceed rate."""
    def __init__(self, max_per_minute, name=""):
        self.interval = 60.0 / max_per_minute
        self.last_call = 0.0
        self.name = name
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                wait_time = self.interval - elapsed
                time.sleep(wait_time)
            self.last_call = time.time()


_yfinance_limiter = _RateLimiter(30, "yfinance")


def _alpaca_headers():
    return get_alpaca_headers()


_cache = {}
DXY_TTL = 3600
VOLUME_TTL = 300


def get_dxy():
    now = time.time()
    cached = _cache.get("dxy")
    if cached and now - cached["time"] < DXY_TTL:
        return cached["data"]

    import yfinance as yf

    _yfinance_limiter.wait()
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
    if source_type in ("binance", "binance_fapi"):
        base_url = BINANCE_FAPI_BASE if source_type == "binance_fapi" else BINANCE_BASE
        time.sleep(0.1)  # rate limit: space out Binance calls
        r = fetch_with_retry(
            f"{base_url}/klines",
            params={"symbol": symbol, "interval": "15m", "limit": 100},
            timeout=10,
        )
        if r is None:
            return None
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
        from datetime import datetime, timezone

        time.sleep(0.4)  # rate limit: space out Alpaca calls (150/min target)
        end = datetime.now(timezone.utc)
        start = end - pd.Timedelta(days=5)
        r = fetch_with_retry(
            f"{ALPACA_BASE}/stocks/{symbol}/bars",
            headers=_alpaca_headers(),
            params={
                "timeframe": "15Min",
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "iex",
            },
            timeout=10,
        )
        if r is None:
            return None
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            return None
        df = pd.DataFrame(bars)
        df = df.rename(
            columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
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
    if len(vol) < 2:
        return None
    last_vol = float(vol.iloc[-2])
    avg20 = (
        float(vol.iloc[:-1].rolling(20).mean().iloc[-1])
        if len(vol) >= 22
        else float(vol.iloc[:-1].mean())
    )
    ratio = last_vol / avg20 if avg20 > 0 else 1.0

    # Price direction over last 3 completed candles
    if len(close) >= 5:
        price_change = float(close.iloc[-2] / close.iloc[-5] - 1)
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


TREASURY_TTL = 3600

from portfolio.fomc_dates import FOMC_DATES_ISO as FOMC_DATES


def get_treasury():
    now = time.time()
    cached = _cache.get("treasury")
    if cached and now - cached["time"] < TREASURY_TTL:
        return cached["data"]

    import yfinance as yf

    tickers = {"10y": "^TNX", "2y": "2YY=F", "30y": "^TYX"}
    result = {}
    for label, sym in tickers.items():
        try:
            _yfinance_limiter.wait()
            t = yf.Ticker(sym)
            h = t.history(period="30d")
            if h.empty:
                continue
            close = h["Close"]
            current = float(close.iloc[-1])
            pct_5d = (
                float((close.iloc[-1] / close.iloc[-5] - 1) * 100)
                if len(close) >= 5
                else 0
            )
            result[label] = {
                "yield_pct": round(current, 3),
                "change_5d": round(pct_5d, 2),
            }
        except Exception:
            pass

    if "10y" in result and "2y" in result:
        spread = result["10y"]["yield_pct"] - result["2y"]["yield_pct"]
        result["spread_2s10s"] = round(spread, 3)
        if spread < 0:
            result["curve"] = "inverted"
        elif spread < 0.2:
            result["curve"] = "flat"
        else:
            result["curve"] = "normal"

    if result:
        _cache["treasury"] = {"data": result, "time": now}
    return result or None


def get_fed_calendar():
    from datetime import datetime, timedelta, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    upcoming = [d for d in FOMC_DATES if d >= today]
    if not upcoming:
        return None

    next_date = upcoming[0]
    days_until = (
        datetime.strptime(next_date, "%Y-%m-%d") - datetime.strptime(today, "%Y-%m-%d")
    ).days

    is_meeting_day = today in FOMC_DATES
    is_day_before = any(
        (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        == today
        for d in FOMC_DATES
    )

    result = {
        "next_fomc": next_date,
        "days_until": days_until,
        "meetings_remaining": len(upcoming) // 2,
    }
    if is_meeting_day:
        result["warning"] = "FOMC meeting TODAY — expect volatility"
    elif is_day_before:
        result["warning"] = "FOMC meeting TOMORROW — positioning risk"
    elif days_until <= 7:
        result["warning"] = f"FOMC in {days_until} days — pre-meeting drift possible"

    return result


if __name__ == "__main__":
    dxy = get_dxy()
    print(f"DXY: {dxy}")
    treasury = get_treasury()
    print(f"Treasury: {treasury}")
    fed = get_fed_calendar()
    print(f"Fed: {fed}")
    for t in list(TICKER_MAP.keys()):
        print(f"{t}: {get_volume_signal(t)}")
