"""Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from portfolio.circuit_breaker import CircuitBreaker
from portfolio.http_retry import fetch_with_retry
from portfolio.api_utils import load_config as _load_config, get_alpaca_headers, BINANCE_BASE, BINANCE_FAPI_BASE, ALPACA_BASE
from portfolio.indicators import compute_indicators, technical_signal
import portfolio.shared_state as _ss

logger = logging.getLogger("portfolio.data_collector")

# --- Circuit breakers for each data source ---

binance_spot_cb = CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)
binance_fapi_cb = CircuitBreaker("binance_fapi", failure_threshold=5, recovery_timeout=60)
alpaca_cb = CircuitBreaker("alpaca", failure_threshold=5, recovery_timeout=60)
ALPACA_INTERVAL_MAP = {
    "15m": ("15Min", 5),
    "1h": ("1Hour", 10),
    "1d": ("1Day", 365),
    "1w": ("1Week", 730),
    "1M": ("1Month", 1825),
}

# yfinance interval mapping: our interval → (yf_interval, yf_period)
_YF_INTERVAL_MAP = {
    "15m": ("15m", "5d"),       # yfinance max for intraday <=60d
    "1h": ("1h", "30d"),
    "1d": ("1d", "365d"),
    "1w": ("1wk", "730d"),
    "1M": ("1mo", "1825d"),
}

# Multi-timeframe analysis — (label, binance_interval, num_candles, cache_ttl_seconds)
TIMEFRAMES = [
    ("Now", "15m", 100, 0),  # ~25h data, refresh every cycle
    ("12h", "1h", 100, 300),  # ~4d data, cache 5min
    ("2d", "4h", 100, 900),  # ~17d data, cache 15min
    ("7d", "1d", 100, 3600),  # ~100d data, cache 1hr
    ("1mo", "3d", 100, 14400),  # ~300d data, cache 4hr
    ("3mo", "1w", 100, 43200),  # ~2yr data, cache 12hr
    ("6mo", "1M", 48, 86400),  # ~4yr data, cache 24hr
]

STOCK_TIMEFRAMES = [
    ("Now", "15m", 100, 0),
    ("12h", "1h", 100, 300),
    ("2d", "1h", 48, 900),
    ("7d", "1d", 30, 3600),
    ("1mo", "1d", 100, 3600),
    ("3mo", "1w", 100, 43200),
    ("6mo", "1M", 48, 86400),
]


# --- Binance API ---


def binance_klines(symbol, interval="5m", limit=100):
    if not binance_spot_cb.allow_request():
        logger.warning("Binance spot circuit OPEN — skipping %s", symbol)
        raise ConnectionError(f"Binance spot circuit open for {symbol}")
    try:
        r = fetch_with_retry(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r is None:
            raise ConnectionError(f"Binance klines request failed for {symbol}")
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(
            data,
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
                "taker_buy_vol",
                "taker_buy_quote_vol",
                "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        binance_spot_cb.record_success()
        return df
    except Exception:
        binance_spot_cb.record_failure()
        raise


def binance_fapi_klines(symbol, interval="5m", limit=100):
    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
    if not binance_fapi_cb.allow_request():
        logger.warning("Binance FAPI circuit OPEN — skipping %s", symbol)
        raise ConnectionError(f"Binance FAPI circuit open for {symbol}")
    try:
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r is None:
            raise ConnectionError(f"Binance FAPI klines request failed for {symbol}")
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(
            data,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy_vol",
                "taker_buy_quote_vol", "ignore",
            ],
        )
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        binance_fapi_cb.record_success()
        return df
    except Exception:
        binance_fapi_cb.record_failure()
        raise


# --- Alpaca API ---


def alpaca_klines(ticker, interval="1d", limit=100):
    if interval not in ALPACA_INTERVAL_MAP:
        raise ValueError(f"Unsupported Alpaca interval: {interval}")
    if not alpaca_cb.allow_request():
        logger.warning("Alpaca circuit OPEN — skipping %s", ticker)
        raise ConnectionError(f"Alpaca circuit open for {ticker}")
    try:
        alpaca_tf, lookback_days = ALPACA_INTERVAL_MAP[interval]
        end = datetime.now(timezone.utc)
        start = end - pd.Timedelta(days=lookback_days)
        r = fetch_with_retry(
            f"{ALPACA_BASE}/stocks/{ticker}/bars",
            headers=get_alpaca_headers(),
            params={
                "timeframe": alpaca_tf,
                "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "feed": "iex",
                "adjustment": "split",
            },
            timeout=10,
        )
        if r is None:
            raise ConnectionError(f"Alpaca request failed for {ticker}")
        r.raise_for_status()
        bars = r.json().get("bars") or []
        if not bars:
            raise ValueError(f"No Alpaca data for {ticker} interval={interval}")
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
        df["time"] = pd.to_datetime(df["time"])
        alpaca_cb.record_success()
        return df.tail(limit)
    except Exception:
        alpaca_cb.record_failure()
        raise


# --- yfinance API ---


def yfinance_klines(ticker, interval="1d", limit=100):
    """Fetch candles via yfinance with extended-hours data (prepost=True).

    Returns a DataFrame matching alpaca_klines() format:
    columns: open, high, low, close, volume, time
    """
    import yfinance as yf
    from portfolio.tickers import YF_MAP

    yf_ticker = YF_MAP.get(ticker, ticker)
    if interval not in _YF_INTERVAL_MAP:
        raise ValueError(f"Unsupported yfinance interval: {interval}")
    yf_interval, yf_period = _YF_INTERVAL_MAP[interval]

    df = yf.download(
        yf_ticker,
        period=yf_period,
        interval=yf_interval,
        prepost=True,
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        raise ValueError(f"No yfinance data for {yf_ticker} interval={interval}")

    # yfinance returns MultiIndex columns when downloading single ticker too
    # (e.g. ('Close', 'NVDA')); flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    df["time"] = df.index
    df = df.reset_index(drop=True)
    return df.tail(limit)


# --- Kline dispatcher ---


def _fetch_klines(source, interval, limit):
    if "binance_fapi" in source:
        _ss._binance_limiter.wait()
        return binance_fapi_klines(source["binance_fapi"], interval=interval, limit=limit)
    elif "binance" in source:
        _ss._binance_limiter.wait()
        return binance_klines(source["binance"], interval=interval, limit=limit)
    elif "alpaca" in source:
        ticker = source["alpaca"]
        if _ss._current_market_state in ("closed", "weekend"):
            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
            _ss._yfinance_limiter.wait()
            return yfinance_klines(ticker, interval=interval, limit=limit)
        _ss._alpaca_limiter.wait()
        return alpaca_klines(ticker, interval=interval, limit=limit)
    raise ValueError(f"Unknown source: {source}")


# --- Multi-timeframe collector ---


def collect_timeframes(source):
    is_stock = "alpaca" in source
    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
    source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")
    results = []
    for label, interval, limit, ttl in tfs:
        cache_key = f"tf_{source_key}_{label}"
        if ttl > 0:
            with _ss._cache_lock:
                cached = _ss._tool_cache.get(cache_key)
                if cached and time.time() - cached["time"] < ttl:
                    results.append((label, cached["data"]))
                    continue
        try:
            df = _fetch_klines(source, interval, limit)
            ind = compute_indicators(df)
            if ind is None:
                continue
            if label == "Now":
                action, conf = None, None
            else:
                action, conf = technical_signal(ind)
            entry = {"indicators": ind, "action": action, "confidence": conf}
            if label == "Now":
                entry["_df"] = df  # preserve raw DataFrame for enhanced signals
            if ttl > 0:
                with _ss._cache_lock:
                    _ss._tool_cache[cache_key] = {"data": entry, "time": time.time()}
            results.append((label, entry))
        except Exception as e:
            results.append((label, {"error": str(e)}))
    return results
