"""Data collection — Binance, Alpaca, yfinance kline fetchers + multi-timeframe collector."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import pandas as pd

import portfolio.shared_state as _ss
from portfolio.api_utils import ALPACA_BASE, BINANCE_BASE, BINANCE_FAPI_BASE, get_alpaca_headers
from portfolio.circuit_breaker import CircuitBreaker
from portfolio.http_retry import fetch_with_retry
from portfolio.indicators import compute_indicators, technical_signal

logger = logging.getLogger("portfolio.data_collector")

# --- Circuit breakers for each data source ---

binance_spot_cb = CircuitBreaker("binance_spot", failure_threshold=5, recovery_timeout=60)
binance_fapi_cb = CircuitBreaker("binance_fapi", failure_threshold=5, recovery_timeout=60)
alpaca_cb = CircuitBreaker("alpaca", failure_threshold=5, recovery_timeout=60)

# BUG-179: Timeout for parallel timeframe fetches (seconds)
_TF_POOL_TIMEOUT = 60
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

_BINANCE_KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_vol",
    "taker_buy_quote_vol", "ignore",
]


def _binance_fetch(base_url, cb, label, symbol, interval="5m", limit=100):
    """Shared Binance kline fetcher for spot and FAPI endpoints."""
    if not cb.allow_request():
        logger.warning("Binance %s circuit OPEN — skipping %s", label, symbol)
        raise ConnectionError(f"Binance {label} circuit open for {symbol}")
    try:
        r = fetch_with_retry(
            f"{base_url}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r is None:
            raise ConnectionError(f"Binance {label} klines request failed for {symbol}")
        r.raise_for_status()
        data = r.json()
        # BUG-100: Empty response (200 OK but no data) should not count as success
        if not data:
            logger.warning("Binance %s returned empty data for %s %s", label, symbol, interval)
            cb.record_failure()
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=_BINANCE_KLINE_COLS)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        cb.record_success()
        return df
    except Exception:
        cb.record_failure()
        raise


def binance_klines(symbol, interval="5m", limit=100):
    return _binance_fetch(BINANCE_BASE, binance_spot_cb, "spot", symbol, interval, limit)


def binance_fapi_klines(symbol, interval="5m", limit=100):
    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
    return _binance_fetch(BINANCE_FAPI_BASE, binance_fapi_cb, "FAPI", symbol, interval, limit)


# --- Alpaca API ---


def alpaca_klines(ticker, interval="1d", limit=100):
    if interval not in ALPACA_INTERVAL_MAP:
        raise ValueError(f"Unsupported Alpaca interval: {interval}")
    if not alpaca_cb.allow_request():
        logger.warning("Alpaca circuit OPEN — skipping %s", ticker)
        raise ConnectionError(f"Alpaca circuit open for {ticker}")
    try:
        alpaca_tf, lookback_days = ALPACA_INTERVAL_MAP[interval]
        end = datetime.now(UTC)
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


def fetch_vix():
    """Fetch current VIX level via yfinance. Returns dict or None."""
    try:
        import yfinance as yf

        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist is None or hist.empty:
            return None
        # Flatten MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else last
        current = float(last["Close"])
        prev_close = float(prev["Close"])
        change_pct = ((current - prev_close) / prev_close * 100) if prev_close > 0 else 0

        # VIX regime classification
        if current >= 30:
            regime_hint = "high-vol"
        elif current >= 20:
            regime_hint = "elevated"
        elif current >= 15:
            regime_hint = "normal"
        else:
            regime_hint = "complacent"

        return {
            "value": round(current, 2),
            "prev_close": round(prev_close, 2),
            "change_pct": round(change_pct, 2),
            "regime_hint": regime_hint,
        }
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        return None


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
        if _ss._current_market_state in ("closed", "weekend", "holiday"):
            logger.debug("%s: using yfinance (market %s)", ticker, _ss._current_market_state)
            _ss._yfinance_limiter.wait()
            return yfinance_klines(ticker, interval=interval, limit=limit)
        _ss._alpaca_limiter.wait()
        return alpaca_klines(ticker, interval=interval, limit=limit)
    raise ValueError(f"Unknown source: {source}")


# --- Multi-timeframe collector ---


# yfinance is not thread-safe; serialize calls with a lock
_yfinance_lock = threading.Lock()


def _fetch_one_timeframe(source, source_key, label, interval, limit, ttl):
    """Fetch and process a single timeframe. Thread-safe."""
    cache_key = f"tf_{source_key}_{label}"
    if ttl > 0:
        with _ss._cache_lock:
            cached = _ss._tool_cache.get(cache_key)
            if cached and time.time() - cached["time"] < ttl:
                return (label, cached["data"])
    try:
        # yfinance is not thread-safe — serialize its calls
        if "alpaca" in source and _ss._current_market_state in ("closed", "weekend", "holiday"):
            with _yfinance_lock:
                df = _fetch_klines(source, interval, limit)
        else:
            df = _fetch_klines(source, interval, limit)
        ind = compute_indicators(df)
        if ind is None:
            logger.debug("%s/%s: insufficient data (%d rows), skipping",
                         source_key, label, len(df) if df is not None else 0)
            return None
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
        return (label, entry)
    except Exception as e:
        return (label, {"error": str(e)})


def collect_timeframes(source):
    """Collect all timeframes for a source, fetching in parallel."""
    is_stock = "alpaca" in source
    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
    source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")

    # BUG-179: Submit all timeframe fetches with timeout to prevent hangs
    with ThreadPoolExecutor(max_workers=len(tfs), thread_name_prefix=f"tf_{source_key}") as pool:
        futures = {
            pool.submit(_fetch_one_timeframe, source, source_key, label, interval, limit, ttl): label
            for label, interval, limit, ttl in tfs
        }
        raw_results = []
        try:
            for future in as_completed(futures, timeout=_TF_POOL_TIMEOUT):
                result = future.result()
                if result is not None:
                    raw_results.append(result)
        except TimeoutError:
            stuck = [lbl for f, lbl in futures.items() if not f.done()]
            logger.error("BUG-179: Timeframe pool timeout for %s. Stuck: %s",
                         source_key, stuck)
            for f in futures:
                f.cancel()

    # Maintain original timeframe order
    tf_order = {label: i for i, (label, _, _, _) in enumerate(tfs)}
    raw_results.sort(key=lambda x: tf_order.get(x[0], 999))
    return raw_results
