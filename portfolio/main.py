#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data"""

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
COMPACT_SUMMARY_FILE = DATA_DIR / "agent_summary_compact.json"
CONFIG_FILE = BASE_DIR / "config.json"

from portfolio.tickers import SYMBOLS, CRYPTO_SYMBOLS, STOCK_SYMBOLS, METALS_SYMBOLS

# --- Config caching (delegates to shared api_utils for mtime-based caching) ---
from portfolio.api_utils import load_config as _load_config


# Market hours (UTC) — EU open to US close, DST-aware for NYSE
# EU open is always ~07:00 UTC (CET/CEST difference is small for our purposes).
# NYSE close shifts: EST (Nov-Mar) = 21:00 UTC, EDT (Mar-Nov) = 20:00 UTC.
MARKET_OPEN_HOUR = 7  # ~Frankfurt/London open

# Loop intervals by market state
INTERVAL_MARKET_OPEN = 60  # 1 min — full speed
INTERVAL_MARKET_CLOSED = 300  # 5 min — crypto only weekday nights
INTERVAL_WEEKEND = 600  # 10 min — crypto only weekends


def _is_us_dst(dt):
    """Check if a UTC datetime falls within US Eastern Daylight Time (EDT).

    US DST rule (since 2007):
      Starts: second Sunday of March at 02:00 local (07:00 UTC)
      Ends:   first Sunday of November at 02:00 local (06:00 UTC)

    Returns True during EDT (Mar-Nov), False during EST (Nov-Mar).
    """
    year = dt.year

    # Second Sunday of March
    mar1_wd = date(year, 3, 1).weekday()  # 0=Mon..6=Sun
    first_sun_mar = 1 + (6 - mar1_wd) % 7
    second_sun_mar = first_sun_mar + 7
    dst_start = datetime(year, 3, second_sun_mar, 7, 0, tzinfo=timezone.utc)  # 02:00 ET = 07:00 UTC

    # First Sunday of November
    nov1_wd = date(year, 11, 1).weekday()
    first_sun_nov = 1 + (6 - nov1_wd) % 7
    dst_end = datetime(year, 11, first_sun_nov, 6, 0, tzinfo=timezone.utc)  # 02:00 ET = 06:00 UTC (still EDT)

    return dst_start <= dt < dst_end


def _market_close_hour_utc(dt):
    """Return the NYSE close hour in UTC, adjusted for DST.

    NYSE closes at 16:00 ET.
    EDT (Mar-Nov): 16:00 ET = 20:00 UTC
    EST (Nov-Mar): 16:00 ET = 21:00 UTC
    """
    if _is_us_dst(dt):
        return 20
    return 21


def _is_agent_window(now=None):
    """Check if current time is within the Layer 2 invocation window.

    Window: 1h before EU market open through 1h after US market close.
    EU open (Frankfurt): 08:00 CET / 07:00 UTC  -> start 06:00 UTC
    US close (NYSE):     16:00 ET
      EST (Nov-Mar): 21:00 UTC  -> end 22:00 UTC
      EDT (Mar-Nov): 20:00 UTC  -> end 21:00 UTC
    Weekends: no agent invocation.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    current_minutes = now.hour * 60 + now.minute
    start = 6 * 60  # 06:00 UTC (1h before Frankfurt 07:00)
    if _is_us_dst(now):
        end = 21 * 60  # 21:00 UTC (1h after NYSE 20:00 EDT)
    else:
        end = 22 * 60  # 22:00 UTC (1h after NYSE 21:00 EST)
    return start <= current_minutes < end


def get_market_state():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    all_symbols = set(SYMBOLS.keys())
    if weekday >= 5:
        return "weekend", all_symbols, INTERVAL_WEEKEND
    close_hour = _market_close_hour_utc(now)
    if MARKET_OPEN_HOUR <= hour < close_hour:
        return "open", all_symbols, INTERVAL_MARKET_OPEN
    return "closed", all_symbols, INTERVAL_MARKET_CLOSED


INITIAL_CASH_SEK = 500_000
FEE_CRYPTO = 0.0005  # 0.05% taker fee (Binance futures)
FEE_STOCK = 0.001  # 0.10% (typical broker commission)

BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"

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

# Tool cache — avoid re-running expensive tools every cycle
_tool_cache = {}
FEAR_GREED_TTL = 300  # 5 min

# --- Rate limiter — space out API calls to stay under provider limits ---
import threading

class _RateLimiter:
    """Token-bucket rate limiter. Sleeps when calls exceed rate."""
    def __init__(self, max_per_minute, name=""):
        self.interval = 60.0 / max_per_minute  # min seconds between calls
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

# Alpaca IEX: 200 req/min → target 150/min to leave headroom
_alpaca_limiter = _RateLimiter(150, "alpaca")
# Binance: 1200 weight/min → very generous, but space out slightly
_binance_limiter = _RateLimiter(600, "binance")
# Yahoo Finance (yfinance): no official limit, but be polite — 30/min
_yfinance_limiter = _RateLimiter(30, "yfinance")
SENTIMENT_TTL = 900  # 15 min
MINISTRAL_TTL = 900  # 15 min
ML_SIGNAL_TTL = 900  # 15 min
FUNDING_RATE_TTL = 900  # 15 min
VOLUME_TTL = 300  # 5 min


_RETRY_COOLDOWN = 60


def _cached(key, ttl, func, *args):
    now = time.time()
    if key in _tool_cache and now - _tool_cache[key]["time"] < ttl:
        return _tool_cache[key]["data"]
    try:
        data = func(*args)
        _tool_cache[key] = {"data": data, "time": now}
        return data
    except Exception as e:
        print(f"    [{key}] error: {e}")
        if key in _tool_cache:
            _tool_cache[key]["time"] = now - ttl + _RETRY_COOLDOWN
            return _tool_cache[key]["data"]
        return None


# --- Binance API ---


def binance_klines(symbol, interval="5m", limit=100):
    r = requests.get(
        f"{BINANCE_BASE}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
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
    return df


def binance_fapi_klines(symbol, interval="5m", limit=100):
    """Fetch klines from Binance Futures API (for metals like XAUUSDT, XAGUSDT)."""
    r = requests.get(
        f"{BINANCE_FAPI_BASE}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=10,
    )
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
    return df


ALPACA_BASE = "https://data.alpaca.markets/v2"
ALPACA_INTERVAL_MAP = {
    "15m": ("15Min", 5),
    "1h": ("1Hour", 10),
    "1d": ("1Day", 365),
    "1w": ("1Week", 730),
    "1M": ("1Month", 1825),
}


# NOTE: Shared Alpaca header logic is available in portfolio.api_utils.get_alpaca_headers().
# This local copy is kept to avoid a large refactor of main.py config loading path.
def _get_alpaca_headers():
    cfg = _load_config()
    acfg = cfg.get("alpaca", {})
    return {
        "APCA-API-KEY-ID": acfg.get("key", ""),
        "APCA-API-SECRET-KEY": acfg.get("secret", ""),
    }


def alpaca_klines(ticker, interval="1d", limit=100):
    if interval not in ALPACA_INTERVAL_MAP:
        raise ValueError(f"Unsupported Alpaca interval: {interval}")
    alpaca_tf, lookback_days = ALPACA_INTERVAL_MAP[interval]
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=lookback_days)
    r = requests.get(
        f"{ALPACA_BASE}/stocks/{ticker}/bars",
        headers=_get_alpaca_headers(),
        params={
            "timeframe": alpaca_tf,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feed": "iex",
            "adjustment": "split",
        },
        timeout=10,
    )
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
    return df.tail(limit)


# yfinance interval mapping: our interval → (yf_interval, yf_period)
_YF_INTERVAL_MAP = {
    "15m": ("15m", "5d"),       # yfinance max for intraday <=60d
    "1h": ("1h", "30d"),
    "1d": ("1d", "365d"),
    "1w": ("1wk", "730d"),
    "1M": ("1mo", "1825d"),
}


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


_fx_cache = {"rate": None, "time": 0}
_FX_STALE_THRESHOLD = 7200  # 2 hours — warn if FX rate hasn't been refreshed


def fetch_usd_sek():
    now = time.time()
    if _fx_cache["rate"] and now - _fx_cache["time"] < 3600:
        # Check staleness even when returning cached value
        age_secs = now - _fx_cache["time"]
        if age_secs > _FX_STALE_THRESHOLD:
            print(f"  WARNING: FX rate is stale ({age_secs / 3600:.1f}h old)")
        return _fx_cache["rate"]
    try:
        r = requests.get(
            "https://api.frankfurter.app/latest",
            params={"from": "USD", "to": "SEK"},
            timeout=10,
        )
        r.raise_for_status()
        rate = float(r.json()["rates"]["SEK"])
        _fx_cache["rate"] = rate
        _fx_cache["time"] = now
        return rate
    except Exception as e:
        print(f"  WARNING: FX rate fetch failed: {e}")
    if _fx_cache["rate"]:
        age_secs = now - _fx_cache["time"]
        if age_secs > _FX_STALE_THRESHOLD:
            print(f"  WARNING: Using stale FX rate ({age_secs / 3600:.1f}h old)")
            _fx_alert_telegram(age_secs)
        return _fx_cache["rate"]
    # Last resort: hardcoded fallback
    print("  WARNING: Using hardcoded FX fallback rate 10.50 SEK — no cached or live rate available")
    _fx_alert_telegram(None)
    return 10.50


def _fx_alert_telegram(age_secs):
    """Send a one-shot Telegram alert about FX rate issues. Fires at most once per 4h."""
    global _fx_cache
    last_alert = _fx_cache.get("_last_fx_alert", 0)
    now = time.time()
    if now - last_alert < 14400:  # 4h cooldown between alerts
        return
    try:
        config = _load_config()
        if age_secs is not None:
            msg = f"_FX WARNING: USD/SEK rate is {age_secs / 3600:.1f}h stale. API may be down._"
        else:
            msg = "_FX WARNING: Using hardcoded fallback rate 10.50 SEK. No live or cached rate available._"
        send_telegram(msg, config)
        _fx_cache["_last_fx_alert"] = now
    except Exception:
        pass  # non-critical


# --- Indicators ---


def compute_indicators(df):
    if len(df) < 26:
        return None
    close = df["close"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    # EMA(9, 21)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    # Bollinger Bands(20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ATR(14) — Average True Range for volatility measurement
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean().iloc[-1]

    # RSI rolling percentiles for adaptive thresholds
    rsi_series = rsi
    rsi_p20 = rsi_series.rolling(100, min_periods=20).quantile(0.2).iloc[-1]
    rsi_p80 = rsi_series.rolling(100, min_periods=20).quantile(0.8).iloc[-1]

    return {
        "close": float(close.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "macd_hist_prev": float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
        "ema9": float(ema9.iloc[-1]),
        "ema21": float(ema21.iloc[-1]),
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "bb_mid": float(bb_mid.iloc[-1]),
        "price_vs_bb": (
            "below_lower"
            if float(close.iloc[-1]) <= float(bb_lower.iloc[-1])
            else (
                "above_upper"
                if float(close.iloc[-1]) >= float(bb_upper.iloc[-1])
                else "inside"
            )
        ),
        "atr": float(atr14),
        "atr_pct": float(atr14 / close.iloc[-1]) * 100,
        "rsi_p20": float(rsi_p20) if not pd.isna(rsi_p20) else 30.0,
        "rsi_p80": float(rsi_p80) if not pd.isna(rsi_p80) else 70.0,
    }


# Cycle counter — incremented at the start of each run() to invalidate per-cycle caches
_run_cycle_id = 0
# Current market state — updated each run() cycle, used by _fetch_klines for yfinance fallback
_current_market_state = "open"

_regime_cache = {}
_regime_cache_cycle = 0


def detect_regime(indicators, is_crypto=True):
    global _regime_cache, _regime_cache_cycle
    # Invalidate cache on new cycle
    if _run_cycle_id != _regime_cache_cycle:
        _regime_cache = {}
        _regime_cache_cycle = _run_cycle_id
    # Cache key: use id of the indicators dict + is_crypto flag
    cache_key = (id(indicators), is_crypto)
    if cache_key in _regime_cache:
        return _regime_cache[cache_key]

    atr_pct = indicators.get("atr_pct", 0)
    ema9 = indicators.get("ema9", 0)
    ema21 = indicators.get("ema21", 0)
    rsi = indicators.get("rsi", 50)

    high_vol_threshold = 4.0 if is_crypto else 3.0
    if atr_pct > high_vol_threshold:
        result = "high-vol"
    elif ema21 != 0 and abs(ema9 - ema21) / ema21 * 100 > 1.0:
        if ema9 > ema21 and rsi > 45:
            result = "trending-up"
        elif ema9 < ema21 and rsi < 55:
            result = "trending-down"
        else:
            result = "ranging"
    else:
        result = "ranging"

    _regime_cache[cache_key] = result
    return result


# --- Technical-only signal (for longer timeframes) ---


def technical_signal(ind):
    buy = 0
    sell = 0
    if ind["rsi"] < 50:
        buy += 1
    else:
        sell += 1
    if ind["macd_hist"] > 0:
        buy += 1
    else:
        sell += 1
    if ind["ema9"] > ind["ema21"]:
        buy += 1
    else:
        sell += 1
    if ind["close"] > ind["bb_mid"]:
        buy += 1
    else:
        sell += 1
    total = buy + sell
    if buy > sell:
        return "BUY", buy / total
    elif sell > buy:
        return "SELL", sell / total
    return "HOLD", 0.5


def _fetch_klines(source, interval, limit):
    if "binance_fapi" in source:
        _binance_limiter.wait()
        return binance_fapi_klines(source["binance_fapi"], interval=interval, limit=limit)
    elif "binance" in source:
        _binance_limiter.wait()
        return binance_klines(source["binance"], interval=interval, limit=limit)
    elif "alpaca" in source:
        ticker = source["alpaca"]
        if _current_market_state in ("closed", "weekend"):
            _yfinance_limiter.wait()
            return yfinance_klines(ticker, interval=interval, limit=limit)
        _alpaca_limiter.wait()
        return alpaca_klines(ticker, interval=interval, limit=limit)
    raise ValueError(f"Unknown source: {source}")


STOCK_TIMEFRAMES = [
    ("Now", "15m", 100, 0),
    ("12h", "1h", 100, 300),
    ("2d", "1h", 48, 900),
    ("7d", "1d", 30, 3600),
    ("1mo", "1d", 100, 3600),
    ("3mo", "1w", 100, 43200),
    ("6mo", "1M", 48, 86400),
]


def collect_timeframes(source):
    is_stock = "alpaca" in source
    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
    source_key = source.get("alpaca") or source.get("binance") or source.get("binance_fapi")
    results = []
    for label, interval, limit, ttl in tfs:
        cache_key = f"tf_{source_key}_{label}"
        if ttl > 0:
            cached = _tool_cache.get(cache_key)
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
                _tool_cache[cache_key] = {"data": entry, "time": time.time()}
            results.append((label, entry))
        except Exception as e:
            results.append((label, {"error": str(e)}))
    return results


# --- Signal (full 7-signal for "Now" timeframe) ---


MIN_VOTERS_CRYPTO = 3  # crypto has 25 signals (11 original + 14 enhanced) — need 3 active voters
MIN_VOTERS_STOCK = 3  # stocks have 21 signals (7 original + 14 enhanced) — need 3 active voters

# Sentiment hysteresis — prevents rapid flip spam from ~50% confidence oscillation
_prev_sentiment = {}  # in-memory cache; seeded from trigger_state.json on first call
_prev_sentiment_loaded = False


def _load_prev_sentiments():
    global _prev_sentiment, _prev_sentiment_loaded
    if _prev_sentiment_loaded:
        return
    try:
        ts_file = DATA_DIR / "trigger_state.json"
        if ts_file.exists():
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
            _prev_sentiment = ts.get("prev_sentiment", {})
    except Exception:
        pass
    _prev_sentiment_loaded = True


def _get_prev_sentiment(ticker):
    _load_prev_sentiments()
    return _prev_sentiment.get(ticker)


def _set_prev_sentiment(ticker, direction):
    _load_prev_sentiments()
    _prev_sentiment[ticker] = direction
    # Persist to trigger_state.json alongside other trigger state
    try:
        ts_file = DATA_DIR / "trigger_state.json"
        ts = json.loads(ts_file.read_text(encoding="utf-8")) if ts_file.exists() else {}
        ts["prev_sentiment"] = _prev_sentiment
        import tempfile as _tmp, os as _os

        fd, tmp = _tmp.mkstemp(dir=ts_file.parent, suffix=".tmp")
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ts, f, indent=2, default=str)
        _os.replace(tmp, ts_file)
    except Exception:
        pass


REGIME_WEIGHTS = {
    "trending-up": {"ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7},
    "trending-down": {"ema": 1.5, "macd": 1.3, "rsi": 0.7, "bb": 0.7},
    "ranging": {"rsi": 1.5, "bb": 1.5, "ema": 0.5, "macd": 0.5},
    "high-vol": {"bb": 1.5, "volume": 1.3, "funding": 1.3, "ema": 0.5},
}


def _weighted_consensus(votes, accuracy_data, regime, activation_rates=None):
    """Compute weighted consensus using accuracy, regime, and activation frequency.

    Weight per signal = accuracy_weight * regime_mult * normalized_weight
    where normalized_weight = rarity_bonus * bias_penalty (from activation rates).
    Rare, balanced signals get more weight; noisy/biased signals get less.
    """
    buy_weight = 0.0
    sell_weight = 0.0
    regime_mults = REGIME_WEIGHTS.get(regime, {})
    activation_rates = activation_rates or {}
    for signal_name, vote in votes.items():
        if vote == "HOLD":
            continue
        # Accuracy weight
        stats = accuracy_data.get(signal_name, {})
        acc = stats.get("accuracy", 0.5)
        samples = stats.get("total", 0)
        weight = acc if samples >= 20 else 0.5
        # Regime adjustment
        weight *= regime_mults.get(signal_name, 1.0)
        # Activation frequency normalization (rarity * bias correction)
        act_data = activation_rates.get(signal_name, {})
        norm_weight = act_data.get("normalized_weight", 1.0)
        weight *= norm_weight
        if vote == "BUY":
            buy_weight += weight
        elif vote == "SELL":
            sell_weight += weight
    total_weight = buy_weight + sell_weight
    if total_weight == 0:
        return "HOLD", 0.0
    buy_conf = buy_weight / total_weight
    sell_conf = sell_weight / total_weight
    if buy_conf > sell_conf and buy_conf >= 0.5:
        return "BUY", round(buy_conf, 4)
    if sell_conf > buy_conf and sell_conf >= 0.5:
        return "SELL", round(sell_conf, 4)
    return "HOLD", round(max(buy_conf, sell_conf), 4)


def _confluence_score(votes, indicators):
    active = {k: v for k, v in votes.items() if v != "HOLD"}
    if not active:
        return 0.0
    buy_count = sum(1 for v in active.values() if v == "BUY")
    sell_count = sum(1 for v in active.values() if v == "SELL")
    majority = max(buy_count, sell_count)
    score = majority / len(active)
    if indicators.get("volume_action") in ("BUY", "SELL"):
        vol_dir = indicators.get("volume_action")
        majority_dir = "BUY" if buy_count >= sell_count else "SELL"
        if vol_dir == majority_dir:
            score += 0.1
    return min(round(score, 4), 1.0)


def _time_of_day_factor():
    hour = datetime.now(timezone.utc).hour
    if 2 <= hour <= 6:
        return 0.8
    return 1.0


def generate_signal(ind, ticker=None, config=None, timeframes=None, df=None):
    votes = {}
    extra_info = {}

    # RSI — only votes at extremes (adaptive thresholds from rolling percentiles)
    rsi_lower = ind.get("rsi_p20", 30)
    rsi_upper = ind.get("rsi_p80", 70)
    rsi_lower = max(rsi_lower, 15)
    rsi_upper = min(rsi_upper, 85)
    if ind["rsi"] < rsi_lower:
        votes["rsi"] = "BUY"
    elif ind["rsi"] > rsi_upper:
        votes["rsi"] = "SELL"
    else:
        votes["rsi"] = "HOLD"

    # MACD — only votes on crossover
    if ind["macd_hist"] > 0 and ind["macd_hist_prev"] <= 0:
        votes["macd"] = "BUY"
    elif ind["macd_hist"] < 0 and ind["macd_hist_prev"] >= 0:
        votes["macd"] = "SELL"
    else:
        votes["macd"] = "HOLD"

    # EMA trend — votes only when gap is meaningful (>0.5%)
    ema_gap_pct = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100 if ind["ema21"] != 0 else 0
    )
    if ema_gap_pct >= 0.5:
        votes["ema"] = "BUY" if ind["ema9"] > ind["ema21"] else "SELL"
    else:
        votes["ema"] = "HOLD"

    # Bollinger Bands — only votes at extremes
    if ind["price_vs_bb"] == "below_lower":
        votes["bb"] = "BUY"
    elif ind["price_vs_bb"] == "above_upper":
        votes["bb"] = "SELL"
    else:
        votes["bb"] = "HOLD"

    # --- Extended signals from tools (optional) ---

    # Fear & Greed Index (per-ticker: crypto->alternative.me, stocks->VIX)
    votes["fear_greed"] = "HOLD"
    try:
        from portfolio.fear_greed import get_fear_greed

        fg_key = f"fear_greed_{ticker}" if ticker else "fear_greed"
        fg = _cached(fg_key, FEAR_GREED_TTL, get_fear_greed, ticker)
        if fg:
            extra_info["fear_greed"] = fg["value"]
            extra_info["fear_greed_class"] = fg["classification"]
            if fg["value"] <= 20:
                votes["fear_greed"] = "BUY"
            elif fg["value"] >= 80:
                votes["fear_greed"] = "SELL"
    except ImportError:
        pass

    # Social media posts (Reddit) — fetched separately, merged into sentiment
    social_posts = []
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.social_sentiment import get_reddit_posts

            reddit = _cached(
                f"reddit_{short_ticker}",
                SENTIMENT_TTL,
                get_reddit_posts,
                short_ticker,
            )
            if reddit:
                social_posts.extend(reddit)
        except ImportError:
            pass

    # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM) — includes social posts
    # Hysteresis: flipping direction requires confidence > 0.55, same direction > 0.40
    votes["sentiment"] = "HOLD"
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.sentiment import get_sentiment

            newsapi_key = (config or {}).get("newsapi_key", "")
            sent = _cached(
                f"sentiment_{short_ticker}",
                SENTIMENT_TTL,
                get_sentiment,
                short_ticker,
                newsapi_key or None,
                social_posts or None,
            )
            if sent and sent.get("num_articles", 0) > 0:
                extra_info["sentiment"] = sent["overall_sentiment"]
                extra_info["sentiment_conf"] = sent["confidence"]
                extra_info["sentiment_model"] = sent.get("model", "unknown")
                if sent.get("sources"):
                    extra_info["sentiment_sources"] = sent["sources"]

                prev_sent_dir = _get_prev_sentiment(ticker)
                current_dir = sent["overall_sentiment"]
                if (
                    prev_sent_dir
                    and current_dir != prev_sent_dir
                    and current_dir != "neutral"
                ):
                    sent_threshold = 0.55
                else:
                    sent_threshold = 0.40

                if (
                    sent["overall_sentiment"] == "positive"
                    and sent["confidence"] > sent_threshold
                ):
                    votes["sentiment"] = "BUY"
                    _set_prev_sentiment(ticker, "positive")
                elif (
                    sent["overall_sentiment"] == "negative"
                    and sent["confidence"] > sent_threshold
                ):
                    votes["sentiment"] = "SELL"
                    _set_prev_sentiment(ticker, "negative")
        except ImportError:
            pass

    # ML Classifier (HistGradientBoosting on BTC/ETH 1h data)
    votes["ml"] = "HOLD"
    if ticker:
        try:
            from portfolio.ml_signal import get_ml_signal

            ml = _cached(f"ml_{ticker}", ML_SIGNAL_TTL, get_ml_signal, ticker)
            if ml:
                extra_info["ml_action"] = ml["action"]
                extra_info["ml_confidence"] = ml["confidence"]
                votes["ml"] = ml["action"]
        except ImportError:
            pass

    # Funding Rate (Binance perpetuals, crypto only — contrarian)
    votes["funding"] = "HOLD"
    if ticker:
        try:
            from portfolio.funding_rate import get_funding_rate

            fr = _cached(
                f"funding_{ticker}", FUNDING_RATE_TTL, get_funding_rate, ticker
            )
            if fr:
                extra_info["funding_rate"] = fr["rate_pct"]
                extra_info["funding_action"] = fr["action"]
                votes["funding"] = fr["action"]
        except ImportError:
            pass

    # Volume Confirmation (spike + price direction = vote)
    votes["volume"] = "HOLD"
    if ticker:
        try:
            from portfolio.macro_context import get_volume_signal

            vs = _cached(f"volume_{ticker}", VOLUME_TTL, get_volume_signal, ticker)
            if vs:
                extra_info["volume_ratio"] = vs["ratio"]
                extra_info["volume_action"] = vs["action"]
                votes["volume"] = vs["action"]
        except ImportError:
            pass

    # Ministral-8B LLM reasoning (original CryptoTrader-LM, crypto only)
    # custom_lora fully disabled: 20.9% accuracy, 97% SELL bias (worse than random).
    # Model no longer invoked. Shadow A/B testing data preserved in ab_test_log.jsonl.
    votes["ministral"] = "HOLD"
    if ticker and ticker in CRYPTO_SYMBOLS:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.ministral_signal import get_ministral_signal

            tf_summary = ""
            if timeframes:
                parts = []
                for label, entry in timeframes:
                    if (
                        isinstance(entry, dict)
                        and "action" in entry
                        and entry["action"]
                    ):
                        ti = entry.get("indicators", {})
                        parts.append(
                            f"{label}: {entry['action']} (RSI={ti.get('rsi', 0):.0f})"
                        )
                if parts:
                    tf_summary = " | ".join(parts)

            ema_gap = (
                abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100
                if ind["ema21"] != 0
                else 0
            )

            ctx = {
                "ticker": short_ticker,
                "price_usd": ind["close"],
                "rsi": round(ind["rsi"], 1),
                "macd_hist": round(ind["macd_hist"], 2),
                "ema_bullish": ind["ema9"] > ind["ema21"],
                "ema_gap_pct": round(ema_gap, 2),
                "bb_position": ind["price_vs_bb"],
                "fear_greed": extra_info.get("fear_greed", "N/A"),
                "fear_greed_class": extra_info.get("fear_greed_class", ""),
                "news_sentiment": extra_info.get("sentiment", "N/A"),
                "sentiment_confidence": extra_info.get("sentiment_conf", "N/A"),
                "volume_ratio": extra_info.get("volume_ratio", "N/A"),
                "funding_rate": extra_info.get("funding_action", "N/A"),
                "timeframe_summary": tf_summary,
                "headlines": "",
            }
            ms = _cached(
                f"ministral_{short_ticker}",
                MINISTRAL_TTL,
                get_ministral_signal,
                ctx,
            )
            if ms:
                orig = ms.get("original") or ms
                extra_info["ministral_action"] = orig["action"]
                extra_info["ministral_reasoning"] = orig.get("reasoning", "")
                votes["ministral"] = orig["action"]

                # custom_lora fully disabled — not even stored in extra.
                # Shadow A/B data preserved in data/ab_test_log.jsonl.
        except ImportError:
            pass

    # --- Enhanced signal modules (composite indicators computed from raw OHLCV) ---
    _enhanced_modules = [
        ("trend", "portfolio.signals.trend", "compute_trend_signal"),
        ("momentum", "portfolio.signals.momentum", "compute_momentum_signal"),
        ("volume_flow", "portfolio.signals.volume_flow", "compute_volume_flow_signal"),
        ("volatility_sig", "portfolio.signals.volatility", "compute_volatility_signal"),
        ("candlestick", "portfolio.signals.candlestick", "compute_candlestick_signal"),
        ("structure", "portfolio.signals.structure", "compute_structure_signal"),
        ("fibonacci", "portfolio.signals.fibonacci", "compute_fibonacci_signal"),
        ("smart_money", "portfolio.signals.smart_money", "compute_smart_money_signal"),
        ("oscillators", "portfolio.signals.oscillators", "compute_oscillator_signal"),
        ("heikin_ashi", "portfolio.signals.heikin_ashi", "compute_heikin_ashi_signal"),
        ("mean_reversion", "portfolio.signals.mean_reversion", "compute_mean_reversion_signal"),
        ("calendar", "portfolio.signals.calendar_seasonal", "compute_calendar_signal"),
        ("momentum_factors", "portfolio.signals.momentum_factors", "compute_momentum_factors_signal"),
    ]
    # macro_regime is special — it takes an extra macro dict parameter
    _macro_regime_module = ("macro_regime", "portfolio.signals.macro_regime", "compute_macro_regime_signal")

    # Calendar signal re-enabled: quorum fix (min 2 active votes) eliminates
    # the old 100% BUY bias from single-sub-signal noise.
    _NON_VOTING_ENHANCED = set()

    if df is not None and isinstance(df, pd.DataFrame) and len(df) >= 26:
        for sig_name, module_path, func_name in _enhanced_modules:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                compute_fn = getattr(mod, func_name)
                result = compute_fn(df)
                if result and isinstance(result, dict):
                    extra_info[f"{sig_name}_action"] = result.get("action", "HOLD")
                    extra_info[f"{sig_name}_confidence"] = result.get("confidence", 0.0)
                    extra_info[f"{sig_name}_sub_signals"] = result.get("sub_signals", {})
                    if sig_name in _NON_VOTING_ENHANCED:
                        # Store in extra only, do NOT add to votes
                        pass
                    else:
                        votes[sig_name] = result.get("action", "HOLD")
                else:
                    if sig_name not in _NON_VOTING_ENHANCED:
                        votes[sig_name] = "HOLD"
            except Exception:
                if sig_name not in _NON_VOTING_ENHANCED:
                    votes[sig_name] = "HOLD"

        # macro_regime gets macro context from cache if available
        try:
            import importlib
            macro_data = None
            try:
                from portfolio.macro_context import get_dxy, get_fed_calendar, get_treasury
                macro_data = {}
                dxy = _cached("dxy", 3600, get_dxy)
                if dxy:
                    macro_data["dxy"] = dxy
                treasury = _cached("treasury", 3600, get_treasury)
                if treasury:
                    macro_data["treasury"] = treasury
                fed = get_fed_calendar()
                if fed:
                    macro_data["fed"] = fed
            except Exception:
                pass
            mr_name, mr_path, mr_func = _macro_regime_module
            mod = importlib.import_module(mr_path)
            compute_fn = getattr(mod, mr_func)
            result = compute_fn(df, macro=macro_data or None)
            if result and isinstance(result, dict):
                votes[mr_name] = result.get("action", "HOLD")
                extra_info[f"{mr_name}_action"] = result.get("action", "HOLD")
                extra_info[f"{mr_name}_confidence"] = result.get("confidence", 0.0)
                extra_info[f"{mr_name}_sub_signals"] = result.get("sub_signals", {})
            else:
                votes[mr_name] = "HOLD"
        except Exception:
            votes[_macro_regime_module[0]] = "HOLD"
    else:
        for sig_name, _, _ in _enhanced_modules:
            if sig_name not in _NON_VOTING_ENHANCED:
                votes[sig_name] = "HOLD"
        votes[_macro_regime_module[0]] = "HOLD"

    # Derive buy/sell counts from named votes
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")

    # Core signal gate: at least 1 core signal must be active for non-HOLD consensus.
    # Enhanced signals can strengthen/weaken a consensus but never create one alone.
    CORE_SIGNAL_NAMES = {
        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
        "ml", "funding", "volume", "ministral",
    }
    core_buy = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "BUY")
    core_sell = sum(1 for s in CORE_SIGNAL_NAMES if votes.get(s) == "SELL")
    core_active = core_buy + core_sell

    # Total applicable signals:
    # Crypto: 10 core (11 original - custom_lora removed) + 14 enhanced (all voting) = 24
    # Metals: 7 core + 14 enhanced = 21
    # Stocks: 7 core + 14 enhanced = 21
    is_crypto = ticker in CRYPTO_SYMBOLS
    is_metal = ticker in METALS_SYMBOLS
    if is_crypto:
        total_applicable = 24  # 10 core + 14 enhanced
    elif is_metal:
        total_applicable = 21  # 7 core + 14 enhanced
    else:
        total_applicable = 21  # 7 core + 14 enhanced

    active_voters = buy + sell
    if ticker in STOCK_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK
    elif ticker in METALS_SYMBOLS:
        min_voters = MIN_VOTERS_STOCK  # metals use same threshold
    else:
        min_voters = MIN_VOTERS_CRYPTO

    # Core gate: if no core signal is active, force HOLD regardless of enhanced votes
    if core_active == 0:
        action = "HOLD"
        conf = 0.0
    elif active_voters < min_voters:
        action = "HOLD"
        conf = 0.0
    else:
        buy_conf = buy / active_voters
        sell_conf = sell / active_voters
        if buy_conf > sell_conf and buy_conf >= 0.5:
            action = "BUY"
            conf = buy_conf
        elif sell_conf > buy_conf and sell_conf >= 0.5:
            action = "SELL"
            conf = sell_conf
        else:
            action = "HOLD"
            conf = max(buy_conf, sell_conf)

    # Weighted consensus using accuracy data, regime, and activation frequency
    regime = detect_regime(ind, is_crypto=ticker in CRYPTO_SYMBOLS)
    accuracy_data = {}
    activation_rates = {}
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy, signal_accuracy, write_accuracy_cache,
            load_cached_activation_rates,
        )

        cached = load_cached_accuracy("1d")
        if cached:
            accuracy_data = cached
        else:
            accuracy_data = signal_accuracy("1d")
            if accuracy_data:
                write_accuracy_cache("1d", accuracy_data)
        activation_rates = load_cached_activation_rates()
    except Exception:
        pass
    weighted_action, weighted_conf = _weighted_consensus(
        votes, accuracy_data, regime, activation_rates
    )

    # Apply core gate to weighted consensus too — no core signal active = HOLD
    if core_active == 0:
        weighted_action = "HOLD"
        weighted_conf = 0.0

    # Confluence score
    confluence = _confluence_score(votes, extra_info)

    # Time-of-day confidence adjustment
    tod_factor = _time_of_day_factor()
    conf *= tod_factor
    weighted_conf *= tod_factor

    # Store raw consensus in extra for debugging, then use weighted as primary
    extra_info["_raw_action"] = action
    extra_info["_raw_confidence"] = conf
    extra_info["_voters"] = active_voters
    extra_info["_total_applicable"] = total_applicable
    extra_info["_buy_count"] = buy
    extra_info["_sell_count"] = sell
    extra_info["_core_buy"] = core_buy
    extra_info["_core_sell"] = core_sell
    extra_info["_core_active"] = core_active
    extra_info["_votes"] = votes
    extra_info["_regime"] = regime
    extra_info["_weighted_action"] = weighted_action
    extra_info["_weighted_confidence"] = weighted_conf
    extra_info["_confluence_score"] = confluence

    # Primary action = weighted consensus (accounts for accuracy + bias penalties)
    action = weighted_action
    conf = weighted_conf
    return action, conf, extra_info


# --- State ---


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "cash_sek": INITIAL_CASH_SEK,
        "holdings": {},
        "transactions": [],
        "start_date": datetime.now(timezone.utc).isoformat(),
        "initial_value_sek": INITIAL_CASH_SEK,
    }


def _atomic_write_json(path, data):
    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_state(state):
    _atomic_write_json(STATE_FILE, state)


def _cross_asset_signals(all_signals):
    btc = all_signals.get("BTC-USD", {})
    btc_action = btc.get("action", "HOLD")
    if btc_action == "HOLD":
        return {}

    followers = {"ETH-USD": "BTC-USD", "MSTR": "BTC-USD"}
    leads = {}
    for follower, leader in followers.items():
        f_data = all_signals.get(follower, {})
        f_action = f_data.get("action", "HOLD")
        if f_action == "HOLD" and btc_action != "HOLD":
            leads[follower] = {
                "leader": leader,
                "leader_action": btc_action,
                "note": f"{leader} is {btc_action} but {follower} hasn't moved yet",
            }
    return leads


# --- Agent summary + invocation ---


def write_agent_summary(
    signals, prices_usd, fx_rate, state, tf_data, trigger_reasons=None
):
    total = portfolio_value(state, prices_usd, fx_rate)
    pnl_pct = ((total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_reasons": trigger_reasons or [],
        "fx_rate": round(fx_rate, 2),
        "portfolio": {
            "total_sek": round(total),
            "pnl_pct": round(pnl_pct, 2),
            "cash_sek": round(state["cash_sek"]),
            "holdings": state.get("holdings", {}),
            "num_transactions": len(state.get("transactions", [])),
        },
        "signals": {},
        "timeframes": {},
        "fear_greed": {},
    }

    for name, sig in signals.items():
        extra = sig.get("extra", {})
        ind = sig["indicators"]
        # Collect enhanced signal summaries (compact, no sub_signals detail in top-level)
        enhanced = {}
        for esig_name in ("trend", "momentum", "volume_flow", "volatility_sig",
                          "candlestick", "structure", "fibonacci", "smart_money",
                          "oscillators", "heikin_ashi", "mean_reversion", "calendar",
                          "macro_regime", "momentum_factors"):
            eaction = extra.get(f"{esig_name}_action", "HOLD")
            econf = extra.get(f"{esig_name}_confidence", 0.0)
            enhanced[esig_name] = {"action": eaction, "confidence": econf}

        # Strip bulky sub_signals from extra to keep agent_summary.json lean
        extra_clean = {k: v for k, v in extra.items() if not k.endswith("_sub_signals")}

        sig_entry = {
            "action": sig["action"],
            "confidence": sig["confidence"],
            "weighted_confidence": extra.get("_weighted_confidence", 0.0),
            "confluence_score": extra.get("_confluence_score", 0.0),
            "price_usd": ind["close"],
            "rsi": round(ind["rsi"], 1),
            "macd_hist": round(ind["macd_hist"], 2),
            "bb_position": ind["price_vs_bb"],
            "atr": round(ind.get("atr", 0), 4),
            "atr_pct": round(ind.get("atr_pct", 0), 2),
            "regime": detect_regime(ind, is_crypto=name in CRYPTO_SYMBOLS),
            "enhanced_signals": enhanced,
            "extra": extra_clean,
        }
        # Mark extended-hours data for stocks (yfinance prepost during off-hours)
        if name in STOCK_SYMBOLS and _current_market_state in ("closed", "weekend"):
            sig_entry["extended_hours"] = True
        summary["signals"][name] = sig_entry
        if "fear_greed" in extra:
            summary["fear_greed"][name] = {
                "value": extra["fear_greed"],
                "classification": extra.get("fear_greed_class", ""),
            }

        tf_list = []
        for label, entry in tf_data.get(name, []):
            if "error" in entry:
                tf_list.append({"horizon": label, "error": entry["error"]})
            else:
                ei = entry["indicators"]
                tf_list.append(
                    {
                        "horizon": label,
                        "action": entry["action"] if label != "Now" else sig["action"],
                        "confidence": (
                            entry["confidence"] if label != "Now" else sig["confidence"]
                        ),
                        "rsi": round(ei["rsi"], 1),
                        "macd_hist": round(ei["macd_hist"], 2),
                        "ema_bullish": ei["ema9"] > ei["ema21"],
                        "bb_position": ei["price_vs_bb"],
                    }
                )
        summary["timeframes"][name] = tf_list

    # Macro context (non-voting, for Claude Code reasoning)
    try:
        from portfolio.macro_context import get_dxy, get_fed_calendar, get_treasury

        macro = {}
        dxy = _cached("dxy", 3600, get_dxy)
        if dxy:
            macro["dxy"] = dxy
        treasury = _cached("treasury", 3600, get_treasury)
        if treasury:
            macro["treasury"] = treasury
        fed = get_fed_calendar()
        if fed:
            macro["fed"] = fed
        if macro:
            summary["macro"] = macro
    except (ImportError, Exception):
        pass

    try:
        from portfolio.accuracy_stats import (
            signal_accuracy,
            consensus_accuracy,
            best_worst_signals,
        )

        sig_acc = signal_accuracy("1d")
        cons_acc = consensus_accuracy("1d")
        bw = best_worst_signals("1d")
        qualified = {k: v for k, v in sig_acc.items() if v["total"] >= 5}
        if qualified:
            summary["signal_accuracy_1d"] = {
                "signals": {
                    k: {"accuracy": round(v["accuracy"], 3), "samples": v["total"]}
                    for k, v in qualified.items()
                },
                "consensus": {
                    "accuracy": round(cons_acc["accuracy"], 3),
                    "samples": cons_acc["total"],
                },
                "best": bw.get("best"),
                "worst": bw.get("worst"),
            }
    except Exception:
        pass

    # Signal activation rates (normalized weights for Layer 2 reference)
    try:
        from portfolio.accuracy_stats import load_cached_activation_rates
        act_rates = load_cached_activation_rates()
        if act_rates:
            summary["signal_weights"] = {
                name: {
                    "activation_rate": d["activation_rate"],
                    "normalized_weight": d["normalized_weight"],
                    "bias": d["bias"],
                }
                for name, d in act_rates.items()
                if d.get("samples", 0) > 0
            }
    except Exception:
        pass

    cross_leads = _cross_asset_signals(signals)
    if cross_leads:
        summary["cross_asset_leads"] = cross_leads

    # Avanza-tracked instruments (Tier 2: Nordic equities, Tier 3: warrants)
    try:
        from portfolio.avanza_tracker import fetch_avanza_prices
        avanza_prices = fetch_avanza_prices()
        if avanza_prices:
            summary["avanza_instruments"] = avanza_prices
    except Exception:
        pass

    # Preserve stale data for instruments not in current cycle (e.g. stocks off-hours)
    # so Layer 2 always sees all instruments
    if AGENT_SUMMARY_FILE.exists():
        try:
            prev = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
            for section in ("signals", "timeframes", "fear_greed"):
                prev_section = prev.get(section, {})
                for ticker, data in prev_section.items():
                    if ticker not in summary[section]:
                        if section == "signals" and isinstance(data, dict):
                            data["stale"] = True
                        summary[section][ticker] = data
        except Exception:
            pass

    _atomic_write_json(AGENT_SUMMARY_FILE, summary)
    _write_compact_summary(summary)
    return summary


def _write_compact_summary(summary):
    """Write a stripped version of agent_summary for Layer 2 (<25K tokens).

    Removes enhanced_signals (already in extra._votes) and verbose extra fields,
    keeping only the essentials Layer 2 needs for decision-making.
    """
    import copy

    KEEP_EXTRA = {
        "fear_greed", "fear_greed_class",
        "sentiment", "sentiment_conf",
        "ml_action", "ml_confidence",
        "funding_rate", "funding_action",
        "volume_ratio", "volume_action",
        "ministral_action",
        "_voters", "_total_applicable", "_buy_count", "_sell_count",
        "_votes", "_weighted_action", "_weighted_confidence",
        "_confluence_score",
    }

    compact = copy.deepcopy(summary)

    for ticker_data in compact.get("signals", {}).values():
        ticker_data.pop("enhanced_signals", None)
        if "extra" in ticker_data:
            ticker_data["extra"] = {
                k: v for k, v in ticker_data["extra"].items()
                if k in KEEP_EXTRA
            }

    # Compact timeframes: keep only horizon + action (indicators duplicate top-level)
    for ticker, tf_list in compact.get("timeframes", {}).items():
        compact["timeframes"][ticker] = [
            {"horizon": tf["horizon"], "action": tf.get("action", "HOLD")}
            if "error" not in tf else {"horizon": tf["horizon"], "error": tf["error"]}
            for tf in tf_list
        ]

    # Remove sections redundant in compact (already in per-ticker extra or baked in)
    compact.pop("fear_greed", None)      # in extra.fear_greed per ticker
    compact.pop("signal_weights", None)  # already applied in _weighted_confidence

    _atomic_write_json(COMPACT_SUMMARY_FILE, compact)


INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"


_agent_proc = None
_agent_log = None
_agent_start = 0
AGENT_TIMEOUT = 600


def _log_trigger(reasons, status):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
        "status": status,
    }
    with open(INVOCATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def invoke_agent(reasons):
    global _agent_proc, _agent_log, _agent_start
    if _agent_proc and _agent_proc.poll() is None:
        elapsed = time.time() - _agent_start
        if elapsed > AGENT_TIMEOUT:
            print(f"  Agent pid={_agent_proc.pid} timed out ({elapsed:.0f}s), killing")
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(_agent_proc.pid)],
                    capture_output=True,
                )
            else:
                _agent_proc.kill()
            try:
                _agent_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if _agent_log:
                _agent_log.close()
                _agent_log = None
        else:
            print(
                f"  Agent still running (pid {_agent_proc.pid}, {elapsed:.0f}s), skipping"
            )
            return False

    if _agent_log:
        _agent_log.close()
        _agent_log = None

    try:
        from portfolio.journal import write_context

        n = write_context()
        print(f"  Layer 2 context: {n} journal entries")
    except Exception as e:
        print(f"  WARNING: journal context failed: {e}")

    agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
    if not agent_bat.exists():
        print(f"  WARNING: Agent script not found at {agent_bat}")
        return False
    try:
        _agent_log = open(DATA_DIR / "agent.log", "a", encoding="utf-8")
        # Strip Claude Code session markers to avoid "nested session" error
        # when the parent process tree has Claude Code running
        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        _agent_proc = subprocess.Popen(
            ["cmd", "/c", str(agent_bat)],
            cwd=str(BASE_DIR),
            stdout=_agent_log,
            stderr=subprocess.STDOUT,
            env=agent_env,
        )
        _agent_start = time.time()
        print(f"  Agent invoked pid={_agent_proc.pid} ({', '.join(reasons)})")
        # Send brief Telegram notification that Layer 2 was triggered
        try:
            config = _load_config()
            reason_str = escape_markdown_v1(", ".join(reasons[:3]))
            if len(reasons) > 3:
                reason_str += f" (+{len(reasons) - 3} more)"
            notify_msg = f"_Layer 2 invoked: {reason_str}_"
            send_telegram(notify_msg, config)
        except Exception:
            pass  # non-critical
        return True
    except Exception as e:
        print(f"  ERROR invoking agent: {e}")
        return False


def portfolio_value(state, prices_usd, fx_rate):
    total = state["cash_sek"]
    for ticker, h in state.get("holdings", {}).items():
        if h["shares"] > 0 and ticker in prices_usd:
            total += h["shares"] * prices_usd[ticker] * fx_rate
    return total


# --- Telegram ---

import re

# Characters that can break Telegram Markdown v1 parsing when used in dynamic content.
# Markdown v1 interprets _ * ` [ as formatting. We escape them in dynamic strings
# (ticker names, prices, reasons) but NOT in our own formatting (bold/italic wrappers).
_MD_V1_SPECIAL = re.compile(r'([_*`\[\]])')


def escape_markdown_v1(text):
    """Escape special Markdown v1 characters in dynamic content to prevent parse failures.

    Use this on user-facing dynamic strings (ticker names, error messages, reason text)
    that are inserted into Markdown-formatted Telegram messages. Do NOT apply to the
    entire message — it would break intentional formatting like *bold* and _italic_.
    """
    return _MD_V1_SPECIAL.sub(r'\\\1', str(text))


def send_telegram(msg, config):
    if os.environ.get("NO_TELEGRAM"):
        print("  [NO_TELEGRAM] Skipping send")
        return True
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=30,
    )
    if r.ok:
        return True
    # Markdown parse failure (HTTP 400) — retry without parse_mode so the message
    # still arrives (unformatted) rather than being silently lost.
    if r.status_code == 400:
        err_desc = ""
        try:
            err_desc = r.json().get("description", "")
        except Exception:
            pass
        if "parse" in err_desc.lower() or "markdown" in err_desc.lower() or "entity" in err_desc.lower():
            print(f"  WARNING: Telegram Markdown parse failed ({err_desc}), resending without formatting")
            r2 = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg},
                timeout=30,
            )
            return r2.ok
    return False


BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
_COOLDOWN_PREFIXES = ("cooldown", "crypto check-in", "startup")


def _maybe_send_alert(config, signals, prices_usd, fx_rate, state, reasons, tf_data):
    significant = [r for r in reasons if not r.startswith(_COOLDOWN_PREFIXES)]
    if not significant:
        return
    headline = escape_markdown_v1(significant[0])
    lines = [f"*ALERT: {headline}*", ""]
    # Actionable-only: show BUY/SELL tickers, compress HOLDs
    hold_count = 0
    for ticker in SYMBOLS:
        sig = signals.get(ticker)
        if not sig:
            continue
        action = sig["action"]
        if action == "HOLD":
            hold_count += 1
            continue
        price = prices_usd.get(ticker, 0)
        extra = sig.get("extra", {})
        b = extra.get("_buy_count", 0)
        s = extra.get("_sell_count", 0)
        total = extra.get("_total_applicable", 0)
        h = max(0, total - b - s)
        if price >= 1000:
            p_str = f"${price:,.0f}"
        else:
            p_str = f"${price:,.2f}"
        lines.append(f"`{ticker:<7} {p_str:>9}  {action:<4} {b}B/{s}S/{h}H`")
    if hold_count > 0:
        lines.append(f"_+ {hold_count} HOLD_")
    fg_val = ""
    for ticker, sig in signals.items():
        extra = sig.get("extra", {})
        if "fear_greed" in extra:
            fg_class = escape_markdown_v1(extra.get("fear_greed_class", ""))
            fg_val = f"{extra['fear_greed']} ({fg_class})"
            break
    patient_total = portfolio_value(state, prices_usd, fx_rate)
    patient_pnl = (
        (patient_total - state["initial_value_sek"]) / state["initial_value_sek"]
    ) * 100
    lines.append("")
    if fg_val:
        lines.append(f"_F&G: {fg_val}_")
    lines.append(f"_Patient: {patient_total:,.0f} SEK ({patient_pnl:+.1f}%)_")
    if BOLD_STATE_FILE.exists():
        bold = json.loads(BOLD_STATE_FILE.read_text(encoding="utf-8"))
        bold_total = portfolio_value(bold, prices_usd, fx_rate)
        bold_pnl = (
            (bold_total - bold["initial_value_sek"]) / bold["initial_value_sek"]
        ) * 100
        lines.append(f"_Bold: {bold_total:,.0f} SEK ({bold_pnl:+.1f}%)_")
    msg = "\n".join(lines)
    try:
        send_telegram(msg, config)
        print(f"  Alert sent: {headline}")
    except Exception as e:
        print(f"  WARNING: alert send failed: {e}")


# --- 4h Digest ---

DIGEST_INTERVAL = 14400  # 4 hours


def _get_last_digest_time():
    try:
        state = json.loads(
            (DATA_DIR / "trigger_state.json").read_text(encoding="utf-8")
        )
        return state.get("last_digest_time", 0)
    except Exception:
        return 0


def _set_last_digest_time(t):
    path = DATA_DIR / "trigger_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state["last_digest_time"] = t
    _atomic_write_json(path, state)


JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


def _build_digest_message():
    from portfolio.stats import load_jsonl

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=DIGEST_INTERVAL)

    entries = load_jsonl(INVOCATIONS_FILE)
    recent = [e for e in entries if datetime.fromisoformat(e["ts"]) >= cutoff]
    reason_counts = Counter()
    status_counts = Counter()
    for e in recent:
        status_counts[e.get("status", "invoked")] += 1
        for r in e.get("reasons", []):
            if "flipped" in r:
                reason_counts["signal_flip"] += 1
            elif "moved" in r:
                reason_counts["price_move"] += 1
            elif "F&G" in r:
                reason_counts["fear_greed"] += 1
            elif "sentiment" in r:
                reason_counts["sentiment"] += 1
            elif "cooldown" in r or "check-in" in r:
                reason_counts["check_in"] += 1
            else:
                reason_counts["other"] += 1

    # Count Layer 2 decisions from journal
    journal = load_jsonl(JOURNAL_FILE)
    recent_journal = [e for e in journal if datetime.fromisoformat(e["ts"]) >= cutoff]
    l2_decisions = {"patient": Counter(), "bold": Counter()}
    for e in recent_journal:
        decisions = e.get("decisions", {})
        for strat in ("patient", "bold"):
            action = decisions.get(strat, {}).get("action", "HOLD")
            l2_decisions[strat][action] += 1

    summary = json.loads(AGENT_SUMMARY_FILE.read_text(encoding="utf-8"))
    fx_rate = summary.get("fx_rate", 10.5)
    prices_usd = {t: s["price_usd"] for t, s in summary.get("signals", {}).items()}

    state = load_state()
    p_total = portfolio_value(state, prices_usd, fx_rate)
    p_pnl = ((p_total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
    p_holdings = [
        t for t, h in state.get("holdings", {}).items() if h.get("shares", 0) > 0
    ]

    lines = ["*4H DIGEST*", ""]
    lines.append(
        f"_{cutoff.strftime('%H:%M')} - {now.strftime('%H:%M UTC')} ({now.strftime('%b %d')})_"
    )

    # Layer 2 (Claude) activity
    invoked = status_counts.get("invoked", 0)
    skipped = status_counts.get("skipped_busy", 0)
    l2_total = len(recent_journal)
    lines.append(
        f"_Layer 2: {invoked} invoked, {l2_total} decisions, {skipped} skipped_"
    )
    for strat in ("patient", "bold"):
        counts = l2_decisions[strat]
        if counts:
            parts = []
            for action in ("HOLD", "BUY", "SELL"):
                if counts[action]:
                    parts.append(f"{counts[action]} {action}")
            lines.append(f"`  {strat:<8} {', '.join(parts)}`")

    lines.append("")
    lines.append(f"_Triggers: {len(recent)}_")
    for reason, count in reason_counts.most_common():
        lines.append(f"`{reason:<14} {count}`")

    lines.append("")
    p_holdings_str = escape_markdown_v1(', '.join(p_holdings)) if p_holdings else 'cash'
    lines.append(
        f"_Patient: {p_total:,.0f} SEK ({p_pnl:+.1f}%) · {p_holdings_str}_"
    )

    if BOLD_STATE_FILE.exists():
        bold = json.loads(BOLD_STATE_FILE.read_text(encoding="utf-8"))
        b_total = portfolio_value(bold, prices_usd, fx_rate)
        b_pnl = (
            (b_total - bold["initial_value_sek"]) / bold["initial_value_sek"]
        ) * 100
        b_holdings = [
            t for t, h in bold.get("holdings", {}).items() if h.get("shares", 0) > 0
        ]
        b_holdings_str = escape_markdown_v1(', '.join(b_holdings)) if b_holdings else 'cash'
        lines.append(
            f"_Bold: {b_total:,.0f} SEK ({b_pnl:+.1f}%) · {b_holdings_str}_"
        )

    return "\n".join(lines)


def _maybe_send_digest(config):
    last = _get_last_digest_time()
    if last and (time.time() - last) < DIGEST_INTERVAL:
        return
    try:
        msg = _build_digest_message()
        send_telegram(msg, config)
        _set_last_digest_time(time.time())
        print("  4h digest sent")
    except Exception as e:
        print(f"  WARNING: digest failed: {e}")


# --- Main ---


def run(force_report=False, active_symbols=None):
    global _run_cycle_id, _current_market_state
    _run_cycle_id += 1

    config = _load_config()
    state = load_state()
    fx_rate = fetch_usd_sek()

    market_state, default_symbols, _ = get_market_state()
    _current_market_state = market_state
    active = active_symbols or default_symbols

    skipped = set(SYMBOLS.keys()) - active
    skip_note = f" (skipped: {', '.join(sorted(skipped))})" if skipped else ""
    print(
        f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] USD/SEK: {fx_rate:.2f} | market: {market_state}{skip_note}"
    )

    signals = {}
    prices_usd = {}
    tf_data = {}

    for name, source in SYMBOLS.items():
        if name not in active:
            continue
        try:
            tfs = collect_timeframes(source)
            tf_data[name] = tfs

            now_entry = tfs[0][1] if tfs else None
            now_df = None
            if now_entry and "indicators" in now_entry:
                ind = now_entry["indicators"]
                now_df = now_entry.get("_df")  # raw DataFrame from collect_timeframes
            else:
                now_df = _fetch_klines(source, interval="15m", limit=100)
                ind = compute_indicators(now_df)

            if ind is None:
                print(f"  {name}: insufficient data, skipping")
                continue
            price = ind["close"]
            prices_usd[name] = price

            action, conf, extra = generate_signal(
                ind, ticker=name, config=config, timeframes=tfs, df=now_df
            )
            signals[name] = {
                "action": action,
                "confidence": conf,
                "indicators": ind,
                "extra": extra,
            }

            extra_str = ""
            if extra:
                parts = []
                if "fear_greed" in extra:
                    parts.append(f"F&G:{extra['fear_greed']}")
                if "sentiment" in extra:
                    parts.append(f"News:{extra['sentiment']}")
                if "ministral_action" in extra:
                    parts.append(f"8B:{extra['ministral_action']}")
                if "ml_action" in extra:
                    parts.append(f"ML:{extra['ml_action']}")
                if "funding_action" in extra:
                    parts.append(f"FR:{extra['funding_rate']}%")
                if "volume_action" in extra and extra["volume_action"] != "HOLD":
                    parts.append(f"Vol:{extra['volume_ratio']}x")
                if parts:
                    extra_str = f" | {' '.join(parts)}"
            # Enhanced signals summary
            enh_parts = []
            for esig in ("trend", "momentum", "volume_flow", "volatility_sig",
                         "candlestick", "structure", "fibonacci", "smart_money",
                         "oscillators", "heikin_ashi", "mean_reversion", "calendar",
                         "macro_regime", "momentum_factors"):
                ea = extra.get(f"{esig}_action", "HOLD")
                if ea != "HOLD":
                    enh_parts.append(f"{esig[:4].title()}:{ea[0]}")
            enh_str = f" | Enh: {' '.join(enh_parts)}" if enh_parts else ""

            print(
                f"  {name}: ${price:,.2f} | RSI {ind['rsi']:.0f} | MACD {ind['macd_hist']:+.1f}{extra_str}{enh_str} | {action} ({conf:.0%})"
            )

            # Print multi-timeframe summary
            for label, entry in tfs[1:]:
                if "error" in entry:
                    print(f"    {label}: ERROR - {entry['error']}")
                else:
                    ei = entry["indicators"]
                    print(
                        f"    {label}: {entry['action']} {entry['confidence']:.0%} | RSI {ei['rsi']:.0f} | MACD {ei['macd_hist']:+.1f}"
                    )

        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    total = portfolio_value(state, prices_usd, fx_rate)
    pnl_pct = ((total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
    print(
        f"\n  Portfolio: {total:,.0f} SEK ({pnl_pct:+.2f}%) | Cash: {state['cash_sek']:,.0f} SEK"
    )

    # Ensure portfolio state file exists (first run)
    if not STATE_FILE.exists():
        save_state(state)

    # Smart trigger — invoke Claude Code agent when something meaningful changed
    from portfolio.trigger import check_triggers

    fear_greeds = {}
    sentiments = {}
    for name, sig in signals.items():
        extra = sig.get("extra", {})
        if "fear_greed" in extra:
            fear_greeds[name] = {
                "value": extra["fear_greed"],
                "classification": extra.get("fear_greed_class", ""),
            }
        if "sentiment" in extra:
            sentiments[name] = extra["sentiment"]

    triggered, reasons = check_triggers(signals, prices_usd, fear_greeds, sentiments)

    if triggered or force_report:
        reasons_list = reasons if reasons else ["startup"]
        write_agent_summary(signals, prices_usd, fx_rate, state, tf_data, reasons_list)
        print(f"\n  Trigger: {', '.join(reasons_list)}")

        # Log signal snapshot for forward tracking
        try:
            from portfolio.outcome_tracker import log_signal_snapshot

            log_signal_snapshot(signals, prices_usd, fx_rate, reasons_list)
        except Exception as e:
            print(f"  WARNING: signal logging failed: {e}")

        layer2_cfg = config.get("layer2", {})
        if os.environ.get("NO_TELEGRAM"):
            print("  [NO_TELEGRAM] Skipping agent invocation")
            _log_trigger(reasons_list, "skipped_test")
        elif layer2_cfg.get("enabled", True):
            if _is_agent_window():
                result = invoke_agent(reasons_list)
                _log_trigger(reasons_list, "invoked" if result else "skipped_busy")
            else:
                print("  Layer 2: outside market window (1h before open to 1h after close), skipping")
                _log_trigger(reasons_list, "skipped_offhours")
        else:
            print("  Layer 2 disabled — skipping agent invocation")
            _maybe_send_alert(
                config, signals, prices_usd, fx_rate, state, reasons_list, tf_data
            )
            _log_trigger(reasons_list, "alert_only")
    else:
        write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
        print("\n  No trigger — nothing changed.")

    # Big Bet detection (runs every cycle, alerts on extreme setups)
    bigbet_cfg = config.get("bigbet", {})
    if bigbet_cfg.get("enabled", False):
        try:
            from portfolio.bigbet import check_bigbet

            check_bigbet(signals, prices_usd, fx_rate, tf_data, config)
        except Exception as e:
            print(f"  WARNING: Big Bet check failed: {e}")

    # ISKBETS monitoring (runs every cycle; session config is the on/off switch)
    try:
        from portfolio.iskbets import check_iskbets

        check_iskbets(signals, prices_usd, fx_rate, tf_data, config)
    except Exception as e:
        print(f"  WARNING: ISKBETS check failed: {e}")


def _crash_alert(error_msg):
    """Send Telegram alert on loop crash."""
    try:
        import json, requests
        config_path = Path(__file__).resolve().parent.parent / "config.json"
        config = json.load(open(config_path))
        token = config.get("telegram", {}).get("token", "")
        chat_id = config.get("telegram", {}).get("chat_id", "")
        if token and chat_id:
            text = f"LOOP CRASH\n\n{error_msg[:3000]}"
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10
            )
    except Exception:
        pass  # Can't alert about alert failure


def loop(interval=None):
    config = _load_config()
    print("Starting loop with market-aware scheduling. Ctrl+C to stop.")

    # Start Telegram poller for ISKBETS commands (lightweight daemon, no-ops when idle)
    try:
        from portfolio.telegram_poller import TelegramPoller
        from portfolio.iskbets import handle_command

        poller = TelegramPoller(config, on_command=handle_command)
        poller.start()
        print("  ISKBETS Telegram poller started")
    except Exception as e:
        print(f"  WARNING: ISKBETS poller failed to start: {e}")

    try:
        run(force_report=True)
        _maybe_send_digest(config)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        import traceback
        _crash_alert(traceback.format_exc())
        print(f"  ERROR in initial run: {e}")
        time.sleep(10)
    last_state = None
    while True:
        market_state, active_symbols, sleep_interval = get_market_state()
        if interval:
            sleep_interval = interval
        if market_state != last_state:
            print(
                f"\n  Schedule: {market_state} — {len(active_symbols)} instruments, {sleep_interval}s interval"
            )
            last_state = market_state
        time.sleep(sleep_interval)
        try:
            run(force_report=False, active_symbols=active_symbols)
            _maybe_send_digest(config)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            import traceback
            _crash_alert(traceback.format_exc())
            print(f"  ERROR in run: {e}")
            time.sleep(10)
        # Touch heartbeat file after each cycle
        try:
            (DATA_DIR / "heartbeat.txt").write_text(datetime.now(timezone.utc).isoformat())
        except Exception:
            pass


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check-outcomes" in args:
        print("=== Outcome Backfill ===")
        from portfolio.outcome_tracker import backfill_outcomes

        updated = backfill_outcomes()
        print(f"Updated {updated} entries")
    elif "--accuracy" in args:
        from portfolio.accuracy_stats import print_accuracy_report

        print_accuracy_report()
    elif "--retrain" in args:
        print("=== ML Retraining ===")
        print("Refreshing data from Binance API...")
        from portfolio.data_refresh import refresh_all

        refresh_all(days=365)
        print("\nTraining model...")
        from portfolio.ml_trainer import load_data, train_final

        data = load_data()
        feature_cols = [c for c in data.columns if c not in ("target", "month")]
        print(f"Dataset: {len(data):,} rows, {len(feature_cols)} features")
        train_final(data, feature_cols)
        print("Done.")
    elif "--analyze" in args:
        idx = args.index("--analyze")
        if idx + 1 >= len(args):
            print("Usage: --analyze TICKER (e.g. --analyze ETH-USD)")
            sys.exit(1)
        ticker = args[idx + 1].upper()
        from portfolio.analyze import run_analysis

        run_analysis(ticker)
    elif "--watch" in args:
        idx = args.index("--watch")
        pos_args = args[idx + 1 :]
        if not pos_args:
            print("Usage: --watch TICKER:ENTRY [TICKER:ENTRY ...]")
            print("Example: --watch BTC:66500 ETH:1920 MSTR:125")
            sys.exit(1)
        from portfolio.analyze import watch_positions

        watch_positions(pos_args)
    elif "--avanza-status" in args:
        from portfolio.avanza_client import get_positions, get_portfolio_value

        positions = get_positions()
        value = get_portfolio_value()
        print(f"Portfolio value: {value:,.0f} SEK")
        if positions:
            for p in positions:
                print(f"  {p}")
    elif "--loop" in args:
        idx = args.index("--loop")
        override = int(args[idx + 1]) if idx + 1 < len(args) else None
        loop(interval=override)
    else:
        run(force_report="--report" in args)
