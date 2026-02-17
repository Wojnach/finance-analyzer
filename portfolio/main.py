#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data"""

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "portfolio_state.json"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
CONFIG_FILE = BASE_DIR / "config.json"

SYMBOLS = {
    "BTC-USD": {"binance": "BTCUSDT"},
    "ETH-USD": {"binance": "ETHUSDT"},
    "MSTR": {"alpaca": "MSTR"},
    "PLTR": {"alpaca": "PLTR"},
    "NVDA": {"alpaca": "NVDA"},
}
CRYPTO_SYMBOLS = {"BTC-USD", "ETH-USD"}
STOCK_SYMBOLS = {"MSTR", "PLTR", "NVDA"}

# Market hours (UTC) — EU open to US close
MARKET_OPEN_HOUR = 7  # ~Frankfurt/London open
MARKET_CLOSE_HOUR = 21  # ~NYSE close

# Loop intervals by market state
INTERVAL_MARKET_OPEN = 60  # 1 min — full speed
INTERVAL_MARKET_CLOSED = 300  # 5 min — crypto only weekday nights
INTERVAL_WEEKEND = 600  # 10 min — crypto only weekends


def get_market_state():
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon, 6=Sun
    hour = now.hour
    if weekday >= 5:
        return "weekend", CRYPTO_SYMBOLS, INTERVAL_WEEKEND
    if MARKET_OPEN_HOUR <= hour < MARKET_CLOSE_HOUR:
        return "open", set(SYMBOLS.keys()), INTERVAL_MARKET_OPEN
    return "closed", CRYPTO_SYMBOLS, INTERVAL_MARKET_CLOSED


USDSEK_RATE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=USDTSEK"

INITIAL_CASH_SEK = 500_000
CONFIDENCE_TELEGRAM = 0.75  # 3/4 signals must agree to alert
BUY_ALLOC = 0.20  # 20% of cash per buy
SELL_ALLOC = 0.50  # sell 50% of position per sell
MIN_TRADE_SEK = 500
FEE_CRYPTO = 0.0005  # 0.05% taker fee (Binance futures)
FEE_STOCK = 0.001  # 0.10% (typical broker commission)

BINANCE_BASE = "https://api.binance.com/api/v3"

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
SENTIMENT_TTL = 900  # 15 min
MINISTRAL_TTL = 900  # 15 min
ML_SIGNAL_TTL = 900  # 15 min
FUNDING_RATE_TTL = 900  # 15 min
VOLUME_TTL = 300  # 5 min


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
            return _tool_cache[key]["data"]
        return None


# --- Binance API ---


def binance_price(symbol):
    r = requests.get(
        f"{BINANCE_BASE}/ticker/price", params={"symbol": symbol}, timeout=5
    )
    r.raise_for_status()
    return float(r.json()["price"])


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


ALPACA_BASE = "https://data.alpaca.markets/v2"
ALPACA_INTERVAL_MAP = {
    "15m": ("15Min", 5),
    "1h": ("1Hour", 10),
    "4h": ("4Hour", 30),
    "1d": ("1Day", 365),
    "1w": ("1Week", 730),
    "1M": ("1Month", 1825),
}


def _get_alpaca_headers():
    cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
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


_fx_cache = {"rate": None, "time": 0}


def fetch_usd_sek():
    now = time.time()
    if _fx_cache["rate"] and now - _fx_cache["time"] < 3600:
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
    except Exception:
        pass
    return _fx_cache["rate"] or 10.50


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
    }


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
    if "binance" in source:
        return binance_klines(source["binance"], interval=interval, limit=limit)
    elif "alpaca" in source:
        return alpaca_klines(source["alpaca"], interval=interval, limit=limit)
    raise ValueError(f"Unknown source: {source}")


STOCK_TIMEFRAMES = [
    ("Now", "15m", 100, 0),
    ("12h", "1h", 100, 300),
    ("2d", "4h", 100, 900),
    ("7d", "1d", 30, 3600),
    ("1mo", "1d", 100, 3600),
    ("3mo", "1w", 100, 43200),
    ("6mo", "1M", 48, 86400),
]


def collect_timeframes(source):
    is_stock = "alpaca" in source
    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
    source_key = source.get("alpaca") or source.get("binance")
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
            if ttl > 0:
                _tool_cache[cache_key] = {"data": entry, "time": time.time()}
            results.append((label, entry))
        except Exception as e:
            results.append((label, {"error": str(e)}))
    return results


# --- Signal (full 7-signal for "Now" timeframe) ---


MIN_VOTERS_CRYPTO = 3  # crypto has 11 signals — need 3 active voters
MIN_VOTERS_STOCK = 2  # stocks have 7 signals (~71% abstention) — 2 suffices

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


def generate_signal(ind, ticker=None, config=None, timeframes=None):
    buy = 0
    sell = 0
    extra_info = {}

    # RSI — only votes at extremes
    if ind["rsi"] < 30:
        buy += 1
    elif ind["rsi"] > 70:
        sell += 1

    # MACD — only votes on crossover
    if ind["macd_hist"] > 0 and ind["macd_hist_prev"] <= 0:
        buy += 1
    elif ind["macd_hist"] < 0 and ind["macd_hist_prev"] >= 0:
        sell += 1

    # EMA trend — votes only when gap is meaningful (>0.5%)
    ema_gap_pct = (
        abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] * 100 if ind["ema21"] != 0 else 0
    )
    if ema_gap_pct >= 0.5:
        if ind["ema9"] > ind["ema21"]:
            buy += 1
        else:
            sell += 1

    # Bollinger Bands — only votes at extremes
    if ind["price_vs_bb"] == "below_lower":
        buy += 1
    elif ind["price_vs_bb"] == "above_upper":
        sell += 1

    # --- Extended signals from tools (optional) ---

    # Fear & Greed Index (per-ticker: crypto->alternative.me, stocks->VIX)
    try:
        from portfolio.fear_greed import get_fear_greed

        fg_key = f"fear_greed_{ticker}" if ticker else "fear_greed"
        fg = _cached(fg_key, FEAR_GREED_TTL, get_fear_greed, ticker)
        if fg:
            extra_info["fear_greed"] = fg["value"]
            extra_info["fear_greed_class"] = fg["classification"]
            if fg["value"] <= 20:
                buy += 1
            elif fg["value"] >= 80:
                sell += 1
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

                # Determine confidence threshold with hysteresis
                prev_sent_dir = _get_prev_sentiment(ticker)
                current_dir = sent["overall_sentiment"]
                if (
                    prev_sent_dir
                    and current_dir != prev_sent_dir
                    and current_dir != "neutral"
                ):
                    sent_threshold = 0.55  # flipping direction — higher bar
                else:
                    sent_threshold = 0.40  # same direction or first reading

                if (
                    sent["overall_sentiment"] == "positive"
                    and sent["confidence"] > sent_threshold
                ):
                    buy += 1
                    _set_prev_sentiment(ticker, "positive")
                elif (
                    sent["overall_sentiment"] == "negative"
                    and sent["confidence"] > sent_threshold
                ):
                    sell += 1
                    _set_prev_sentiment(ticker, "negative")
        except ImportError:
            pass

    # ML Classifier (HistGradientBoosting on BTC/ETH 1h data)
    if ticker:
        try:
            from portfolio.ml_signal import get_ml_signal

            ml = _cached(f"ml_{ticker}", ML_SIGNAL_TTL, get_ml_signal, ticker)
            if ml:
                extra_info["ml_action"] = ml["action"]
                extra_info["ml_confidence"] = ml["confidence"]
                if ml["action"] == "BUY":
                    buy += 1
                elif ml["action"] == "SELL":
                    sell += 1
        except ImportError:
            pass

    # Funding Rate (Binance perpetuals, crypto only — contrarian)
    if ticker:
        try:
            from portfolio.funding_rate import get_funding_rate

            fr = _cached(
                f"funding_{ticker}", FUNDING_RATE_TTL, get_funding_rate, ticker
            )
            if fr:
                extra_info["funding_rate"] = fr["rate_pct"]
                extra_info["funding_action"] = fr["action"]
                if fr["action"] == "BUY":
                    buy += 1
                elif fr["action"] == "SELL":
                    sell += 1
        except ImportError:
            pass

    # Volume Confirmation (spike + price direction = vote)
    if ticker:
        try:
            from portfolio.macro_context import get_volume_signal

            vs = _cached(f"volume_{ticker}", VOLUME_TTL, get_volume_signal, ticker)
            if vs:
                extra_info["volume_ratio"] = vs["ratio"]
                extra_info["volume_action"] = vs["action"]
                if vs["action"] == "BUY":
                    buy += 1
                elif vs["action"] == "SELL":
                    sell += 1
        except ImportError:
            pass

    # Ministral-8B LLM reasoning (original CryptoTrader-LM + custom LoRA, crypto only)
    # Runs AFTER all other signals so it can include their results in its context
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
                if orig["action"] == "BUY":
                    buy += 1
                elif orig["action"] == "SELL":
                    sell += 1

                cust = ms.get("custom")
                if cust:
                    extra_info["custom_lora_action"] = cust["action"]
                    extra_info["custom_lora_reasoning"] = cust.get("reasoning", "")
                    if cust["action"] == "BUY":
                        buy += 1
                    elif cust["action"] == "SELL":
                        sell += 1
        except ImportError:
            pass

    # Total applicable signals: crypto has 4 extra (CryptoTrader-LM, Custom LoRA, ML, Funding Rate)
    is_crypto = ticker in CRYPTO_SYMBOLS
    total_applicable = 11 if is_crypto else 7

    active_voters = buy + sell
    min_voters = MIN_VOTERS_STOCK if ticker in STOCK_SYMBOLS else MIN_VOTERS_CRYPTO
    if active_voters < min_voters:
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

    extra_info["_voters"] = active_voters
    extra_info["_total_applicable"] = total_applicable
    extra_info["_buy_count"] = buy
    extra_info["_sell_count"] = sell
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
        summary["signals"][name] = {
            "action": sig["action"],
            "confidence": sig["confidence"],
            "price_usd": ind["close"],
            "rsi": round(ind["rsi"], 1),
            "macd_hist": round(ind["macd_hist"], 2),
            "bb_position": ind["price_vs_bb"],
            "extra": extra,
        }
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

    _atomic_write_json(AGENT_SUMMARY_FILE, summary)
    return summary


INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"


_agent_proc = None
_agent_log = None
_agent_start = 0
AGENT_TIMEOUT = 600


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

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
    }
    with open(INVOCATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
    if not agent_bat.exists():
        print(f"  WARNING: Agent script not found at {agent_bat}")
        return False
    try:
        _agent_log = open(DATA_DIR / "agent.log", "a", encoding="utf-8")
        _agent_proc = subprocess.Popen(
            ["cmd", "/c", str(agent_bat)],
            cwd=str(BASE_DIR),
            stdout=_agent_log,
            stderr=subprocess.STDOUT,
        )
        _agent_start = time.time()
        print(f"  Agent invoked pid={_agent_proc.pid} ({', '.join(reasons)})")
        return True
    except Exception as e:
        print(f"  ERROR invoking agent: {e}")
        return False


# --- Trade gating ---


def should_trade(state, ticker, action):
    """Check if trade is allowed. Returns (allowed, reason)."""
    if action == "HOLD":
        return False, "HOLD"
    return True, ""


# --- Trading ---


def execute_trade(state, ticker, action, confidence, price_usd, fx_rate):
    if action == "HOLD":
        return None

    price_sek = price_usd * fx_rate
    fee_rate = FEE_CRYPTO if ticker in CRYPTO_SYMBOLS else FEE_STOCK
    holdings = state.setdefault("holdings", {})

    if action == "BUY":
        alloc = state["cash_sek"] * BUY_ALLOC
        if alloc < MIN_TRADE_SEK:
            return None
        fee = alloc * fee_rate
        net_alloc = alloc - fee
        shares = net_alloc / price_sek
        cur = holdings.get(ticker, {"shares": 0, "avg_cost_usd": 0})
        total = cur["shares"] + shares
        avg = (
            (cur["shares"] * cur["avg_cost_usd"] + shares * price_usd) / total
            if total > 0
            else price_usd
        )
        holdings[ticker] = {"shares": total, "avg_cost_usd": avg}
        state["cash_sek"] -= alloc

    elif action == "SELL":
        cur = holdings.get(ticker, {"shares": 0, "avg_cost_usd": 0})
        if cur["shares"] <= 0:
            return None
        sell_shares = cur["shares"] * SELL_ALLOC
        proceeds = sell_shares * price_sek
        fee = proceeds * fee_rate
        proceeds -= fee
        cur["shares"] -= sell_shares
        holdings[ticker] = cur
        state["cash_sek"] += proceeds
        shares = sell_shares

    trade = {
        "time": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "action": action,
        "shares": shares,
        "price_usd": price_usd,
        "price_sek": price_sek,
        "confidence": confidence,
        "fx_rate": fx_rate,
        "fee_sek": round(fee, 2),
    }
    state.setdefault("transactions", []).append(trade)
    state["total_fees_sek"] = round(state.get("total_fees_sek", 0) + fee, 2)
    return trade


def portfolio_value(state, prices_usd, fx_rate):
    total = state["cash_sek"]
    for ticker, h in state.get("holdings", {}).items():
        if h["shares"] > 0 and ticker in prices_usd:
            total += h["shares"] * prices_usd[ticker] * fx_rate
    return total


# --- Telegram ---


def send_telegram(msg, config):
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=30,
    )
    return r.ok


BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"
_COOLDOWN_PREFIXES = ("cooldown", "crypto check-in", "startup")


def _maybe_send_alert(config, signals, prices_usd, fx_rate, state, reasons, tf_data):
    significant = [r for r in reasons if not r.startswith(_COOLDOWN_PREFIXES)]
    if not significant:
        return
    headline = significant[0]
    lines = [f"*ALERT: {headline}*", ""]
    for ticker in SYMBOLS:
        sig = signals.get(ticker)
        if not sig:
            continue
        price = prices_usd.get(ticker, 0)
        extra = sig.get("extra", {})
        b = extra.get("_buy_count", 0)
        s = extra.get("_sell_count", 0)
        total = extra.get("_total_applicable", 0)
        h = total - b - s
        action = sig["action"]
        if price >= 1000:
            p_str = f"${price:,.0f}"
        else:
            p_str = f"${price:,.2f}"
        lines.append(f"`{ticker:<7} {p_str:>9}  {action:<4} {b}B/{s}S/{h}H`")
    fg_val = ""
    for ticker, sig in signals.items():
        extra = sig.get("extra", {})
        if "fear_greed" in extra:
            fg_val = f"{extra['fear_greed']} ({extra.get('fear_greed_class', '')})"
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


def _rsi_arrow(rsi):
    if rsi < 30:
        return "oversold"
    if rsi < 45:
        return "low"
    if rsi > 70:
        return "overbought"
    if rsi > 55:
        return "high"
    return ""


def build_report(state, signals, trades, prices_usd, fx_rate, tf_data=None):
    total = portfolio_value(state, prices_usd, fx_rate)
    pnl = total - state["initial_value_sek"]
    pnl_pct = (pnl / state["initial_value_sek"]) * 100
    lines = []

    # --- Lead with the action ---
    if trades:
        for t in trades:
            amt_sek = t["shares"] * t["price_sek"]
            lines.append(
                f"*{t['action']} {t['ticker']}* — {t['shares']:.6f} @ ${t['price_usd']:,.2f} ({amt_sek:,.0f} SEK)"
            )
    else:
        for ticker, sig in signals.items():
            voters = sig.get("extra", {}).get("_voters", 0)
            lines.append(
                f"*{sig['action']} {ticker}* ({sig['confidence']:.0%}, {voters} voting)"
            )

    lines.append("")

    # --- Per-instrument compact breakdown ---
    for ticker, sig in signals.items():
        extra = sig.get("extra", {})
        now_ind = sig["indicators"]

        # Header: ticker + price + signals summary
        parts = [f"*{ticker}*  ${now_ind['close']:,.2f}"]
        sig_parts = []
        if "fear_greed" in extra:
            sig_parts.append(f"F&G:{extra['fear_greed']}")
        if "sentiment" in extra:
            sig_parts.append(f"News:{extra['sentiment']}")
        if "ministral_action" in extra:
            sig_parts.append(f"8B:{extra['ministral_action']}")
        if "custom_lora_action" in extra:
            sig_parts.append(f"LoRA:{extra['custom_lora_action']}")
        if "ml_action" in extra:
            sig_parts.append(f"ML:{extra['ml_action']}")
        if sig_parts:
            parts.append(" | ".join(sig_parts))
        lines.append("  ".join(parts))

        # Timeframe table — compact
        ticker_tfs = tf_data.get(ticker, []) if tf_data else []
        for label, entry in ticker_tfs:
            if "error" in entry:
                continue
            ind = entry["indicators"]
            rsi = ind["rsi"]
            macd_dir = "+" if ind["macd_hist"] > 0 else "-"
            ema_dir = "+" if ind["ema9"] > ind["ema21"] else "-"
            rsi_note = _rsi_arrow(rsi)

            if label == "Now":
                action = sig["action"]
                conf = sig["confidence"]
            else:
                action = entry["action"]
                conf = entry["confidence"]

            rsi_str = f"{rsi:.0f}"
            if rsi_note:
                rsi_str += f"({rsi_note})"
            lines.append(
                f"  {label:<4} *{action}* {conf:.0%} R{rsi_str} M{macd_dir} E{ema_dir}"
            )

        # Ministral reasoning — truncate to last full sentence
        reasoning = extra.get("ministral_reasoning", "")
        if reasoning:
            if len(reasoning) > 120:
                cut = reasoning[:120]
                last_stop = max(cut.rfind("."), cut.rfind("!"), cut.rfind(";"))
                if last_stop > 40:
                    reasoning = cut[: last_stop + 1]
                else:
                    reasoning = cut.rstrip() + "..."
            lines.append(f"  _8B: {reasoning}_")
        lines.append("")

    # --- Portfolio ---
    lines.append(f"Portfolio: *{total:,.0f} SEK* ({pnl_pct:+.2f}%)")

    holdings_parts = []
    for ticker, h in state.get("holdings", {}).items():
        if h["shares"] > 0:
            val = h["shares"] * prices_usd.get(ticker, 0) * fx_rate
            cost = h["avg_cost_usd"]
            cur = prices_usd.get(ticker, cost)
            ticker_pnl = ((cur - cost) / cost * 100) if cost > 0 else 0
            holdings_parts.append(f"{ticker}: {val:,.0f} SEK ({ticker_pnl:+.1f}%)")
    if holdings_parts:
        lines.append(" | ".join(holdings_parts))

    lines.append(f"Cash: {state['cash_sek']:,.0f} SEK")

    # --- Date at bottom ---
    lines.append("")
    lines.append(f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")

    return "\n".join(lines)


# --- Main ---


def run(force_report=False, active_symbols=None):
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    state = load_state()
    fx_rate = fetch_usd_sek()

    market_state, default_symbols, _ = get_market_state()
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
            if now_entry and "indicators" in now_entry:
                ind = now_entry["indicators"]
            else:
                df = _fetch_klines(source, interval="15m", limit=100)
                ind = compute_indicators(df)

            if ind is None:
                print(f"  {name}: insufficient data, skipping")
                continue
            price = ind["close"]
            prices_usd[name] = price

            action, conf, extra = generate_signal(
                ind, ticker=name, config=config, timeframes=tfs
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
                if "custom_lora_action" in extra:
                    parts.append(f"LoRA:{extra['custom_lora_action']}")
                if "ml_action" in extra:
                    parts.append(f"ML:{extra['ml_action']}")
                if "funding_action" in extra:
                    parts.append(f"FR:{extra['funding_rate']}%")
                if "volume_action" in extra and extra["volume_action"] != "HOLD":
                    parts.append(f"Vol:{extra['volume_ratio']}x")
                if parts:
                    extra_str = f" | {' '.join(parts)}"
            print(
                f"  {name}: ${price:,.2f} | RSI {ind['rsi']:.0f} | MACD {ind['macd_hist']:+.1f}{extra_str} | {action} ({conf:.0%})"
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
        if layer2_cfg.get("enabled", True):
            invoke_agent(reasons_list)
        else:
            print("  Layer 2 disabled — skipping agent invocation")
            _maybe_send_alert(
                config, signals, prices_usd, fx_rate, state, reasons_list, tf_data
            )
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


def loop(interval=None):
    print("Starting loop with market-aware scheduling. Ctrl+C to stop.")
    try:
        run(force_report=True)
    except KeyboardInterrupt:
        raise
    except Exception as e:
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
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ERROR in run: {e}")
            time.sleep(10)


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
    elif "--loop" in args:
        idx = args.index("--loop")
        override = int(args[idx + 1]) if idx + 1 < len(args) else None
        loop(interval=override)
    else:
        run(force_report="--report" in args)
