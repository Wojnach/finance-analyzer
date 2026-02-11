#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data"""

import json
import os
import subprocess
import sys
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
    "MSTR": {"yfinance": "MSTR"},
    "PLTR": {"yfinance": "PLTR"},
    "NVDA": {"yfinance": "NVDA"},
}
USDSEK_RATE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=USDTSEK"

INITIAL_CASH_SEK = 500_000
CONFIDENCE_TELEGRAM = 0.75  # 3/4 signals must agree to alert
BUY_ALLOC = 0.20  # 20% of cash per buy
SELL_ALLOC = 0.50  # sell 50% of position per sell
MIN_TRADE_SEK = 500
TRADE_COOLDOWN_SECONDS = 3600  # 1 hour between trades on same ticker

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


YF_INTERVAL_MAP = {
    "15m": ("15m", "5d"),
    "1h": ("1h", "30d"),
    "4h": ("1h", "60d"),
    "1d": ("1d", "1y"),
    "3d": ("1d", "2y"),
    "1w": ("1wk", "5y"),
    "1M": ("1mo", "10y"),
}


def yfinance_klines(yf_ticker, interval="1d", limit=100):
    import yfinance as yf

    yf_interval, period = YF_INTERVAL_MAP.get(interval, ("1d", "1y"))
    stock = yf.Ticker(yf_ticker)
    df = stock.history(period=period, interval=yf_interval)
    if df.empty:
        raise ValueError(f"No data for {yf_ticker} interval={yf_interval}")
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df.reset_index()
    if "4h" == interval:
        df = df.set_index("Datetime" if "Datetime" in df.columns else "Date")
        df = (
            df.resample("4h")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
    df["time"] = (
        pd.to_datetime(df.iloc[:, 0]) if "time" not in df.columns else df["time"]
    )
    return df.tail(limit)


_fx_cache = {"rate": None, "time": 0}


def fetch_usd_sek():
    now = time.time()
    if _fx_cache["rate"] and now - _fx_cache["time"] < 3600:
        return _fx_cache["rate"]
    try:
        import yfinance as yf

        t = yf.Ticker("USDSEK=X")
        h = t.history(period="5d")
        if not h.empty:
            rate = float(h["Close"].iloc[-1])
            _fx_cache["rate"] = rate
            _fx_cache["time"] = now
            return rate
    except Exception:
        pass
    return _fx_cache["rate"] or 10.50


# --- Indicators ---


def compute_indicators(df):
    close = df["close"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
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
    elif "yfinance" in source:
        return yfinance_klines(source["yfinance"], interval=interval, limit=limit)
    raise ValueError(f"Unknown source: {source}")


# Stocks use daily+ candles (no intraday without market data subscription)
STOCK_TIMEFRAMES = [
    ("Now", "1d", 100, 0),
    ("7d", "1d", 100, 3600),
    ("1mo", "1d", 100, 3600),
    ("3mo", "1w", 100, 43200),
    ("6mo", "1M", 48, 86400),
]


def collect_timeframes(source):
    is_stock = "yfinance" in source
    tfs = STOCK_TIMEFRAMES if is_stock else TIMEFRAMES
    source_key = source.get("yfinance") or source.get("binance")
    results = []
    for label, interval, limit, ttl in tfs:
        cache_key = f"tf_{source_key}_{interval}"
        if ttl > 0:
            cached = _tool_cache.get(cache_key)
            if cached and time.time() - cached["time"] < ttl:
                results.append((label, cached["data"]))
                continue
        try:
            df = _fetch_klines(source, interval, limit)
            ind = compute_indicators(df)
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


MIN_VOTERS = 3  # need at least 3 signals voting to act


def generate_signal(ind, ticker=None, config=None):
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

    # EMA trend — always votes
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

    # Sentiment (crypto->CryptoBERT, stocks->Trading-Hero-LLM)
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
            )
            if sent and sent.get("num_articles", 0) > 0:
                extra_info["sentiment"] = sent["overall_sentiment"]
                extra_info["sentiment_conf"] = sent["confidence"]
                extra_info["sentiment_model"] = sent.get("model", "unknown")
                if sent["overall_sentiment"] == "positive" and sent["confidence"] > 0.4:
                    buy += 1
                elif (
                    sent["overall_sentiment"] == "negative" and sent["confidence"] > 0.4
                ):
                    sell += 1
        except ImportError:
            pass

    # Ministral-8B LLM reasoning
    if ticker:
        short_ticker = ticker.replace("-USD", "")
        try:
            from portfolio.ministral_signal import get_ministral_signal

            ctx = {
                "ticker": short_ticker,
                "price_usd": ind["close"],
                "rsi": round(ind["rsi"], 1),
                "macd_hist": round(ind["macd_hist"], 2),
                "ema_bullish": ind["ema9"] > ind["ema21"],
                "bb_position": ind["price_vs_bb"],
                "fear_greed": extra_info.get("fear_greed", "N/A"),
                "fear_greed_class": extra_info.get("fear_greed_class", ""),
                "news_sentiment": extra_info.get("sentiment", "N/A"),
                "timeframe_summary": "",
                "headlines": "",
            }
            ms = _cached(
                f"ministral_{short_ticker}",
                MINISTRAL_TTL,
                get_ministral_signal,
                ctx,
            )
            if ms:
                extra_info["ministral_action"] = ms["action"]
                extra_info["ministral_reasoning"] = ms.get("reasoning", "")
                if ms["action"] == "BUY":
                    buy += 1
                elif ms["action"] == "SELL":
                    sell += 1
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

    total = buy + sell
    if total < MIN_VOTERS:
        action = "HOLD"
        conf = 0.0
    else:
        buy_conf = buy / total
        sell_conf = sell / total
        if buy_conf > sell_conf and buy_conf >= 0.5:
            action = "BUY"
            conf = buy_conf
        elif sell_conf > buy_conf and sell_conf >= 0.5:
            action = "SELL"
            conf = sell_conf
        else:
            action = "HOLD"
            conf = max(buy_conf, sell_conf)

    extra_info["_voters"] = total
    return action, conf, extra_info


# --- State ---


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "cash_sek": INITIAL_CASH_SEK,
        "holdings": {},
        "transactions": [],
        "start_date": datetime.now(timezone.utc).isoformat(),
        "initial_value_sek": INITIAL_CASH_SEK,
    }


def save_state(state):
    DATA_DIR.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


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
        from portfolio.macro_context import get_dxy

        dxy = _cached("dxy", 3600, get_dxy)
        if dxy:
            summary["macro"] = {"dxy": dxy}
    except (ImportError, Exception):
        pass

    AGENT_SUMMARY_FILE.parent.mkdir(exist_ok=True)
    AGENT_SUMMARY_FILE.write_text(json.dumps(summary, indent=2, default=str))
    return summary


INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"


def invoke_agent(reasons):
    # Log invocation (Layer 1 side — tracks even if agent crashes)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reasons": reasons,
    }
    with open(INVOCATIONS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    agent_bat = BASE_DIR / "scripts" / "win" / "pf-agent.bat"
    if not agent_bat.exists():
        print(f"  WARNING: Agent script not found at {agent_bat}")
        return False
    try:
        log_file = open(DATA_DIR / "agent.log", "a")
        subprocess.Popen(
            ["cmd", "/c", str(agent_bat)],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"  Agent invoked ({', '.join(reasons)})")
        return True
    except Exception as e:
        print(f"  ERROR invoking agent: {e}")
        return False


# --- Trade gating ---


def should_trade(state, ticker, action):
    """Check if trade is allowed. Returns (allowed, reason)."""
    if action == "HOLD":
        return False, "HOLD"

    last_trade = state.get("last_trade", {}).get(ticker, {})

    # State-change gating: block repeat actions
    if last_trade.get("action") == action:
        return False, f"repeat {action}, no state change"

    # Per-symbol cooldown
    if last_trade.get("time"):
        last_time = datetime.fromisoformat(last_trade["time"])
        elapsed = (datetime.now(timezone.utc) - last_time).total_seconds()
        if elapsed < TRADE_COOLDOWN_SECONDS:
            remaining = int(TRADE_COOLDOWN_SECONDS - elapsed)
            return False, f"cooldown ({remaining}s remaining)"

    return True, ""


# --- Trading ---


def execute_trade(state, ticker, action, confidence, price_usd, fx_rate):
    if action == "HOLD":
        return None

    price_sek = price_usd * fx_rate
    holdings = state.setdefault("holdings", {})

    if action == "BUY":
        alloc = state["cash_sek"] * BUY_ALLOC
        if alloc < MIN_TRADE_SEK:
            return None
        shares = alloc / price_sek
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
    }
    state.setdefault("transactions", []).append(trade)
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


def run(force_report=False):
    config = json.loads(CONFIG_FILE.read_text())
    state = load_state()
    fx_rate = fetch_usd_sek()

    print(
        f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] USD/SEK: {fx_rate:.2f}"
    )

    signals = {}
    prices_usd = {}
    tf_data = {}

    for name, source in SYMBOLS.items():
        try:
            tfs = collect_timeframes(source)
            tf_data[name] = tfs

            now_entry = tfs[0][1] if tfs else None
            if now_entry and "indicators" in now_entry:
                ind = now_entry["indicators"]
            else:
                df = _fetch_klines(
                    source, interval="15m" if "binance" in source else "1d", limit=100
                )
                ind = compute_indicators(df)

            price = ind["close"]
            prices_usd[name] = price

            action, conf, extra = generate_signal(ind, ticker=name, config=config)
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

    # Write agent_summary.json every cycle (data for Layer 2)
    write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)

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
        invoke_agent(reasons_list)
    else:
        print("\n  No trigger — nothing changed.")


def loop(interval=60):
    print(f"Starting loop — running every {interval}s. Ctrl+C to stop.")
    # Send a startup report
    run(force_report=True)
    while True:
        time.sleep(interval)
        try:
            run(force_report=False)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  ERROR in run: {e}")
            time.sleep(10)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--retrain" in args:
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
        interval = int(args[idx + 1]) if idx + 1 < len(args) else 60
        loop(interval=interval)
    else:
        run(force_report="--report" in args)
