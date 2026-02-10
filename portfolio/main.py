#!/usr/bin/env python3
"""Portfolio Intelligence System — Simulated Trading on Binance Real-Time Data"""

import json
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
CONFIG_FILE = BASE_DIR / "config.json"

SYMBOLS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
USDSEK_RATE_URL = "https://api.binance.com/api/v3/ticker/price?symbol=USDTSEK"

INITIAL_CASH_SEK = 500_000
CONFIDENCE_TELEGRAM = 0.75  # 3/4 signals must agree to alert
BUY_ALLOC = 0.20  # 20% of cash per buy
SELL_ALLOC = 0.50  # sell 50% of position per sell
MIN_TRADE_SEK = 500

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


def collect_timeframes(symbol):
    results = []
    for label, interval, limit, ttl in TIMEFRAMES:
        cache_key = f"tf_{symbol}_{interval}"
        if ttl > 0:
            cached = _tool_cache.get(cache_key)
            if cached and time.time() - cached["time"] < ttl:
                results.append((label, cached["data"]))
                continue
        try:
            df = binance_klines(symbol, interval=interval, limit=limit)
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

    # Fear & Greed Index (per-ticker: crypto→alternative.me, stocks→VIX)
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

    # Sentiment (crypto→CryptoBERT, stocks→Trading-Hero-LLM)
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
        timeout=10,
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
    trades = []
    prices_usd = {}
    tf_data = {}

    for name, symbol in SYMBOLS.items():
        try:
            # Collect all timeframes
            tfs = collect_timeframes(symbol)
            tf_data[name] = tfs

            # "Now" is the first timeframe — use it for trading with all 6 signals
            now_entry = tfs[0][1] if tfs else None
            if now_entry and "indicators" in now_entry:
                ind = now_entry["indicators"]
            else:
                df = binance_klines(symbol, interval="15m", limit=100)
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

            trade = execute_trade(state, name, action, conf, price, fx_rate)
            if trade:
                trades.append(trade)
                print(f"    >>> {action} {trade['shares']:.6f} @ ${price:,.2f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    total = portfolio_value(state, prices_usd, fx_rate)
    pnl_pct = ((total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
    print(
        f"\n  Portfolio: {total:,.0f} SEK ({pnl_pct:+.2f}%) | Cash: {state['cash_sek']:,.0f} SEK"
    )

    save_state(state)

    # Send Telegram if high-confidence signal, trade executed, or forced
    high_conf = any(s["confidence"] >= CONFIDENCE_TELEGRAM for s in signals.values())
    if high_conf or trades or force_report:
        msg = build_report(state, signals, trades, prices_usd, fx_rate, tf_data)
        reason = (
            "forced" if force_report else "high-confidence" if high_conf else "trade"
        )
        print(f"\n  Sending Telegram ({reason})...")
        if send_telegram(msg, config):
            print("  Sent!")
        else:
            print("  FAILED to send!")
    else:
        print("\n  No alert needed — low confidence, no trades.")


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
    if "--loop" in args:
        idx = args.index("--loop")
        interval = int(args[idx + 1]) if idx + 1 < len(args) else 60
        loop(interval=interval)
    else:
        run(force_report="--report" in args)
