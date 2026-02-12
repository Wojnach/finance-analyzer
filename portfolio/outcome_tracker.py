import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"

HORIZONS = {"1d": 86400, "3d": 259200, "5d": 432000, "10d": 864000}
BINANCE_MAP = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
YF_MAP = {"MSTR": "MSTR", "PLTR": "PLTR", "NVDA": "NVDA"}

SIGNAL_NAMES = [
    "rsi",
    "macd",
    "ema",
    "bb",
    "fear_greed",
    "sentiment",
    "ministral",
    "ml",
    "funding",
    "volume",
    "custom_lora",
]


def _derive_signal_vote(name, indicators, extra):
    if name == "rsi":
        rsi = indicators.get("rsi")
        if rsi is None:
            return "HOLD"
        if rsi < 30:
            return "BUY"
        if rsi > 70:
            return "SELL"
        return "HOLD"

    if name == "macd":
        hist = indicators.get("macd_hist")
        hist_prev = indicators.get("macd_hist_prev")
        if hist is None or hist_prev is None:
            return "HOLD"
        if hist > 0 and hist_prev <= 0:
            return "BUY"
        if hist < 0 and hist_prev >= 0:
            return "SELL"
        return "HOLD"

    if name == "ema":
        ema9 = indicators.get("ema9")
        ema21 = indicators.get("ema21")
        if ema9 is None or ema21 is None:
            return "HOLD"
        return "BUY" if ema9 > ema21 else "SELL"

    if name == "bb":
        pos = indicators.get("price_vs_bb")
        if pos == "below_lower":
            return "BUY"
        if pos == "above_upper":
            return "SELL"
        return "HOLD"

    if name == "fear_greed":
        fg = extra.get("fear_greed")
        if fg is None:
            return "HOLD"
        if fg <= 20:
            return "BUY"
        if fg >= 80:
            return "SELL"
        return "HOLD"

    if name == "sentiment":
        sent = extra.get("sentiment")
        conf = extra.get("sentiment_conf", 0)
        if sent == "positive" and conf > 0.4:
            return "BUY"
        if sent == "negative" and conf > 0.4:
            return "SELL"
        return "HOLD"

    if name == "ministral":
        return extra.get("ministral_action", "HOLD")

    if name == "ml":
        return extra.get("ml_action", "HOLD")

    if name == "funding":
        return extra.get("funding_action", "HOLD")

    if name == "volume":
        return extra.get("volume_action", "HOLD")

    if name == "custom_lora":
        return extra.get("custom_lora_action", "HOLD")

    return "HOLD"


def log_signal_snapshot(signals_dict, prices_usd, fx_rate, trigger_reasons):
    ts = datetime.now(timezone.utc).isoformat()
    tickers = {}

    for ticker, sig_data in signals_dict.items():
        indicators = sig_data.get("indicators", {})
        extra = sig_data.get("extra", {})
        price = prices_usd.get(ticker, indicators.get("close"))

        signals = {}
        buy_count = 0
        sell_count = 0
        for name in SIGNAL_NAMES:
            vote = _derive_signal_vote(name, indicators, extra)
            signals[name] = vote
            if vote == "BUY":
                buy_count += 1
            elif vote == "SELL":
                sell_count += 1

        consensus = sig_data.get("action", "HOLD")
        total_voters = buy_count + sell_count

        tickers[ticker] = {
            "price_usd": price,
            "consensus": consensus,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_voters": total_voters,
            "signals": signals,
        }

    entry = {
        "ts": ts,
        "trigger_reasons": trigger_reasons,
        "fx_rate": fx_rate,
        "tickers": tickers,
        "outcomes": {},
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def _fetch_current_price(ticker):
    if ticker in BINANCE_MAP:
        symbol = BINANCE_MAP[ticker]
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": symbol},
            timeout=5,
        )
        r.raise_for_status()
        return float(r.json()["price"])

    if ticker in YF_MAP:
        import yfinance as yf

        t = yf.Ticker(YF_MAP[ticker])
        h = t.history(period="5d")
        if h.empty:
            return None
        return float(h["Close"].iloc[-1])

    return None


def backfill_outcomes():
    if not SIGNAL_LOG.exists():
        return 0

    entries = []
    with open(SIGNAL_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()
    price_cache = {}
    updated = 0

    for entry in entries:
        entry_ts = datetime.fromisoformat(entry["ts"]).timestamp()
        tickers = entry.get("tickers", {})
        outcomes = entry.get("outcomes", {})

        all_filled = True
        for ticker in tickers:
            if ticker not in outcomes:
                outcomes[ticker] = {"1d": None, "3d": None, "5d": None, "10d": None}
            for h_key in HORIZONS:
                if outcomes[ticker].get(h_key) is None:
                    all_filled = False

        if all_filled and all(
            all(outcomes[t].get(h) is not None for h in HORIZONS) for t in tickers
        ):
            continue

        entry_updated = False
        for ticker in tickers:
            if ticker not in outcomes:
                outcomes[ticker] = {"1d": None, "3d": None, "5d": None, "10d": None}

            base_price = tickers[ticker].get("price_usd")
            for h_key, h_seconds in HORIZONS.items():
                if outcomes[ticker].get(h_key) is not None:
                    continue
                if now_ts < entry_ts + h_seconds:
                    continue

                if ticker not in price_cache:
                    try:
                        price_cache[ticker] = _fetch_current_price(ticker)
                    except Exception:
                        price_cache[ticker] = None

                current_price = price_cache[ticker]
                if current_price is None:
                    continue

                change_pct = 0.0
                if base_price and base_price > 0:
                    change_pct = round(
                        ((current_price - base_price) / base_price) * 100, 2
                    )

                outcomes[ticker][h_key] = {
                    "price_usd": round(current_price, 2),
                    "change_pct": change_pct,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                entry_updated = True

        entry["outcomes"] = outcomes
        if entry_updated:
            updated += 1

    with open(SIGNAL_LOG, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return updated


if __name__ == "__main__":
    count = backfill_outcomes()
    print(f"Backfilled {count} entries")
