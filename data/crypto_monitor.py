"""
BTC / ETH / MSTR Price Monitor with Claude Analysis

Three correlated instruments tracked together:
  - BTC-USD: Binance spot (BTCUSDT)
  - ETH-USD: Binance spot (ETHUSDT)
  - MSTR: Yahoo Finance (after-hours capable)

Two layers:
  - Fast (15s): price checks for all three, console status line
  - Analysis (10min): comprehensive data -> Claude Code analyzes (Sonnet)
    Telegram warnings are DISABLED by default (set SEND_WARNINGS=True to enable)

Run: .venv/Scripts/python.exe -u data/crypto_monitor.py
"""
import json, time, datetime, requests, sys, os, subprocess, shutil, platform
from collections import deque
from pathlib import Path

# === Config ===
CHECK_INTERVAL = 15              # price check every 15s
ANALYSIS_INTERVAL = 600          # Claude analysis every 10 min
VELOCITY_WINDOW = 12             # 12 readings at 15s = 3 min window
VELOCITY_ALERT_PCT = -2.0        # BTC/ETH can move fast, wider threshold
SEND_WARNINGS = False            # Telegram warnings disabled by default

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INSTRUMENTS = {
    "BTC": {
        "symbol": "BTCUSDT",
        "source": "binance",
        "name": "Bitcoin",
    },
    "ETH": {
        "symbol": "ETHUSDT",
        "source": "binance",
        "name": "Ethereum",
    },
    "MSTR": {
        "symbol": "MSTR",
        "source": "yahoo",
        "name": "Strategy Inc (MSTR)",
    },
}

BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_24HR = "https://api.binance.com/api/v3/ticker/24hr"
CRYPTOCOMPARE_NEWS = "https://min-api.cryptocompare.com/data/v2/news/"
FEAR_GREED_API = "https://api.alternative.me/fng/"

# Load Telegram config
with open(BASE_DIR / "config.json") as f:
    cfg = json.load(f)
TG_TOKEN = cfg["telegram"]["token"]
TG_CHAT = cfg["telegram"]["chat_id"]

# === State ===
price_history = {k: deque(maxlen=VELOCITY_WINDOW) for k in INSTRUMENTS}
session_prices = {k: [] for k in INSTRUMENTS}
session_lows = {}
session_highs = {}
start_prices = {}  # first price captured this session
last_analysis_ts = 0
start_time = None
analysis_count = 0
analysis_history = []


last_news_ts = 0
NEWS_INTERVAL = 300  # fetch news every 5 min
cached_news = []
cached_fear_greed = None


def fetch_crypto_news():
    """Fetch latest crypto news from CryptoCompare + Fear & Greed."""
    global cached_news, cached_fear_greed, last_news_ts

    now = time.time()
    if now - last_news_ts < NEWS_INTERVAL and cached_news:
        return cached_news, cached_fear_greed

    headlines = []

    # CryptoCompare news (free, no key needed)
    try:
        r = requests.get(CRYPTOCOMPARE_NEWS, params={"lang": "EN", "sortOrder": "latest"},
                         timeout=8)
        if r.status_code == 200:
            articles = r.json().get("Data", [])
            for a in articles[:15]:
                title = a.get("title", "")
                source = a.get("source", "")
                cats = a.get("categories", "")
                published = a.get("published_on", 0)
                # Filter for BTC/ETH/MSTR relevance
                relevant = any(kw in title.lower() or kw in cats.lower()
                               for kw in ["bitcoin", "btc", "ethereum", "eth", "microstrategy",
                                           "strategy inc", "mstr", "crypto", "etf", "halving",
                                           "whale", "institutional"])
                if relevant:
                    headlines.append({
                        "title": title,
                        "source": source,
                        "categories": cats,
                        "age_min": round((now - published) / 60),
                    })
    except Exception as e:
        print(f"  [!] CryptoCompare news error: {e}")

    # Fear & Greed Index
    fg = None
    try:
        r = requests.get(FEAR_GREED_API, params={"limit": 2}, timeout=5)
        if r.status_code == 200:
            data = r.json().get("data", [])
            if data:
                fg = {
                    "value": int(data[0]["value"]),
                    "class": data[0]["value_classification"],
                    "yesterday": int(data[1]["value"]) if len(data) > 1 else None,
                }
    except Exception as e:
        print(f"  [!] Fear & Greed error: {e}")

    cached_news = headlines
    cached_fear_greed = fg
    last_news_ts = now
    return headlines, fg


def fetch_binance_price(symbol):
    try:
        r = requests.get(BINANCE_TICKER, params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        return None


def fetch_binance_24hr(symbol):
    try:
        r = requests.get(BINANCE_24HR, params={"symbol": symbol}, timeout=5)
        r.raise_for_status()
        d = r.json()
        return {
            "price": float(d["lastPrice"]),
            "change_pct": float(d["priceChangePercent"]),
            "high_24h": float(d["highPrice"]),
            "low_24h": float(d["lowPrice"]),
            "volume": float(d["volume"]),
        }
    except Exception:
        return None


def fetch_yahoo_price(symbol):
    """Fetch stock price from Yahoo Finance v8 API."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, params={"interval": "1m", "range": "1d"}, timeout=8)
        r.raise_for_status()
        data = r.json()
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("previousClose", price)
        return {
            "price": price,
            "prev_close": prev_close,
            "change_pct": round((price - prev_close) / prev_close * 100, 2) if prev_close else 0,
        }
    except Exception as e:
        return None


def fetch_binance_klines(symbol, interval="5m", limit=50):
    try:
        r = requests.get(BINANCE_KLINES,
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def calc_ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def calc_bb(closes, period=20, std_mult=2):
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    sma = sum(window) / period
    std = (sum((x - sma) ** 2 for x in window) / period) ** 0.5
    return sma - std_mult * std, sma, sma + std_mult * std


def calc_macd(closes, fast=12, slow=26):
    fast_ema = calc_ema(closes, fast)
    slow_ema = calc_ema(closes, slow)
    if fast_ema is None or slow_ema is None:
        return None, None
    macd_line = fast_ema - slow_ema
    prev_fast = calc_ema(closes[:-1], fast)
    prev_slow = calc_ema(closes[:-1], slow)
    if prev_fast and prev_slow:
        prev_macd = prev_fast - prev_slow
        return macd_line, macd_line - prev_macd
    return macd_line, None


def send_telegram(msg):
    if not SEND_WARNINGS:
        print(f"  >> TG DISABLED (would send: {msg[:60]}...)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
        ok = r.json().get("ok", False)
        print(f"  >> TG {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  [!] TG error: {e}")
        return False


def fetch_all_prices():
    """Fetch prices for all three instruments."""
    prices = {}
    for key, inst in INSTRUMENTS.items():
        if inst["source"] == "binance":
            p = fetch_binance_price(inst["symbol"])
            if p is not None:
                prices[key] = p
        elif inst["source"] == "yahoo":
            data = fetch_yahoo_price(inst["symbol"])
            if data and data["price"] > 0:
                prices[key] = data["price"]
    return prices


def gather_analysis_data(prices):
    """Collect comprehensive data for Claude analysis."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    data = {
        "timestamp": now_utc.isoformat(),
        "instruments": {},
        "correlations": {},
        "context": {},
    }

    # Per-instrument data
    for key, inst in INSTRUMENTS.items():
        price = prices.get(key)
        if price is None:
            continue

        inst_data = {
            "name": inst["name"],
            "price": price,
            "session_low": session_lows.get(key),
            "session_high": session_highs.get(key),
        }

        # Session change
        if key in start_prices and start_prices[key] > 0:
            inst_data["session_change_pct"] = round(
                (price - start_prices[key]) / start_prices[key] * 100, 3)

        # 24h data for Binance instruments
        if inst["source"] == "binance":
            d24 = fetch_binance_24hr(inst["symbol"])
            if d24:
                inst_data["change_24h_pct"] = d24["change_pct"]
                inst_data["high_24h"] = d24["high_24h"]
                inst_data["low_24h"] = d24["low_24h"]
                inst_data["volume_24h"] = d24["volume"]

            # Technicals from klines
            inst_data["technicals"] = {}
            for interval, label, limit in [("5m", "5m", 60), ("15m", "15m", 50), ("1h", "1h", 48), ("4h", "4h", 30)]:
                raw = fetch_binance_klines(inst["symbol"], interval, limit)
                if raw:
                    closes = [float(k[4]) for k in raw]
                    volumes = [float(k[5]) for k in raw]

                    rsi = calc_rsi(closes, 14)
                    macd_line, macd_delta = calc_macd(closes)
                    bb_lower, bb_mid, bb_upper = calc_bb(closes)

                    changes = {}
                    for n in [5, 10, 20]:
                        if len(closes) >= n:
                            changes[f"last_{n}"] = round(
                                (closes[-1] - closes[-n]) / closes[-n] * 100, 3)

                    vol_ratio = None
                    if len(volumes) >= 20:
                        avg_vol = sum(volumes[-20:]) / 20
                        recent_vol = sum(volumes[-5:]) / 5
                        vol_ratio = round(recent_vol / avg_vol, 2) if avg_vol > 0 else None

                    inst_data["technicals"][label] = {
                        "rsi": round(rsi, 1) if rsi else None,
                        "macd_line": round(macd_line, 4) if macd_line else None,
                        "macd_delta": round(macd_delta, 4) if macd_delta else None,
                        "bb_lower": round(bb_lower, 2) if bb_lower else None,
                        "bb_mid": round(bb_mid, 2) if bb_mid else None,
                        "bb_upper": round(bb_upper, 2) if bb_upper else None,
                        "bb_position": ("above_upper" if bb_upper and closes[-1] > bb_upper
                                        else "below_lower" if bb_lower and closes[-1] < bb_lower
                                        else "inside") if bb_upper else None,
                        "price_changes_pct": changes,
                        "volume_ratio": vol_ratio,
                    }
                time.sleep(0.15)  # rate limit

        elif inst["source"] == "yahoo":
            ydata = fetch_yahoo_price(inst["symbol"])
            if ydata:
                inst_data["change_today_pct"] = ydata["change_pct"]
                inst_data["prev_close"] = ydata["prev_close"]

        # Session context
        if key in session_prices and session_prices[key]:
            sp = session_prices[key]
            prices_only = [p for _, p in sp]
            s_low, s_high = min(prices_only), max(prices_only)
            s_range_pct = round((s_high - s_low) / s_low * 100, 3) if s_low > 0 else 0
            # Last 30 min trend
            cutoff_30m = time.time() - 1800
            recent = [p for t, p in sp if t >= cutoff_30m]
            if len(recent) >= 4:
                q1_avg = sum(recent[:len(recent)//4]) / (len(recent)//4)
                q4_avg = sum(recent[-len(recent)//4:]) / (len(recent)//4)
                trend = "rising" if q4_avg > q1_avg * 1.001 else "falling" if q4_avg < q1_avg * 0.999 else "flat"
            else:
                trend = "insufficient_data"
            inst_data["session_context"] = {
                "duration_min": round((time.time() - sp[0][0]) / 60, 1),
                "ticks": len(sp),
                "range_pct": s_range_pct,
                "last_30m_trend": trend,
            }

        data["instruments"][key] = inst_data

    # Cross-instrument context
    btc_p = prices.get("BTC")
    eth_p = prices.get("ETH")
    if btc_p and eth_p:
        data["correlations"]["eth_btc_ratio"] = round(eth_p / btc_p, 6)

    # Market context
    data["context"]["analysis_cycle"] = analysis_count + 1
    data["context"]["previous_analyses"] = [
        {"cycle": h["cycle"], "summary": h.get("summary", ""), "prices": h.get("prices", {})}
        for h in analysis_history[-5:]
    ]

    # News headlines
    headlines, fg = fetch_crypto_news()
    if headlines:
        data["news"] = headlines[:10]  # top 10 relevant headlines
    if fg:
        data["fear_greed"] = fg

    # Cycle reference data
    data["context"]["btc_cycle"] = {
        "halving_date": "2024-04-20",
        "months_post_halving": 22,
        "ath": 126198,
        "ath_date": "2025-10-06",
        "drawdown_from_ath_pct": round((btc_p - 126198) / 126198 * 100, 1) if btc_p else None,
        "key_levels": {
            "support": [63000, 60000, 55000],
            "resistance": [72500, 84000, 107000],
        },
        "mvrv_zscore": 0.32,
        "whale_accumulation": "270K BTC in 30 days (largest in 13 years)",
    }

    return data


def invoke_claude_analysis(data_path):
    """Invoke Claude Code for BTC/ETH/MSTR analysis."""
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        print("  [!] claude not found on PATH, skipping analysis")
        return False

    prompt = f"""You are a crypto/equities analyst monitoring BTC, ETH, and MSTR together. These three instruments are highly correlated (BTC leads, ETH follows with higher beta, MSTR is leveraged BTC equity exposure).

The user is watching for a LONG-TERM RE-ENTRY opportunity. They sold after a -50% drawdown from the Oct 2025 ATH ($126K BTC). They need to know: is NOW the time to re-enter, or should they wait?

## Step 1: Read technical + news data
Read data/crypto_analysis.json — it contains prices, technicals (RSI, MACD, BB across 5m/15m/1h/4h), session context, Bitcoin cycle data, AND latest news headlines with Fear & Greed index.

## Step 2: Web news search
Search for 2-3 queries to supplement the headlines:
- "bitcoin price news today" — what's driving the current move?
- "bitcoin ETF flows institutional" — is smart money buying?
- "MSTR strategy stock bitcoin" — MSTR-specific catalysts

Focus on FACTS: ETF inflow/outflow numbers, whale wallet movements, regulatory news, macro events.

## Step 3: Analyze for re-entry
Consider:
- BTC cycle position (22mo post-halving, ATH was $126K Oct 2025)
- Key levels: BTC support $63K, resistance $72.5K (break above = bottom confirmed)
- Is the current bounce dead-cat or structural reversal? Volume tells the story.
- ETH/BTC ratio: is ETH catching up (bullish rotation) or lagging?
- MSTR NAV discount: what does institutional positioning say?
- Fear & Greed index: extreme fear = contrarian buy signal?
- News sentiment: are headlines turning positive or still fear-driven?

## Step 4: Write summary
Write and execute data/crypto_tg_send.py:
```python
import json, datetime
prices = {{}}  # fill with current BTC, ETH, MSTR prices
summary = '...'  # 3-5 sentence analysis with re-entry assessment
# Log locally (NO Telegram — warnings disabled)
entry = {{
    'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'text': summary,
    'prices': prices,
    'news_headlines': [],  # top 3 headlines that influenced analysis
    'fear_greed': None,    # current F&G value
    'category': 'crypto_monitor',
    'sent': False,
}}
with open('data/crypto_monitor_log.jsonl', 'a') as f:
    f.write(json.dumps(entry) + '\\n')
print(f'Logged: {{summary[:100]}}')
```

IMPORTANT: Do NOT send Telegram messages. Only log locally. Include re-entry assessment in every analysis."""

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    log_path = DATA_DIR / "crypto_agent.log"
    print(f"  -> Invoking Claude for analysis...")
    try:
        with open(log_path, "a", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                [claude_cmd, "-p", prompt,
                 "--allowedTools", "Read,Bash,Write,WebSearch,WebFetch",
                 "--model", "haiku",
                 "--max-turns", "10"],
                cwd=str(BASE_DIR),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            proc.wait(timeout=120)
            print(f"  -> Claude finished (exit {proc.returncode})")
            return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [!] Claude timed out, killing")
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.kill()
        return False
    except Exception as e:
        print(f"  [!] Claude invoke error: {e}")
        return False


def format_price(key, price):
    if key == "BTC":
        return f"${price:,.0f}"
    elif key == "ETH":
        return f"${price:,.2f}"
    else:
        return f"${price:,.2f}"


def main():
    global last_analysis_ts, start_time, analysis_count

    start_time = time.time()
    last_analysis_ts = time.time()

    print("=== BTC / ETH / MSTR Monitor (with News) ===")
    print(f"Fast checks: {CHECK_INTERVAL}s | Analysis: {ANALYSIS_INTERVAL}s | News: {NEWS_INTERVAL}s")
    print(f"Telegram warnings: {'ENABLED' if SEND_WARNINGS else 'DISABLED'}")
    print(f"Mode: Long-term re-entry monitoring")
    print()

    # Initial fetch
    prices = fetch_all_prices()
    for key, price in prices.items():
        start_prices[key] = price
        session_lows[key] = price
        session_highs[key] = price
        print(f"  {key}: {format_price(key, price)}")
    print()

    # First analysis after 60 seconds
    last_analysis_ts = time.time() - ANALYSIS_INTERVAL + 60

    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            prices = fetch_all_prices()
            now = datetime.datetime.now()
            ts = now.strftime("%H:%M:%S")

            parts = []
            for key in ["BTC", "ETH", "MSTR"]:
                price = prices.get(key)
                if price is None:
                    parts.append(f"{key}:---")
                    continue

                price_history[key].append(price)
                session_prices[key].append((time.time(), price))

                if key not in session_lows or price < session_lows[key]:
                    session_lows[key] = price
                if key not in session_highs or price > session_highs[key]:
                    session_highs[key] = price

                # Session change
                chg = ""
                if key in start_prices and start_prices[key] > 0:
                    pct = (price - start_prices[key]) / start_prices[key] * 100
                    chg = f"({pct:+.2f}%)"

                parts.append(f"{key}:{format_price(key, price)} {chg}")

            # ETH/BTC ratio
            ratio = ""
            if "BTC" in prices and "ETH" in prices:
                ratio = f" R:{prices['ETH']/prices['BTC']:.5f}"

            next_analysis = max(0, ANALYSIS_INTERVAL - (time.time() - last_analysis_ts))
            fg_str = ""
            if cached_fear_greed:
                fg_str = f" F&G:{cached_fear_greed['value']}"
            print(f"[{ts}] {' | '.join(parts)}{ratio}{fg_str} [next: {next_analysis:.0f}s]")

            # === Velocity alerts (BTC only, biggest mover) ===
            if "BTC" in prices and len(price_history["BTC"]) >= 3:
                oldest = price_history["BTC"][0]
                vel = (prices["BTC"] - oldest) / oldest * 100
                if abs(vel) >= abs(VELOCITY_ALERT_PCT):
                    direction = "PUMP" if vel > 0 else "DUMP"
                    print(f"\n  *** BTC {direction}: {vel:+.1f}% in {len(price_history['BTC']) * CHECK_INTERVAL}s ***")
                    if SEND_WARNINGS:
                        msg = f"*BTC {direction}: {vel:+.1f}%*\n`BTC {format_price('BTC', prices['BTC'])} | ETH {format_price('ETH', prices.get('ETH', 0))}`"
                        send_telegram(msg)
                    last_analysis_ts = 0  # trigger immediate analysis

            # === Claude Analysis Cycle ===
            if time.time() - last_analysis_ts >= ANALYSIS_INTERVAL:
                last_analysis_ts = time.time()
                analysis_count += 1
                print(f"\n{'='*60}")
                print(f"  ANALYSIS CYCLE #{analysis_count}")
                print(f"{'='*60}")

                data = gather_analysis_data(prices)
                analysis_path = DATA_DIR / "crypto_analysis.json"
                with open(analysis_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  Data written to {analysis_path}")

                success = invoke_claude_analysis(analysis_path)
                analysis_history.append({
                    "cycle": analysis_count,
                    "prices": {k: prices.get(k) for k in INSTRUMENTS},
                    "summary": "",
                    "ts": time.time(),
                })
                # Read back summary
                try:
                    lines = open(DATA_DIR / "crypto_monitor_log.jsonl", "r", encoding="utf-8").readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        analysis_history[-1]["summary"] = last.get("text", "")[:200]
                except Exception:
                    pass

                if not success:
                    print(f"  [fallback] Claude analysis failed")
                print(f"{'='*60}\n")

        except KeyboardInterrupt:
            print(f"\n=== Crypto Monitor stopped ===")
            for key in ["BTC", "ETH", "MSTR"]:
                if key in prices:
                    chg = ""
                    if key in start_prices and start_prices[key] > 0:
                        pct = (prices[key] - start_prices[key]) / start_prices[key] * 100
                        chg = f" ({pct:+.2f}%)"
                    print(f"  {key}: {format_price(key, prices[key])}{chg}")
            break
        except Exception as e:
            print(f"  [!] Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
