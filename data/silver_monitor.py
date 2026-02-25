"""
Silver Price Monitor with Claude Analysis
MINI L SILVER AVA 301 | 150K SEK | 4.76x leverage

Two layers:
  - Fast (10s): price checks, instant alerts only on -3%+ drops
  - Analysis (5min): comprehensive data -> Claude Code analyzes
    Claude only sends Telegram on WARNING or EXIT, NOT on HOLD

Run: .venv/Scripts/python.exe -u data/silver_monitor.py
"""
import json, time, datetime, requests, sys, os, subprocess, shutil, platform
from collections import deque
from pathlib import Path

# === Config ===
REFERENCE_PRICE = 90.55
LEVERAGE = 4.76
POSITION_SEK = 150_000
CHECK_INTERVAL = 10              # fast price check every 10s
ANALYSIS_INTERVAL = 300          # Claude analysis every 5 min
VELOCITY_WINDOW = 18             # 18 readings at 10s = 3 min window
VELOCITY_ALERT_PCT = -0.3

BINANCE_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_SYMBOL = "XAGUSDT"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Mechanical alert thresholds — only serious drops get instant Telegram
# Minor moves are handled by Claude analysis every 5 min
ALERT_LEVELS = [
    (-3.0, "WARNING"),       # warrant -14.3%, -21K SEK — instant TG
    (-5.0, "DANGER"),        # warrant -23.8%, -36K SEK — instant TG
    (-7.0, "HIGH RISK"),     # warrant -33.3%, -50K SEK — instant TG
    (-10.0, "CRITICAL"),     # warrant -47.6%, -71K SEK — instant TG
    (-12.5, "EMERGENCY"),    # warrant -59.5%, -89K SEK — instant TG
]

# Load Telegram config
with open(BASE_DIR / "config.json") as f:
    cfg = json.load(f)
TG_TOKEN = cfg["telegram"]["token"]
TG_CHAT = cfg["telegram"]["chat_id"]

# === State ===
price_history = deque(maxlen=VELOCITY_WINDOW)
session_prices = []              # ALL prices this session (for trend context)
analysis_history = []            # previous Claude decisions for context
alerted_levels = set()
session_low = None
session_high = None
last_analysis_ts = 0
start_time = None
consecutive_down = 0
prev_price = None
analysis_count = 0


def fetch_price():
    try:
        r = requests.get(BINANCE_URL, params={"symbol": BINANCE_SYMBOL}, timeout=5)
        r.raise_for_status()
        return float(r.json()["price"])
    except Exception as e:
        print(f"  [!] Price fetch error: {e}")
        return None


def fetch_klines(interval="1m", limit=50):
    try:
        r = requests.get(BINANCE_KLINES,
                         params={"symbol": BINANCE_SYMBOL, "interval": interval, "limit": limit},
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


def calc_macd(closes, fast=12, slow=26, signal=9):
    fast_ema = calc_ema(closes, fast)
    slow_ema = calc_ema(closes, slow)
    if fast_ema is None or slow_ema is None:
        return None, None
    macd_line = fast_ema - slow_ema
    # Simplified: just return the line value and direction
    prev_fast = calc_ema(closes[:-1], fast)
    prev_slow = calc_ema(closes[:-1], slow)
    if prev_fast and prev_slow:
        prev_macd = prev_fast - prev_slow
        return macd_line, macd_line - prev_macd  # line, histogram-like delta
    return macd_line, None


def calc_warrant(price):
    pct = (price - REFERENCE_PRICE) / REFERENCE_PRICE * 100
    wpct = pct * LEVERAGE
    wsek = POSITION_SEK * wpct / 100
    return pct, wpct, wsek


def send_telegram(msg):
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


def gather_analysis_data(price):
    """Collect comprehensive data for Claude analysis."""
    data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "position": {
            "instrument": "MINI L SILVER AVA 301",
            "invested_sek": POSITION_SEK,
            "leverage": LEVERAGE,
            "reference_price": REFERENCE_PRICE,
            "strategy": "hold for a few hours, exit today"
        },
        "price": {
            "current": price,
            "pct_from_entry": round((price - REFERENCE_PRICE) / REFERENCE_PRICE * 100, 3),
            "session_low": session_low,
            "session_high": session_high,
            "consecutive_down_ticks": consecutive_down,
        },
        "warrant": {},
        "technicals": {},
        "context": {}
    }

    pct, wpct, wsek = calc_warrant(price)
    data["warrant"] = {
        "pct_change": round(wpct, 2),
        "sek_pnl": round(wsek, 0),
        "position_value": round(POSITION_SEK + wsek, 0),
    }

    # Fetch multiple timeframe klines
    for interval, label, limit in [("1m", "1m", 50), ("5m", "5m", 50), ("15m", "15m", 50), ("1h", "1h", 30)]:
        raw = fetch_klines(interval, limit)
        if raw:
            closes = [float(k[4]) for k in raw]
            highs = [float(k[2]) for k in raw]
            lows = [float(k[3]) for k in raw]
            volumes = [float(k[5]) for k in raw]

            rsi = calc_rsi(closes, 14)
            macd_line, macd_delta = calc_macd(closes)
            bb_lower, bb_mid, bb_upper = calc_bb(closes)

            # Momentum: % change over last N candles
            changes = {}
            for n in [5, 10, 20]:
                if len(closes) >= n:
                    changes[f"last_{n}"] = round((closes[-1] - closes[-n]) / closes[-n] * 100, 3)

            # Volume trend
            if len(volumes) >= 20:
                avg_vol = sum(volumes[-20:]) / 20
                recent_vol = sum(volumes[-5:]) / 5
                vol_ratio = round(recent_vol / avg_vol, 2) if avg_vol > 0 else None
            else:
                vol_ratio = None

            data["technicals"][label] = {
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
                "candle_range": round(highs[-1] - lows[-1], 4),
            }
        time.sleep(0.2)  # rate limit

    # Fetch DXY proxy and gold for context
    for sym, key in [("XAUUSDT", "gold_price")]:
        try:
            r = requests.get(BINANCE_URL, params={"symbol": sym}, timeout=5)
            r.raise_for_status()
            data["context"][key] = float(r.json()["price"])
        except Exception:
            pass

    # Gold/silver ratio
    if "gold_price" in data["context"] and price > 0:
        data["context"]["gold_silver_ratio"] = round(data["context"]["gold_price"] / price, 2)

    return data


def invoke_claude_analysis(data_path):
    """Invoke Claude Code to analyze silver data and send Telegram."""
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        print("  [!] claude not found on PATH, skipping analysis")
        return False

    prompt = (
        "Read data/silver_analysis.json. This is a live silver LONG position (MINI L SILVER AVA 301, 150K SEK, 4.76x leverage, holding a few hours).\n\n"
        "Look at RSI across 1m/5m/15m/1h, MACD deltas, BB position, volume ratios, and price momentum.\n\n"
        "Decide: HOLD, WARNING, or EXIT.\n"
        "- HOLD = trend OK, no danger. Do NOT send any Telegram.\n"
        "- WARNING = signs of reversal (RSI rolling, momentum fading, volume dying). Send Telegram.\n"
        "- EXIT = clear reversal or overbought breakdown. Send Telegram.\n\n"
        "Write and execute data/silver_tg_send.py:\n"
        "```python\n"
        "import json, datetime, requests\n"
        "decision = 'HOLD'  # or 'WARNING' or 'EXIT' -- set based on your analysis\n"
        "bold = '...'  # 1 sentence: breakout continuation or exhaustion?\n"
        "patient = '...'  # 1 sentence: risk assessment\n"
        "price = ...  # from silver_analysis.json\n"
        "warrant_pct = ...  # from silver_analysis.json\n"
        "warrant_sek = ...  # from silver_analysis.json\n"
        "msg = f'*XAG {decision}: ${price:.2f}*\\n`W: {warrant_pct:+.1f}% = {warrant_sek:+,.0f} SEK`\\nBold: {bold}\\nPatient: {patient}'\n"
        "config = json.load(open('config.json'))\n"
        "entry = {'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'text': msg, 'category': 'analysis', 'decision': decision, 'sent': decision != 'HOLD'}\n"
        "with open('data/telegram_messages.jsonl', 'a') as f: f.write(json.dumps(entry) + '\\n')\n"
        "if decision != 'HOLD':\n"
        "    requests.post(f\"https://api.telegram.org/bot{config['telegram']['token']}/sendMessage\", json={'chat_id': config['telegram']['chat_id'], 'text': msg, 'parse_mode': 'Markdown'})\n"
        "    print(f'Telegram sent: {decision}')\n"
        "else:\n"
        "    print(f'HOLD -- no Telegram')\n"
        "```\n"
        "Adjust the decision, bold, patient, and values based on your analysis. Then run it."
    )

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    log_path = DATA_DIR / "silver_agent.log"
    print(f"  -> Invoking Claude for analysis...")
    try:
        with open(log_path, "a", encoding="utf-8") as log_fh:
            proc = subprocess.Popen(
                [claude_cmd, "-p", prompt,
                 "--allowedTools", "Read,Bash,Write",
                 "--model", "haiku",
                 "--max-turns", "15"],
                cwd=str(BASE_DIR),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            # Wait up to 120s
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


def main():
    global session_low, session_high, last_analysis_ts, start_time
    global consecutive_down, prev_price, analysis_count

    start_time = time.time()
    last_analysis_ts = time.time()  # don't analyze immediately, let data accumulate

    print("=== Silver Monitor + Claude Analysis ===")
    print(f"Ref: ${REFERENCE_PRICE} | Lev: {LEVERAGE}x | Pos: {POSITION_SEK:,} SEK")
    print(f"Fast checks: {CHECK_INTERVAL}s | Claude analysis: {ANALYSIS_INTERVAL}s (5 min)")
    print(f"Alerts: {', '.join(f'{t[0]}%' for t in ALERT_LEVELS)}")
    print()

    price = fetch_price()
    if price:
        pct, wpct, wsek = calc_warrant(price)
        session_low = price
        session_high = price
        print(f"Start: ${price:.2f} ({pct:+.2f}%) | W:{wpct:+.1f}%")
        # No startup Telegram — only alert on WARNING/EXIT

        # Run first Claude analysis after 60 seconds
        last_analysis_ts = time.time() - ANALYSIS_INTERVAL + 60
    print()

    while True:
        try:
            time.sleep(CHECK_INTERVAL)
            price = fetch_price()
            if price is None:
                continue

            now = datetime.datetime.now()
            pct_change, warrant_pct, warrant_sek = calc_warrant(price)

            # Consecutive down tracking
            if prev_price is not None:
                if price < prev_price - 0.001:
                    consecutive_down += 1
                else:
                    consecutive_down = 0
            prev_price = price

            price_history.append(price)

            if session_low is None or price < session_low:
                session_low = price
            if session_high is None or price > session_high:
                session_high = price

            # Status line
            ts = now.strftime("%H:%M:%S")
            next_analysis = max(0, ANALYSIS_INTERVAL - (time.time() - last_analysis_ts))
            print(f"[{ts}] ${price:.2f} ({pct_change:+.2f}%) W:{warrant_pct:+.1f}% ({warrant_sek:+,.0f}) "
                  f"Lo:{session_low:.2f} Hi:{session_high:.2f} "
                  f"{'v' + str(consecutive_down) if consecutive_down >= 3 else ''} "
                  f"[next analysis: {next_analysis:.0f}s]")

            # === Mechanical threshold alerts (instant, no Claude needed) ===
            for threshold, level_name in ALERT_LEVELS:
                if pct_change <= threshold and threshold not in alerted_levels:
                    alerted_levels.add(threshold)
                    msg = (
                        f"*{level_name}: XAG ${price:.2f} ({pct_change:+.1f}%)*\n"
                        f"`Warrant: {warrant_pct:+.1f}% = {warrant_sek:+,.0f} SEK`\n"
                        f"`Position: {POSITION_SEK + warrant_sek:,.0f} SEK`\n"
                        f"_Ref ${REFERENCE_PRICE} | {LEVERAGE}x_"
                    )
                    print(f"\n  *** {level_name}: {pct_change:.1f}% ***")
                    send_telegram(msg)
                    # Also trigger immediate Claude analysis on danger+ levels
                    if threshold <= -3.0:
                        print("  -> Triggering immediate Claude analysis")
                        last_analysis_ts = 0  # force next cycle
                    print()

            # === Velocity alert ===
            if len(price_history) >= 3:
                oldest = price_history[0]
                vel = (price - oldest) / oldest * 100
                if vel <= VELOCITY_ALERT_PCT:
                    vel_key = f"vel_{int(time.time() // 300)}"
                    if vel_key not in alerted_levels:
                        alerted_levels.add(vel_key)
                        msg = (
                            f"*RAPID DROP: XAG {vel:.1f}% in {len(price_history) * CHECK_INTERVAL}s*\n"
                            f"`${price:.2f} | W:{warrant_pct:+.1f}%`\n"
                            f"_Check now_"
                        )
                        print(f"\n  *** VELOCITY: {vel:.1f}% ***")
                        send_telegram(msg)
                        last_analysis_ts = 0  # trigger Claude analysis
                        print()

            # === Claude Analysis Cycle (every 10 min) ===
            if time.time() - last_analysis_ts >= ANALYSIS_INTERVAL:
                last_analysis_ts = time.time()
                analysis_count += 1
                print(f"\n{'='*50}")
                print(f"  ANALYSIS CYCLE #{analysis_count}")
                print(f"{'='*50}")

                # Gather comprehensive data
                data = gather_analysis_data(price)
                analysis_path = DATA_DIR / "silver_analysis.json"
                with open(analysis_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"  Data written to {analysis_path}")

                # Invoke Claude
                success = invoke_claude_analysis(analysis_path)
                if not success:
                    # Log locally only — no Telegram on fallback
                    t = data.get("technicals", {})
                    rsi_1m = t.get("1m", {}).get("rsi", "?")
                    rsi_5m = t.get("5m", {}).get("rsi", "?")
                    rsi_15m = t.get("15m", {}).get("rsi", "?")
                    rsi_1h = t.get("1h", {}).get("rsi", "?")
                    print(f"  [fallback] RSI: 1m={rsi_1m} 5m={rsi_5m} 15m={rsi_15m} 1h={rsi_1h}")
                print(f"{'='*50}\n")

        except KeyboardInterrupt:
            print(f"\n=== Monitor stopped ===")
            if price:
                pct, wpct, wsek = calc_warrant(price)
                print(f"Final: ${price:.2f} ({pct:+.2f}%) | W:{wpct:+.1f}% ({wsek:+,.0f} SEK)")
                print(f"Session: Lo ${session_low:.2f} Hi ${session_high:.2f}")
                print(f"Analyses run: {analysis_count}")
            sys.exit(0)


if __name__ == "__main__":
    main()
