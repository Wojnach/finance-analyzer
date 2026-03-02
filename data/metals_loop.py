"""
Metals Intraday Trading Loop v6 (Layer 1).
Collects price data every 60-90 seconds from Avanza.
When trigger conditions are met, invokes Claude Code (Layer 2) for trading decisions.
Claude reads data/metals_context.json and decides to buy/sell/hold.

Features:
- Tiered Claude invocation (Haiku/Sonnet/Opus)
- Local LLM inference (Ministral-8B + Chronos, 5min cycle)
- Monte Carlo VaR for leveraged warrants
- Trade guards (cooldowns, session limits, loss escalation)
- Drawdown circuit breaker (-15% emergency liquidation)
- Multi-level stop-loss (L1 warn / L2 alert / L3 emergency auto-sell)
- Short instrument tracking (BEAR SILVER X5)
- Time server (timeapi.io) for accurate CET
- Daily range analysis (historical percentiles + intraday assessment)
- Spike catcher (limit sell orders before US open)
- Invocation logging (tier/model/trigger tracking)

Run: .venv/Scripts/python.exe data/metals_loop.py
"""
import json, os, sys, time, datetime, traceback, subprocess, shutil, platform
os.chdir(r"Q:/finance-analyzer")

import requests
from playwright.sync_api import sync_playwright

# --- Optional modules (graceful fallback) ---
try:
    sys.path.insert(0, "data")
    from metals_llm import (
        start_llm_thread, stop_llm_thread, get_llm_signals,
        get_llm_accuracy, get_llm_summary, get_llm_age,
    )
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_llm import failed: {e}", flush=True)
    LLM_AVAILABLE = False

try:
    from metals_risk import (
        get_risk_summary, log_portfolio_value, check_portfolio_drawdown,
        check_trade_guard, record_metals_trade, simulate_all_positions,
        compute_daily_range_stats, compute_intraday_assessment,
        compute_spike_targets, load_spike_state, save_spike_state,
    )
    RISK_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_risk import failed: {e}", flush=True)
    RISK_AVAILABLE = False

# --- CONFIG ---
CHECK_INTERVAL = 90           # seconds between price checks
TRIGGER_PRICE_MOVE = 2.0      # % move from last invocation to trigger
TRIGGER_TRAILING = 3.0        # % drop from peak to trigger
TRIGGER_PROFIT = 4.0          # % profit from entry to trigger
TRIGGER_STOP_NEAR = 5.0       # % from stop-loss to trigger
HEARTBEAT_CHECKS = 20         # invoke every N checks (~30 min at 90s)
MIN_INVOKE_INTERVAL = 300     # minimum 5 min between invocations
EOD_HOUR_UTC = 16             # 17:00 CET = 16:00 UTC

# Spike catcher config (US open limit sell orders)
SPIKE_ENABLED = True
SPIKE_PLACE_CET = 15.25       # 15:15 CET — place orders before US open (15:30)
SPIKE_CANCEL_CET = 16.5       # 16:30 CET — cancel unfilled orders
SPIKE_PERCENTILE = 75          # P75 of daily open_to_high as target
SPIKE_PARTIAL_PCT = 50         # sell 50% of position to capture spike profit

# Invocation log
INVOCATION_LOG = "data/metals_invocations.jsonl"

# Stop levels (distance from barrier as % of bid)
STOP_L1_PCT = 8.0   # L1: warning — log + flag in context
STOP_L2_PCT = 5.0   # L2: alert — Telegram + force Claude invocation
STOP_L3_PCT = 2.0   # L3: emergency — auto-sell immediately

# Tier config: model, timeout, max_turns
TIER_CONFIG = {
    1: {"model": "haiku",  "timeout": 60,   "max_turns": 8,  "label": "QUICK"},
    2: {"model": "sonnet", "timeout": 180,  "max_turns": 15, "label": "ANALYSIS"},
    3: {"model": None,     "timeout": 300,  "max_turns": 20, "label": "CRITICAL"},
}

# --- POSITIONS ---
POSITIONS = {
    "gold": {
        "name": "BULL GULD X8 N", "ob_id": "856394", "api_type": "certificate",
        "units": 5, "entry": 972.4, "stop": 900.0, "active": True,
    },
    "silver79": {
        "name": "MINI L SILVER AVA 79", "ob_id": "1078198", "api_type": "warrant",
        "units": 78, "entry": 65.13, "stop": 59.92, "active": True,
    },
    "silver301": {
        "name": "MINI L SILVER AVA 301", "ob_id": "2334960", "api_type": "warrant",
        "units": 240, "entry": 20.70, "stop": 18.69, "active": True,
    },
}

SHORT_INSTRUMENTS = {
    "bear_silver_x5": {
        "name": "BEAR SILVER X5 AVA 12", "ob_id": "2286417", "api_type": "certificate",
    },
}

ACCOUNT_ID = "1625505"

config = json.load(open("config.json"))
TG_TOKEN = config["telegram"]["token"]
TG_CHAT = config["telegram"]["chat_id"]

# --- STATE ---
check_count = 0
last_invoke_prices = {}   # prices at last Claude invocation
last_invoke_time = 0      # timestamp of last invocation
peak_bids = {}            # session peak for trailing stop
price_history = []        # circular buffer of snapshots
last_signal_data = {}
prev_signal_actions = {}  # for signal flip detection
claude_proc = None
claude_log_fh = None
claude_start = 0
claude_timeout = 300
invoke_count = 0
startup_grace = True      # skip first check to establish baseline
short_prices = {}         # latest prices for short instruments
daily_range_stats = {}    # historical daily range percentiles (computed at startup)

def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
    except:
        pass

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def pnl_pct(current, entry):
    if entry == 0: return 0
    return ((current - entry) / entry) * 100

def get_cet_time():
    """Get current CET/CEST time from timeapi.io, fallback to UTC+1."""
    try:
        r = requests.get(
            "http://timeapi.io/api/time/current/zone?timeZone=Europe/Stockholm",
            timeout=3
        )
        if r.status_code == 200:
            data = r.json()
            h = data["hour"]
            m = data["minute"]
            return h + m / 60, f"{h:02d}:{m:02d} CET", "timeapi"
    except:
        pass
    # Fallback: UTC+1 (doesn't handle DST)
    now = datetime.datetime.now(datetime.timezone.utc)
    h = (now.hour + 1) % 24
    m = now.minute
    return h + m / 60, f"{h:02d}:{m:02d} CET", "system_utc+1"

def cet_hour():
    h, _, _ = get_cet_time()
    return h

def cet_time_str():
    _, ts, _ = get_cet_time()
    return ts

def is_market_hours():
    """Check if Avanza warrant market is open (Mon-Fri 09:00-17:25 CET)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    weekday = now.weekday()
    h = cet_hour()
    return weekday < 5 and 9.0 <= h <= 17.42

def fetch_price(page, ob_id, api_type):
    result = page.evaluate("""async (args) => {
        const [id, type] = args;
        const resp = await fetch('https://www.avanza.se/_api/market-guide/' + type + '/' + id, {credentials:'include'});
        if (resp.status !== 200) return null;
        const d = await resp.json();
        return {
            bid: d.quote?.buy, ask: d.quote?.sell, last: d.quote?.last,
            change_pct: d.quote?.changePercent,
            high: d.quote?.highest, low: d.quote?.lowest,
            underlying: d.underlying?.quote?.last,
            underlying_name: d.underlying?.name,
            leverage: d.keyIndicators?.leverage,
            barrier: d.keyIndicators?.barrierLevel,
        };
    }""", [ob_id, api_type])
    return result

def read_signal_data():
    """Read XAG/XAU signal data from the main loop's agent_summary.json."""
    try:
        path = "data/agent_summary.json"
        if not os.path.exists(path):
            path = "data/agent_summary_compact.json"
        if not os.path.exists(path):
            return {}

        mtime = os.path.getmtime(path)
        age_min = (time.time() - mtime) / 60

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = {"age_min": round(age_min, 1)}

        tickers = data.get("tickers", {})
        if not tickers:
            for key in ["forecast_signals", "cumulative_gains"]:
                if key in data:
                    result[key] = data[key]
            return result

        for ticker in ["XAG-USD", "XAU-USD"]:
            if ticker in tickers:
                t = tickers[ticker]
                extra = t.get("extra", {})
                result[ticker] = {
                    "action": t.get("action", "?"),
                    "confidence": round(t.get("confidence", 0), 3),
                    "weighted_confidence": round(t.get("weighted_confidence", 0), 3),
                    "rsi": round(t.get("rsi", 0), 1),
                    "macd_hist": t.get("macd_hist", 0),
                    "bb_position": t.get("bb_position", "?"),
                    "regime": t.get("regime", "?"),
                    "atr_pct": t.get("atr_pct", 0),
                    "buy_count": extra.get("_buy_count", 0),
                    "sell_count": extra.get("_sell_count", 0),
                    "voters": extra.get("_voters", 0),
                    "vote_detail": extra.get("_vote_detail", ""),
                }

        timeframes = data.get("timeframe_heatmap", {})
        for ticker in ["XAG-USD", "XAU-USD"]:
            if ticker in timeframes and ticker in result:
                result[ticker]["timeframes"] = timeframes[ticker]

        return result
    except Exception as e:
        log(f"Signal read error: {e}")
        return {}

def read_decision_history(n=5):
    """Read the last N decisions from metals_decisions.jsonl."""
    try:
        path = "data/metals_decisions.jsonl"
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line.strip()))
            except:
                pass
        return entries
    except:
        return []

def emergency_sell(page, key, pos, bid):
    """L3 emergency auto-sell via Avanza API."""
    log(f"!!! L3 EMERGENCY SELL: {key} at {bid} (entry: {pos['entry']}, stop: {pos['stop']})")
    send_telegram(f"*L3 EMERGENCY SELL* {pos['name']}\nBid: {bid} | Entry: {pos['entry']}\nAuto-selling {pos['units']} units")

    try:
        # Get CSRF token
        cookies = page.context.cookies()
        csrf = None
        for c in cookies:
            if c["name"] == "AZACSRF":
                csrf = c["value"]
                break

        if not csrf:
            log("EMERGENCY SELL FAILED: no CSRF token")
            return

        result = page.evaluate("""async (args) => {
            const [payload, token] = args;
            const resp = await fetch('https://www.avanza.se/_api/trading-critical/rest/order/new', {
                method: 'POST',
                headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                credentials: 'include',
                body: JSON.stringify(payload),
            });
            return {status: resp.status, body: await resp.text()};
        }""", [{
            "accountId": ACCOUNT_ID,
            "orderbookId": pos["ob_id"],
            "side": "SELL",
            "condition": "NORMAL",
            "price": bid,
            "validUntil": (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "volume": pos["units"],
        }, csrf])

        log(f"Emergency sell result: {result}")
        send_telegram(f"*L3 SELL RESULT*: status={result.get('status')}")

        # Log trade
        trade = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": "EMERGENCY_SELL",
            "position": key,
            "name": pos["name"],
            "units": pos["units"],
            "price": bid,
            "entry": pos["entry"],
            "pnl_pct": round(pnl_pct(bid, pos["entry"]), 2),
            "result": result,
        }
        with open("data/metals_trades.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trade, ensure_ascii=False) + "\n")

        # Record in trade guards
        if RISK_AVAILABLE:
            record_metals_trade(key, "SELL", pnl_pct_value=pnl_pct(bid, pos["entry"]))

        pos["active"] = False
    except Exception as e:
        log(f"Emergency sell FAILED: {e}")
        send_telegram(f"*L3 SELL FAILED*: {e}")

def _get_csrf(page):
    """Extract CSRF token from Avanza cookies."""
    for c in page.context.cookies():
        if c["name"] == "AZACSRF":
            return c["value"]
    return None


def place_spike_orders(page, positions, prices, targets):
    """Place limit sell orders for US open spike capture.

    Returns dict of {position_key: order_id} for placed orders.
    """
    csrf = _get_csrf(page)
    if not csrf:
        log("Spike: no CSRF token, skipping")
        return {}

    placed = {}
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    for key, target in targets.items():
        pos = positions.get(key)
        if not pos or not pos.get("active"):
            continue

        try:
            payload = {
                "accountId": ACCOUNT_ID,
                "orderbookId": pos["ob_id"],
                "side": "SELL",
                "condition": "NORMAL",
                "price": target["target_price"],
                "validUntil": today_str,
                "volume": target["units_to_sell"],
            }

            result = page.evaluate("""async (args) => {
                const [payload, token] = args;
                const resp = await fetch('https://www.avanza.se/_api/trading-critical/rest/order/new', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                    credentials: 'include',
                    body: JSON.stringify(payload),
                });
                return {status: resp.status, body: await resp.text()};
            }""", [payload, csrf])

            status = result.get("status", 0)
            body = result.get("body", "")

            if status == 200 or status == 201:
                # Try to parse orderId from response
                try:
                    resp_data = json.loads(body)
                    order_id = resp_data.get("orderId", "")
                except Exception:
                    order_id = ""

                placed[key] = order_id
                log(f"Spike SELL placed: {pos['name']} {target['units_to_sell']}u @ {target['target_price']} "
                    f"(+{target['target_pnl_pct']:.1f}% from entry) [order: {order_id}]")
            else:
                log(f"Spike SELL failed for {key}: status={status}, body={body[:200]}")

        except Exception as e:
            log(f"Spike order error for {key}: {e}")

    return placed


def cancel_spike_orders(page, spike_state):
    """Cancel all unfilled spike orders."""
    csrf = _get_csrf(page)
    if not csrf:
        log("Spike cancel: no CSRF token")
        return

    orders = spike_state.get("orders", {})
    for key, order_id in list(orders.items()):
        if not order_id:
            continue

        try:
            result = page.evaluate("""async (args) => {
                const [accountId, orderId, token] = args;
                const resp = await fetch(
                    'https://www.avanza.se/_api/trading-critical/rest/order/' + accountId + '/' + orderId,
                    {
                        method: 'DELETE',
                        headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                        credentials: 'include',
                    }
                );
                return {status: resp.status, body: await resp.text()};
            }""", [ACCOUNT_ID, order_id, csrf])

            log(f"Spike cancel {key}: status={result.get('status')}")
        except Exception as e:
            log(f"Spike cancel error {key}: {e}")


def check_spike_fills(page, spike_state, positions):
    """Check if any spike orders were filled. Returns list of filled position keys."""
    csrf = _get_csrf(page)
    if not csrf:
        return []

    filled = []
    orders = spike_state.get("orders", {})

    for key, order_id in orders.items():
        if not order_id:
            continue

        try:
            result = page.evaluate("""async (args) => {
                const [accountId, orderId, token] = args;
                const resp = await fetch(
                    'https://www.avanza.se/_api/trading-critical/rest/order/' + accountId + '/' + orderId,
                    {
                        method: 'GET',
                        headers: {'Content-Type': 'application/json', 'X-SecurityToken': token},
                        credentials: 'include',
                    }
                );
                if (resp.status !== 200) return {status: resp.status};
                return {status: 200, body: await resp.json()};
            }""", [ACCOUNT_ID, order_id, csrf])

            if result.get("status") == 200:
                body = result.get("body", {})
                state = body.get("state", "").upper()
                if state in ("FILLED", "EXECUTED", "DONE"):
                    filled.append(key)
                    target = spike_state.get("targets", {}).get(key, {})
                    pnl = target.get("target_pnl_pct", 0)
                    log(f"SPIKE FILLED: {key} sold at target (+{pnl:.1f}% from entry)")
                    send_telegram(
                        f"*SPIKE FILLED* {positions.get(key, {}).get('name', key)}\n"
                        f"Sold {target.get('units_to_sell', '?')}u at {target.get('target_price', '?')}\n"
                        f"P&L from entry: +{pnl:.1f}%"
                    )

                    # Log the trade
                    trade = {
                        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "action": "SPIKE_SELL",
                        "position": key,
                        "units": target.get("units_to_sell"),
                        "price": target.get("target_price"),
                        "entry": positions.get(key, {}).get("entry"),
                        "pnl_pct": pnl,
                        "reason": target.get("reason", "US open spike capture"),
                    }
                    with open("data/metals_trades.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(trade, ensure_ascii=False) + "\n")

        except Exception as e:
            log(f"Spike check error {key}: {e}")

    return filled


def log_invocation(tier, model, trigger, check_num, invoke_num, elapsed_s=None, rc=None):
    """Log a Claude invocation to the invocations JSONL file."""
    entry = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tier": tier,
        "model": model or "opus",
        "trigger": trigger,
        "check_count": check_num,
        "invoke_count": invoke_num,
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "return_code": rc,
    }
    try:
        with open(INVOCATION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def write_context(prices, trigger_reason, tier=2):
    """Write context JSON for Claude Layer 2."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ctx = {
        "timestamp": now.isoformat(),
        "cet_time": cet_time_str(),
        "check_count": check_count,
        "invoke_count": invoke_count,
        "trigger_reason": trigger_reason,
        "tier": tier,
        "market_close_cet": "17:25",
        "hours_remaining": round(max(0, 16.42 - (now.hour + now.minute/60)), 1),
        "positions": {},
        "underlying": {},
        "totals": {},
        "price_history_recent": price_history[-10:] if price_history else [],
        "signals": last_signal_data,
        "recent_decisions": read_decision_history(5),
        "short_instruments": {},
        "llm_predictions": {},
        "risk": {},
        "trades_today_file": "data/metals_trades.jsonl",
    }

    # Short instrument prices
    for sk, si in SHORT_INSTRUMENTS.items():
        sp = short_prices.get(sk, {})
        ctx["short_instruments"][sk] = {
            "name": si["name"],
            "ob_id": si["ob_id"],
            "api_type": si["api_type"],
            "bid": sp.get("bid"),
            "ask": sp.get("ask"),
            "note": "5x short silver certificate, available for hedging",
        }

    # Historical stats
    try:
        with open("data/metals_history.json", "r", encoding="utf-8") as f:
            history = json.load(f)
        ctx["historical_ytd"] = {
            ticker: data["stats"]
            for ticker, data in history.get("metals", {}).items()
        }
    except:
        pass

    # Daily range analysis — historical percentiles + today vs typical
    if RISK_AVAILABLE and daily_range_stats:
        ctx["daily_ranges"] = daily_range_stats
        try:
            ctx["intraday_assessment"] = compute_intraday_assessment(
                POSITIONS, prices, price_history, daily_range_stats
            )
        except Exception as e:
            ctx["intraday_assessment"] = {"error": str(e)}

    # LLM predictions
    if LLM_AVAILABLE:
        try:
            ctx["llm_predictions"] = get_llm_summary()
        except Exception:
            pass

    # Risk summary (Monte Carlo + drawdown + guards)
    if RISK_AVAILABLE:
        try:
            llm_sigs = get_llm_signals() if LLM_AVAILABLE else None
            ctx["risk"] = get_risk_summary(POSITIONS, prices, last_signal_data, llm_sigs)
        except Exception as e:
            ctx["risk"] = {"error": str(e)}

    total_val = 0
    total_inv = 0

    for key, pos in POSITIONS.items():
        p = prices.get(key) or {}
        bid = p.get('bid') or 0
        pnl = pnl_pct(bid, pos["entry"]) if bid > 0 else 0
        val = bid * pos["units"]
        peak = peak_bids.get(key, 0)
        from_peak = pnl_pct(bid, peak) if peak > 0 and bid > 0 else 0
        dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
        invested = pos["entry"] * pos["units"]

        ctx["positions"][key] = {
            "name": pos["name"],
            "ob_id": pos["ob_id"],
            "api_type": pos["api_type"],
            "units": pos["units"],
            "entry": pos["entry"],
            "bid": bid,
            "ask": p.get('ask', 0),
            "pnl_pct": round(pnl, 2),
            "value_sek": round(val, 1),
            "invested_sek": round(invested, 1),
            "profit_sek": round(val - invested, 1),
            "peak_bid": peak,
            "from_peak_pct": round(from_peak, 2),
            "stop": pos["stop"],
            "dist_to_stop_pct": round(dist_stop, 2),
            "day_change_pct": p.get('change_pct', 0),
            "leverage": p.get('leverage'),
            "barrier": p.get('barrier'),
            "active": pos["active"],
        }

        if p.get('underlying'):
            if 'silver' in key.lower():
                ctx["underlying"]["silver"] = {"price": p['underlying'], "bid": p.get('bid'), "ask": p.get('ask')}
            elif 'gold' in key.lower():
                ctx["underlying"]["gold"] = {"price": p['underlying'], "bid": p.get('bid'), "ask": p.get('ask')}

        if pos["active"]:
            total_val += val
            total_inv += invested

    total_pnl = pnl_pct(total_val, total_inv) if total_inv > 0 else 0
    ctx["totals"] = {
        "invested": round(total_inv, 0),
        "current": round(total_val, 0),
        "pnl_pct": round(total_pnl, 2),
        "profit_sek": round(total_val - total_inv, 0),
    }

    with open("data/metals_context.json", "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)

    return ctx

def classify_tier(reasons):
    """Classify trigger into tier (1=cheap workhorse, 2=deeper analysis, 3=critical)."""
    critical_patterns = ["stop-loss", "end_of_day", "profit target", "L2 ALERT", "L3 EMERGENCY",
                         "drawdown", "EMERGENCY"]
    if any(p in r for r in reasons for p in critical_patterns):
        return 3

    # LLM high-confidence triggers get T2
    if any("LLM consensus" in r for r in reasons):
        return 2

    if len(reasons) >= 2:
        return 2

    return 1

def check_triggers(prices):
    """Check if any trigger condition is met. Returns (triggered, reasons)."""
    global prev_signal_actions
    reasons = []

    for key, pos in POSITIONS.items():
        if not pos["active"]:
            continue

        bid = prices.get(key, {}).get('bid') or 0
        if bid <= 0:
            continue

        pnl = pnl_pct(bid, pos["entry"])
        peak = peak_bids.get(key, 0)
        from_peak = pnl_pct(bid, peak) if peak > 0 else 0

        # Price moved significantly from last invocation
        last_price = last_invoke_prices.get(key, pos["entry"])
        price_move = abs(pnl_pct(bid, last_price))
        if price_move >= TRIGGER_PRICE_MOVE:
            reasons.append(f"{key} moved {price_move:+.1f}% since last check")

        # Trailing stop zone (only if we've been up at least 1%)
        peak_pnl = pnl_pct(peak, pos["entry"])
        if peak_pnl >= 1.0 and from_peak <= -TRIGGER_TRAILING:
            reasons.append(f"{key} dropped {from_peak:.1f}% from peak {peak}")

        # Profit target zone
        if pnl >= TRIGGER_PROFIT:
            reasons.append(f"{key} profit target zone +{pnl:.1f}%")

        # Multi-level stop proximity
        dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
        if 0 < dist_stop < STOP_L3_PCT:
            reasons.append(f"{key} L3 EMERGENCY: {dist_stop:.1f}% from stop")
        elif 0 < dist_stop < STOP_L2_PCT:
            reasons.append(f"{key} L2 ALERT: {dist_stop:.1f}% from stop-loss")
        elif 0 < dist_stop < STOP_L1_PCT:
            reasons.append(f"{key} L1 WARNING: {dist_stop:.1f}% from stop-loss")

    # Signal flip detection
    if last_signal_data:
        for ticker in ["XAG-USD", "XAU-USD"]:
            if ticker in last_signal_data:
                current_action = last_signal_data[ticker].get("action", "?")
                prev_action = prev_signal_actions.get(ticker)
                if prev_action and current_action != prev_action and current_action in ("BUY", "SELL"):
                    reasons.append(f"signal flip {ticker}: {prev_action}->{current_action}")
                prev_signal_actions[ticker] = current_action

    # LLM consensus trigger (high confidence + proven accuracy)
    if LLM_AVAILABLE and check_count > 5:
        try:
            llm_sigs = get_llm_signals()
            llm_acc = get_llm_accuracy()
            for ticker, data in llm_sigs.items():
                consensus = data.get("consensus", {})
                direction = consensus.get("direction", "flat")
                confidence = consensus.get("confidence", 0)
                has_accuracy = any(
                    v.get("total", 0) >= 10 and v.get("accuracy", 0) >= 0.6
                    for k, v in llm_acc.items()
                )
                if direction in ("up", "down") and confidence >= 0.7 and has_accuracy:
                    action = "BUY" if direction == "up" else "SELL"
                    reasons.append(f"LLM consensus {ticker}: {action} ({confidence:.0%})")
        except Exception:
            pass

    # Drawdown circuit breaker
    if RISK_AVAILABLE and check_count % 10 == 0 and check_count > 0:
        try:
            dd = check_portfolio_drawdown(POSITIONS, prices)
            if dd.get("breached"):
                reasons.append(f"EMERGENCY drawdown breached: {dd['current_drawdown_pct']:.1f}%")
            elif dd.get("level") == "WARNING":
                reasons.append(f"drawdown warning: {dd['current_drawdown_pct']:.1f}%")
        except Exception:
            pass

    # Heartbeat (every ~30 min)
    if check_count > 0 and check_count % HEARTBEAT_CHECKS == 0:
        if not reasons:
            reasons.append("periodic heartbeat")

    # End of day
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if now_utc.hour == EOD_HOUR_UTC and now_utc.minute < 3 and check_count > 10:
        reasons.append("end_of_day_summary")

    return len(reasons) > 0, reasons

def _kill_claude():
    """Kill a running Claude process."""
    global claude_proc, claude_log_fh
    if claude_proc and claude_proc.poll() is None:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(claude_proc.pid)],
                capture_output=True,
            )
        else:
            claude_proc.kill()
        try:
            claude_proc.wait(timeout=10)
        except:
            pass
    claude_proc = None
    if claude_log_fh:
        try:
            claude_log_fh.close()
        except:
            pass
        claude_log_fh = None

def invoke_claude(trigger_reasons, tier=2):
    """Invoke Claude Code as Layer 2 trading agent with tier-based model selection."""
    global claude_proc, claude_log_fh, claude_start, claude_timeout
    global invoke_count, last_invoke_time

    tier_cfg = TIER_CONFIG.get(tier, TIER_CONFIG[2])

    # Guard 1: Check if previous invocation is still running
    if claude_proc and claude_proc.poll() is None:
        elapsed = time.time() - claude_start
        if elapsed > claude_timeout:
            log(f"Claude timed out ({elapsed:.0f}s), killing")
            _kill_claude()
        else:
            log(f"Claude still running ({elapsed:.0f}s), skipping invocation")
            return False

    # Guard 2: Cooldown
    since_last = time.time() - last_invoke_time
    if last_invoke_time > 0 and since_last < MIN_INVOKE_INTERVAL:
        remaining = MIN_INVOKE_INTERVAL - since_last
        log(f"Cooldown: {remaining:.0f}s remaining, skipping (tier {tier})")
        return False

    # Read prompt template
    prompt_file = "data/metals_agent_prompt.txt"
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            base_prompt = f.read()
    except:
        log(f"Cannot read {prompt_file}")
        return False

    reason_str = "; ".join(trigger_reasons[:5])
    tier_label = tier_cfg["label"]
    prompt = f"{base_prompt}\n\n## This Invocation\nTier: {tier} ({tier_label})\nTrigger: {reason_str}\nTime: {cet_time_str()}\nCheck #{check_count}, Invocation #{invoke_count + 1}"

    # Tier 1 (Haiku) gets focused prompt
    if tier == 1:
        prompt = (
            "You are the metals intraday trading agent (QUICK ASSESSMENT).\n"
            "Read data/metals_context.json — analyze the trigger and current positions.\n"
            "Read data/metals_decisions.jsonl — check your last 3 decisions for continuity.\n"
            "Read memory/trading_rules.md — follow the mandatory checklist.\n\n"
            "For EACH position: assess P&L from ENTRY, distance from peak, distance from stop.\n"
            "Check the `risk` section for Monte Carlo VaR and drawdown status.\n"
            "Check `llm_predictions` for model consensus and accuracy.\n"
            "If a position is in danger (near stop, big drop from peak), flag it clearly.\n"
            "If everything is stable, confirm HOLD with brief reasoning.\n\n"
            "Strategic thesis: Silver bull 2026, target ATH. Bias HOLD. Only sell on structure break.\n\n"
            "ALWAYS: (1) Log decision to data/metals_decisions.jsonl, (2) Send Telegram with P&L per position.\n"
            "Keep it concise — you are Haiku, optimize for speed.\n"
            f"\nTrigger: {reason_str}\nTime: {cet_time_str()}\n"
            f"Check #{check_count}, Invocation #{invoke_count + 1}"
        )

    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        log("claude not found on PATH!")
        return False

    cmd = [
        claude_cmd, "-p", prompt,
        "--allowedTools", "Edit,Read,Bash,Write",
        "--max-turns", str(tier_cfg["max_turns"]),
    ]
    if tier_cfg["model"]:
        cmd.extend(["--model", tier_cfg["model"]])

    try:
        log_fh = open("data/metals_agent.log", "a", encoding="utf-8")
        log_fh.write(f"\n{'='*60}\n")
        log_fh.write(f"Invocation #{invoke_count + 1} | T{tier} ({tier_label}) | "
                      f"{datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"Model: {tier_cfg['model'] or 'default (opus)'} | "
                      f"Max turns: {tier_cfg['max_turns']} | "
                      f"Timeout: {tier_cfg['timeout']}s\n")
        log_fh.write(f"Trigger: {reason_str}\n")
        log_fh.write(f"{'='*60}\n")

        agent_env = os.environ.copy()
        agent_env.pop("CLAUDECODE", None)
        agent_env.pop("CLAUDE_CODE_ENTRYPOINT", None)

        claude_proc = subprocess.Popen(
            cmd,
            cwd=r"Q:\finance-analyzer",
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=agent_env,
        )
        claude_log_fh = log_fh
        claude_start = time.time()
        claude_timeout = tier_cfg["timeout"]
        invoke_count += 1
        last_invoke_time = time.time()

        log(f"Claude T{tier} invoked (pid={claude_proc.pid}, model={tier_cfg['model'] or 'opus'}, "
            f"max_turns={tier_cfg['max_turns']}, timeout={tier_cfg['timeout']}s)")

        # Log invocation start
        log_invocation(tier, tier_cfg["model"], reason_str, check_count, invoke_count)

        if tier >= 2:
            send_telegram(f"_Metals L2 T{tier} ({tier_label}): {reason_str}_")

        return True
    except Exception as e:
        log(f"Claude invocation failed: {e}")
        if log_fh:
            try:
                log_fh.close()
            except:
                pass
        return False

def main():
    global check_count, last_signal_data, last_invoke_prices, startup_grace
    global claude_proc, claude_log_fh, claude_start, claude_timeout
    global short_prices, daily_range_stats

    # Probe time server on startup
    h, ts, src = get_cet_time()
    log(f"Starting metals trading loop (v6 — LLM + Monte Carlo + Daily Ranges + Spike Catcher)...")
    log(f"Time: {ts} (source: {src})")
    log(f"Check interval: {CHECK_INTERVAL}s | Heartbeat: every {HEARTBEAT_CHECKS} checks (~{HEARTBEAT_CHECKS*CHECK_INTERVAL//60}min)")
    log(f"Triggers: price>{TRIGGER_PRICE_MOVE}% | trail>{TRIGGER_TRAILING}% | profit>{TRIGGER_PROFIT}%")
    log(f"Stop levels: L1(warn)<{STOP_L1_PCT}% | L2(alert)<{STOP_L2_PCT}% | L3(emergency)<{STOP_L3_PCT}%")
    log(f"Cooldown: {MIN_INVOKE_INTERVAL}s between invocations")
    log(f"Tiers: T1=haiku(8t,60s) | T2=sonnet(15t,180s) | T3=opus(20t,300s)")
    log(f"Short instruments: {', '.join(v['name'] for v in SHORT_INSTRUMENTS.values())}")
    if SPIKE_ENABLED:
        log(f"Spike catcher: place@{SPIKE_PLACE_CET:.2f} cancel@{SPIKE_CANCEL_CET:.2f} "
            f"P{SPIKE_PERCENTILE} {SPIKE_PARTIAL_PCT}% partial")
    log(f"Invocation log: {INVOCATION_LOG}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(storage_state="data/avanza_storage_state.json")
        page = ctx.new_page()
        page.goto("https://www.avanza.se/min-ekonomi/oversikt.html", wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        log("Avanza session loaded")

        # Initialize peaks and last-invoke prices
        for key, pos in POSITIONS.items():
            if pos["active"]:
                p = fetch_price(page, pos["ob_id"], pos["api_type"])
                if p and p.get('bid'):
                    peak_bids[key] = p.get('high') or p['bid']
                    last_invoke_prices[key] = p['bid']
                    log(f"  {key}: bid={p['bid']}, peak={peak_bids[key]}, entry={pos['entry']}, "
                        f"pnl={pnl_pct(p['bid'], pos['entry']):+.1f}%")

        # Read initial signal data
        last_signal_data = read_signal_data()
        if last_signal_data:
            log(f"  Signal data loaded (age: {last_signal_data.get('age_min', '?')}min)")

        log("Token budget estimate: ~75K tokens/day (25 haiku + 4 sonnet + 1 opus)")

        # Start local LLM background thread
        if LLM_AVAILABLE:
            def _get_signal_data():
                return last_signal_data

            def _get_underlying_prices():
                result = {}
                if price_history:
                    snap = price_history[-1]
                    silver_und = snap.get("silver79_und") or snap.get("silver301_und")
                    gold_und = snap.get("gold_und")
                    if silver_und and silver_und > 0:
                        result["XAG-USD"] = silver_und
                    if gold_und and gold_und > 0:
                        result["XAU-USD"] = gold_und
                return result

            start_llm_thread(_get_signal_data, _get_underlying_prices)
            log("LLM thread: Ministral + Chronos running every 5min")
        else:
            log("LLM thread: NOT available (import failed)")

        if RISK_AVAILABLE:
            log("Risk module: Monte Carlo + Trade Guards + Drawdown active")
            # Compute daily range stats at startup
            daily_range_stats.update(compute_daily_range_stats())
            if daily_range_stats:
                for ticker, rs in daily_range_stats.items():
                    dr = rs.get("daily_range", {})
                    log(f"  {ticker} daily range: P50={dr.get('p50',0)}% P90={dr.get('p90',0)}% "
                        f"({rs.get('trading_days',0)} days)")
            else:
                log("  Daily range stats: no data available")
        else:
            log("Risk module: NOT available (import failed)")

        send_telegram(f"""*METALS LOOP v6 STARTED*
Tiered: T1=haiku T2=sonnet T3=opus
LLM: {"Ministral+Chronos (5min)" if LLM_AVAILABLE else "DISABLED"}
Risk: {"MC+Guards+Drawdown+DailyRanges" if RISK_AVAILABLE else "DISABLED"}
Spike: {"P" + str(SPIKE_PERCENTILE) + " " + str(SPIKE_PARTIAL_PCT) + "% @15:15-16:30" if SPIKE_ENABLED else "OFF"}
Positions: gold(8x), silver79(5x), silver301(4.3x)""")

        try:
            while True:
                check_count += 1

                if not is_market_hours():
                    if check_count % 40 == 0:
                        log("Outside market hours")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Check if all positions closed
                active = sum(1 for p in POSITIONS.values() if p["active"])
                if active == 0:
                    log("All positions closed.")
                    send_telegram("*METALS LOOP* All positions closed. Loop exiting.")
                    break

                # Fetch prices
                prices = {}
                try:
                    for key, pos in POSITIONS.items():
                        if pos["active"]:
                            p = fetch_price(page, pos["ob_id"], pos["api_type"])
                            if p:
                                prices[key] = p
                                bid = p.get('bid') or 0
                                if bid > peak_bids.get(key, 0):
                                    peak_bids[key] = bid
                except Exception as e:
                    log(f"Price error: {e}")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Fetch short instrument prices (every 4th check)
                if check_count % 4 == 0:
                    for sk, si in SHORT_INSTRUMENTS.items():
                        try:
                            sp = fetch_price(page, si["ob_id"], si["api_type"])
                            if sp:
                                short_prices[sk] = sp
                        except Exception:
                            pass

                # Read signal data periodically (every ~6 min)
                if check_count % 4 == 0:
                    last_signal_data = read_signal_data()

                # Store price snapshot
                snap = {
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "check": check_count,
                }
                for key in POSITIONS:
                    if key in prices:
                        snap[key] = prices[key].get('bid', 0)
                        snap[f"{key}_und"] = prices[key].get('underlying', 0)
                price_history.append(snap)
                if len(price_history) > 120:
                    price_history.pop(0)

                # Log portfolio value for drawdown tracking (every 10th check)
                if RISK_AVAILABLE and check_count % 10 == 0:
                    try:
                        log_portfolio_value(POSITIONS, prices)
                    except Exception:
                        pass

                # --- SPIKE CATCHER: US open limit sell orders ---
                if SPIKE_ENABLED and RISK_AVAILABLE and daily_range_stats and check_count > 3:
                    h_now = cet_hour()
                    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    spike_st = load_spike_state()

                    # Reset state on new day
                    if spike_st.get("date") != today_str:
                        spike_st = {"orders": {}, "targets": {}, "date": today_str,
                                    "placed": False, "cancelled": False}
                        save_spike_state(spike_st)

                    # Phase 1: Place orders at 15:15 CET
                    if (not spike_st["placed"] and SPIKE_PLACE_CET <= h_now < SPIKE_CANCEL_CET):
                        targets = compute_spike_targets(
                            POSITIONS, prices, daily_range_stats,
                            percentile=SPIKE_PERCENTILE, partial_pct=SPIKE_PARTIAL_PCT)
                        if targets:
                            log(f"Spike catcher: placing {len(targets)} limit sell orders")
                            for k, t in targets.items():
                                log(f"  {k}: sell {t['units_to_sell']}u @ {t['target_price']} "
                                    f"(+{t['target_pnl_pct']:.1f}%)")
                            placed = place_spike_orders(page, POSITIONS, prices, targets)
                            spike_st["orders"] = placed
                            spike_st["targets"] = targets
                            spike_st["placed"] = True
                            save_spike_state(spike_st)
                            send_telegram(
                                f"*SPIKE CATCHER*\n"
                                + "\n".join(f"`{k}: SELL {t['units_to_sell']}u @ {t['target_price']} "
                                           f"(+{t['target_pnl_pct']:.1f}%)`"
                                           for k, t in targets.items())
                                + f"\n_P{SPIKE_PERCENTILE} target, cancels at 16:30_"
                            )
                        else:
                            log("Spike catcher: no eligible positions for spike targets")
                            spike_st["placed"] = True  # don't retry
                            save_spike_state(spike_st)

                    # Phase 2: Check for fills (every 2nd check while orders active)
                    if spike_st["placed"] and not spike_st["cancelled"] and check_count % 2 == 0:
                        if spike_st.get("orders"):
                            filled = check_spike_fills(page, spike_st, POSITIONS)
                            for fk in filled:
                                spike_st["orders"].pop(fk, None)
                                # Update position units if partial sell
                                if fk in POSITIONS and SPIKE_PARTIAL_PCT < 100:
                                    sold = spike_st.get("targets", {}).get(fk, {}).get("units_to_sell", 0)
                                    POSITIONS[fk]["units"] = max(0, POSITIONS[fk]["units"] - sold)
                                    if POSITIONS[fk]["units"] == 0:
                                        POSITIONS[fk]["active"] = False
                                if RISK_AVAILABLE and fk in spike_st.get("targets", {}):
                                    record_metals_trade(fk, "SELL",
                                                        pnl_pct_value=spike_st["targets"][fk].get("target_pnl_pct", 0))
                            if filled:
                                save_spike_state(spike_st)

                    # Phase 3: Cancel unfilled at 16:30 CET
                    if spike_st["placed"] and not spike_st["cancelled"] and h_now >= SPIKE_CANCEL_CET:
                        if spike_st.get("orders"):
                            log(f"Spike catcher: cancelling {len(spike_st['orders'])} unfilled orders")
                            cancel_spike_orders(page, spike_st)
                            send_telegram("_Spike orders cancelled (16:30 CET, unfilled)_")
                        spike_st["cancelled"] = True
                        save_spike_state(spike_st)

                # Startup grace
                if startup_grace:
                    startup_grace = False
                    log(f"#{check_count} Baseline established (grace period)")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Check triggers
                triggered, reasons = check_triggers(prices)

                # L3 EMERGENCY: auto-sell positions near barrier
                for r in reasons[:]:
                    if "L3 EMERGENCY" in r:
                        for key, pos in POSITIONS.items():
                            if not pos["active"] or key not in prices:
                                continue
                            bid = prices[key].get('bid') or 0
                            if bid <= 0:
                                continue
                            dist = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
                            if dist < STOP_L3_PCT:
                                emergency_sell(page, key, pos, bid)
                        break

                # Log status (every 3rd check)
                if check_count % 3 == 0:
                    parts = []
                    for key, pos in POSITIONS.items():
                        if pos["active"] and key in prices:
                            bid = prices[key].get('bid', 0)
                            pnl = pnl_pct(bid, pos["entry"])
                            parts.append(f"{key}:{bid}({pnl:+.1f}%)")
                    cet = cet_time_str()
                    # Add LLM consensus tag
                    llm_tag = ""
                    if LLM_AVAILABLE:
                        try:
                            llm_sigs = get_llm_signals()
                            for t, d in llm_sigs.items():
                                c = d.get("consensus", {})
                                if c.get("direction") in ("up", "down"):
                                    short_t = t.split("-")[0]
                                    llm_tag += f" {short_t}={'UP' if c['direction']=='up' else 'DN'}({c.get('confidence',0):.0%})"
                        except Exception:
                            pass
                    # Add risk score
                    risk_tag = ""
                    if RISK_AVAILABLE:
                        try:
                            dd = check_portfolio_drawdown(POSITIONS, prices)
                            risk_tag = f" DD:{dd.get('current_pnl_pct', 0):+.1f}%"
                        except Exception:
                            pass
                    log(f"#{check_count} [{cet}] {' | '.join(parts)}{llm_tag}{risk_tag}" +
                        (f" [TRIGGER: {reasons[0]}]" if triggered else ""))

                # Invoke Claude if triggered
                if triggered:
                    tier = classify_tier(reasons)
                    write_context(prices, "; ".join(reasons), tier=tier)
                    for key in prices:
                        if prices[key].get('bid'):
                            last_invoke_prices[key] = prices[key]['bid']
                    invoke_claude(reasons, tier=tier)

                # Check if Claude finished (non-blocking)
                if claude_proc and claude_proc.poll() is not None:
                    elapsed = time.time() - claude_start
                    retcode = claude_proc.returncode
                    log(f"Claude finished (rc={retcode}, {elapsed:.0f}s)")
                    # Log completion with elapsed time and return code
                    log_invocation(0, None, "completed", check_count, invoke_count,
                                   elapsed_s=elapsed, rc=retcode)
                    claude_proc = None
                    if claude_log_fh:
                        try:
                            claude_log_fh.close()
                        except:
                            pass

                    # Re-read trade log in case Claude executed a trade
                    if os.path.exists("data/metals_trades.jsonl"):
                        try:
                            with open("data/metals_trades.jsonl", "r") as f:
                                lines = f.readlines()
                            if lines:
                                last_trade = json.loads(lines[-1])
                                log(f"Last trade: {last_trade.get('action','')} {last_trade.get('name','')}")
                        except:
                            pass

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Stopped by user")
        except Exception as e:
            log(f"FATAL: {e}")
            traceback.print_exc()
            send_telegram(f"*METALS LOOP CRASH*: {e}")
        finally:
            _kill_claude()
            if LLM_AVAILABLE:
                try:
                    stop_llm_thread()
                except:
                    pass
            browser.close()
            log(f"Loop stopped: {check_count} checks, {invoke_count} invocations")

if __name__ == "__main__":
    main()
