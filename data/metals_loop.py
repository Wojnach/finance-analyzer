"""
Metals Intraday Trading Loop (Layer 1).
Collects price data every 60-90 seconds from Avanza.
When trigger conditions are met, invokes Claude Code (Layer 2) for trading decisions.
Claude reads data/metals_context.json and decides to buy/sell/hold.

Token cost management:
- Heartbeat (every 30 min): uses Haiku model (cheapest, ~2s)
- Price/trailing triggers: uses Sonnet model (moderate, ~10s)
- Critical triggers (stop proximity, profit target, EOD): uses Opus (full, ~60s)
- Minimum 5 min cooldown between invocations
- Never invokes if previous invocation still running

Run: .venv/Scripts/python.exe data/metals_loop.py
"""
import json, os, sys, time, datetime, traceback, subprocess, shutil, platform
os.chdir(r"Q:/finance-analyzer")

import requests
from playwright.sync_api import sync_playwright

# --- CONFIG ---
CHECK_INTERVAL = 90           # seconds between price checks
TRIGGER_PRICE_MOVE = 2.0      # % move from last invocation to trigger
TRIGGER_TRAILING = 3.0        # % drop from peak to trigger
TRIGGER_PROFIT = 4.0          # % profit from entry to trigger
TRIGGER_STOP_NEAR = 5.0       # % from stop-loss to trigger
HEARTBEAT_CHECKS = 20         # invoke every N checks (~30 min at 90s)
MIN_INVOKE_INTERVAL = 300     # minimum 5 min between invocations
EOD_HOUR_UTC = 16             # 17:00 CET = 16:00 UTC

# Tier config: model, timeout, max_turns
# Matches the main loop's tiered approach
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

SHORT_INSTRUMENT = {
    "name": "BEAR SILVER X5 AVA 12", "ob_id": "2286417", "api_type": "certificate",
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

def is_market_hours():
    now = datetime.datetime.now(datetime.timezone.utc)
    hour_utc = now.hour + now.minute / 60
    weekday = now.weekday()
    return weekday < 5 and 8.0 <= hour_utc <= 16.42

def cet_hour():
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.hour + 1 + now.minute / 60

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
        # Try the full summary first (has per-ticker data)
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

        # agent_summary.json has tickers at top level
        tickers = data.get("tickers", {})
        if not tickers:
            # compact format: look for signal data in different structure
            # The compact JSON has forecast_signals, cumulative_gains, etc.
            # but ticker-level signals are in the full JSON only
            # Fall back to using whatever we can find
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

        # Timeframe heatmap
        timeframes = data.get("timeframe_heatmap", {})
        for ticker in ["XAG-USD", "XAU-USD"]:
            if ticker in timeframes and ticker in result:
                result[ticker]["timeframes"] = timeframes[ticker]

        return result
    except Exception as e:
        log(f"Signal read error: {e}")
        return {}

def read_decision_history(n=5):
    """Read the last N decisions from metals_decisions.jsonl for context."""
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

def write_context(prices, trigger_reason, tier=2):
    """Write context JSON for Claude Layer 2."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ctx = {
        "timestamp": now.isoformat(),
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
        "short_instrument": {
            "name": SHORT_INSTRUMENT["name"],
            "ob_id": SHORT_INSTRUMENT["ob_id"],
            "api_type": SHORT_INSTRUMENT["api_type"],
            "note": "5x short silver certificate, available for hedging",
        },
        "trades_today_file": "data/metals_trades.jsonl",
    }

    # Include historical stats if available (lightweight — just the stats summary)
    try:
        with open("data/metals_history.json", "r", encoding="utf-8") as f:
            history = json.load(f)
        ctx["historical_ytd"] = {
            ticker: data["stats"]
            for ticker, data in history.get("metals", {}).items()
        }
    except:
        pass

    total_val = 0
    total_inv = 0

    for key, pos in POSITIONS.items():
        p = prices.get(key) or {}
        bid = p.get('bid') or 0
        pnl = pnl_pct(bid, pos["entry"]) if bid > 0 else 0
        val = bid * pos["units"]
        peak = peak_bids.get(key, 0)
        from_peak = pnl_pct(bid, peak) if peak > 0 and bid > 0 else 0
        dist_stop = pnl_pct(bid, pos["stop"]) if bid > 0 else 999
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
                ctx["underlying"]["silver"] = p['underlying']
            elif 'gold' in key.lower():
                ctx["underlying"]["gold"] = p['underlying']

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
    """Classify trigger into tier (1=cheap workhorse, 2=deeper analysis, 3=critical).

    Haiku handles the bulk of invocations (price moves, trailing, heartbeats).
    Sonnet only for multi-trigger events or large moves needing deeper analysis.
    Opus reserved for critical decisions (stop proximity, profit targets, EOD).
    """
    # Tier 3 (Critical — Opus): stop proximity, profit target, EOD
    critical_patterns = ["stop-loss", "end_of_day", "profit target"]
    if any(p in r for r in reasons for p in critical_patterns):
        return 3

    # Tier 2 (Deeper — Sonnet): only when multiple triggers fire simultaneously,
    # or very large moves that need more careful analysis
    if len(reasons) >= 2:
        return 2

    # Tier 1 (Workhorse — Haiku): single triggers, heartbeats, routine
    # This handles: price moves, trailing drops, signal flips, heartbeats
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

        # Near stop-loss
        dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
        if 0 < dist_stop < TRIGGER_STOP_NEAR:
            reasons.append(f"{key} {dist_stop:.1f}% from stop-loss")

    # Signal flip detection (sustained — check if action changed from previous)
    if last_signal_data:
        for ticker in ["XAG-USD", "XAU-USD"]:
            if ticker in last_signal_data:
                current_action = last_signal_data[ticker].get("action", "?")
                prev_action = prev_signal_actions.get(ticker)
                if prev_action and current_action != prev_action and current_action in ("BUY", "SELL"):
                    reasons.append(f"signal flip {ticker}: {prev_action}->{current_action}")
                prev_signal_actions[ticker] = current_action

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

    # Guard 2: Cooldown — minimum interval between invocations
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

    # Add trigger context and tier instruction to prompt
    reason_str = "; ".join(trigger_reasons[:5])
    tier_label = tier_cfg["label"]
    prompt = f"{base_prompt}\n\n## This Invocation\nTier: {tier} ({tier_label})\nTrigger: {reason_str}\nTime: {datetime.datetime.now().strftime('%H:%M CET')}\nCheck #{check_count}, Invocation #{invoke_count + 1}"

    # Tier 1 (Haiku workhorse) gets a focused prompt — reads full context but
    # keeps analysis brief. Can recommend trades but prefers HOLD.
    if tier == 1:
        prompt = (
            "You are the metals intraday trading agent (QUICK ASSESSMENT).\n"
            "Read data/metals_context.json — analyze the trigger and current positions.\n"
            "Read data/metals_decisions.jsonl — check your last 3 decisions for continuity.\n"
            "Read memory/trading_rules.md — follow the mandatory checklist.\n\n"
            "For EACH position: assess P&L from ENTRY, distance from peak, distance from stop.\n"
            "If a position is in danger (near stop, big drop from peak), flag it clearly.\n"
            "If everything is stable, confirm HOLD with brief reasoning.\n\n"
            "Strategic thesis: Silver bull 2026, target ATH. Bias HOLD. Only sell on structure break.\n\n"
            "ALWAYS: (1) Log decision to data/metals_decisions.jsonl, (2) Send Telegram with P&L per position.\n"
            "Keep it concise — you are Haiku, optimize for speed.\n"
            f"\nTrigger: {reason_str}\nTime: {datetime.datetime.now().strftime('%H:%M CET')}\n"
            f"Check #{check_count}, Invocation #{invoke_count + 1}"
        )

    # Find claude executable
    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        log("claude not found on PATH!")
        return False

    cmd = [
        claude_cmd, "-p", prompt,
        "--allowedTools", "Edit,Read,Bash,Write",
        "--max-turns", str(tier_cfg["max_turns"]),
    ]

    # Add model flag for cheaper tiers
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

        # Strip Claude Code session markers to avoid nested session error
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

        # Brief Telegram notification (skip for heartbeats to reduce noise)
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

    log("Starting metals trading loop (v2 — tiered invocation)...")
    log(f"Check interval: {CHECK_INTERVAL}s | Heartbeat: every {HEARTBEAT_CHECKS} checks (~{HEARTBEAT_CHECKS*CHECK_INTERVAL//60}min)")
    log(f"Triggers: price>{TRIGGER_PRICE_MOVE}% | trail>{TRIGGER_TRAILING}% | profit>{TRIGGER_PROFIT}% | stop<{TRIGGER_STOP_NEAR}%")
    log(f"Cooldown: {MIN_INVOKE_INTERVAL}s between invocations")
    log(f"Tiers: T1=haiku(8t,60s) | T2=sonnet(15t,180s) | T3=opus(20t,300s)")

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

        # Estimate daily token budget
        # At 90s intervals over 8.5h market day: ~340 checks
        # Most triggers: T1 haiku (~25/day, ~1K tokens each = ~25K)
        # Multi-trigger events: T2 sonnet (~3-5/day, ~5K tokens each = ~20K)
        # Critical decisions: T3 opus (~1-2/day, ~20K tokens = ~30K)
        # Total: ~75K tokens/day — well within Max subscription budget
        log("Token budget estimate: ~75K tokens/day (25 haiku + 4 sonnet + 1 opus)")

        send_telegram(f"""*METALS LOOP v2 STARTED*
Tiered invocation: T1=haiku(30min) T2=sonnet(triggers) T3=opus(critical)
Cooldown: {MIN_INVOKE_INTERVAL//60}min between invocations
Positions: gold(5x), silver79(1.3x), silver301(4.3x)""")

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
                                # Update peak
                                bid = p.get('bid') or 0
                                if bid > peak_bids.get(key, 0):
                                    peak_bids[key] = bid
                except Exception as e:
                    log(f"Price error: {e}")
                    time.sleep(CHECK_INTERVAL)
                    continue

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

                # Startup grace: first check establishes baseline without triggering
                if startup_grace:
                    startup_grace = False
                    log(f"#{check_count} Baseline established (grace period)")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Check triggers
                triggered, reasons = check_triggers(prices)

                # Log status (every 3rd check)
                if check_count % 3 == 0:
                    parts = []
                    for key, pos in POSITIONS.items():
                        if pos["active"] and key in prices:
                            bid = prices[key].get('bid', 0)
                            pnl = pnl_pct(bid, pos["entry"])
                            parts.append(f"{key}:{bid}({pnl:+.1f}%)")
                    log(f"#{check_count} {' | '.join(parts)}" +
                        (f" [TRIGGER: {reasons[0]}]" if triggered else ""))

                # Invoke Claude if triggered
                if triggered:
                    tier = classify_tier(reasons)

                    # Write fresh context
                    write_context(prices, "; ".join(reasons), tier=tier)

                    # Update last-invoke prices
                    for key in prices:
                        if prices[key].get('bid'):
                            last_invoke_prices[key] = prices[key]['bid']

                    invoke_claude(reasons, tier=tier)

                # Check if Claude finished (non-blocking)
                if claude_proc and claude_proc.poll() is not None:
                    elapsed = time.time() - claude_start
                    retcode = claude_proc.returncode
                    log(f"Claude finished (rc={retcode}, {elapsed:.0f}s)")
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
            browser.close()
            log(f"Loop stopped: {check_count} checks, {invoke_count} invocations")

if __name__ == "__main__":
    main()
