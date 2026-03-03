"""
Metals Intraday Trading Loop v7 (Layer 1).
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

try:
    from metals_signal_tracker import (
        log_snapshot, backfill_outcomes, get_accuracy_report,
        get_accuracy_summary, get_accuracy_for_context, get_snapshot_count,
    )
    TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_signal_tracker import failed: {e}", flush=True)
    TRACKER_AVAILABLE = False

try:
    from metals_swing_trader import SwingTrader
    SWING_TRADER_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] swing trader import failed: {e}", flush=True)
    SWING_TRADER_AVAILABLE = False

try:
    from metals_swing_config import WARRANT_CATALOG
    CATALOG_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_swing_config import failed: {e}", flush=True)
    WARRANT_CATALOG = {}
    CATALOG_AVAILABLE = False

from metals_avanza_helpers import (
    get_csrf,
    fetch_price,
    fetch_account_cash,
    place_order,
    place_stop_loss,
    check_session_alive,
)

# --- CONFIG ---
CHECK_INTERVAL = 90           # seconds between price checks
TRIGGER_PRICE_MOVE = 2.0      # % move from last invocation to trigger
TRIGGER_TRAILING = 3.0        # % drop from peak to trigger
TRIGGER_PROFIT = 4.0          # % profit from entry to trigger
TRIGGER_STOP_NEAR = 5.0       # % from stop-loss to trigger
HEARTBEAT_CHECKS = 20         # invoke every N checks (~30 min at 90s)
MIN_INVOKE_INTERVAL = 300     # minimum 5 min between invocations
EOD_HOUR_CET = 17.0           # 17:00 CET (DST-safe via cet_hour())

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

# Cascading stop-loss orders (hardware protection via Avanza limit orders)
STOP_ORDER_ENABLED = True
STOP_ORDER_LEVELS = 3          # number of stop orders per position
STOP_ORDER_SPREAD_PCT = 1.0    # spread between levels (1% of stop price)
STOP_ORDER_FILE = "data/metals_stop_orders.json"

# Trade queue (Layer 2 writes intent, Layer 1 executes)
TRADE_QUEUE_ENABLED = True
TRADE_QUEUE_FILE = "data/metals_trade_queue.json"
TRADE_QUEUE_MAX_AGE_S = 300     # expire orders older than 5 min
TRADE_QUEUE_MAX_SLIPPAGE = 2.0  # reject if price moved > 2% from queued

# Session health monitoring
SESSION_HEALTH_CHECK_INTERVAL = 20   # check every N loops (~30 min at 90s)
SESSION_EXPIRY_WARNING_H = 20        # warn when storage state is >20h old (session lasts ~24h)
SESSION_STORAGE_FILE = "data/avanza_storage_state.json"

# Trailing stop config
TRAIL_START_PCT = 2.0           # start trailing after 2% gain from entry
TRAIL_DISTANCE_PCT = 3.0        # trail 3% below current bid
TRAIL_MIN_MOVE_PCT = 1.0        # minimum move to update (avoid excessive API calls)

# Momentum exit (derivative-based early exit)
MOMENTUM_ENABLED = True
MOMENTUM_LOOKBACK = 5           # checks (5 * 90s = ~7.5 min)
MOMENTUM_MIN_VELOCITY = -0.5    # must be dropping at least 0.5%/check
MOMENTUM_ACCEL_THRESHOLD = -0.1 # acceleration must be negative (accelerating decline)
MOMENTUM_REQUIRE_L1 = True      # only trigger if already in L1+ danger zone

# Auto-exit override (prevent Claude HOLD paralysis)
AUTO_EXIT_L2_CHECKS = 5         # auto-sell after position in L2+ zone for N checks

# Periodic holdings verification (detect broker-triggered stop-losses)
HOLDINGS_CHECK_INTERVAL = 20    # verify Avanza holdings every N checks (~30 min at 90s)

# Tier config: model, timeout, max_turns
TIER_CONFIG = {
    1: {"model": "haiku",  "timeout": 60,   "max_turns": 8,  "label": "QUICK"},
    2: {"model": "sonnet", "timeout": 180,  "max_turns": 15, "label": "ANALYSIS"},
    3: {"model": None,     "timeout": 300,  "max_turns": 20, "label": "CRITICAL"},
}

# --- POSITIONS (defaults — overridden by persisted state on startup) ---
POSITIONS_DEFAULTS = {
    "gold": {
        "name": "BULL GULD X8 N", "ob_id": "856394", "api_type": "certificate",
        "units": 4, "entry": 907.5, "stop": 780.0, "active": True,
    },
    "silver301": {
        "name": "MINI L SILVER AVA 301", "ob_id": "2334960", "api_type": "warrant",
        "units": 130, "entry": 15.36, "stop": 12.50, "active": True,
    },
    "silver_sg": {
        "name": "MINI L SILVER SG", "ob_id": "2043157", "api_type": "warrant",
        "units": 441, "entry": 52.0, "stop": 46.0, "active": True,
    },
}
POSITIONS_STATE_FILE = "data/metals_positions_state.json"

def _load_positions():
    """Load position state from disk, falling back to defaults."""
    import copy
    positions = copy.deepcopy(POSITIONS_DEFAULTS)
    try:
        if os.path.exists(POSITIONS_STATE_FILE):
            with open(POSITIONS_STATE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, state in saved.items():
                if key in positions:
                    # Restore persisted active/sold status
                    positions[key]["active"] = state.get("active", True)
                    # Restore updated units/entry/stop if they were modified
                    if "units" in state:
                        positions[key]["units"] = state["units"]
                    if "entry" in state:
                        positions[key]["entry"] = state["entry"]
                    if "stop" in state:
                        positions[key]["stop"] = state["stop"]
                    # Preserve sell metadata
                    if "sold_ts" in state:
                        positions[key]["sold_ts"] = state["sold_ts"]
                    if "sold_price" in state:
                        positions[key]["sold_price"] = state["sold_price"]
                    if "sold_reason" in state:
                        positions[key]["sold_reason"] = state["sold_reason"]
            print(f"Position state loaded from {POSITIONS_STATE_FILE}", flush=True)
    except Exception as e:
        print(f"Position state load failed (using defaults): {e}", flush=True)
    return positions

def _save_positions(positions):
    """Persist position state to disk (survives restarts)."""
    state = {}
    for key, pos in positions.items():
        state[key] = {
            "active": pos.get("active", True),
            "units": pos.get("units"),
            "entry": pos.get("entry"),
            "stop": pos.get("stop"),
        }
        # Include sell metadata if present
        for field in ("sold_ts", "sold_price", "sold_reason"):
            if field in pos:
                state[key][field] = pos[field]
    try:
        with open(POSITIONS_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"Position state save failed: {e}")

def _verify_position_holdings(page, positions):
    """At startup, verify actual Avanza holdings match our position state.

    Uses the canonical positions API: /_api/position-data/positions
    Response has withOrderbook[] where each item has:
      - account.id
      - instrument.orderbook.id  (the orderbook ID we match on)
      - volume.value             (units held)
    All numeric fields are wrapped in {"value": N, "unit": "..."} objects.
    Falls back to price check if positions API fails.
    """
    held_ob_ids = set()
    try:
        result = page.evaluate("""async (accountId) => {
            const resp = await fetch(
                'https://www.avanza.se/_api/position-data/positions',
                {credentials: 'include'}
            );
            if (resp.status !== 200) return {error: resp.status};
            const data = await resp.json();
            const ids = [];
            for (const item of (data.withOrderbook || [])) {
                if (item.account && String(item.account.id) === accountId) {
                    const obId = item.instrument && item.instrument.orderbook
                        ? String(item.instrument.orderbook.id) : null;
                    if (obId) {
                        const vol = item.volume && item.volume.value != null
                            ? item.volume.value : 0;
                        ids.push({id: obId, units: vol});
                    }
                }
            }
            return {positions: ids};
        }""", ACCOUNT_ID)

        if result and "positions" in result:
            held_map = {p["id"]: p["units"] for p in result["positions"]}
            held_ob_ids = set(held_map.keys())
            log(f"  Avanza account {ACCOUNT_ID} has {len(held_ob_ids)} positions")

            for key, pos in positions.items():
                if not pos.get("active"):
                    log(f"  {key}: already inactive (sold)")
                    continue
                if pos["ob_id"] in held_ob_ids:
                    api_units = held_map[pos["ob_id"]]
                    log(f"  {key}: confirmed held on Avanza ({api_units}u)")
                    if api_units != pos["units"]:
                        log(f"  {key}: units mismatch — state={pos['units']}, Avanza={api_units}, updating")
                        pos["units"] = api_units
                else:
                    log(f"  {key}: NOT found in Avanza holdings — deactivating")
                    pos["active"] = False
                    pos["sold_reason"] = "startup_verify_not_held"
                    pos["sold_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return
        else:
            log(f"  Positions API returned: {result}")
    except Exception as e:
        log(f"  Positions API failed ({e}), falling back to price check")

    # Fallback: just verify prices exist (can't check holdings)
    for key, pos in positions.items():
        if not pos.get("active"):
            log(f"  {key}: already inactive")
            continue
        try:
            data = fetch_price(page, pos["ob_id"], pos["api_type"])
            if data is None:
                log(f"  {key}: API returned null — possibly delisted")
                continue
            bid = data.get("bid") or data.get("last") or 0
            if bid <= 0:
                log(f"  {key}: bid=0, last=0 — instrument may be dead, deactivating")
                pos["active"] = False
                pos["sold_reason"] = "startup_verify_no_price"
                pos["sold_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            else:
                log(f"  {key}: bid={bid} (holdings unverified, keeping active)")
        except Exception as e:
            log(f"  {key}: verify failed ({e}), keeping current state")

# Load positions (persisted state overrides defaults)
POSITIONS = _load_positions()

SHORT_INSTRUMENTS = {
    "bear_silver_x5": {
        "name": "BEAR SILVER X5 AVA 12", "ob_id": "2286417", "api_type": "certificate",
    },
}

ACCOUNT_ID = "1625505"

try:
    with open("config.json", "r", encoding="utf-8") as _cf:
        config = json.load(_cf)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"[FATAL] Cannot load config.json: {e}", flush=True)
    sys.exit(1)
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
l2_zone_checks = {}           # tracks how many consecutive checks each position has been in L2+ zone
cached_account_data = {}      # latest account buying power (refreshed periodically)
cached_warrant_catalog = {}   # warrant catalog with live prices (refreshed periodically)
session_healthy = True            # tracks Avanza session health
session_alert_sent = False        # debounce: only send one alert per outage
session_expiry_warned = False     # debounce: only warn once about approaching expiry

def send_telegram(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        print(f"[TG ERROR] {e}", flush=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def pnl_pct(current, entry):
    if entry == 0: return 0
    return ((current - entry) / entry) * 100

def get_cet_time():
    """Get current CET/CEST time from timeapi.io, fallback to zoneinfo (DST-safe)."""
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
    except Exception as e:
        print(f"[WARN] timeapi.io failed: {e}", flush=True)
    # Fallback: zoneinfo handles DST correctly (CET/CEST)
    try:
        from zoneinfo import ZoneInfo
        now = datetime.datetime.now(ZoneInfo("Europe/Stockholm"))
        h = now.hour
        m = now.minute
        return h + m / 60, f"{h:02d}:{m:02d} CET", "zoneinfo"
    except ImportError:
        # Last resort: UTC+1 (wrong during summer DST)
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
            except json.JSONDecodeError:
                pass
        return entries
    except Exception as e:
        log(f"Decision history read error: {e}")
        return []

def emergency_sell(page, key, pos, bid):
    """L3 emergency auto-sell via Avanza API.

    Returns True if position was successfully sold or confirmed already sold.
    Returns False if sell failed and position may still be active.
    """
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
            return False

        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
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
            "validUntil": today_str,
            "volume": pos["units"],
        }, csrf])

        log(f"Emergency sell result: {result}")

        # Log trade
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        trade = {
            "ts": now_ts,
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

        # Parse API response to determine outcome
        body_str = result.get("body", "")
        try:
            body = json.loads(body_str)
        except (json.JSONDecodeError, TypeError):
            body = {}

        order_status = body.get("orderRequestStatus", "")
        message_code = body.get("messageCode", "")

        if order_status == "SUCCESS":
            # Sell order placed successfully
            send_telegram(f"*L3 SELL OK* {pos['name']} — order placed")
            pos["active"] = False
            pos["sold_ts"] = now_ts
            pos["sold_price"] = bid
            pos["sold_reason"] = "L3_emergency"
            _save_positions(POSITIONS)
            # Cancel any remaining stop orders for this position
            _cleanup_stop_orders_for(page, key)

            # Record in trade guards
            if RISK_AVAILABLE:
                record_metals_trade(key, "SELL", pnl_pct_value=pnl_pct(bid, pos["entry"]))
            return True

        elif "short.sell.not.allowed" in message_code:
            # "Short sell not allowed" means we tried to sell more than we hold.
            # This happens when: (a) position already sold by stop-loss on Avanza,
            # (b) position was sold in a previous emergency_sell this session, or
            # (c) we never held this position on this account.
            # In all cases, deactivate — we clearly don't hold it.
            log(f"  {key}: short-sell-not-allowed — position already sold or not held, deactivating")
            send_telegram(f"*L3* {pos['name']}: not held (already sold?), deactivating")
            pos["active"] = False
            pos["sold_ts"] = now_ts
            pos["sold_price"] = bid
            pos["sold_reason"] = "L3_already_sold"
            _save_positions(POSITIONS)
            # Cancel any remaining stop orders for this position
            _cleanup_stop_orders_for(page, key)
            return True

        else:
            # Other error — keep position active, may need manual intervention
            error_msg = body.get("message", body_str[:100])
            log(f"  {key}: sell failed with: {error_msg}")
            send_telegram(f"*L3 SELL FAILED* {pos['name']}: {error_msg}")
            return False

    except Exception as e:
        log(f"Emergency sell FAILED: {e}")
        send_telegram(f"*L3 SELL FAILED*: {e}")
        return False

def _cleanup_stop_orders_for(page, key):
    """Cancel any remaining stop orders for a sold position and clean up state."""
    try:
        stop_state = _load_stop_orders()
        if key in stop_state and stop_state[key].get("orders"):
            csrf = get_csrf(page)
            if csrf:
                _cancel_stop_orders(page, key, stop_state[key], csrf)
                log(f"  Cancelled stale stop orders for {key} (position sold)")
            del stop_state[key]
            _save_stop_orders(stop_state)
    except Exception as e:
        log(f"  Stop order cleanup error for {key}: {e}")


def _load_stop_orders():
    """Load stop order state from disk."""
    try:
        if os.path.exists(STOP_ORDER_FILE):
            with open(STOP_ORDER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"Stop order state load failed: {e}")
    return {}

def _save_stop_orders(state):
    """Save stop order state to disk."""
    try:
        with open(STOP_ORDER_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log(f"Stop order state save failed: {e}")


# ---------------------------------------------------------------------------
# Trade queue — Layer 2 writes intent, Layer 1 executes
# ---------------------------------------------------------------------------

def _check_session_and_alert(page):
    """Periodic session health check with Telegram alerting.

    Checks two things:
    1. Live 401 check — is the session actually dead?
    2. Storage state file age — is it approaching the ~24h expiry?

    Sends Telegram alert on failure (once per outage, not spam).
    Sends recovery alert when session comes back.
    """
    global session_healthy, session_alert_sent, session_expiry_warned

    # --- 1. Live health check ---
    alive = check_session_alive(page)

    if alive and not session_healthy:
        # Session recovered
        session_healthy = True
        session_alert_sent = False
        log("Avanza session recovered")
        send_telegram("*METALS SESSION* Avanza session recovered — API responding normally.")
    elif alive and session_healthy:
        # All good, reset alert flag if it was set
        if session_alert_sent:
            session_alert_sent = False
    elif not alive and session_healthy:
        # Session just died
        session_healthy = False
        log("WARNING: Avanza session is DEAD (401)")
        if not session_alert_sent:
            session_alert_sent = True
            send_telegram(
                "*AVANZA SESSION EXPIRED*\n"
                "API returning 401 — BankID session is dead.\n"
                "Price fetching and trade execution will FAIL.\n\n"
                "*Action needed:* Run `scripts/avanza_login.py` to re-authenticate via BankID, "
                "then restart the metals loop."
            )
    elif not alive and not session_healthy:
        # Still dead — don't spam, already alerted
        pass

    # --- 2. Proactive expiry warning (file age) ---
    try:
        if os.path.exists(SESSION_STORAGE_FILE):
            mtime = os.path.getmtime(SESSION_STORAGE_FILE)
            age_h = (time.time() - mtime) / 3600
            if age_h >= SESSION_EXPIRY_WARNING_H and not session_expiry_warned:
                session_expiry_warned = True
                log(f"WARNING: Avanza storage state is {age_h:.1f}h old (session expires ~24h)")
                send_telegram(
                    f"*AVANZA SESSION WARNING*\n"
                    f"Storage state is *{age_h:.1f}h* old (expires at ~24h).\n"
                    f"Session will die in ~{24 - age_h:.1f}h.\n\n"
                    f"*Renew soon:* Run `scripts/avanza_login.py` to refresh BankID session."
                )
            elif age_h < SESSION_EXPIRY_WARNING_H and session_expiry_warned:
                # File was renewed — reset warning
                session_expiry_warned = False
                log("Avanza storage state renewed")
    except Exception as e:
        log(f"Session age check error: {e}")

    return alive


def _fetch_warrant_catalog_prices(page):
    """Fetch live bid/ask for all warrants in WARRANT_CATALOG."""
    catalog_with_prices = {}
    for wkey, winfo in WARRANT_CATALOG.items():
        try:
            p = fetch_price(page, winfo["ob_id"], winfo["api_type"])
            entry = dict(winfo)  # copy static metadata
            if p:
                entry["bid"] = p.get("bid")
                entry["ask"] = p.get("ask")
                entry["last"] = p.get("last")
                entry["underlying_price"] = p.get("underlying")
                entry["current_leverage"] = p.get("leverage") or winfo.get("leverage")
                # Compute barrier distance
                und = p.get("underlying") or 0
                barrier = winfo.get("barrier") or 0
                if und > 0 and barrier > 0:
                    entry["barrier_distance_pct"] = round((und - barrier) / und * 100, 1)
                else:
                    entry["barrier_distance_pct"] = None
                # Spread %
                bid = p.get("bid") or 0
                ask = p.get("ask") or 0
                if bid > 0 and ask > 0:
                    entry["spread_pct"] = round((ask - bid) / bid * 100, 2)
                else:
                    entry["spread_pct"] = None
            catalog_with_prices[wkey] = entry
        except Exception as e:
            log(f"  Warrant catalog price error for {wkey}: {e}")
            catalog_with_prices[wkey] = dict(winfo)
    return catalog_with_prices


def _load_trade_queue():
    """Load trade queue from disk."""
    try:
        if os.path.exists(TRADE_QUEUE_FILE):
            with open(TRADE_QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log(f"Trade queue load error: {e}")
    return {"version": 1, "orders": []}


def _save_trade_queue(queue):
    """Save trade queue to disk."""
    try:
        with open(TRADE_QUEUE_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, indent=2, ensure_ascii=False)
    except IOError as e:
        log(f"Trade queue save error: {e}")


def process_trade_queue(page):
    """Process pending orders from the trade queue file.

    Called after Claude exits and on each loop cycle.
    For each pending order:
      1. Check session health
      2. Check order age (expire > 5 min)
      3. Re-fetch live price, reject if slippage > 2%
      4. Execute via Avanza API
      5. On BUY: add to POSITIONS, place hardware stop-loss, log trade
      6. On SELL: deactivate position, cancel stops, log trade
      7. Send Telegram confirmation
    """
    global POSITIONS

    if not TRADE_QUEUE_ENABLED:
        return

    queue = _load_trade_queue()
    orders = queue.get("orders", [])
    if not orders:
        return

    pending = [o for o in orders if o.get("status") == "pending"]
    if not pending:
        return

    log(f"Trade queue: {len(pending)} pending order(s)")

    # Session health check (once per batch)
    if not check_session_alive(page):
        log("Trade queue: Avanza session unhealthy (401), skipping execution")
        send_telegram("*TRADE QUEUE* Session expired — cannot execute orders. Re-login needed.")
        # Mark all pending as failed
        for order in pending:
            order["status"] = "failed"
            order["result"] = {"error": "session_unhealthy"}
            order["executed_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _save_trade_queue(queue)
        return

    now = datetime.datetime.now(datetime.timezone.utc)

    for order in pending:
        order_id_short = order.get("id", "?")[:8]
        action = order.get("action", "?")
        warrant_name = order.get("warrant_name", order.get("warrant_key", "?"))

        # --- Age check ---
        try:
            order_ts = datetime.datetime.fromisoformat(order["timestamp"])
            if order_ts.tzinfo is None:
                order_ts = order_ts.replace(tzinfo=datetime.timezone.utc)
            age_s = (now - order_ts).total_seconds()
        except (ValueError, KeyError):
            age_s = 9999

        if age_s > TRADE_QUEUE_MAX_AGE_S:
            log(f"  Order {order_id_short} expired ({age_s:.0f}s old)")
            order["status"] = "expired"
            order["result"] = {"error": f"expired ({age_s:.0f}s > {TRADE_QUEUE_MAX_AGE_S}s)"}
            order["executed_ts"] = now.isoformat()
            send_telegram(f"_Trade queue: {action} {warrant_name} expired ({age_s:.0f}s old)_")
            continue

        # --- Deduplicate: same ob_id + action within 5 min ---
        already_done = False
        for other in orders:
            if (other is not order and
                other.get("ob_id") == order.get("ob_id") and
                other.get("action") == action and
                other.get("status") in ("filled", "executed")):
                try:
                    other_ts = datetime.datetime.fromisoformat(other["executed_ts"])
                    if other_ts.tzinfo is None:
                        other_ts = other_ts.replace(tzinfo=datetime.timezone.utc)
                    if (now - other_ts).total_seconds() < 300:
                        already_done = True
                        break
                except (ValueError, KeyError):
                    pass
        if already_done:
            log(f"  Order {order_id_short} deduplicated (same {action} recently filled)")
            order["status"] = "deduplicated"
            order["executed_ts"] = now.isoformat()
            continue

        # --- Re-fetch live price ---
        live_price_data = fetch_price(page, order["ob_id"], order.get("api_type", "warrant"))
        if not live_price_data:
            log(f"  Order {order_id_short}: cannot fetch live price, skipping")
            order["status"] = "failed"
            order["result"] = {"error": "live_price_fetch_failed"}
            order["executed_ts"] = now.isoformat()
            continue

        if action == "BUY":
            live_price = live_price_data.get("ask") or live_price_data.get("last") or 0
        else:
            live_price = live_price_data.get("bid") or live_price_data.get("last") or 0

        queued_price = order.get("price", 0)
        if queued_price > 0 and live_price > 0:
            slippage = abs(live_price - queued_price) / queued_price * 100
            if slippage > TRADE_QUEUE_MAX_SLIPPAGE:
                log(f"  Order {order_id_short}: slippage {slippage:.1f}% > {TRADE_QUEUE_MAX_SLIPPAGE}% "
                    f"(queued={queued_price}, live={live_price})")
                order["status"] = "rejected_slippage"
                order["result"] = {"error": f"slippage {slippage:.1f}%", "queued": queued_price, "live": live_price}
                order["executed_ts"] = now.isoformat()
                send_telegram(f"_Trade queue: {action} {warrant_name} rejected — "
                              f"price moved {slippage:.1f}% (queued {queued_price}, now {live_price})_")
                continue
            # Use live price for execution (better fill)
            exec_price = live_price
        else:
            exec_price = queued_price

        # Update order price to live price for execution
        order["price"] = exec_price
        if order.get("volume", 0) > 0:
            order["total_sek"] = round(exec_price * order["volume"], 2)

        # --- Execute ---
        log(f"  Executing {action} {warrant_name}: {order['volume']}u @ {exec_price}")
        success, result = place_order(
            page, ACCOUNT_ID, order["ob_id"], order["action"],
            order["price"], order["volume"],
        )
        order["result"] = result
        order["executed_ts"] = now.isoformat()

        if success:
            order["status"] = "filled"
            log(f"  Order {order_id_short} FILLED: {action} {order['volume']}u @ {exec_price}")

            # Log trade
            trade_entry = {
                "ts": now.isoformat(),
                "action": action,
                "queue_id": order.get("id"),
                "warrant_key": order.get("warrant_key"),
                "name": warrant_name,
                "ob_id": order.get("ob_id"),
                "units": order["volume"],
                "price": exec_price,
                "total_sek": order.get("total_sek", 0),
                "reasoning": order.get("reasoning", ""),
                "result": result,
            }
            try:
                with open("data/metals_trades.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(trade_entry, ensure_ascii=False) + "\n")
            except IOError as e:
                log(f"  Trade log write error: {e}")

            if action == "BUY":
                _handle_buy_fill(page, order, exec_price, live_price_data)
            elif action == "SELL":
                _handle_sell_fill(page, order, exec_price)

            send_telegram(
                f"*TRADE FILLED* {action} {warrant_name}\n"
                f"Volume: {order['volume']}u @ {exec_price} SEK\n"
                f"Total: {order.get('total_sek', 0):.0f} SEK\n"
                f"_{order.get('reasoning', '')}_"
            )
        else:
            order["status"] = "failed"
            error_msg = result.get("error", result.get("parsed", {}).get("message", "unknown"))
            log(f"  Order {order_id_short} FAILED: {error_msg}")
            send_telegram(
                f"*TRADE FAILED* {action} {warrant_name}\n"
                f"Error: {error_msg}\n"
                f"_{order.get('reasoning', '')}_"
            )

    _save_trade_queue(queue)


def _handle_buy_fill(page, order, exec_price, price_data):
    """After a BUY fill: add position to POSITIONS, place hardware stop-loss."""
    global POSITIONS

    wkey = order.get("warrant_key", "")
    pos_key = wkey.lower().replace("_", "")  # e.g. "minilsilverava301"
    # Use a more readable key
    if "silver" in wkey.lower():
        # Find a unique silver key
        idx = sum(1 for k in POSITIONS if "silver" in k.lower() and POSITIONS[k].get("active"))
        pos_key = f"silver_q{idx}" if idx > 0 else "silver_queue"
        # If the ob_id already matches an existing position, use that key
        for k, p in POSITIONS.items():
            if p.get("ob_id") == order.get("ob_id"):
                pos_key = k
                break
    elif "gold" in wkey.lower():
        pos_key = "gold_queue"
        for k, p in POSITIONS.items():
            if p.get("ob_id") == order.get("ob_id"):
                pos_key = k
                break

    # Check if position already exists (add to existing)
    if pos_key in POSITIONS and POSITIONS[pos_key].get("active"):
        existing = POSITIONS[pos_key]
        old_units = existing["units"]
        old_entry = existing["entry"]
        new_units = order["volume"]
        # Weighted average entry price
        total_units = old_units + new_units
        avg_entry = (old_units * old_entry + new_units * exec_price) / total_units
        existing["units"] = total_units
        existing["entry"] = round(avg_entry, 4)
        log(f"  Added to existing position {pos_key}: {old_units}+{new_units}={total_units}u, "
            f"avg entry {old_entry}->{avg_entry:.4f}")
    else:
        # New position
        catalog_info = WARRANT_CATALOG.get(wkey, {})
        POSITIONS[pos_key] = {
            "name": order.get("warrant_name", wkey),
            "ob_id": order.get("ob_id"),
            "api_type": order.get("api_type", "warrant"),
            "units": order["volume"],
            "entry": exec_price,
            "stop": order.get("stop_trigger", exec_price * 0.85),  # fallback: 15% below
            "active": True,
            "swing": True,  # mark as swing trade from queue
            "bought_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        peak_bids[pos_key] = exec_price
        last_invoke_prices[pos_key] = exec_price
        log(f"  New position added: {pos_key} = {order['volume']}u @ {exec_price}")

    _save_positions(POSITIONS)

    # Place hardware stop-loss
    stop_trigger = order.get("stop_trigger")
    stop_sell = order.get("stop_sell")
    if stop_trigger and stop_sell and order["volume"] > 0:
        vol = POSITIONS[pos_key]["units"]  # use total units (may have added to existing)
        ok, stop_id = place_stop_loss(page, ACCOUNT_ID, order["ob_id"], stop_trigger, stop_sell, vol)
        if ok:
            log(f"  Stop-loss placed for {pos_key}: trigger={stop_trigger}, sell={stop_sell}")
        else:
            log(f"  Stop-loss FAILED for {pos_key} — manual intervention needed")
            send_telegram(f"*WARNING* Stop-loss failed for {POSITIONS[pos_key]['name']} — set manually!")


def _handle_sell_fill(page, order, exec_price):
    """After a SELL fill: deactivate position, cancel stop-losses."""
    global POSITIONS

    ob_id = order.get("ob_id")
    sold_key = None
    for k, p in POSITIONS.items():
        if p.get("ob_id") == ob_id and p.get("active"):
            sold_key = k
            break

    if sold_key:
        pos = POSITIONS[sold_key]
        entry = pos.get("entry", 0)
        pnl = pnl_pct(exec_price, entry) if entry > 0 else 0
        pos["active"] = False
        pos["sold_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        pos["sold_price"] = exec_price
        pos["sold_reason"] = "trade_queue_sell"
        _save_positions(POSITIONS)
        _cleanup_stop_orders_for(page, sold_key)

        if RISK_AVAILABLE:
            record_metals_trade(sold_key, "SELL", pnl_pct_value=pnl)

        log(f"  Position {sold_key} sold: {exec_price} (entry={entry}, PnL={pnl:+.1f}%)")
    else:
        log(f"  SELL fill but no matching active position for ob_id={ob_id}")


def place_stop_loss_orders(page, positions):
    """Place cascading stop-loss orders for all active positions.

    Places STOP_ORDER_LEVELS orders per position, spread across levels:
    - S1 (1/3 units): at stop price
    - S2 (1/3 units): at stop - STOP_ORDER_SPREAD_PCT%
    - S3 (remaining): at stop - 2*STOP_ORDER_SPREAD_PCT%

    Returns stop order state dict.
    """
    csrf = get_csrf(page)
    if not csrf:
        log("Stop orders: no CSRF token")
        return {}

    state = _load_stop_orders()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    for key, pos in positions.items():
        if not pos.get("active"):
            continue

        # Skip if orders already placed today for this position at current stop level
        existing = state.get(key, {})
        if (existing.get("date") == today_str and
            existing.get("stop_base") == pos["stop"] and
            existing.get("orders")):
            log(f"  Stop orders already placed for {key} today")
            continue

        # Cancel any existing orders first
        if existing.get("orders"):
            _cancel_stop_orders(page, key, existing, csrf)

        units = pos["units"]
        stop_base = pos["stop"]
        orders = []

        for level in range(STOP_ORDER_LEVELS):
            # Calculate price for this level
            spread = level * STOP_ORDER_SPREAD_PCT / 100.0
            price = round(stop_base * (1 - spread), 2)

            # Calculate units for this level (split evenly, last gets remainder)
            if level < STOP_ORDER_LEVELS - 1:
                level_units = units // STOP_ORDER_LEVELS
            else:
                level_units = units - (units // STOP_ORDER_LEVELS) * (STOP_ORDER_LEVELS - 1)

            if level_units <= 0:
                continue

            try:
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
                    "price": price,
                    "validUntil": today_str,
                    "volume": level_units,
                }, csrf])

                body_str = result.get("body", "")
                try:
                    body = json.loads(body_str)
                except (json.JSONDecodeError, TypeError):
                    body = {}

                order_status = body.get("orderRequestStatus", "")
                order_id = body.get("orderId", "")

                if order_status == "SUCCESS":
                    orders.append({
                        "level": level + 1,
                        "order_id": order_id,
                        "price": price,
                        "units": level_units,
                        "status": "placed",
                    })
                    log(f"  Stop S{level+1} placed: {key} {level_units}u @ {price} [order {order_id}]")
                else:
                    error_msg = body.get("message", body_str[:100])
                    log(f"  Stop S{level+1} FAILED: {key} — {error_msg}")
                    orders.append({
                        "level": level + 1,
                        "price": price,
                        "units": level_units,
                        "status": "failed",
                        "error": error_msg,
                    })
            except Exception as e:
                log(f"  Stop S{level+1} error: {key} — {e}")

        state[key] = {
            "date": today_str,
            "stop_base": stop_base,
            "orders": orders,
            "placed_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    _save_stop_orders(state)
    return state


def _cancel_stop_orders(page, key, order_state, csrf=None):
    """Cancel existing stop orders for a position."""
    if not csrf:
        csrf = get_csrf(page)
    if not csrf:
        return

    for order in order_state.get("orders", []):
        order_id = order.get("order_id")
        if not order_id or order.get("status") != "placed":
            continue
        try:
            result = page.evaluate("""async (args) => {
                const [accountId, orderId, token] = args;
                const resp = await fetch(
                    'https://www.avanza.se/_api/trading-critical/rest/order/' + accountId + '/' + orderId,
                    {method: 'DELETE', headers: {'Content-Type': 'application/json', 'X-SecurityToken': token}, credentials: 'include'}
                );
                return {status: resp.status};
            }""", [ACCOUNT_ID, order_id, csrf])
            log(f"  Cancel stop S{order['level']} {key}: status={result.get('status')}")
            order["status"] = "cancelled"
        except Exception as e:
            log(f"  Cancel stop error {key} S{order['level']}: {e}")


def check_stop_order_fills(page, stop_state, positions):
    """Check if any stop-loss orders were filled. Returns list of filled keys."""
    csrf = get_csrf(page)
    if not csrf:
        return []

    filled_keys = []

    for key, state in stop_state.items():
        if key not in positions or not positions[key].get("active"):
            continue

        any_filled = False
        total_filled_units = 0

        for order in state.get("orders", []):
            order_id = order.get("order_id")
            if not order_id or order.get("status") != "placed":
                continue

            try:
                result = page.evaluate("""async (args) => {
                    const [accountId, orderId, token] = args;
                    const resp = await fetch(
                        'https://www.avanza.se/_api/trading-critical/rest/order/' + accountId + '/' + orderId,
                        {method: 'GET', headers: {'Content-Type': 'application/json', 'X-SecurityToken': token}, credentials: 'include'}
                    );
                    if (resp.status !== 200) return {status: resp.status};
                    return {status: 200, body: await resp.json()};
                }""", [ACCOUNT_ID, order_id, csrf])

                if result.get("status") == 200:
                    body = result.get("body", {})
                    order_state_str = (body.get("state") or "").upper()
                    if order_state_str in ("FILLED", "EXECUTED", "DONE"):
                        order["status"] = "filled"
                        total_filled_units += order["units"]
                        any_filled = True
                        log(f"STOP FILLED: {key} S{order['level']} {order['units']}u @ {order['price']}")
            except Exception as e:
                log(f"Stop check error {key} S{order['level']}: {e}")

        if any_filled:
            filled_keys.append(key)
            pos = positions[key]
            remaining = pos["units"] - total_filled_units

            send_telegram(
                f"*STOP FILLED* {pos['name']}\n"
                f"Sold {total_filled_units}u (stop orders)\n"
                f"Remaining: {remaining}u"
            )

            # Log the trade
            trade = {
                "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "action": "STOP_ORDER_SELL",
                "position": key,
                "name": pos["name"],
                "units": total_filled_units,
                "price": state["stop_base"],
                "entry": pos["entry"],
                "pnl_pct": round(pnl_pct(state["stop_base"], pos["entry"]), 2),
            }
            with open("data/metals_trades.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(trade, ensure_ascii=False) + "\n")

            if remaining <= 0:
                pos["active"] = False
                pos["sold_ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                pos["sold_price"] = state["stop_base"]
                pos["sold_reason"] = "stop_order_filled"
            else:
                pos["units"] = remaining

            _save_positions(positions)

    if filled_keys:
        _save_stop_orders(stop_state)

    return filled_keys


def update_trailing_stops(page, positions, stop_state, prices):
    """Update stop-loss orders when positions gain — ratchet stops higher.

    Only updates when:
    1. Position has gained >= TRAIL_START_PCT from entry
    2. New stop would be >= TRAIL_MIN_MOVE_PCT higher than current stop
    """
    updated = False

    for key, pos in positions.items():
        if not pos.get("active") or key not in prices:
            continue

        bid = prices[key].get("bid") or 0
        if bid <= 0:
            continue

        gain_pct = pnl_pct(bid, pos["entry"])
        if gain_pct < TRAIL_START_PCT:
            continue

        # Calculate new trailing stop
        new_stop = round(bid * (1 - TRAIL_DISTANCE_PCT / 100.0), 2)
        current_stop = pos["stop"]

        # Only move stop UP (never down)
        if new_stop <= current_stop:
            continue

        # Check minimum move threshold
        move_pct = ((new_stop - current_stop) / current_stop) * 100
        if move_pct < TRAIL_MIN_MOVE_PCT:
            continue

        log(f"TRAILING STOP: {key} raising stop {current_stop} -> {new_stop} "
            f"(bid={bid}, gain={gain_pct:+.1f}%, move={move_pct:.1f}%)")

        # Cancel old orders and place new ones at higher level
        if key in stop_state and stop_state[key].get("orders"):
            csrf = get_csrf(page)
            if csrf:
                _cancel_stop_orders(page, key, stop_state[key], csrf)

        # Update position stop level
        pos["stop"] = new_stop
        _save_positions(positions)

        # Place new stop orders at higher level
        updated = True
        send_telegram(
            f"*TRAILING STOP* {pos['name']}\n"
            f"Stop raised: {current_stop} -> {new_stop}\n"
            f"Bid: {bid} | Gain: {gain_pct:+.1f}%"
        )

    if updated:
        # Re-place all stop orders with new levels
        place_stop_loss_orders(page, positions)

    return updated


def check_momentum_exit(positions, prices, price_history_buf):
    """Check for accelerating price decline (derivative-based exit).

    Returns list of (key, reason) tuples for positions that should be sold immediately.
    """
    if not MOMENTUM_ENABLED or len(price_history_buf) < MOMENTUM_LOOKBACK + 2:
        return []

    exits = []

    for key, pos in positions.items():
        if not pos.get("active"):
            continue

        bid = prices.get(key, {}).get("bid") or 0
        if bid <= 0:
            continue

        # Check if in L1+ danger zone (if required)
        if MOMENTUM_REQUIRE_L1:
            dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
            if dist_stop >= STOP_L1_PCT:
                continue

        # Get price history for this position
        recent = []
        for snap in price_history_buf[-(MOMENTUM_LOOKBACK + 2):]:
            p = snap.get(key, 0)
            if p > 0:
                recent.append(p)

        if len(recent) < MOMENTUM_LOOKBACK + 1:
            continue

        # Calculate velocity (1st derivative): % change per check
        velocities = []
        for i in range(1, len(recent)):
            v = ((recent[i] - recent[i-1]) / recent[i-1]) * 100
            velocities.append(v)

        if len(velocities) < 2:
            continue

        current_velocity = velocities[-1]
        prev_velocity = velocities[-2]

        # Calculate acceleration (2nd derivative)
        acceleration = current_velocity - prev_velocity

        # Check momentum exit conditions
        if (current_velocity < MOMENTUM_MIN_VELOCITY and
            acceleration < MOMENTUM_ACCEL_THRESHOLD):
            reason = (f"MOMENTUM EXIT: {key} velocity={current_velocity:.2f}%/check, "
                      f"accel={acceleration:.2f}, declining and accelerating")
            exits.append((key, reason))
            log(f"!!! {reason}")

    return exits


def place_spike_orders(page, positions, prices, targets):
    """Place limit sell orders for US open spike capture.

    Returns dict of {position_key: order_id} for placed orders.
    """
    csrf = get_csrf(page)
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
    csrf = get_csrf(page)
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
    csrf = get_csrf(page)
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
        "hours_remaining": round(max(0, EOD_HOUR_CET + 25/60 - cet_hour()), 1),
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
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        log(f"  Historical stats load failed: {e}")

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

    # Signal accuracy tracking
    if TRACKER_AVAILABLE:
        try:
            ctx["signal_accuracy"] = get_accuracy_for_context()
        except Exception as e:
            ctx["signal_accuracy"] = {"error": str(e)}

    # Account data (buying power) — for trade sizing
    if cached_account_data:
        ctx["account"] = cached_account_data

    # Warrant catalog with live prices — for BUY warrant selection
    if TRADE_QUEUE_ENABLED and cached_warrant_catalog:
        ctx["warrant_catalog"] = cached_warrant_catalog

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
            # Track L2+ zone checks for auto-exit override
            l2_zone_checks[key] = l2_zone_checks.get(key, 0) + 1
            if l2_zone_checks[key] >= AUTO_EXIT_L2_CHECKS:
                # Check if price trend is downward (last 3 bids declining)
                recent_bids = [s.get(key, 0) for s in price_history[-3:] if s.get(key, 0) > 0]
                if len(recent_bids) >= 3 and recent_bids[-1] < recent_bids[-2] < recent_bids[-3]:
                    reasons.append(f"{key} AUTO-EXIT: L2+ for {l2_zone_checks[key]} checks, declining trend")
        elif 0 < dist_stop < STOP_L1_PCT:
            reasons.append(f"{key} L1 WARNING: {dist_stop:.1f}% from stop-loss")

        # Reset L2 counter when not in L2 danger zone
        if dist_stop >= STOP_L2_PCT:
            l2_zone_checks.pop(key, None)

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

    # End of day (CET-based)
    h_cet = cet_hour()
    if int(h_cet) == int(EOD_HOUR_CET) and (h_cet % 1) < 0.05 and check_count > 10:
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
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[WARN] claude_proc.wait failed: {e}", flush=True)
    claude_proc = None
    if claude_log_fh:
        try:
            claude_log_fh.close()
        except OSError as e:
            print(f"[WARN] claude_log_fh close failed: {e}", flush=True)
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
    except (FileNotFoundError, IOError) as e:
        log(f"Cannot read {prompt_file}: {e}")
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
            except OSError:
                pass
        return False

def main():
    global check_count, last_signal_data, last_invoke_prices, startup_grace
    global claude_proc, claude_log_fh, claude_start, claude_timeout
    global short_prices, daily_range_stats

    # Probe time server on startup
    h, ts, src = get_cet_time()
    log(f"Starting metals trading loop (v7 — LLM + MC + Ranges + Spikes + Cascading Stops)...")
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

        # Check session health at startup
        if not _check_session_and_alert(page):
            log("WARNING: Avanza session is DEAD at startup! Continuing but trades will fail.")

        # Verify actual holdings at startup (detect already-sold positions)
        log("Verifying position holdings...")
        _verify_position_holdings(page, POSITIONS)
        # Persist any corrections from verification
        _save_positions(POSITIONS)

        # Log position summary
        active_count = sum(1 for p in POSITIONS.values() if p["active"])
        sold_count = sum(1 for p in POSITIONS.values() if not p["active"])
        log(f"  Positions: {active_count} active, {sold_count} sold/inactive")
        if active_count == 0 and not SWING_TRADER_AVAILABLE and not TRADE_QUEUE_ENABLED:
            log("All positions already sold and no swing trader / trade queue — nothing to monitor. Exiting.")
            send_telegram("*METALS LOOP* All positions already sold. Not starting.")
            return

        # Place cascading stop-loss orders
        stop_order_state = {}
        if STOP_ORDER_ENABLED:
            log("Placing cascading stop-loss orders...")
            stop_order_state = place_stop_loss_orders(page, POSITIONS)
            placed_count = sum(
                len([o for o in s.get("orders", []) if o.get("status") == "placed"])
                for s in stop_order_state.values()
            )
            log(f"  {placed_count} stop orders placed across {len(stop_order_state)} positions")

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
            log("LLM thread: Ministral every 5min, Chronos every 60s")
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

        if TRACKER_AVAILABLE:
            snap_count = get_snapshot_count()
            log(f"Signal tracker: active ({snap_count} existing snapshots)")
            if snap_count > 0:
                acc_summary = get_accuracy_summary()
                log(f"  Accuracy: {acc_summary}")
        else:
            log("Signal tracker: NOT available (import failed)")

        # Initialize swing trader
        swing_trader = None
        if SWING_TRADER_AVAILABLE:
            try:
                swing_trader = SwingTrader(page)
                log(f"Swing trader: ACTIVE (cash={swing_trader.state['cash_sek']:.0f} SEK, "
                    f"DRY_RUN={swing_trader.state.get('_dry', 'see config')})")
            except Exception as e:
                log(f"Swing trader init failed: {e}")
                swing_trader = None
        else:
            log("Swing trader: NOT available (import failed)")

        # Initialize trade queue: fetch account data + warrant catalog
        if TRADE_QUEUE_ENABLED:
            log("Trade queue: ENABLED")
            try:
                acct = fetch_account_cash(page, ACCOUNT_ID)
                if acct:
                    cached_account_data.update(acct)
                    log(f"  Account: buying_power={acct.get('buying_power')} SEK")
                else:
                    log("  Account data: fetch returned None")
            except Exception as e:
                log(f"  Account data fetch error: {e}")
            if CATALOG_AVAILABLE:
                try:
                    cat = _fetch_warrant_catalog_prices(page)
                    if cat:
                        cached_warrant_catalog.update(cat)
                        log(f"  Warrant catalog: {len(cat)} instruments loaded")
                        for wk, wi in cat.items():
                            log(f"    {wk}: bid={wi.get('bid')}, ask={wi.get('ask')}, "
                                f"lev={wi.get('current_leverage')}, barrier_dist={wi.get('barrier_distance_pct')}%")
                    else:
                        log("  Warrant catalog: empty")
                except Exception as e:
                    log(f"  Warrant catalog fetch error: {e}")
            else:
                log("  Warrant catalog: NOT available (import failed)")
        else:
            log("Trade queue: DISABLED")

        # Build dynamic positions summary
        pos_parts = []
        for key, pos in POSITIONS.items():
            status = "ACTIVE" if pos["active"] else "SOLD"
            pos_parts.append(f"{key}({status})")
        pos_summary = ", ".join(pos_parts)

        send_telegram(f"""*METALS LOOP v7 STARTED*
Tiered: T1=haiku T2=sonnet T3=opus
LLM: {"Ministral+Chronos (60s)" if LLM_AVAILABLE else "DISABLED"}
Risk: {"MC+Guards+Drawdown+DailyRanges" if RISK_AVAILABLE else "DISABLED"}
Tracker: {"ON (" + str(get_snapshot_count()) + " snaps)" if TRACKER_AVAILABLE else "OFF"}
Spike: {"P" + str(SPIKE_PERCENTILE) + " " + str(SPIKE_PARTIAL_PCT) + "% @15:15-16:30" if SPIKE_ENABLED else "OFF"}
Stops: {"3x cascaded + trailing + momentum" if STOP_ORDER_ENABLED else "L3 only"}
Swing: {"ACTIVE (DRY_RUN)" if swing_trader else "OFF"}
TradeQ: {"ENABLED" if TRADE_QUEUE_ENABLED else "OFF"}
Session: {"ALIVE" if session_healthy else "DEAD — re-login needed!"}
Positions: {pos_summary}""")

        try:
            while True:
                check_count += 1

                if not is_market_hours():
                    if check_count % 40 == 0:
                        log("Outside market hours")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Check if all positions closed (keep running if swing trader is active)
                active = sum(1 for p in POSITIONS.values() if p["active"])
                if active == 0 and not swing_trader and not TRADE_QUEUE_ENABLED:
                    log("All positions closed and no swing trader / trade queue.")
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

                # Refresh account data + warrant catalog (every 10th check ~15 min)
                if TRADE_QUEUE_ENABLED and check_count % 10 == 0:
                    try:
                        acct = fetch_account_cash(page, ACCOUNT_ID)
                        if acct:
                            cached_account_data.clear()
                            cached_account_data.update(acct)
                    except Exception as e:
                        log(f"Account data fetch error: {e}")
                    if CATALOG_AVAILABLE:
                        try:
                            cat = _fetch_warrant_catalog_prices(page)
                            if cat:
                                cached_warrant_catalog.clear()
                                cached_warrant_catalog.update(cat)
                        except Exception as e:
                            log(f"Warrant catalog fetch error: {e}")

                # Session health check (~every 30 min)
                if check_count % SESSION_HEALTH_CHECK_INTERVAL == 0:
                    _check_session_and_alert(page)

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
                                _save_positions(POSITIONS)  # persist after spike fills

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

                # Check stop order fills (every 2nd check)
                if STOP_ORDER_ENABLED and stop_order_state and check_count % 2 == 0:
                    filled = check_stop_order_fills(page, stop_order_state, POSITIONS)
                    if filled:
                        log(f"Stop orders filled for: {', '.join(filled)}")

                # Periodic holdings verification (detect broker-triggered sells)
                if check_count % HOLDINGS_CHECK_INTERVAL == 0:
                    active_before = sum(1 for p in POSITIONS.values() if p["active"])
                    if active_before > 0:
                        _verify_position_holdings(page, POSITIONS)
                        active_after = sum(1 for p in POSITIONS.values() if p["active"])
                        if active_after < active_before:
                            lost = active_before - active_after
                            log(f"Holdings check: {lost} position(s) no longer held on Avanza")
                            _save_positions(POSITIONS)
                            # Clean up stop orders for deactivated positions
                            for key, pos in POSITIONS.items():
                                if not pos["active"] and pos.get("sold_reason", "").startswith("startup_verify"):
                                    _cleanup_stop_orders_for(page, key)
                            send_telegram(f"_Holdings check: {lost} position(s) sold by broker_")
                            if active_after == 0 and not swing_trader and not TRADE_QUEUE_ENABLED:
                                log("All positions sold — exiting loop")
                                send_telegram("*METALS LOOP* All positions sold by broker. Stopping.")
                                return

                # Check momentum exit
                momentum_exits = check_momentum_exit(POSITIONS, prices, price_history)
                for mkey, mreason in momentum_exits:
                    if POSITIONS[mkey].get("active"):
                        mbid = prices.get(mkey, {}).get("bid") or 0
                        if mbid > 0:
                            log(f"!!! MOMENTUM SELL: {mkey} at {mbid}")
                            send_telegram(f"*MOMENTUM EXIT* {POSITIONS[mkey]['name']}\nBid: {mbid} | Accelerating decline detected")
                            emergency_sell(page, mkey, POSITIONS[mkey], mbid)

                # Trailing stop updates (every 5th check)
                if STOP_ORDER_ENABLED and check_count % 5 == 0:
                    update_trailing_stops(page, POSITIONS, stop_order_state, prices)

                # Swing trader: autonomous BUY/SELL evaluation
                if swing_trader:
                    try:
                        swing_trader.evaluate_and_execute(prices, last_signal_data)
                    except Exception as e:
                        log(f"Swing trader error: {e}")

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
                        # State already persisted inside emergency_sell()
                        break

                # AUTO-EXIT: sell positions stuck in L2 zone with declining trend
                for r in reasons[:]:
                    if "AUTO-EXIT" in r:
                        for key, pos in POSITIONS.items():
                            if not pos["active"] or key not in prices:
                                continue
                            bid = prices[key].get('bid') or 0
                            if bid <= 0:
                                continue
                            dist = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999
                            if dist < STOP_L2_PCT:
                                log(f"!!! AUTO-EXIT SELL: {key} at {bid} (L2+ for {l2_zone_checks.get(key, 0)} checks)")
                                send_telegram(f"*AUTO-EXIT* {pos['name']}\nBid: {bid} | L2 zone {l2_zone_checks.get(key, 0)} checks, declining")
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
                    # Add signal accuracy tag (every 12th check to avoid spam)
                    acc_tag = ""
                    if TRACKER_AVAILABLE and check_count % 12 == 0:
                        try:
                            acc_tag = f" ACC:[{get_accuracy_summary()}]"
                        except Exception:
                            pass
                    log(f"#{check_count} [{cet}] {' | '.join(parts)}{llm_tag}{risk_tag}{acc_tag}" +
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
                        except OSError:
                            pass
                        claude_log_fh = None

                    # Process trade queue (Layer 2 may have written orders)
                    if TRADE_QUEUE_ENABLED:
                        try:
                            process_trade_queue(page)
                        except Exception as e:
                            log(f"Trade queue processing error: {e}")
                            traceback.print_exc()

                    # Re-read trade log in case Claude executed a trade
                    if os.path.exists("data/metals_trades.jsonl"):
                        try:
                            with open("data/metals_trades.jsonl", "r") as f:
                                lines = f.readlines()
                            if lines:
                                last_trade = json.loads(lines[-1])
                                log(f"Last trade: {last_trade.get('action','')} {last_trade.get('name','')}")
                        except (json.JSONDecodeError, IOError) as e:
                            log(f"Trade log read error: {e}")

                # --- SIGNAL TRACKER: log snapshot + backfill accuracy ---
                if TRACKER_AVAILABLE:
                    try:
                        llm_sigs = get_llm_signals() if LLM_AVAILABLE else {}
                        log_snapshot(
                            check_count, prices, POSITIONS,
                            last_signal_data, llm_sigs,
                            triggered, reasons if triggered else [],
                        )
                    except Exception as e:
                        log(f"Signal tracker log error: {e}")

                    # Backfill outcomes every 10th check
                    if check_count % 10 == 0:
                        try:
                            und_prices = {}
                            for key, p in prices.items():
                                if isinstance(p, dict) and p.get("underlying"):
                                    if "silver" in key.lower():
                                        und_prices["XAG-USD"] = p["underlying"]
                                    elif "gold" in key.lower():
                                        und_prices["XAU-USD"] = p["underlying"]
                            if und_prices:
                                backfill_outcomes(und_prices)
                        except Exception as e:
                            log(f"Signal tracker backfill error: {e}")

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
                except Exception as e:
                    print(f"[WARN] LLM thread stop failed: {e}", flush=True)
            browser.close()
            log(f"Loop stopped: {check_count} checks, {invoke_count} invocations")

if __name__ == "__main__":
    main()
