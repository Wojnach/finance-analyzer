"""
Unified Market Monitoring Loop v10 (Layer 1 — Autonomous).
Runs every 60s, fully autonomous without Claude Code dependency.
Tracks: XAG/XAU (Binance FAPI), BTC/ETH (Binance SPOT), MSTR (Yahoo).
Core features: probability-focused Telegram, momentum-aware trailing stops,
auto-detect holdings, per-signal accuracy, crypto Fear & Greed, on-chain metrics.

v10: Silver fast-tick monitor merged from silver_monitor.py — 10-second price
checks with instant threshold alerts (-3% to -12.5%) and 3-minute velocity
flush detection.  Replaces the standalone silver_monitor.py process.

Features:
- Silver fast-tick: 10s price checks during 60s cycle sleep (threshold + velocity alerts)
- Tiered Claude invocation (Haiku/Sonnet, no Opus)
- Local LLM inference (Ministral-8B + Chronos for all tracked symbols)
- Monte Carlo VaR for leveraged warrants
- Trade guards (cooldowns, session limits, loss escalation)
- Drawdown circuit breaker (-15% emergency liquidation)
- Multi-level stop-loss (L1 warn / L2 alert / L3 emergency auto-sell)
- Short instrument tracking (BEAR SILVER X5)
- Time server (timeapi.io) for accurate CET
- Daily range analysis (historical percentiles + intraday assessment)
- Spike catcher (limit sell orders before US open)
- Invocation logging (tier/model/trigger tracking)
- Crypto data: Fear & Greed, CryptoCompare news, on-chain (MVRV/SOPR)
- MSTR-BTC NAV premium tracking

Run: .venv/Scripts/python.exe data/metals_loop.py
"""
import atexit
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import requests
from playwright.sync_api import sync_playwright

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json
from portfolio.loop_contract import MetalsCycleReport, ViolationTracker, verify_and_act, verify_metals_contract

try:
    from portfolio.notification_text import (
        format_tier_footer,
        format_vote_summary,
        humanize_thesis_status,
        humanize_ticker,
    )
except ImportError:
    def format_tier_footer(source, tier, check_number, cet_str):
        return f"_{source} T{tier} · #{check_number} · {cet_str}_"

    def format_vote_summary(buy_count, sell_count):
        return f"{int(buy_count)}B/{int(sell_count)}S"

    def humanize_thesis_status(status):
        return str(status or "neutral").replace("_", " ").title()

    def humanize_ticker(ticker):
        return str(ticker or "").replace("-USD", "")

try:
    import msvcrt  # Windows file locking for single-instance guard
except ImportError:
    msvcrt = None

# --- Optional modules (graceful fallback) ---
try:
    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))
    from metals_llm import (
        get_llm_accuracy,
        get_llm_age,
        get_llm_signals,
        get_llm_summary,
        start_llm_thread,
        stop_llm_thread,
    )
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_llm import failed: {e}", flush=True)
    LLM_AVAILABLE = False

try:
    from metals_risk import (
        check_portfolio_drawdown,
        check_trade_guard,
        compute_daily_range_stats,
        compute_intraday_assessment,
        compute_spike_targets,
        get_risk_summary,
        load_spike_state,
        log_portfolio_value,
        record_metals_trade,
        save_spike_state,
        simulate_all_positions,
    )
    RISK_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_risk import failed: {e}", flush=True)
    RISK_AVAILABLE = False

try:
    from metals_signal_tracker import (
        backfill_outcomes,
        get_accuracy_for_context,
        get_accuracy_report,
        get_accuracy_summary,
        get_snapshot_count,
        log_snapshot,
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
    from metals_execution_engine import build_execution_recommendations, hours_to_metals_close
    EXECUTION_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_execution_engine import failed: {e}", flush=True)
    EXECUTION_ENGINE_AVAILABLE = False

try:
    from metals_swing_config import WARRANT_CATALOG
    CATALOG_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] metals_swing_config import failed: {e}", flush=True)
    WARRANT_CATALOG = {}
    CATALOG_AVAILABLE = False

try:
    from portfolio.avanza_control import (
        check_session_alive,
        fetch_account_cash,
        fetch_price,
        get_csrf,
        place_order,
        place_stop_loss,
    )
    AVANZA_CONTROL_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] portfolio.avanza_control import failed: {e}", flush=True)
    AVANZA_CONTROL_AVAILABLE = False

    def _missing_avanza_control(*_args, **_kwargs):
        raise RuntimeError("portfolio.avanza_control unavailable in this environment")

    get_csrf = _missing_avanza_control
    fetch_price = _missing_avanza_control
    fetch_account_cash = _missing_avanza_control
    place_order = _missing_avanza_control
    place_stop_loss = _missing_avanza_control
    check_session_alive = _missing_avanza_control

try:
    from crypto_data import (
        compute_mstr_btc_nav,
        fetch_mstr_price,
        get_crypto_news,
        get_fear_greed,
        get_onchain_summary,
        is_us_market_hours,
    )
    CRYPTO_DATA_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] crypto_data import failed: {e}", flush=True)
    CRYPTO_DATA_AVAILABLE = False

try:
    from portfolio.news_keywords import score_headline
    NEWS_KEYWORDS_AVAILABLE = True
except ImportError:
    NEWS_KEYWORDS_AVAILABLE = False

    def score_headline(title):
        return 1.0, []

# --- CONFIG ---
CLAUDE_ENABLED = False        # Master switch: set True to re-enable Claude invocations
CHECK_INTERVAL = 60           # target seconds between checks
TRIGGER_PRICE_MOVE = 5.0      # % move from last invocation to trigger (was 2.0)
TRIGGER_TRAILING = 8.0        # % drop from peak to trigger (was 3.0)
TRIGGER_PROFIT = 4.0          # % profit from entry to trigger
TRIGGER_STOP_NEAR = 5.0       # % from stop-loss to trigger
HEARTBEAT_CHECKS = 60         # invoke every N checks (~60 min at 60s)
# Per-tier cooldowns (seconds) — replaces flat MIN_INVOKE_INTERVAL
TIER_COOLDOWNS = {
    1: 120,    # Haiku: 2 min cooldown
    2: 600,    # Sonnet: 10 min cooldown
    3: 0,      # Critical: no cooldown (immediate)
}
EOD_HOUR_CET = 17.0           # 17:00 Stockholm time (legacy summary trigger)

# Spike catcher config (US open limit sell orders)
SPIKE_ENABLED = True
SPIKE_PLACE_ET = (9, 15)      # place 15 min before NYSE open
SPIKE_OPEN_ET = (9, 30)       # NYSE regular session open
SPIKE_CANCEL_ET = (10, 30)    # cancel 1h after open if unfilled
SPIKE_PERCENTILE = 75          # P75 of daily open_to_high as target
SPIKE_PARTIAL_PCT = 50         # sell 50% of position to capture spike profit

# Invocation log
INVOCATION_LOG = "data/metals_invocations.jsonl"

# Stop levels (distance from barrier as % of bid)
STOP_L1_PCT = 8.0   # L1: warning — log + flag in context
STOP_L2_PCT = 5.0   # L2: alert — Telegram + force Claude invocation
STOP_L3_PCT = 2.0   # L3: emergency — auto-sell immediately

# Cascading stop-loss orders (hardware protection via Avanza limit orders)
STOP_ORDER_ENABLED = False      # default OFF: only place stop orders on explicit request
STOP_ORDER_LEVELS = 3          # number of stop orders per position
STOP_ORDER_SPREAD_PCT = 1.0    # spread between levels (1% of stop price)
STOP_ORDER_FILE = "data/metals_stop_orders.json"

# Emergency auto-sell (L3) safety
EMERGENCY_SELL_ENABLED = False  # default OFF: requires explicit enablement

_STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm") if ZoneInfo else None
_US_EASTERN_TZ = ZoneInfo("America/New_York") if ZoneInfo else None

# Trade queue (Layer 2 writes intent, Layer 1 executes)
TRADE_QUEUE_ENABLED = True
TRADE_QUEUE_FILE = "data/metals_trade_queue.json"
TRADE_QUEUE_MAX_AGE_S = 300     # expire orders older than 5 min
TRADE_QUEUE_MAX_SLIPPAGE = 2.0  # reject if price moved > 2% from queued

# Session health monitoring
SESSION_HEALTH_CHECK_INTERVAL = 20   # check every N loops (~30 min at 90s)
SESSION_EXPIRY_WARNING_H = 20        # warn when storage state is >20h old (session lasts ~24h)
SESSION_STORAGE_FILE = "data/avanza_storage_state.json"

# Hardware trailing stop (Avanza-managed, FOLLOW_DOWNWARDS)
# When True: places a single trailing stop via Avanza API on position open.
# Avanza tracks the price peak and triggers a sell when it drops trail_pct%.
# This works even if our process crashes. No polling needed.
HARDWARE_TRAILING_ENABLED = True
HARDWARE_TRAILING_PCT = 5.0     # trail 5% below peak (matches TRAIL_DISTANCE_PCT)
HARDWARE_TRAILING_VALID_DAYS = 8  # stop expires after 8 days

# Software trailing stop config — momentum-aware (fallback when hardware disabled)
TRAIL_START_PCT = 1.0           # start trailing after 1% gain from entry (was 2%)
TRAIL_DISTANCE_PCT = 5.0        # trail 5% below current bid (was 3% — user wants 5% safety)
TRAIL_MIN_MOVE_PCT = 0.5        # minimum move to update (was 1% — more responsive)
TRAIL_TIGHTEN_MOMENTUM = 3.0    # tighten to 3% when momentum negative
TRAIL_TIGHTEN_ACCEL = 2.0       # tighten to 2% when acceleration is negative (fast decline)

# Momentum exit (derivative-based early exit)
MOMENTUM_ENABLED = True
MOMENTUM_LOOKBACK = 5           # checks (5 * 90s = ~7.5 min)
MOMENTUM_MIN_VELOCITY = -0.5    # must be dropping at least 0.5%/check
MOMENTUM_ACCEL_THRESHOLD = -0.1 # acceleration must be negative (accelerating decline)
MOMENTUM_REQUIRE_L1 = True      # only trigger if already in L1+ danger zone

# Auto-exit override (prevent Claude HOLD paralysis)
AUTO_EXIT_L2_CHECKS = 5         # auto-sell after position in L2+ zone for N checks

# Holdings reconciliation cadence (always compare local state vs Avanza)
HOLDINGS_DIFF_INTERVAL_S = 30   # run diff/reconcile every 30s
# Legacy periodic verification cadence (active positions only)
HOLDINGS_CHECK_INTERVAL = 4     # every N checks

# Tier config: model, timeout, max_turns (no Opus — Sonnet handles critical too)
TIER_CONFIG = {
    1: {"model": "haiku",  "timeout": 60,   "max_turns": 8,  "label": "QUICK"},
    2: {"model": "sonnet", "timeout": 180,  "max_turns": 15, "label": "ANALYSIS"},
    3: {"model": "sonnet", "timeout": 180,  "max_turns": 15, "label": "CRITICAL"},
}

# --- SILVER FAST-TICK MONITOR (merged from silver_monitor.py) ---
# Provides 10-second price checks with instant threshold alerts and velocity detection
# for active silver positions. Runs during _sleep_for_cycle() between main 60s cycles.
SILVER_FAST_TICK_ENABLED = True
SILVER_FAST_TICK_INTERVAL = 10   # seconds between fast price checks
SILVER_ALERT_LEVELS = [
    (-3.0, "WARNING"),       # -14.3% warrant at ~4.76x
    (-5.0, "DANGER"),        # -23.8% warrant
    (-7.0, "HIGH RISK"),     # -33.3% warrant
    (-10.0, "CRITICAL"),     # -47.6% warrant
    (-12.5, "EMERGENCY"),    # -59.5% warrant
]
SILVER_VELOCITY_WINDOW = 18      # 18 × 10s = 3 min rolling window
SILVER_VELOCITY_ALERT_PCT = -0.8 # % drop threshold over the velocity window
SILVER_VELOCITY_TELEGRAM = True  # send Telegram on velocity alerts

# --- POSITIONS (defaults — overridden by persisted state on startup) ---
POSITIONS_DEFAULTS = {
    "gold": {
        "name": "BULL GULD X8 N", "ob_id": "856394", "api_type": "certificate",
        "units": 0, "entry": 0, "stop": 0, "active": False,
    },
    "silver301": {
        "name": "MINI L SILVER AVA 301", "ob_id": "2334960", "api_type": "warrant",
        "units": 0, "entry": 0, "stop": 0, "active": False,
    },
    "silver_sg": {
        "name": "MINI L SILVER SG", "ob_id": "2043157", "api_type": "warrant",
        "units": 0, "entry": 0, "stop": 0, "active": False,
    },
}
POSITIONS_STATE_FILE = "data/metals_positions_state.json"


def _load_json_state(path, default, label):
    """Load a JSON state file with explicit logging on corrupt/unreadable content."""
    import copy

    fallback = copy.deepcopy(default)
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        message = f"{label} load failed: {e}"
        if "log" in globals():
            log(message)
        else:
            print(message, flush=True)
        return fallback


def _default_trade_queue():
    return {"version": 1, "orders": []}

def _load_positions():
    """Load position state from disk, falling back to defaults."""
    import copy
    positions = copy.deepcopy(POSITIONS_DEFAULTS)
    try:
        saved = _load_json_state(POSITIONS_STATE_FILE, {}, "Position state")
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
        if saved:
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
        atomic_write_json(POSITIONS_STATE_FILE, state, ensure_ascii=False)
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
                    pos["sold_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
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
                pos["sold_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
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

def _load_runtime_config():
    """Load config.json for the current checkout, with safe import-time fallback."""
    candidates = [
        BASE_DIR / "config.json",
        Path(r"Q:/finance-analyzer/config.json"),
    ]
    last_error = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as _cf:
                return json.load(_cf)
        except (OSError, json.JSONDecodeError) as e:
            last_error = e
            print(f"[WARN] Cannot load config from {path}: {e}", flush=True)

    if __name__ == "__main__":
        if last_error:
            print(f"[FATAL] Cannot load config.json: {last_error}", flush=True)
        else:
            print("[FATAL] Cannot load config.json from the current checkout or live repo root", flush=True)
        sys.exit(1)

    if last_error:
        print(f"[WARN] Proceeding without config.json during import: {last_error}", flush=True)
    return {}


config = _load_runtime_config()
telegram_cfg = config.get("telegram", {}) if isinstance(config, dict) else {}
TG_TOKEN = telegram_cfg.get("token", "")
TG_CHAT = telegram_cfg.get("chat_id", "")

# --- STATE ---
check_count = 0
last_invoke_prices = {}   # prices at last Claude invocation
last_invoke_times = {1: 0.0, 2: 0.0, 3: 0.0}  # per-tier last invocation timestamps
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
_last_auto_telegram = 0           # timestamp of last autonomous Telegram (for throttling)
AUTO_TELEGRAM_COOLDOWN = 1800     # 30 min between routine autonomous messages
_last_news_fetch_ts = 0.0         # timestamp of last metals news fetch
NEWS_FETCH_INTERVAL = 1800        # fetch news every 30 min (1800s)

# --- FISH ENGINE (integrated intraday fishing) ---
FISH_ENGINE_ENABLED = False       # disabled by default — enable manually per session
_fish_engine = None               # FishEngine instance (lazy init)
PROB_REPORT_INTERVAL = 5          # compute probability report every N checks (~2.5 min)
PROB_TELEGRAM_INTERVAL = 20       # send probability telegram every N checks (~10 min)
session_alert_sent = False        # debounce: only send one alert per outage
session_expiry_warned = False     # debounce: only warn once about approaching expiry

SINGLETON_LOCK_FILE = os.path.join("data", "metals_loop.singleton.lock")
DUPLICATE_INSTANCE_EXIT_CODE = 11
_singleton_lock_fh = None

# --- Silver fast-tick state (merged from silver_monitor.py) ---
from collections import deque

_silver_fast_prices = deque(maxlen=SILVER_VELOCITY_WINDOW)
_silver_alerted_levels = set()       # thresholds already alerted this session
_silver_session_low = None
_silver_session_high = None
_silver_consecutive_down = 0
_silver_prev_price = None
_silver_underlying_ref = None        # XAG-USD price at position entry (reference for alerts)

# --- Fish precompute state ---
_gold_price_history = deque(maxlen=12)  # ~12 minutes at 60s cycle, for 5-min change
_silver_price_history_fish = deque(maxlen=12)  # same for silver
_orb_computed_today = None                # date string "YYYY-MM-DD" when ORB was last computed
_orb_range_cache = {"high": 0, "low": 0, "formed": False}
_vol_scalar_cache = {"value": 1.0, "ts": 0.0}  # hourly refresh
_signal_action_history = deque(maxlen=10)  # last 10 XAG signal actions for flip detection

def acquire_singleton_lock(lock_path=SINGLETON_LOCK_FILE):
    """Acquire single-instance lock for metals loop (non-blocking)."""
    global _singleton_lock_fh
    if _singleton_lock_fh is not None:
        return True
    if msvcrt is None:
        return True

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        return False

    try:
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} started={datetime.datetime.now(datetime.UTC).isoformat()}\n")
        fh.flush()
    except Exception:
        pass

    _singleton_lock_fh = fh
    return True

def release_singleton_lock():
    """Release single-instance lock if held."""
    global _singleton_lock_fh
    if _singleton_lock_fh is None:
        return
    try:
        if msvcrt is not None:
            _singleton_lock_fh.seek(0)
            msvcrt.locking(_singleton_lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        try:
            _singleton_lock_fh.close()
        except Exception:
            pass
        _singleton_lock_fh = None

def _safe_print(msg):
    """Print text without crashing on Windows non-UTF console encodings."""
    try:
        print(msg, flush=True)
        return
    except UnicodeEncodeError:
        pass

    try:
        safe = msg.encode("ascii", "replace").decode("ascii")
        print(safe, flush=True)
        return
    except Exception:
        pass

    try:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", "replace"))
        sys.stdout.flush()
    except Exception:
        pass


def send_telegram(msg):
    if not TG_TOKEN or not TG_CHAT:
        return
    if telegram_cfg.get("mute_all", False):
        log("[TG muted] " + msg[:80].replace("\n", " "))
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10
        )
    except Exception as e:
        _safe_print(f"[TG ERROR] {e}")


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _safe_print(f"[{ts}] {msg}")

def pnl_pct(current, entry):
    if entry == 0: return 0
    return ((current - entry) / entry) * 100


def _sleep_for_cycle(cycle_started, interval_s, label):
    """Sleep until the next scheduled cycle start, running silver fast ticks.

    This keeps cadence anchored to cycle start time rather than drifting by
    `interval + work_duration` on every iteration.  During the sleep window,
    runs 10-second silver fast ticks for any active silver position.
    """
    if not SILVER_FAST_TICK_ENABLED or not _has_active_silver():
        # No silver position — simple sleep
        elapsed = time.monotonic() - cycle_started
        remaining = interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)
            return
        log(f"{label} overran by {abs(remaining):.1f}s; continuing immediately")
        return

    # Silver fast-tick sub-loop during sleep
    min_remaining = SILVER_FAST_TICK_INTERVAL * 0.5  # don't bother if less than half a tick left
    while True:
        elapsed = time.monotonic() - cycle_started
        remaining = interval_s - elapsed
        if remaining <= min_remaining:
            break
        tick_sleep = min(SILVER_FAST_TICK_INTERVAL, remaining - min_remaining)
        if tick_sleep <= 0:
            break
        time.sleep(tick_sleep)
        try:
            _silver_fast_tick()
        except Exception as e:
            _safe_print(f"[silver tick] error: {e}")


# ---------------------------------------------------------------------------
# Silver fast-tick monitor (merged from silver_monitor.py)
# ---------------------------------------------------------------------------

def _has_active_silver():
    """Return True if any active silver position exists."""
    for key, pos in POSITIONS.items():
        if "silver" in key.lower() and pos.get("active"):
            return True
    return False


def _get_active_silver():
    """Return (key, pos) for the first active silver position, or (None, None)."""
    for key, pos in POSITIONS.items():
        if "silver" in key.lower() and pos.get("active"):
            return key, pos
    return None, None


def _silver_fetch_xag():
    """Fetch just XAG-USD from Binance FAPI (lightweight, single HTTP request).

    Updates ``_underlying_prices`` and returns the price, or the cached value
    on failure.
    """
    try:
        r = requests.get(
            f"{BINANCE_FAPI_TICKER}?symbol=XAGUSDT", timeout=5
        )
        if r.status_code == 200:
            price = float(r.json()["price"])
            _underlying_prices["XAG-USD"] = price
            return price
    except Exception:
        pass
    return _underlying_prices.get("XAG-USD")


def _silver_init_ref():
    """Initialize ``_silver_underlying_ref`` from persisted state or live price.

    Priority:
      1. ``underlying_entry`` field in metals_positions_state.json
      2. Current XAG-USD price from Binance (fallback for first run)

    Once computed, persists ``underlying_entry`` back to the state file so
    subsequent restarts use the same reference.
    """
    global _silver_underlying_ref

    # Already initialized
    if _silver_underlying_ref is not None:
        return

    silver_key, silver_pos = _get_active_silver()
    if silver_key is None:
        return

    # Try persisted underlying entry
    ref = silver_pos.get("underlying_entry")
    if ref and ref > 0:
        _silver_underlying_ref = ref
        log(f"Silver ref loaded: ${ref:.2f} (persisted)")
        return

    # Fallback: current XAG-USD price
    xag = _underlying_prices.get("XAG-USD")
    if xag and xag > 0:
        _silver_underlying_ref = xag
        log(f"Silver ref set to current XAG: ${xag:.2f} (session start)")
        # Persist for future restarts
        _silver_persist_ref(silver_key, xag)
        return


def _silver_persist_ref(silver_key, ref_price):
    """Write ``underlying_entry`` to the position state file for restart persistence."""
    try:
        state = _load_json_state(POSITIONS_STATE_FILE, {}, "silver_persist")
        if silver_key in state:
            state[silver_key]["underlying_entry"] = round(ref_price, 4)
            atomic_write_json(POSITIONS_STATE_FILE, state)
    except Exception as e:
        log(f"Silver ref persist error: {e}")


def _silver_reset_session():
    """Reset silver fast-tick session state (e.g., on new position or loop restart)."""
    global _silver_session_low, _silver_session_high
    global _silver_consecutive_down, _silver_prev_price, _silver_underlying_ref
    _silver_fast_prices.clear()
    _silver_alerted_levels.clear()
    _silver_session_low = None
    _silver_session_high = None
    _silver_consecutive_down = 0
    _silver_prev_price = None
    _silver_underlying_ref = None


def _silver_fast_tick():
    """10-second silver price check with threshold and velocity alerts.

    Merged from silver_monitor.py.  Fetches XAG-USD from Binance FAPI,
    checks for significant drops from the entry reference price, and detects
    rapid 3-minute flushes.  Only runs when an active silver position exists.
    """
    global _silver_session_low, _silver_session_high
    global _silver_consecutive_down, _silver_prev_price

    silver_key, silver_pos = _get_active_silver()
    if silver_key is None:
        return

    price = _silver_fetch_xag()
    if price is None or price <= 0:
        return

    # Ensure reference is initialized
    _silver_init_ref()
    ref = _silver_underlying_ref
    if ref is None or ref <= 0:
        return

    # Underlying % change from entry reference
    pct_change = (price - ref) / ref * 100

    # Approximate warrant P&L using position data
    entry_sek = silver_pos.get("entry", 0)
    units = silver_pos.get("units", 0)
    leverage = silver_pos.get("leverage", 4.76)
    invested = entry_sek * units if (entry_sek > 0 and units > 0) else 0
    warrant_pct = pct_change * leverage
    warrant_sek = invested * warrant_pct / 100 if invested > 0 else 0

    # --- Session tracking ---
    if _silver_session_low is None or price < _silver_session_low:
        _silver_session_low = price
    if _silver_session_high is None or price > _silver_session_high:
        _silver_session_high = price

    # Consecutive down ticks
    if _silver_prev_price is not None:
        if price < _silver_prev_price - 0.001:
            _silver_consecutive_down += 1
        else:
            _silver_consecutive_down = 0
    _silver_prev_price = price

    # Velocity tracking
    _silver_fast_prices.append(price)

    # --- Threshold alerts (from entry) ---
    for threshold, level_name in SILVER_ALERT_LEVELS:
        if pct_change <= threshold and threshold not in _silver_alerted_levels:
            _silver_alerted_levels.add(threshold)
            parts = [f"*{level_name}: XAG ${price:.2f} ({pct_change:+.1f}%)*"]
            if invested > 0:
                parts.append(f"`Warrant: {warrant_pct:+.1f}% = {warrant_sek:+,.0f} SEK`")
                parts.append(f"`Position: {invested + warrant_sek:,.0f} SEK`")
            parts.append(f"_Entry ${ref:.2f} | {leverage}x | {silver_key}_")
            msg = "\n".join(parts)
            log(f"*** SILVER {level_name}: XAG ${price:.2f} ({pct_change:+.1f}%) ***")
            send_telegram(msg)

    # --- Velocity alert (3-min rolling drop) ---
    if len(_silver_fast_prices) >= SILVER_VELOCITY_WINDOW:
        oldest = _silver_fast_prices[0]
        if oldest > 0:
            vel = (price - oldest) / oldest * 100
            if vel <= SILVER_VELOCITY_ALERT_PCT:
                vel_key = f"vel_{int(time.time() // 300)}"
                if vel_key not in _silver_alerted_levels:
                    _silver_alerted_levels.add(vel_key)
                    msg = (
                        f"*RAPID DROP: XAG {vel:.1f}% in "
                        f"{SILVER_VELOCITY_WINDOW * SILVER_FAST_TICK_INTERVAL}s*\n"
                        f"`${price:.2f} | W:{warrant_pct:+.1f}%`\n"
                        f"_Check now_"
                    )
                    log(f"*** SILVER VELOCITY: {vel:.1f}% ***")
                    if SILVER_VELOCITY_TELEGRAM:
                        send_telegram(msg)


def get_cet_time():
    """Get current Stockholm time from timeapi.io, fallback to zoneinfo (DST-safe)."""
    tz_label = "CET"
    if _STOCKHOLM_TZ is not None:
        try:
            tz_label = datetime.datetime.now(_STOCKHOLM_TZ).tzname() or "CET"
        except Exception:
            pass
    try:
        r = requests.get(
            "http://timeapi.io/api/time/current/zone?timeZone=Europe/Stockholm",
            timeout=3
        )
        if r.status_code == 200:
            data = r.json()
            h = data["hour"]
            m = data["minute"]
            return h + m / 60, f"{h:02d}:{m:02d} {tz_label}", "timeapi"
    except Exception as e:
        print(f"[WARN] timeapi.io failed: {e}", flush=True)
    # Fallback: zoneinfo handles DST correctly (CET/CEST)
    try:
        now = datetime.datetime.now(_STOCKHOLM_TZ)
        h = now.hour
        m = now.minute
        return h + m / 60, f"{h:02d}:{m:02d} {now.tzname() or tz_label}", "zoneinfo"
    except Exception:
        # Last resort: UTC+1 (wrong during summer DST)
        now = datetime.datetime.now(datetime.UTC)
        h = (now.hour + 1) % 24
        m = now.minute
        return h + m / 60, f"{h:02d}:{m:02d} CET", "system_utc+1"


def get_us_spike_schedule(now=None):
    """Return the daily US-open spike window in Stockholm time.

    Uses New York local times so DST changes in the United States are handled
    automatically, including the spring/fall mismatch against Stockholm.
    """
    if now is None:
        now = datetime.datetime.now(datetime.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.UTC)

    if _STOCKHOLM_TZ is not None and _US_EASTERN_TZ is not None:
        now_utc = now.astimezone(datetime.UTC)
        ny_date = now_utc.astimezone(_US_EASTERN_TZ).date()

        def _mk_ny(hour, minute):
            return datetime.datetime.combine(
                ny_date, datetime.time(hour, minute), tzinfo=_US_EASTERN_TZ
            )

        place_ny = _mk_ny(*SPIKE_PLACE_ET)
        open_ny = _mk_ny(*SPIKE_OPEN_ET)
        cancel_ny = _mk_ny(*SPIKE_CANCEL_ET)
        place_st = place_ny.astimezone(_STOCKHOLM_TZ)
        open_st = open_ny.astimezone(_STOCKHOLM_TZ)
        cancel_st = cancel_ny.astimezone(_STOCKHOLM_TZ)

        return {
            "place_hour": place_st.hour + place_st.minute / 60.0,
            "open_hour": open_st.hour + open_st.minute / 60.0,
            "cancel_hour": cancel_st.hour + cancel_st.minute / 60.0,
            "place_label": place_st.strftime("%H:%M ") + (place_st.tzname() or "Stockholm"),
            "open_label": open_st.strftime("%H:%M ") + (open_st.tzname() or "Stockholm"),
            "cancel_label": cancel_st.strftime("%H:%M ") + (cancel_st.tzname() or "Stockholm"),
            "et_open_label": open_ny.strftime("%H:%M ") + (open_ny.tzname() or "ET"),
        }

    # Fallback: winter Stockholm schedule
    return {
        "place_hour": 15.25,
        "open_hour": 15.5,
        "cancel_hour": 16.5,
        "place_label": "15:15 CET",
        "open_label": "15:30 CET",
        "cancel_label": "16:30 CET",
        "et_open_label": "09:30 ET",
    }

def cet_hour():
    h, _, _ = get_cet_time()
    return h

def cet_time_str():
    _, ts, _ = get_cet_time()
    return ts

def is_market_hours():
    """Check if Avanza commodity warrant market is open (Mon-Fri 08:15-21:55 Stockholm time)."""
    now = datetime.datetime.now(datetime.UTC)
    weekday = now.weekday()
    h = cet_hour()
    return weekday < 5 and 8.25 <= h <= 21.92

def is_avanza_open():
    """Check if Avanza warrant market is open for trading."""
    return is_market_hours()

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

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        result = {"age_min": round(age_min, 1)}
        for key in ["forecast_signals", "cumulative_gains"]:
            if key in data:
                result[key] = data[key]

        tickers = data.get("signals", {})
        if not tickers:
            return result

        for ticker in SIGNAL_TICKERS:
            if ticker in tickers:
                t = tickers[ticker]
                extra = t.get("extra", {})
                result[ticker] = {
                    "action": t.get("action", "?"),
                    "confidence": round(t.get("confidence", 0), 3),
                    "weighted_confidence": round(t.get("weighted_confidence", 0), 3),
                    "raw_action": extra.get("_raw_action", t.get("action", "?")),
                    "raw_confidence": round(extra.get("_raw_confidence", t.get("confidence", 0)), 3),
                    "weighted_action": extra.get("_weighted_action", t.get("action", "?")),
                    "rsi": round(t.get("rsi", 0), 1),
                    "macd_hist": t.get("macd_hist", 0),
                    "bb_position": t.get("bb_position", "?"),
                    "regime": t.get("regime", "?"),
                    "atr_pct": t.get("atr_pct", 0),
                    "buy_count": extra.get("_buy_count", 0),
                    "sell_count": extra.get("_sell_count", 0),
                    "voters": extra.get("_voters", 0),
                    "vote_detail": extra.get("_vote_detail", ""),
                    "price": t.get("price_usd", 0),
                    "extra": extra,
                }

        timeframes = data.get("timeframe_heatmap", {})
        for ticker in SIGNAL_TICKERS:
            if ticker in timeframes and ticker in result:
                result[ticker]["timeframes"] = timeframes[ticker]

        return result
    except Exception as e:
        log(f"Signal read error: {e}")
        return {}

# ---------------------------------------------------------------------------
# Binance FAPI underlying price fetch (always-on, no dependency on positions)
# ---------------------------------------------------------------------------
BINANCE_FAPI_TICKER = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_SPOT_TICKER = "https://api.binance.com/api/v3/ticker/price"

# Metals via FAPI (futures), Crypto via SPOT
UNDERLYING_SYMBOLS = {"XAG-USD": "XAGUSDT", "XAU-USD": "XAUUSDT"}
CRYPTO_SYMBOLS = {"BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT"}
# All tickers tracked for underlying prices
ALL_TRACKED_TICKERS = list(UNDERLYING_SYMBOLS.keys()) + list(CRYPTO_SYMBOLS.keys()) + ["MSTR"]
# Tickers that have signals in agent_summary.json (MSTR was removed Mar 1)
SIGNAL_TICKERS = ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"]

_underlying_prices = {}  # always-fresh: {"XAG-USD": float, ..., "BTC-USD": float, ...}
_underlying_history = {}  # rolling price history for momentum per ticker
_underlying_klines_cache = {}  # {"XAG-USD": {"ts": float, "klines": [...]}}

def fetch_underlying_from_binance():
    """Fetch prices: metals from FAPI, crypto from SPOT, MSTR from Yahoo."""
    global _underlying_prices
    prices = {}

    # Metals via Binance FAPI (futures)
    for ticker, symbol in UNDERLYING_SYMBOLS.items():
        try:
            r = requests.get(
                f"{BINANCE_FAPI_TICKER}?symbol={symbol}", timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                prices[ticker] = float(data["price"])
        except Exception as e:
            log(f"Binance FAPI {ticker} error: {e}")

    # Crypto via Binance SPOT
    for ticker, symbol in CRYPTO_SYMBOLS.items():
        try:
            r = requests.get(
                f"{BINANCE_SPOT_TICKER}?symbol={symbol}", timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                prices[ticker] = float(data["price"])
        except Exception as e:
            log(f"Binance SPOT {ticker} error: {e}")

    # MSTR via Yahoo (only when US market is relevant — always fetch, mark state)
    if CRYPTO_DATA_AVAILABLE:
        try:
            mstr = fetch_mstr_price()
            if mstr and mstr.get("price", 0) > 0:
                prices["MSTR"] = mstr["price"]
        except Exception as e:
            log(f"MSTR Yahoo error: {e}")

    if prices:
        _underlying_prices.update(prices)
        # Update rolling history (keep last 60 = ~1 hour at 60s)
        for ticker, price in prices.items():
            hist = _underlying_history.setdefault(ticker, [])
            hist.append({"ts": time.time(), "price": price})
            if len(hist) > 60:
                hist.pop(0)
    return prices

# ---------------------------------------------------------------------------
# Microstructure snapshot accumulator (order book depth → OFI, spread z-score)
# ---------------------------------------------------------------------------
_MICROSTRUCTURE_AVAILABLE = False
try:
    from portfolio.metals_orderbook import get_orderbook_depth
    from portfolio.microstructure_state import accumulate_snapshot, persist_state
    _MICROSTRUCTURE_AVAILABLE = True
except ImportError:
    pass

_MICROSTRUCTURE_TICKERS = ["XAG-USD", "XAU-USD"]  # metals only for now
_microstructure_persist_counter = 0

def _accumulate_orderbook_snapshots():
    """Poll order book depth and accumulate snapshots for OFI computation.

    Called each cycle (~30-60s).  Fetches depth for metals tickers,
    adds to ring buffer, and persists state every 5th call.
    """
    global _microstructure_persist_counter
    if not _MICROSTRUCTURE_AVAILABLE:
        return
    for ticker in _MICROSTRUCTURE_TICKERS:
        try:
            depth = get_orderbook_depth(ticker, limit=20)
            if depth:
                accumulate_snapshot(ticker, depth)
        except Exception as e:
            if _microstructure_persist_counter % 30 == 0:  # log rarely
                log(f"Microstructure snapshot error for {ticker}: {e}")
    _microstructure_persist_counter += 1
    if _microstructure_persist_counter % 5 == 0:  # persist every ~2.5-5 min
        try:
            persist_state()
        except Exception as e:
            log(f"Microstructure state persist error: {e}")


def fetch_underlying_klines(ticker, interval="1h", limit=100):
    """Fetch OHLCV klines from Binance (FAPI for metals, SPOT for crypto). Cached 5 min."""
    symbol = UNDERLYING_SYMBOLS.get(ticker) or CRYPTO_SYMBOLS.get(ticker)
    if not symbol:
        return None
    cache = _underlying_klines_cache.get(ticker)
    if cache and time.time() - cache["ts"] < 300:
        return cache["klines"]
    # Crypto uses SPOT klines, metals use FAPI
    is_crypto = ticker in CRYPTO_SYMBOLS
    base_url = "https://api.binance.com/api/v3/klines" if is_crypto else BINANCE_FAPI_KLINES
    try:
        r = requests.get(
            base_url,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            klines = [
                {"open": float(k[1]), "high": float(k[2]),
                 "low": float(k[3]), "close": float(k[4]),
                 "volume": float(k[5])}
                for k in data
            ]
            _underlying_klines_cache[ticker] = {"ts": time.time(), "klines": klines}
            return klines
    except Exception as e:
        log(f"Binance klines {ticker} error: {e}")
    return None

def get_underlying_momentum(ticker, lookback=10):
    """Compute price velocity and acceleration for a ticker from rolling history.

    Returns:
        dict with velocity_pct (% per check), acceleration (change in velocity),
        trend ("up"/"down"/"flat"), momentum_score (-1 to +1)
    """
    hist = _underlying_history.get(ticker, [])
    if len(hist) < max(3, lookback):
        return {"velocity_pct": 0, "acceleration": 0, "trend": "flat", "momentum_score": 0}

    recent = hist[-lookback:]
    prices = [h["price"] for h in recent]

    # Velocity: average % change per check over lookback
    changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
    velocity = sum(changes) / len(changes) if changes else 0

    # Acceleration: change in velocity (second derivative)
    if len(changes) >= 4:
        first_half = sum(changes[:len(changes)//2]) / (len(changes)//2)
        second_half = sum(changes[len(changes)//2:]) / (len(changes) - len(changes)//2)
        acceleration = second_half - first_half
    else:
        acceleration = 0

    # Trend classification
    if velocity > 0.02:
        trend = "up"
    elif velocity < -0.02:
        trend = "down"
    else:
        trend = "flat"

    # Momentum score: -1 (strong sell) to +1 (strong buy)
    score = max(-1, min(1, velocity * 10))

    return {
        "velocity_pct": round(velocity, 4),
        "acceleration": round(acceleration, 5),
        "trend": trend,
        "momentum_score": round(score, 3),
    }


# ---------------------------------------------------------------------------
# Dynamic holdings detection — auto-detect instruments bought by user
# ---------------------------------------------------------------------------
# Known warrant orderbook IDs → ticker mapping
KNOWN_WARRANT_OB_IDS = {
    "2334960": {"key": "silver301", "name": "MINI L SILVER AVA 301",
                "api_type": "warrant", "underlying": "XAG-USD", "leverage": 4.3},
    "2043157": {"key": "silver_sg", "name": "MINI L SILVER SG",
                "api_type": "warrant", "underlying": "XAG-USD", "leverage": 1.56},
    "856394":  {"key": "gold", "name": "BULL GULD X8 N",
                "api_type": "certificate", "underlying": "XAU-USD", "leverage": 8.0},
    "2286417": {"key": "bear_silver_x5", "name": "BEAR SILVER X5 AVA 12",
                "api_type": "certificate", "underlying": "XAG-USD", "leverage": 5.0},
}
# Extend with WARRANT_CATALOG if available
if CATALOG_AVAILABLE:
    for wk, wv in WARRANT_CATALOG.items():
        ob_id = str(wv.get("ob_id", ""))
        if ob_id and ob_id not in KNOWN_WARRANT_OB_IDS:
            KNOWN_WARRANT_OB_IDS[ob_id] = {
                "key": wk, "name": wv.get("name", wk),
                "api_type": wv.get("api_type", "warrant"),
                "underlying": wv.get("underlying", "XAG-USD"),
                "leverage": wv.get("leverage", 5.0),
            }

def detect_holdings(page):
    """Detect all held instruments on Avanza. Auto-add new ones to POSITIONS.

    Returns list of change descriptions (for logging/telegram).
    """
    changes = []
    try:
        result = page.evaluate("""async (accountId) => {
            const resp = await fetch(
                'https://www.avanza.se/_api/position-data/positions',
                {credentials: 'include'}
            );
            if (resp.status !== 200) return {error: resp.status};
            const data = await resp.json();
            const items = [];
            for (const item of (data.withOrderbook || [])) {
                if (item.account && String(item.account.id) === accountId) {
                    const obId = item.instrument && item.instrument.orderbook
                        ? String(item.instrument.orderbook.id) : null;
                    const name = item.instrument && item.instrument.orderbook
                        ? item.instrument.orderbook.name || '' : '';
                    const vol = item.volume && item.volume.value != null
                        ? item.volume.value : 0;
                    const avgPrice = item.averageAcquiredPrice && item.averageAcquiredPrice.value != null
                        ? item.averageAcquiredPrice.value : 0;
                    if (obId && vol > 0) {
                        items.push({id: obId, name: name, units: vol, avg_price: avgPrice});
                    }
                }
            }
            return {positions: items};
        }""", ACCOUNT_ID)

        if not result or "positions" not in result:
            return changes

        held_ob_ids = {str(p["id"]): p for p in result["positions"]}

        # Check for NEW instruments not in POSITIONS
        existing_ob_ids = {pos["ob_id"]: key for key, pos in POSITIONS.items()}
        for ob_id, holding in held_ob_ids.items():
            if ob_id in existing_ob_ids:
                key = existing_ob_ids[ob_id]
                pos = POSITIONS[key]
                # Reactivate if was sold
                if not pos["active"]:
                    pos["active"] = True
                    pos["units"] = holding["units"]
                    pos["entry"] = holding["avg_price"] if holding["avg_price"] > 0 else pos["entry"]
                    # Set initial stop at 5% below entry
                    pos["stop"] = round(pos["entry"] * 0.95, 2)
                    pos.pop("sold_ts", None)
                    pos.pop("sold_price", None)
                    pos.pop("sold_reason", None)
                    changes.append(f"REACTIVATED {key}: {holding['units']}u @ {pos['entry']}")
                    log(f"Holdings: reactivated {key} ({holding['units']}u)")
                elif pos["units"] != holding["units"]:
                    old_units = pos["units"]
                    pos["units"] = holding["units"]
                    changes.append(f"UNITS CHANGED {key}: {old_units} -> {holding['units']}")
                    log(f"Holdings: {key} units {old_units} -> {holding['units']}")
            else:
                # Brand new instrument — check if we recognize it
                info = KNOWN_WARRANT_OB_IDS.get(ob_id)
                if info:
                    key = info["key"]
                    entry_price = holding["avg_price"] if holding["avg_price"] > 0 else 0
                    stop_price = round(entry_price * 0.95, 2) if entry_price > 0 else 0
                    POSITIONS[key] = {
                        "name": info["name"], "ob_id": ob_id,
                        "api_type": info["api_type"],
                        "units": holding["units"],
                        "entry": entry_price, "stop": stop_price,
                        "active": True,
                        "_underlying": info["underlying"],
                        "_leverage": info["leverage"],
                    }
                    changes.append(f"NEW {key}: {holding['units']}u @ {entry_price} (auto-detected)")
                    log(f"Holdings: NEW instrument detected: {key} = {info['name']} "
                        f"({holding['units']}u @ {entry_price})")
                else:
                    log(f"Holdings: unknown ob_id {ob_id} ({holding.get('name', '?')}) — skipping")

        # Check for REMOVED instruments (held in POSITIONS but not on Avanza)
        for key, pos in POSITIONS.items():
            if not pos["active"]:
                continue
            if pos["ob_id"] not in held_ob_ids:
                pos["active"] = False
                pos["sold_reason"] = "auto_detect_not_held"
                pos["sold_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
                changes.append(f"SOLD {key}: no longer on Avanza")
                log(f"Holdings: {key} no longer held on Avanza — deactivating")

        if changes:
            _save_positions(POSITIONS)

    except Exception as e:
        log(f"Holdings detection error: {e}")

    return changes


# ---------------------------------------------------------------------------
# Smart trailing stop — momentum-aware, derivative-based
# ---------------------------------------------------------------------------

def compute_smart_trail_distance(position_key, bid, entry, stop):
    """Compute dynamic trailing stop distance based on momentum + derivatives.

    Normal: 5% below current bid
    Momentum reversal: tighten to 3%
    Accelerating decline: tighten to 2%

    Returns: optimal stop price, trail_distance_pct used
    """
    # Determine underlying ticker for momentum
    underlying_ticker = "XAG-USD" if "silver" in position_key else "XAU-USD"
    momentum = get_underlying_momentum(underlying_ticker, lookback=10)

    velocity = momentum["velocity_pct"]
    acceleration = momentum["acceleration"]

    # Base distance: 5%
    trail_dist = TRAIL_DISTANCE_PCT

    # Tighten based on momentum
    if velocity < -0.02 and acceleration < 0:
        # Accelerating decline — tighten to 2%
        trail_dist = TRAIL_TIGHTEN_ACCEL
    elif velocity < -0.01:
        # Negative momentum — tighten to 3%
        trail_dist = TRAIL_TIGHTEN_MOMENTUM

    # Compute new stop price
    new_stop = round(bid * (1 - trail_dist / 100), 2)

    # Never lower the stop (ratchet up only)
    if new_stop <= stop:
        return stop, trail_dist

    return new_stop, trail_dist


def _update_stop_orders_for(page, key, pos, stop_order_state):
    """Cancel existing stop orders for a position and re-place at the new stop level.

    Called when trailing stop moves up — cancels old hardware stops and places
    new cascading stop orders at the updated price.
    """
    csrf = get_csrf(page)
    if not csrf:
        log(f"  Stop update for {key}: no CSRF token")
        return

    # Cancel existing orders
    existing = stop_order_state.get(key, {})
    if existing.get("orders"):
        _cancel_stop_orders(page, key, existing, csrf)

    # Place new cascading stop orders
    units = pos["units"]
    stop_base = pos["stop"]

    # Safety: check stop distance from current bid
    cur_price_data = fetch_price(page, pos["ob_id"], pos.get("api_type", "warrant"))
    cur_bid = (cur_price_data or {}).get("bid", 0)
    if cur_bid > 0:
        distance_pct = (cur_bid - stop_base) / cur_bid * 100
        if distance_pct < 3.0:
            log(f"  SKIP stop update {key}: trigger {stop_base} only {distance_pct:.1f}% "
                f"below bid {cur_bid} — too close")
            return

    orders = []
    for level in range(STOP_ORDER_LEVELS):
        spread = level * STOP_ORDER_SPREAD_PCT / 100.0
        trigger_price = round(stop_base * (1 - spread), 2)
        sell_price = round(trigger_price * 0.99, 2)

        if level < STOP_ORDER_LEVELS - 1:
            level_units = units // STOP_ORDER_LEVELS
        else:
            level_units = units - (units // STOP_ORDER_LEVELS) * (STOP_ORDER_LEVELS - 1)

        if level_units <= 0:
            continue

        try:
            ok, stop_id = place_stop_loss(
                page, ACCOUNT_ID, pos["ob_id"],
                trigger_price=trigger_price,
                sell_price=sell_price,
                volume=level_units,
                valid_days=8,
            )
            if ok:
                orders.append({
                    "level": level + 1,
                    "order_id": stop_id,
                    "trigger": trigger_price,
                    "sell": sell_price,
                    "volume": level_units,
                    "status": "placed",
                })
                log(f"  Stop S{level+1} {key}: trig={trigger_price} sell={sell_price} vol={level_units}")
            else:
                log(f"  Stop S{level+1} FAILED for {key}")
        except Exception as e:
            log(f"  Stop S{level+1} error {key}: {e}")

    if orders:
        stop_order_state[key] = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "stop_base": stop_base,
            "orders": orders,
        }
        _save_stop_orders(stop_order_state)
        log(f"  Trail stop orders updated for {key}: {len(orders)} levels at base={stop_base}")


def update_smart_trailing_stops(page, positions, stop_order_state, prices):
    """Update trailing stops with momentum-aware distance for all active positions."""
    for key, pos in positions.items():
        if not pos.get("active"):
            continue
        p = prices.get(key)
        if not p or not p.get("bid"):
            continue

        bid = p["bid"]
        entry = pos["entry"]
        old_stop = pos["stop"]
        pnl = pnl_pct(bid, entry)

        # Only trail if in profit above threshold
        if pnl < TRAIL_START_PCT:
            continue

        new_stop, dist_used = compute_smart_trail_distance(key, bid, entry, old_stop)

        if new_stop > old_stop:
            move_pct = ((new_stop - old_stop) / old_stop) * 100
            if move_pct >= TRAIL_MIN_MOVE_PCT:
                pos["stop"] = new_stop
                underlying_ticker = "XAG-USD" if "silver" in key else "XAU-USD"
                mom = get_underlying_momentum(underlying_ticker)
                log(f"TRAIL {key}: stop {old_stop} -> {new_stop} "
                    f"(dist={dist_used:.1f}%, vel={mom['velocity_pct']:.3f}%, "
                    f"accel={mom['acceleration']:.4f})")
                _save_positions(positions)

                # Update hardware stop orders on Avanza
                if STOP_ORDER_ENABLED and stop_order_state:
                    try:
                        _update_stop_orders_for(page, key, pos, stop_order_state)
                    except Exception as e:
                        log(f"Stop order update failed for {key}: {e}")


# ---------------------------------------------------------------------------
# Periodic news fetch — metals headlines for fish monitor context
# ---------------------------------------------------------------------------

# Severity keywords to scan for in headlines (subset of news_keywords.py)
_METALS_SEVERITY_KEYWORDS = [
    "tariff", "tariffs", "crash", "war", "sanctions", "ban", "recession",
    "rate hike", "rate cut", "inflation", "collapse", "default", "nuclear",
    "invasion", "trade war", "debt ceiling",
]


def _fetch_metals_news():
    """Fetch silver/gold news headlines and write summary to data/metals_news_summary.json.

    Sources:
    1. CryptoCompare (via get_crypto_news) — general crypto/commodities news
    2. NewsAPI (if key available in config) — targeted silver + gold queries

    Results are merged, deduplicated, and scored for severity keywords.
    Writes to data/metals_news_summary.json using atomic I/O.
    """
    global _last_news_fetch_ts
    headlines = []
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    # Source 1: CryptoCompare news (general crypto/commodities)
    if CRYPTO_DATA_AVAILABLE:
        try:
            crypto_news = get_crypto_news(limit=15)
            for article in (crypto_news or []):
                title = article.get("title", "")
                if not title:
                    continue
                # Filter: only keep articles mentioning metals keywords
                title_lower = title.lower()
                is_metals = any(kw in title_lower for kw in (
                    "gold", "silver", "precious", "metal", "bullion", "xau", "xag",
                    "commodity", "commodities", "fed", "inflation", "tariff", "dollar",
                    "treasury", "yields", "safe haven", "haven",
                ))
                if is_metals:
                    published_ts = article.get("published_on", 0)
                    pub_iso = (
                        datetime.datetime.fromtimestamp(published_ts, tz=datetime.UTC).isoformat()
                        if published_ts else now_iso
                    )
                    # Determine ticker affinity
                    ticker = "XAU-USD"  # default to gold
                    if any(kw in title_lower for kw in ("silver", "xag")):
                        ticker = "XAG-USD"
                    headlines.append({
                        "title": title,
                        "source": article.get("source", "CryptoCompare"),
                        "ticker": ticker,
                        "published": pub_iso,
                    })
        except Exception as e:
            log(f"News fetch (CryptoCompare) error: {e}")

    # Source 2: NewsAPI — targeted silver and gold queries
    newsapi_key = config.get("newsapi_key", "") if isinstance(config, dict) else ""
    if newsapi_key:
        for query, ticker in [
            ("silver AND (price OR market OR ounce OR bullion OR futures)", "XAG-USD"),
            ("gold AND (price OR market OR ounce OR bullion OR futures)", "XAU-USD"),
        ]:
            try:
                resp = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "pageSize": 5,
                    },
                    headers={"User-Agent": "Mozilla/5.0", "X-Api-Key": newsapi_key},
                    timeout=15,
                )
                if resp.status_code == 200:
                    articles = resp.json().get("articles", [])
                    for a in articles:
                        title = a.get("title", "")
                        if not title:
                            continue
                        headlines.append({
                            "title": title,
                            "source": a.get("source", {}).get("name", "NewsAPI"),
                            "ticker": ticker,
                            "published": a.get("publishedAt", now_iso),
                        })
            except Exception as e:
                log(f"News fetch (NewsAPI {ticker}) error: {e}")

    # Deduplicate by title (case-insensitive)
    seen_titles = set()
    unique_headlines = []
    for h in headlines:
        key = h["title"].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_headlines.append(h)

    # Scan for severity keywords
    severity_found = set()
    for h in unique_headlines:
        title_lower = h["title"].lower()
        for kw in _METALS_SEVERITY_KEYWORDS:
            if kw in title_lower:
                severity_found.add(kw)
        # Also use score_headline for richer detection
        _weight, matched = score_headline(h["title"])
        for m in matched:
            severity_found.add(m)

    summary = {
        "timestamp": now_iso,
        "headlines": unique_headlines[:20],  # cap at 20
        "article_count": len(unique_headlines),
        "severity_keywords_found": sorted(severity_found),
    }

    atomic_write_json(str(DATA_DIR / "metals_news_summary.json"), summary)
    _last_news_fetch_ts = time.time()
    log(f"News fetch: {len(unique_headlines)} headlines, severity={sorted(severity_found) or 'none'}")
    return summary


# ---------------------------------------------------------------------------
# Fish engine — integrated intraday fishing decision engine
# ---------------------------------------------------------------------------

def _run_fish_engine_tick():
    """Run one fish engine tick. Called every 60s by the main loop.

    Builds a state dict from metals loop data, calls the engine,
    and executes any buy/sell decisions via Avanza.
    """
    global _fish_engine

    # Lazy init
    if _fish_engine is None:
        try:
            from fish_engine import FishEngine
            _fish_engine = FishEngine()
            # Try to restore state from previous session
            try:
                state_data = _load_json_state(DATA_DIR / "fish_engine_state.json", None, "fish_engine")
                if state_data:
                    _fish_engine.from_dict(state_data)  # instance method, restores state in-place
                    log(f"Fish engine restored: mode={_fish_engine.mode}, pnl={_fish_engine.session_pnl:+.0f}")
            except Exception:
                pass
            log(f"Fish engine initialized: mode={_fish_engine.mode}")
        except ImportError as e:
            log(f"Fish engine import failed: {e}")
            return

    # Build state dict from metals loop data
    xag_price = _underlying_prices.get("XAG-USD", 0)
    xau_price = _underlying_prices.get("XAU-USD", 0)
    if xag_price <= 0:
        return  # no price, skip

    # Read latest signals from agent_summary_compact
    try:
        summary = _load_json_state(DATA_DIR.parent / "data" / "agent_summary_compact.json", {}, "fish_sig")
        if not summary:
            summary = _load_json_state(DATA_DIR / "agent_summary_compact.json", {}, "fish_sig2")
    except Exception:
        summary = {}

    xag_sig = (summary.get("signals") or {}).get("XAG-USD", {})
    xag_extra = xag_sig.get("extra", {})
    xag_mc = (summary.get("monte_carlo") or {}).get("XAG-USD", {})
    xag_focus = (summary.get("focus_probabilities") or {}).get("XAG-USD", {})

    # Read metals loop's own signals
    metals_sig = last_signal_data.get("XAG-USD", {}) if last_signal_data else {}

    # Read fish_precomputed for gold change, orb, vol_scalar
    fish_pre = _load_json_state(DATA_DIR / "fish_precomputed.json", {}, "fish_pre")

    # Read enhanced signals for news/econ
    news_action = "HOLD"
    econ_action = "HOLD"
    event_hours = 999
    high_impact = False
    try:
        full_summary = _load_json_state(DATA_DIR.parent / "data" / "agent_summary.json", {}, "fish_full")
        if not full_summary:
            full_summary = _load_json_state(DATA_DIR / "agent_summary.json", {}, "fish_full2")
        enh = (full_summary.get("signals") or {}).get("XAG-USD", {}).get("enhanced_signals", {})
        news_action = (enh.get("news_event") or {}).get("action", "HOLD")
        econ_action = (enh.get("econ_calendar") or {}).get("action", "HOLD")
        econ_ind = (enh.get("econ_calendar") or {}).get("indicators", {})
        event_hours = econ_ind.get("proximity_hours_until", 999)
        if not isinstance(event_hours, (int, float)):
            event_hours = 999
        high_impact = bool(econ_ind.get("risk_high_impact_within_4h", False))
    except Exception:
        pass

    # Read Layer 2 journal for latest XAG-USD context
    layer2_outlook = ''
    layer2_conviction = 0.0
    layer2_levels = []
    layer2_action = 'HOLD'
    layer2_ts = ''
    try:
        journal_path = DATA_DIR.parent / 'data' / 'layer2_journal.jsonl'
        if not journal_path.exists():
            journal_path = DATA_DIR / 'layer2_journal.jsonl'
        if journal_path.exists():
            with open(str(journal_path), 'r', encoding='utf-8') as _jf:
                lines = _jf.readlines()
            lines = [l.strip() for l in lines if l.strip()]
            # Scan last 10 entries for XAG-USD
            for line in reversed(lines[-10:]):
                try:
                    entry = json.loads(line)
                    tickers = entry.get('tickers', {})
                    if 'XAG-USD' in tickers:
                        xag_j = tickers['XAG-USD']
                        layer2_outlook = xag_j.get('outlook', '')
                        layer2_conviction = float(xag_j.get('conviction', 0))
                        layer2_levels = xag_j.get('levels', [])
                        layer2_ts = entry.get('ts', '')
                        # Also check decisions for action
                        for strategy in ('patient', 'bold'):
                            dec = entry.get('decisions', {}).get(strategy, {})
                            if dec.get('action') in ('BUY', 'SELL'):
                                layer2_action = dec['action']
                                break
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # Monte Carlo bands
    mc_bands_1d = xag_mc.get('price_bands_1d', {})

    # Chronos forecast
    xag_forecast = (summary.get('forecast_signals') or {}).get('XAG-USD', {})
    chronos_1h_pct = float(xag_forecast.get('chronos_1h_pct', 0) or 0)
    chronos_24h_pct = float(xag_forecast.get('chronos_24h_pct', 0) or 0)

    # Prophecy
    prophecy_target = 0.0
    prophecy_conviction = 0.0
    try:
        beliefs = (summary.get('prophecy') or {}).get('beliefs', [])
        for belief in beliefs:
            if belief.get('ticker') == 'XAG-USD':
                prophecy_target = float(belief.get('target_price', 0) or 0)
                prophecy_conviction = float(belief.get('conviction', 0) or 0)
                break
    except Exception:
        pass

    # Check trade guard from metals risk
    trade_guard_ok = True
    try:
        if RISK_AVAILABLE:
            guard = check_trade_guard("XAG-USD", "silver_fish")
            trade_guard_ok = guard.get("allowed", True) if isinstance(guard, dict) else bool(guard)
    except Exception:
        pass

    # Get spread from current quotes
    spread_pct = 0.3  # default
    try:
        from avanza_session import get_quote as _aq
        q = _aq("1650161")  # BULL X5 AVA 4
        bid = float(q.get("buy", 0) or 0)
        ask = float(q.get("sell", 0) or 0)
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / bid * 100
    except Exception:
        pass

    # Build state
    now = datetime.datetime.now()
    f1d = xag_focus.get("1d", {})
    state = {
        "silver_price": xag_price,
        "gold_price": xau_price,
        "gold_5min_change": fish_pre.get("gold_5min_change_pct", 0),
        "signal_action": xag_sig.get("action", "HOLD"),
        "signal_buy_count": xag_extra.get("_buy_count", 0),
        "signal_sell_count": xag_extra.get("_sell_count", 0),
        "rsi": float(xag_sig.get("rsi", 50)),
        "mc_p_up": float(xag_mc.get("p_up", 0.5)),
        "metals_action": xag_sig.get("action", "HOLD"),  # same as signal_action
        "regime": xag_sig.get("regime", "ranging"),
        "news_action": news_action,
        "econ_action": econ_action,
        "focus_1d_dir": f1d.get("direction", "?"),
        "focus_1d_prob": float(f1d.get("probability", 0.5)),
        "orb_range": fish_pre.get("orb_range"),
        "vol_scalar": fish_pre.get("vol_scalar", 1.0),
        "hour_cet": now.hour,
        "minute_cet": now.minute,
        "day_of_week": now.weekday(),
        "velocity": None,  # could hook into silver fast-tick
        "trade_guard_ok": trade_guard_ok,
        "spread_pct": spread_pct,
        "news_spike": False,  # TODO: read from headlines
        "headline_sentiment": "",
        "event_hours": event_hours,
        "high_impact_near": high_impact,
        # Layer 2 journal context
        "layer2_outlook": layer2_outlook,
        "layer2_conviction": layer2_conviction,
        "layer2_levels": layer2_levels,
        "layer2_action": layer2_action,
        "layer2_ts": layer2_ts,
        # Monte Carlo bands
        "mc_bands_1d": mc_bands_1d,
        # Chronos forecast
        "chronos_1h_pct": chronos_1h_pct,
        "chronos_24h_pct": chronos_24h_pct,
        # Prophecy belief
        "prophecy_target": prophecy_target,
        "prophecy_conviction": prophecy_conviction,
    }

    # Call engine
    decision = _fish_engine.tick(state)

    if decision["action"] == "HOLD":
        return  # nothing to do

    # Execute decision
    if decision["action"] == "BUY":
        _fish_engine_execute_buy(decision, xag_price)
    elif decision["action"] == "SELL":
        _fish_engine_execute_sell(decision)

    # Persist engine state
    try:
        atomic_write_json(str(DATA_DIR / "fish_engine_state.json"), _fish_engine.to_dict())
    except Exception:
        pass


def _fish_engine_execute_buy(decision, price):
    """Execute a BUY decision from the fish engine."""
    global _fish_engine
    direction = decision.get("direction", "LONG")
    ob_id = decision.get("instrument_ob", "1650161" if direction == "LONG" else "2286417")
    tactic = decision.get("reason", "")

    try:
        from avanza_session import place_buy_order, get_quote, get_buying_power
        q = get_quote(ob_id)
        ask = float(q.get("sell", 0))
        bp = float(get_buying_power().get("buying_power", 0))

        if ask <= 0 or bp < 1000:
            log(f"[fish] SKIP BUY: ask={ask}, bp={bp:.0f} (need >1000)")
            return

        budget = min(bp * 0.95, 1500) * decision.get("size_scalar", 1.0)
        if budget < 1000:
            log(f"[fish] SKIP BUY: budget {budget:.0f} < 1000 SEK")
            return

        volume = int(budget / ask)
        if volume < 5:
            return

        result = place_buy_order(ob_id, price=ask, volume=volume)
        status = result.get("orderRequestStatus", "?")
        nm = "BULL X5" if direction == "LONG" else "BEAR X5"
        log(f"[fish] BUY {volume}u {nm}@{ask} = {volume*ask:.0f} SEK ({tactic}) [{status}]")

        if status == "SUCCESS":
            _fish_engine.confirm_entry(direction, ask, volume, price)
    except Exception as e:
        log(f"[fish] BUY error: {e}")


def _fish_engine_execute_sell(decision):
    """Execute a SELL decision from the fish engine."""
    global _fish_engine
    if not _fish_engine or not _fish_engine.has_position:
        return

    pos = _fish_engine.position
    ob_id = pos.get("ob_id", "")
    volume = pos.get("volume", 0)
    reason = decision.get("exit_reason", "engine")

    if not ob_id or volume <= 0:
        return

    try:
        from avanza_session import place_sell_order, get_quote
        q = get_quote(ob_id)
        bid = float(q.get("buy", 0))
        if bid <= 0:
            return

        result = place_sell_order(ob_id, price=bid, volume=volume)
        status = result.get("orderRequestStatus", "?")
        entry_price = pos.get("entry_cert", 0)
        pnl = (bid - entry_price) * volume
        nm = "BULL X5" if pos.get("direction") == "LONG" else "BEAR X5"
        log(f"[fish] SELL {volume}u {nm}@{bid} P&L:{pnl:+.0f} ({reason}) [{status}]")

        if status == "SUCCESS":
            _fish_engine.confirm_exit(pnl)
    except Exception as e:
        log(f"[fish] SELL error: {e}")


# ---------------------------------------------------------------------------
# Fish precomputation — provides pre-built data for the fishing monitor
# ---------------------------------------------------------------------------

def _write_fish_precomputed():
    """Write data/fish_precomputed.json for the fishing monitor.

    Computes: gold/silver 5-min price change, gold-leads-silver detection,
    ORB range (once/day after 10:00 UTC), vol scalar (hourly), mode suggestion.
    Never crashes the loop — all exceptions caught internally.
    """
    global _orb_computed_today, _orb_range_cache, _vol_scalar_cache

    try:
        gold_price = _underlying_prices.get("XAU-USD", 0)
        silver_price = _underlying_prices.get("XAG-USD", 0)
        if gold_price <= 0 or silver_price <= 0:
            return  # no prices yet, skip silently

        # --- Update price histories ---
        _gold_price_history.append({"ts": time.time(), "price": gold_price})
        _silver_price_history_fish.append({"ts": time.time(), "price": silver_price})

        # --- 5-min change (need at least 5 entries = ~5 min at 60s cycle) ---
        gold_5min_change = 0.0
        silver_5min_change = 0.0
        if len(_gold_price_history) >= 5:
            old_gold = _gold_price_history[-5]["price"]
            if old_gold > 0:
                gold_5min_change = round((gold_price - old_gold) / old_gold * 100, 4)
        if len(_silver_price_history_fish) >= 5:
            old_silver = _silver_price_history_fish[-5]["price"]
            if old_silver > 0:
                silver_5min_change = round((silver_price - old_silver) / old_silver * 100, 4)

        # --- Gold-leads-silver detection ---
        gold_leads_silver = {"direction": "NEUTRAL", "confidence": 0.0}
        if len(_gold_price_history) >= 5 and len(_silver_price_history_fish) >= 5:
            if gold_5min_change > 0.5 and silver_5min_change < 0.2:
                # Gold rallying but silver lagging => silver should follow up
                gap = gold_5min_change - silver_5min_change
                confidence = min(gap / 1.0, 1.0)  # 1.0% gap = full confidence
                gold_leads_silver = {
                    "direction": "LONG",
                    "confidence": round(confidence, 2),
                }
            elif gold_5min_change < -0.5 and silver_5min_change > -0.2:
                # Gold dropping but silver hasn't followed => silver should follow down
                gap = abs(gold_5min_change) - abs(silver_5min_change)
                confidence = min(gap / 1.0, 1.0)
                gold_leads_silver = {
                    "direction": "SHORT",
                    "confidence": round(confidence, 2),
                }

        # --- ORB range (once per day, after 10:00 UTC) ---
        now_utc = datetime.datetime.now(datetime.UTC)
        today_str = now_utc.strftime("%Y-%m-%d")
        if _orb_computed_today != today_str and now_utc.hour >= 10:
            try:
                from portfolio.orb_predictor import ORBPredictor
                predictor = ORBPredictor()
                # Fetch today's 15m klines from Binance FAPI
                params = {"symbol": "XAGUSDT", "interval": "15m", "limit": 96}
                r = requests.get(BINANCE_FAPI_KLINES, params=params, timeout=10)
                if r.status_code == 200:
                    raw = r.json()
                    parsed = predictor._parse_klines(raw)
                    # Filter to today's candles only
                    today_candles = [c for c in parsed if c["date"] == today_str]
                    morning = predictor.calculate_morning_range(today_candles)
                    if morning:
                        _orb_range_cache = {
                            "high": round(morning.high, 4),
                            "low": round(morning.low, 4),
                            "formed": True,
                        }
                        _orb_computed_today = today_str
                        log(f"Fish ORB computed: {morning.low:.2f}-{morning.high:.2f}")
                    else:
                        _orb_range_cache = {"high": 0, "low": 0, "formed": False}
                        _orb_computed_today = today_str  # don't retry endlessly
            except Exception as e:
                log(f"Fish ORB error (non-fatal): {e}")

        # --- Vol scalar (hourly refresh from signal ATR) ---
        now_ts = time.time()
        if now_ts - _vol_scalar_cache["ts"] >= 3600:
            try:
                klines = fetch_underlying_klines("XAG-USD", interval="1h", limit=24)
                if klines and len(klines) >= 14:
                    # Compute ATR(14) from hourly klines
                    trs = []
                    for i in range(1, len(klines)):
                        k = klines[i]
                        prev_close = klines[i - 1]["close"]
                        tr = max(
                            k["high"] - k["low"],
                            abs(k["high"] - prev_close),
                            abs(k["low"] - prev_close),
                        )
                        trs.append(tr)
                    current_atr = sum(trs[-14:]) / 14
                    if len(trs) >= 14:
                        import statistics as _stats
                        median_atr = _stats.median(trs)
                        if current_atr > 0:
                            vol_scalar = median_atr / current_atr
                            vol_scalar = max(0.25, min(2.0, vol_scalar))
                            _vol_scalar_cache = {
                                "value": round(vol_scalar, 3),
                                "ts": now_ts,
                            }
            except Exception as e:
                log(f"Fish vol_scalar error (non-fatal): {e}")

        # --- Mode suggestion ---
        mode = "momentum"  # default

        # Check yesterday close-to-close for big drop
        xag_hist = _underlying_history.get("XAG-USD", [])
        if len(xag_hist) >= 2:
            # Use oldest vs newest in history buffer (~60 entries = 1 hour)
            # For daily close-to-close, check the klines
            klines_daily = fetch_underlying_klines("XAG-USD", interval="1d", limit=2)
            if klines_daily and len(klines_daily) >= 2:
                prev_close = klines_daily[-2]["close"]
                curr_close = klines_daily[-1]["close"]
                if prev_close > 0:
                    daily_change_pct = (curr_close - prev_close) / prev_close * 100
                    if daily_change_pct < -5.0:
                        mode = "straddle"

        # Check signal flip frequency (>4 flips in last 10 checks)
        current_action = last_signal_data.get("XAG-USD", {}).get("action", "?")
        if current_action in ("BUY", "SELL", "HOLD"):
            _signal_action_history.append(current_action)
        if len(_signal_action_history) >= 5:
            flips = sum(
                1 for i in range(1, len(_signal_action_history))
                if _signal_action_history[i] != _signal_action_history[i - 1]
            )
            if flips > 4:
                mode = "straddle"

        # --- Write the precomputed file ---
        output = {
            "timestamp": now_utc.isoformat(),
            "gold_price": round(gold_price, 2),
            "silver_price": round(silver_price, 4),
            "gold_5min_change_pct": gold_5min_change,
            "silver_5min_change_pct": silver_5min_change,
            "gold_leads_silver": gold_leads_silver,
            "orb_range": _orb_range_cache,
            "vol_scalar": _vol_scalar_cache["value"],
            "mode_suggestion": mode,
        }
        atomic_write_json(str(DATA_DIR / "fish_precomputed.json"), output)

    except Exception as e:
        log(f"Fish precompute error (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Probability report — the core autonomous intelligence
# ---------------------------------------------------------------------------
_last_prob_report = {}  # cached probability report

def compute_probability_report():
    """Compute comprehensive probability report for XAG-USD and XAU-USD.

    Combines: signal consensus, Chronos forecasts, Monte Carlo price ranges,
    per-signal accuracy, and momentum analysis.

    Returns dict keyed by ticker with probability data.
    """
    global _last_prob_report
    report = {}

    for ticker in SIGNAL_TICKERS + ["MSTR"]:
        price = _underlying_prices.get(ticker, 0)
        if price <= 0:
            continue

        entry = {"ticker": ticker, "price": round(price, 4), "ts": time.time()}

        # --- Direction probability from signals ---
        sig = last_signal_data.get(ticker, {})
        buy_count = sig.get("buy_count", 0)
        sell_count = sig.get("sell_count", 0)
        voters = sig.get("voters", 0)
        if buy_count + sell_count > 0:
            entry["signal_up_pct"] = round(buy_count / (buy_count + sell_count) * 100, 1)
            entry["signal_action"] = sig.get("action", "HOLD")
            entry["signal_confidence"] = sig.get("weighted_confidence", 0)
        else:
            entry["signal_up_pct"] = 50.0
            entry["signal_action"] = "HOLD"
            entry["signal_confidence"] = 0
        entry["signal_buy_count"] = buy_count
        entry["signal_sell_count"] = sell_count
        entry["signal_rsi"] = sig.get("rsi", 0)
        entry["signal_regime"] = sig.get("regime", "?")

        # --- Chronos / LLM probability ---
        entry["chronos_1h"] = {"direction": "flat", "pct_move": 0, "confidence": 0}
        entry["chronos_3h"] = {"direction": "flat", "pct_move": 0, "confidence": 0}
        entry["ministral"] = {"action": "HOLD", "confidence": 0}
        entry["llm_consensus"] = {"direction": "flat", "confidence": 0}

        if LLM_AVAILABLE:
            try:
                llm_sigs = get_llm_signals()
                ticker_llm = llm_sigs.get(ticker, {})
                if ticker_llm:
                    for h in ["1h", "3h"]:
                        ckey = f"chronos_{h}"
                        if ticker_llm.get(ckey) and isinstance(ticker_llm[ckey], dict):
                            entry[ckey] = {
                                "direction": ticker_llm[ckey].get("direction", "flat"),
                                "pct_move": round(ticker_llm[ckey].get("pct_move", 0), 4),
                                "confidence": round(ticker_llm[ckey].get("confidence", 0), 3),
                            }
                    if ticker_llm.get("ministral") and isinstance(ticker_llm["ministral"], dict):
                        entry["ministral"] = {
                            "action": ticker_llm["ministral"].get("action", "HOLD"),
                            "confidence": round(ticker_llm["ministral"].get("confidence", 0), 3),
                        }
                    cons = ticker_llm.get("consensus", {})
                    if cons:
                        entry["llm_consensus"] = {
                            "direction": cons.get("direction", "flat"),
                            "confidence": round(cons.get("confidence", 0), 3),
                        }
            except Exception:
                pass

        # --- Combined direction probability ---
        up_score, down_score, total_weight = 0, 0, 0

        # Signal weight (based on accuracy)
        sig_weight = 1.0
        if buy_count + sell_count >= 3:
            if sig.get("action") == "BUY":
                up_score += sig_weight * buy_count / max(buy_count + sell_count, 1)
            elif sig.get("action") == "SELL":
                down_score += sig_weight * sell_count / max(buy_count + sell_count, 1)
            total_weight += sig_weight

        # Chronos weights
        for h in ["1h", "3h"]:
            c = entry.get(f"chronos_{h}", {})
            if c.get("direction") in ("up", "down"):
                w = 0.8  # Chronos weight
                if c["direction"] == "up":
                    up_score += w * c.get("confidence", 0.5)
                else:
                    down_score += w * c.get("confidence", 0.5)
                total_weight += w

        # Ministral weight
        m = entry.get("ministral", {})
        if m.get("action") in ("BUY", "SELL"):
            w = 0.6
            if m["action"] == "BUY":
                up_score += w * m.get("confidence", 0.5)
            else:
                down_score += w * m.get("confidence", 0.5)
            total_weight += w

        # Crypto-specific: Fear & Greed as contrarian weight
        if ticker in ("BTC-USD", "ETH-USD") and CRYPTO_DATA_AVAILABLE:
            try:
                fg = get_fear_greed()
                if fg:
                    fg_val = fg["value"]
                    entry["fear_greed"] = fg
                    w = 0.4  # moderate weight
                    if fg_val <= 20:  # extreme fear → contrarian buy
                        up_score += w * 0.7
                        total_weight += w
                    elif fg_val >= 80:  # extreme greed → contrarian sell
                        down_score += w * 0.7
                        total_weight += w
            except Exception:
                pass

            # On-chain bias for BTC
            if ticker == "BTC-USD":
                try:
                    onchain = get_onchain_summary()
                    if onchain:
                        entry["onchain"] = onchain
                        bias = onchain.get("bias", "neutral")
                        if bias in ("bullish", "bearish"):
                            w = 0.3
                            if bias == "bullish":
                                up_score += w * 0.6
                            else:
                                down_score += w * 0.6
                            total_weight += w
                except Exception:
                    pass

        # MSTR: price-only, no signals — use BTC correlation as proxy
        if ticker == "MSTR":
            btc_report = report.get("BTC-USD", {})
            if btc_report:
                btc_up = btc_report.get("prob_up_pct", 50)
                btc_down = btc_report.get("prob_down_pct", 50)
                # MSTR roughly tracks BTC with ~1.5x beta
                w = 0.5
                if btc_up > btc_down:
                    up_score += w * (btc_up / 100)
                else:
                    down_score += w * (btc_down / 100)
                total_weight += w
                entry["btc_proxy"] = True

        if total_weight > 0:
            prob_up = round(up_score / total_weight * 100, 1)
            prob_down = round(down_score / total_weight * 100, 1)
        else:
            prob_up, prob_down = 50.0, 50.0

        entry["prob_up_pct"] = prob_up
        entry["prob_down_pct"] = prob_down

        # --- Momentum ---
        entry["momentum"] = get_underlying_momentum(ticker)

        # --- Monte Carlo price ranges (1h, 3h from underlying) ---
        from metals_risk import ATR_DEFAULTS
        atr_pct = sig.get("atr_pct", ATR_DEFAULTS.get(ticker, 3.0))
        try:
            import numpy as np
            from metals_risk import _annualized_vol_from_atr
            vol_annual = _annualized_vol_from_atr(atr_pct)
            vol_annual = max(vol_annual, 0.05)
            rng = np.random.default_rng(seed=int(time.time()) % 10000)

            for hours, h_key in [(1, "1h"), (3, "3h"), (8, "8h")]:
                t = hours / (252 * 24)  # metals trade ~24h
                half = 2500
                z = rng.standard_normal(half)
                z_full = np.concatenate([z, -z])
                log_ret = -0.5 * vol_annual**2 * t + vol_annual * (t**0.5) * z_full
                terminal = price * np.exp(log_ret)
                pcts = np.percentile(terminal, [5, 25, 50, 75, 95])
                entry[f"range_{h_key}"] = {
                    "p5": round(float(pcts[0]), 2),
                    "p25": round(float(pcts[1]), 2),
                    "p50": round(float(pcts[2]), 2),
                    "p75": round(float(pcts[3]), 2),
                    "p95": round(float(pcts[4]), 2),
                    "expected_move_pct": round(float(np.std((terminal / price - 1) * 100)), 2),
                }
        except (ImportError, Exception) as e:
            for h_key in ["1h", "3h", "8h"]:
                entry[f"range_{h_key}"] = {"error": str(e)}

        # --- Per-signal accuracy ---
        entry["accuracy"] = {}
        if TRACKER_AVAILABLE:
            try:
                acc = get_accuracy_report()
                short_t = ticker.split("-")[0]
                for k, v in acc.items():
                    if short_t in k and v.get("total", 0) >= 5:
                        entry["accuracy"][k] = {
                            "pct": round(v["accuracy"] * 100, 1),
                            "samples": v["total"],
                        }
            except Exception:
                pass

        # --- Individual signal votes (from vote_detail) ---
        vote_detail = sig.get("vote_detail", "")
        if vote_detail:
            entry["per_signal_votes"] = _parse_vote_detail(vote_detail)

        report[ticker] = entry

    _last_prob_report = report
    return report


def _parse_vote_detail(vote_detail):
    """Parse vote_detail string like 'B:sentiment,volume_flow | S:mean_reversion' into dict."""
    result = {"buy": [], "sell": []}
    try:
        parts = vote_detail.split("|")
        for part in parts:
            part = part.strip()
            if part.startswith("B:"):
                signals = [s.strip() for s in part[2:].split(",") if s.strip()]
                result["buy"] = signals
            elif part.startswith("S:"):
                signals = [s.strip() for s in part[2:].split(",") if s.strip()]
                result["sell"] = signals
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Probability-focused Telegram messages
# ---------------------------------------------------------------------------

def _format_price(price, ticker):
    """Format price compactly: $67.2K for BTC, $1,996 for ETH, $33.45 for metals."""
    if price >= 10000:
        return f"${price/1000:.1f}K"
    elif price >= 1000:
        return f"${price:,.0f}"
    else:
        return f"${price:.2f}"


def build_probability_telegram(prob_report, cet_str):
    """Build a unified probability-focused Telegram message for all instruments.

    Format: metals first (with warrant P&L), then crypto (with F&G), then MSTR.
    Apple Watch first line shows top 2 movers by probability deviation from 50%.
    """
    if not prob_report:
        return None

    # --- First line: Apple Watch — top 2 movers by deviation from 50% ---
    ticker_devs = []
    for ticker, r in prob_report.items():
        prob_up = r.get("prob_up_pct", 50)
        prob_down = r.get("prob_down_pct", 50)
        dev = max(abs(prob_up - 50), abs(prob_down - 50))
        if prob_up > prob_down:
            arrow, prob = "↑", prob_up
        elif prob_down > prob_up:
            arrow, prob = "↓", prob_down
        else:
            arrow, prob = "→", 50
        short = ticker.split("-")[0] if "-" in ticker else ticker
        ticker_devs.append((dev, f"{short} {arrow}{prob:.0f}%"))
    ticker_devs.sort(key=lambda x: -x[0])
    watch_parts = [d[1] for d in ticker_devs[:2]]

    # Add F&G to first line if available
    fg_tag = ""
    if CRYPTO_DATA_AVAILABLE:
        try:
            fg = get_fear_greed()
            if fg:
                fg_tag = f" · F&G {fg['value']}"
        except Exception:
            pass

    first_line = f"*PROB* · {' · '.join(watch_parts)}{fg_tag}"
    lines = [first_line, ""]

    # --- Per-ticker probability data ---
    # Order: metals, crypto, then MSTR
    ticker_order = ["XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD", "MSTR"]
    for ticker in ticker_order:
        r = prob_report.get(ticker)
        if not r:
            continue

        price = r.get("price", 0)
        short = ticker.split("-")[0] if "-" in ticker else ticker
        prob_up = r.get("prob_up_pct", 50)
        prob_down = r.get("prob_down_pct", 50)

        price_str = _format_price(price, ticker)
        lines.append(f"`{short:<4} {price_str}  ↑{prob_up:.0f}%  ↓{prob_down:.0f}%`")

        # Chronos forecasts (metals + crypto, not MSTR)
        if ticker != "MSTR":
            chr_parts = []
            for h in ["1h", "3h"]:
                c = r.get(f"chronos_{h}", {})
                if c.get("direction") in ("up", "down"):
                    arrow = "↑" if c["direction"] == "up" else "↓"
                    pct = abs(c.get("pct_move", 0))
                    chr_parts.append(f"{arrow}{pct:.2f}% {h}")
            if chr_parts:
                lines.append(f"`  chr: {' | '.join(chr_parts)}`")

        # Signal detail (for tickers with signals)
        if ticker in SIGNAL_TICKERS:
            sig = r.get("signal_action", "HOLD")
            buy_c = r.get("signal_buy_count", 0)
            sell_c = r.get("signal_sell_count", 0)
            rsi = r.get("signal_rsi", 0)
            regime = r.get("signal_regime", "?")
            lines.append(f"`  sig: {sig} {buy_c}B/{sell_c}S RSI:{rsi:.0f} {regime}`")

        # Crypto-specific context
        if ticker in ("BTC-USD", "ETH-USD"):
            fg_data = r.get("fear_greed")
            if fg_data:
                lines.append(f"`  F&G: {fg_data['value']} ({fg_data['classification']})`")
            onchain = r.get("onchain")
            if onchain and onchain.get("mvrv"):
                lines.append(f"`  MVRV: {onchain['mvrv']:.2f} ({onchain.get('zone', '?')})`")
            # ETH/BTC ratio
            if ticker == "ETH-USD":
                btc_p = _underlying_prices.get("BTC-USD", 0)
                if btc_p > 0:
                    ratio = price / btc_p
                    lines.append(f"`  ETH/BTC: {ratio:.4f}`")

        # MSTR-specific context
        if ticker == "MSTR":
            if CRYPTO_DATA_AVAILABLE:
                try:
                    mstr_data = fetch_mstr_price()
                    if mstr_data:
                        chg = mstr_data.get("change_pct", 0)
                        state = mstr_data.get("market_state", "CLOSED")
                        state_tag = "" if state == "REGULAR" else f" ({state.lower()})"
                        lines.append(f"`  {chg:+.1f}% today{state_tag}`")
                    btc_p = _underlying_prices.get("BTC-USD", 0)
                    if btc_p > 0:
                        nav = compute_mstr_btc_nav(price, btc_p)
                        if nav:
                            lines.append(f"`  NAV: ${nav['nav_per_share']:.0f} prem:{nav['premium_pct']:+.0f}%`")
                except Exception:
                    pass

        lines.append("")

    # --- Held positions (Avanza warrants) ---
    active_positions = {k: p for k, p in POSITIONS.items() if p.get("active")}
    if active_positions:
        lines.append("_Held:_")
        for key, pos in active_positions.items():
            bid = 0
            if price_history:
                bid = price_history[-1].get(key, 0)
            if bid <= 0:
                bid = pos["entry"]  # fallback
            pnl = pnl_pct(bid, pos["entry"])
            dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999

            # Momentum-aware stop info
            k_lower = key.lower()
            if "silver" in k_lower:
                underlying_ticker = "XAG-USD"
            elif "gold" in k_lower:
                underlying_ticker = "XAU-USD"
            else:
                underlying_ticker = "XAG-USD"  # fallback
            mom = get_underlying_momentum(underlying_ticker)
            trail_tag = ""
            if mom["velocity_pct"] < -0.01:
                trail_tag = " ⚡"
            if mom["acceleration"] < -0.0001:
                trail_tag = " ⚡⚡"

            short_key = key[:8]
            lines.append(
                f"`  {short_key} {pos['units']}u b:{bid:.2f} "
                f"{pnl:+.1f}% stop:{pos['stop']} ({dist_stop:.1f}%){trail_tag}`"
            )
    else:
        lines.append("_No held positions (monitoring only)_")

    lines.append("")
    lines.append(f"_#{check_count} · {cet_str} · {CHECK_INTERVAL}s loop_")

    return "\n".join(lines)


def read_decision_history(n=5):
    """Read the last N decisions from metals_decisions.jsonl."""
    try:
        path = "data/metals_decisions.jsonl"
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
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
    if not EMERGENCY_SELL_ENABLED:
        log(f"[L3 DISABLED] Skipping emergency sell for {key} at {bid}")
        return False

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
        now_ts = datetime.datetime.now(datetime.UTC).isoformat()
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
        atomic_append_jsonl("data/metals_trades.jsonl", trade)

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
            # Ambiguous broker response: can mean already sold OR account/order mismatch.
            # Confirm with live holdings before deactivating local state.
            held_confirmed = False
            try:
                verify = page.evaluate("""async (args) => {
                    const [accountId, orderbookId] = args;
                    const resp = await fetch(
                        'https://www.avanza.se/_api/position-data/positions',
                        {credentials: 'include'}
                    );
                    if (resp.status !== 200) return {held: null, status: resp.status};
                    const data = await resp.json();
                    for (const item of (data.withOrderbook || [])) {
                        if (
                            item.account && String(item.account.id) === String(accountId) &&
                            item.instrument && item.instrument.orderbook &&
                            String(item.instrument.orderbook.id) === String(orderbookId)
                        ) {
                            const vol = item.volume && item.volume.value != null ? item.volume.value : 0;
                            return {held: vol > 0, units: vol};
                        }
                    }
                    return {held: false, units: 0};
                }""", [ACCOUNT_ID, pos["ob_id"]])
                held_confirmed = bool(verify and verify.get("held") is True)
                if held_confirmed and verify.get("units") is not None:
                    pos["units"] = int(verify["units"])
            except Exception as e:
                log(f"  {key}: holdings re-check failed after short-sell-not-allowed: {e}")

            if held_confirmed:
                log(f"  {key}: short-sell-not-allowed but position still held — keeping active")
                send_telegram(f"*L3 WARNING* {pos['name']}: SELL rejected but holding is still live. Kept active.")
                return False

            log(f"  {key}: short-sell-not-allowed and not held — deactivating")
            send_telegram(f"*L3* {pos['name']}: no longer held, deactivating")
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
    return _load_json_state(STOP_ORDER_FILE, {}, "Stop order state")

def _save_stop_orders(state):
    """Save stop order state to disk."""
    try:
        atomic_write_json(STOP_ORDER_FILE, state, ensure_ascii=False)
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
    return _load_json_state(TRADE_QUEUE_FILE, _default_trade_queue(), "Trade queue")


def _save_trade_queue(queue):
    """Save trade queue to disk."""
    try:
        atomic_write_json(TRADE_QUEUE_FILE, queue, ensure_ascii=False)
    except OSError as e:
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
            order["executed_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
        _save_trade_queue(queue)
        return

    now = datetime.datetime.now(datetime.UTC)

    for order in pending:
        order_id_short = order.get("id", "?")[:8]
        action = order.get("action", "?")
        warrant_name = order.get("warrant_name", order.get("warrant_key", "?"))

        # --- Age check ---
        try:
            order_ts = datetime.datetime.fromisoformat(order["timestamp"])
            if order_ts.tzinfo is None:
                order_ts = order_ts.replace(tzinfo=datetime.UTC)
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
                        other_ts = other_ts.replace(tzinfo=datetime.UTC)
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
                atomic_append_jsonl("data/metals_trades.jsonl", trade_entry)
            except OSError as e:
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
            "bought_ts": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        peak_bids[pos_key] = exec_price
        last_invoke_prices[pos_key] = exec_price
        log(f"  New position added: {pos_key} = {order['volume']}u @ {exec_price}")

    _save_positions(POSITIONS)

    # --- Hardware trailing stop (Avanza-managed, no Playwright needed) ---
    if HARDWARE_TRAILING_ENABLED:
        vol = POSITIONS[pos_key]["units"]
        ob_id_str = POSITIONS[pos_key].get("ob_id", order.get("ob_id"))
        try:
            from portfolio.avanza_session import place_trailing_stop as _place_hw_trail
            result = _place_hw_trail(
                orderbook_id=ob_id_str,
                trail_percent=HARDWARE_TRAILING_PCT,
                volume=vol,
                account_id=ACCOUNT_ID,
                valid_days=HARDWARE_TRAILING_VALID_DAYS,
            )
            if result.get("status") == "SUCCESS":
                hw_stop_id = result.get("stoplossOrderId", "?")
                POSITIONS[pos_key]["hw_trailing_stop_id"] = hw_stop_id
                _save_positions(POSITIONS)
                log(f"  HW trailing stop placed for {pos_key}: {HARDWARE_TRAILING_PCT}% trail, "
                    f"vol={vol} [stoploss {hw_stop_id}]")
                send_telegram(f"Trailing stop placed: {POSITIONS[pos_key]['name']} "
                              f"{HARDWARE_TRAILING_PCT}% trail, {vol}u")
            else:
                log(f"  HW trailing stop FAILED for {pos_key}: {result}")
                send_telegram(f"*WARNING* Hardware trailing stop failed for "
                              f"{POSITIONS[pos_key]['name']} — set manually!")
        except Exception as e:
            log(f"  HW trailing stop error for {pos_key}: {e}")
            send_telegram(f"*WARNING* Hardware trailing stop error for "
                          f"{POSITIONS[pos_key]['name']}: {e}")

    # Legacy cascade stop-loss (only if hardware trailing is OFF)
    if STOP_ORDER_ENABLED and not HARDWARE_TRAILING_ENABLED:
        stop_trigger = order.get("stop_trigger")
        stop_sell = order.get("stop_sell")
        if stop_trigger and stop_sell and order["volume"] > 0:
            vol = POSITIONS[pos_key]["units"]
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
        pos["sold_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
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

        # Safety: fetch current bid to verify stop is not too close
        cur_price_data = fetch_price(page, pos["ob_id"], pos.get("api_type", "warrant"))
        cur_bid = (cur_price_data or {}).get("bid", 0)
        if cur_bid > 0:
            distance_pct = (cur_bid - stop_base) / cur_bid * 100
            if distance_pct < 3.0:
                log(f"  SKIP stop for {key}: trigger {stop_base} is only {distance_pct:.1f}% "
                    f"below bid {cur_bid} — too close, would trigger immediately")
                continue

        for level in range(STOP_ORDER_LEVELS):
            # Calculate trigger price for this level
            spread = level * STOP_ORDER_SPREAD_PCT / 100.0
            trigger_price = round(stop_base * (1 - spread), 2)
            # Sell price slightly below trigger (1% slippage buffer)
            sell_price = round(trigger_price * 0.99, 2)

            # Calculate units for this level (split evenly, last gets remainder)
            if level < STOP_ORDER_LEVELS - 1:
                level_units = units // STOP_ORDER_LEVELS
            else:
                level_units = units - (units // STOP_ORDER_LEVELS) * (STOP_ORDER_LEVELS - 1)

            if level_units <= 0:
                continue

            # Use the CORRECT stop-loss API (not regular order API!)
            # Regular order API places immediate sell orders; stop-loss API
            # uses triggerPrice to only activate when price drops to that level.
            try:
                ok, stop_id = place_stop_loss(
                    page, ACCOUNT_ID, pos["ob_id"],
                    trigger_price=trigger_price,
                    sell_price=sell_price,
                    volume=level_units,
                    valid_days=8,
                )

                if ok:
                    orders.append({
                        "level": level + 1,
                        "order_id": stop_id,
                        "trigger": trigger_price,
                        "sell": sell_price,
                        "units": level_units,
                        "status": "placed",
                    })
                    log(f"  Stop S{level+1} placed: {key} {level_units}u trigger={trigger_price} "
                        f"sell={sell_price} [stoploss {stop_id}]")
                else:
                    log(f"  Stop S{level+1} FAILED: {key} trigger={trigger_price}")
                    orders.append({
                        "level": level + 1,
                        "trigger": trigger_price,
                        "sell": sell_price,
                        "units": level_units,
                        "status": "failed",
                    })
            except Exception as e:
                log(f"  Stop S{level+1} error: {key} — {e}")

        state[key] = {
            "date": today_str,
            "stop_base": stop_base,
            "orders": orders,
            "placed_ts": datetime.datetime.now(datetime.UTC).isoformat(),
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
            # Try stop-loss cancel endpoint first, fall back to regular order cancel
            result = page.evaluate("""async (args) => {
                const [orderId, token] = args;
                // Stop-loss cancel endpoint
                let resp = await fetch(
                    'https://www.avanza.se/_api/trading/stoploss/' + orderId,
                    {method: 'DELETE', headers: {'Content-Type': 'application/json', 'X-SecurityToken': token}, credentials: 'include'}
                );
                if (resp.status === 404) {
                    // Fall back to regular order cancel
                    resp = await fetch(
                        'https://www.avanza.se/_api/trading-critical/rest/order/delete/' + orderId,
                        {method: 'DELETE', headers: {'Content-Type': 'application/json', 'X-SecurityToken': token}, credentials: 'include'}
                    );
                }
                return {status: resp.status};
            }""", [order_id, csrf])
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
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                "action": "STOP_ORDER_SELL",
                "position": key,
                "name": pos["name"],
                "units": total_filled_units,
                "price": state["stop_base"],
                "entry": pos["entry"],
                "pnl_pct": round(pnl_pct(state["stop_base"], pos["entry"]), 2),
            }
            atomic_append_jsonl("data/metals_trades.jsonl", trade)

            if remaining <= 0:
                pos["active"] = False
                pos["sold_ts"] = datetime.datetime.now(datetime.UTC).isoformat()
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
                        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                        "action": "SPIKE_SELL",
                        "position": key,
                        "units": target.get("units_to_sell"),
                        "price": target.get("target_price"),
                        "entry": positions.get(key, {}).get("entry"),
                        "pnl_pct": pnl,
                        "reason": target.get("reason", "US open spike capture"),
                    }
                    atomic_append_jsonl("data/metals_trades.jsonl", trade)

        except Exception as e:
            log(f"Spike check error {key}: {e}")

    return filled


def log_invocation(tier, model, trigger, check_num, invoke_num, elapsed_s=None, rc=None):
    """Log a Claude invocation to the invocations JSONL file."""
    entry = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "tier": tier,
        "model": model or "sonnet",
        "trigger": trigger,
        "check_count": check_num,
        "invoke_count": invoke_num,
        "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
        "return_code": rc,
    }
    try:
        atomic_append_jsonl(INVOCATION_LOG, entry)
    except Exception:
        pass


def write_context(prices, trigger_reason, tier=2):
    """Write context JSON for Claude Layer 2."""
    now = datetime.datetime.now(datetime.UTC)
    hours_remaining = hours_to_metals_close(now) if EXECUTION_ENGINE_AVAILABLE else round(
        max(0, EOD_HOUR_CET + 25 / 60 - cet_hour()), 1
    )
    ctx = {
        "timestamp": now.isoformat(),
        "cet_time": cet_time_str(),
        "check_count": check_count,
        "invoke_count": invoke_count,
        "trigger_reason": trigger_reason,
        "tier": tier,
        "market_close_cet": "21:55",
        "hours_remaining": round(hours_remaining, 1),
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
        with open("data/metals_history.json", encoding="utf-8") as f:
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

    if EXECUTION_ENGINE_AVAILABLE:
        try:
            llm_sigs = get_llm_signals() if LLM_AVAILABLE else None
            ctx["execution_targets"] = build_execution_recommendations(
                POSITIONS,
                prices,
                signal_data=last_signal_data,
                llm_signals=llm_sigs,
                warrant_catalog=cached_warrant_catalog or None,
                account=cached_account_data or None,
                hours_remaining=hours_remaining,
            )
        except Exception as e:
            ctx["execution_targets"] = {"error": str(e)}

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
    """Classify trigger into tier (1=cheap workhorse, 2=deeper analysis, 3=critical).

    Cost optimization: only genuine emergencies get T3 (Opus). Everything else is T1 (Haiku).
    """
    # T3 only for genuine emergencies — exact string matches, not substrings
    if any("L3 EMERGENCY" in r or "L2 ALERT" in r or "EMERGENCY drawdown" in r
           or "AUTO-EXIT" in r for r in reasons):
        return 3

    # T2 for end-of-day or profit target (needs deeper analysis)
    if any("end_of_day" in r or "profit target" in r for r in reasons):
        return 2

    # Everything else is T1 (Haiku) — price moves, trailing, heartbeat, LLM consensus
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
            # L1 is log-only — does NOT trigger Claude invocation (cost optimization)
            log(f"L1 WARNING: {key} {dist_stop:.1f}% from stop (log only, no invocation)")

        # Reset L2 counter when not in L2 danger zone
        if dist_stop >= STOP_L2_PCT:
            l2_zone_checks.pop(key, None)

    # Signal flip detection (all tracked tickers with signals)
    if last_signal_data:
        for ticker in SIGNAL_TICKERS:
            if ticker in last_signal_data:
                current_action = last_signal_data[ticker].get("action", "?")
                prev_action = prev_signal_actions.get(ticker)
                if prev_action and current_action != prev_action and current_action in ("BUY", "SELL"):
                    reasons.append(f"signal flip {ticker}: {prev_action}->{current_action}")
                prev_signal_actions[ticker] = current_action

    # Crypto price move triggers (from underlying price history)
    _CRYPTO_TRIGGER_PCT = {"BTC-USD": 3.0, "ETH-USD": 3.0, "MSTR": 5.0}
    for ticker, threshold in _CRYPTO_TRIGGER_PCT.items():
        hist = _underlying_history.get(ticker, [])
        if len(hist) < 2:
            continue
        current_p = hist[-1]["price"]
        # Compare against price 10 checks ago (or oldest available)
        ref_idx = max(0, len(hist) - 10)
        ref_p = hist[ref_idx]["price"]
        if ref_p > 0:
            move_pct = abs((current_p - ref_p) / ref_p * 100)
            if move_pct >= threshold:
                reasons.append(f"{ticker} moved {move_pct:.1f}% (last 10 checks)")

    # Fear & Greed extreme trigger
    if CRYPTO_DATA_AVAILABLE and check_count > 5:
        try:
            fg = get_fear_greed()
            if fg:
                fg_val = fg["value"]
                if fg_val <= 10 or fg_val >= 85:
                    reasons.append(f"F&G extreme: {fg_val} ({fg['classification']})")
        except Exception:
            pass

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

def _make_autonomous_prediction(signals_data, llm_data):
    """Generate an autonomous prediction from signals + LLM for accuracy tracking."""
    directions = []  # (direction, weight)

    for _ticker, sig in signals_data.items():
        action = sig.get("action", "HOLD")
        weight = sig.get("weighted_confidence")
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = None
        if weight is None:
            buy_count = sig.get("buy_count", 0)
            sell_count = sig.get("sell_count", 0)
            total = buy_count + sell_count
            if total == 0:
                continue
            if action == "BUY":
                weight = buy_count / total
            elif action == "SELL":
                weight = sell_count / total
            else:
                continue
        weight = max(0.0, min(weight, 1.0))
        if weight == 0:
            continue
        if action == "BUY":
            directions.append(("up", weight))
        elif action == "SELL":
            directions.append(("down", weight))

    for ticker, data in llm_data.items():
        if ticker.startswith("_"):
            continue
        consensus_dir = data.get("consensus")
        consensus_conf = data.get("consensus_conf", 0)
        if consensus_dir in ("BUY", "SELL"):
            directions.append(("up" if consensus_dir == "BUY" else "down", consensus_conf))
        # Chronos 3h
        chr_dir = data.get("chronos_3h")
        if chr_dir in ("up", "down"):
            directions.append((chr_dir, abs(data.get("chronos_3h_pct", 0)) * 10))

    if not directions:
        return {"action": "HOLD", "direction": "flat", "confidence": 0.0, "horizon": "3h"}

    up_w = sum(w for d, w in directions if d == "up")
    down_w = sum(w for d, w in directions if d == "down")
    total_w = up_w + down_w
    if total_w < 0.1:
        return {"action": "HOLD", "direction": "flat", "confidence": 0.0, "horizon": "3h"}

    if up_w > down_w:
        direction, confidence = "up", round(up_w / total_w, 2)
    else:
        direction, confidence = "down", round(down_w / total_w, 2)

    action = "HOLD"
    if confidence >= 0.7:
        action = "BUY" if direction == "up" else "SELL"

    return {
        "action": action, "direction": direction,
        "confidence": confidence, "horizon": "3h",
        "up_weight": round(up_w, 2), "down_weight": round(down_w, 2),
    }


def _assess_thesis(positions_data, signals_data, trigger_reasons):
    """Assess whether the strategic thesis (silver bull 2026) is intact."""
    threats, supports = [], []

    for key, pos in positions_data.items():
        if pos["pnl_pct"] < -15:
            threats.append(f"{key} deep drawdown {pos['pnl_pct']:+.1f}%")
        if pos["dist_stop_pct"] < 5:
            threats.append(f"{key} near stop ({pos['dist_stop_pct']:.1f}%)")
        if pos["pnl_pct"] > 5:
            supports.append(f"{key} profitable")

    xag = signals_data.get("XAG-USD", {})
    if xag.get("action") == "SELL" and xag.get("sell_count", 0) >= 4:
        threats.append(f"XAG strong SELL ({xag['sell_count']}S)")
    elif xag.get("action") == "BUY" and xag.get("buy_count", 0) >= 3:
        supports.append(f"XAG BUY ({xag['buy_count']}B)")

    if any("EMERGENCY" in r or "AUTO-EXIT" in r for r in trigger_reasons):
        threats.append("Emergency trigger")

    if threats and not supports:
        return "THREATENED"
    elif threats:
        return "MIXED"
    elif supports:
        return "INTACT"
    return "NEUTRAL"


def _build_autonomous_telegram(trigger_reasons, tier, positions_data, signals_data,
                                llm_data, risk_data, prediction, thesis_status,
                                cet_str, is_emergency):
    """Build a rich Telegram message for autonomous mode."""
    action_str = prediction.get("action", "HOLD")
    emergency_tag = " EMG" if is_emergency else ""

    def _fmt_num(value):
        try:
            value = float(value)
        except Exception:
            return str(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")

    # First line: Apple Watch
    headline_detail = ""
    for key, pos in positions_data.items():
        if "silver" in key.lower():
            headline_detail = f"Silver bid {_fmt_num(pos['bid'])}"
            break
    if not headline_detail and positions_data:
        _, first_pos = next(iter(positions_data.items()))
        headline_detail = f"Bid {_fmt_num(first_pos['bid'])}"
    if not headline_detail:
        headline_detail = "Monitoring positions"
    first_line = (
        f"*AUTO {action_str}{emergency_tag}* · "
        f"{headline_detail} · {humanize_thesis_status(thesis_status)}"
    )

    # Position lines
    pos_lines = []
    for key, pos in positions_data.items():
        display_name = pos.get("name") or humanize_ticker(key).upper()
        mc_key = f"{key}_mc_pstop3h"
        pos_lines.append(f"`{display_name} · {pos['units']} units`")
        pos_lines.append(
            f"`Entry {_fmt_num(pos['entry'])} · Bid {_fmt_num(pos['bid'])} · "
            f"Profit/loss {pos['pnl_pct']:+.1f}% · Off peak {pos['from_peak_pct']:+.1f}%`"
        )
        stop_line = f"`Stop-loss {_fmt_num(pos['stop'])} ({pos['dist_stop_pct']:.1f}% away)"
        if mc_key in risk_data:
            stop_line += f" · Monte Carlo stop-hit risk {risk_data[mc_key]:.1f}%"
        stop_line += "`"
        pos_lines.append(stop_line)

    # Signal line
    sig_parts = []
    for ticker, sig in signals_data.items():
        short_t = humanize_ticker(ticker)
        votes = format_vote_summary(sig.get("buy_count", 0), sig.get("sell_count", 0))
        action = sig.get("action", "?")
        raw_action = sig.get("raw_action") or action
        if raw_action != action:
            conf = sig.get("weighted_confidence")
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = None
            conf_part = f"{conf:.0%} confidence; " if conf is not None else ""
            sig_parts.append(f"{short_t} {action} ({conf_part}raw {raw_action}, {votes} votes)")
        else:
            sig_parts.append(f"{short_t} {action} ({votes} votes)")
    sig_line = f"Signals: {' | '.join(sig_parts[:3])}" if sig_parts else ""

    # LLM line
    llm_parts = []
    for ticker, data in llm_data.items():
        if ticker.startswith("_"):
            continue
        ticker_parts = []
        if "ministral" in data:
            conf = data.get("ministral_conf", 0)
            ticker_parts.append(f"Ministral {data['ministral']} ({conf:.0%})")
        if "chronos_3h" in data:
            pct = data.get("chronos_3h_pct", 0)
            direction = data["chronos_3h"]
            move = f"{pct:+.1f}%" if direction in ("up", "down") else f"{abs(pct):.1f}%"
            ticker_parts.append(f"Chronos {direction} {move} over 3h")
        if ticker_parts:
            llm_parts.append(f"{humanize_ticker(ticker)}: {', '.join(ticker_parts)}")
    llm_line = f"AI view: {' | '.join(llm_parts[:3])}" if llm_parts else ""

    # Risk line
    risk_parts = []
    dd_pct = risk_data.get("drawdown_pct")
    if dd_pct is not None:
        risk_parts.append(f"Portfolio drawdown {dd_pct:+.1f}%")
    risk_line = f"Risk: {' | '.join(risk_parts)}" if risk_parts else ""

    # Assemble
    lines = [first_line, ""]
    if pos_lines:
        lines.extend(pos_lines)
        lines.append("")
    if sig_line:
        lines.append(sig_line)
    if llm_line:
        lines.append(llm_line)
    if risk_line:
        lines.append(risk_line)
    lines.append("")
    lines.append(format_tier_footer("Autonomous", tier, check_count, cet_str))

    return "\n".join(lines)


def _build_autonomous_positions_data():
    """Build a compact snapshot of active positions for autonomous decisions."""
    latest_prices = price_history[-1] if price_history else {}
    positions_data = {}

    for key, pos in POSITIONS.items():
        if not pos.get("active"):
            continue

        bid = latest_prices.get(key, 0) or pos.get("entry", 0)
        peak = peak_bids.get(key, bid if bid > 0 else 0)
        pnl = pnl_pct(bid, pos["entry"]) if bid > 0 else 0
        from_peak = pnl_pct(bid, peak) if peak > 0 and bid > 0 else 0
        dist_stop = ((bid - pos["stop"]) / bid * 100) if bid > 0 else 999

        positions_data[key] = {
            "name": pos["name"],
            "units": pos["units"],
            "entry": pos["entry"],
            "bid": bid,
            "stop": pos["stop"],
            "pnl_pct": round(pnl, 2),
            "from_peak_pct": round(from_peak, 2),
            "dist_stop_pct": round(dist_stop, 2),
        }

    return positions_data


def _build_autonomous_risk_data(positions_data, llm_signals):
    """Flatten the risk summary into the small shape used by Telegram/logging."""
    if not RISK_AVAILABLE or not positions_data:
        return {}

    prices = {key: {"bid": pos["bid"]} for key, pos in positions_data.items()}
    risk_data = {}
    try:
        drawdown = check_portfolio_drawdown(POSITIONS, prices)
        drawdown_pct = drawdown.get("current_drawdown_pct")
        if isinstance(drawdown_pct, (int, float)):
            risk_data["drawdown_pct"] = round(drawdown_pct, 2)
    except Exception as e:
        log(f"Autonomous drawdown summary failed: {e}")

    return risk_data


def _autonomous_decision(trigger_reasons, blocked_tier):
    """Handle triggers autonomously — probability-focused decision + Telegram.

    Uses probability report (signals + Chronos + Monte Carlo + momentum) to
    produce a decision log entry and probability-focused Telegram notification.
    Does NOT trade — only the SwingTrader can execute trades.
    """
    EMERGENCY_PATTERNS = ["L3 EMERGENCY", "EMERGENCY drawdown", "AUTO-EXIT"]
    is_emergency = any(p in r for r in trigger_reasons for p in EMERGENCY_PATTERNS)
    if is_emergency and CLAUDE_ENABLED:
        log("Emergency trigger with Claude enabled — escalating to Tier 3")
        invoke_claude(trigger_reasons, tier=3)
        return
    if is_emergency:
        log("EMERGENCY in autonomous mode — Layer 1 handles execution, logging assessment")

    now = datetime.datetime.now(datetime.UTC)
    reason_str = "; ".join(trigger_reasons[:5])
    cet_str = cet_time_str()

    # --- 1. Autonomous prediction (for accuracy tracking) ---
    positions_data = _build_autonomous_positions_data()
    signals_data = {}
    llm_data = {}
    llm_signals = None
    if last_signal_data:
        for ticker in SIGNAL_TICKERS:
            if ticker in last_signal_data:
                s = last_signal_data[ticker]
                signals_data[ticker] = {
                    "action": s.get("action", "?"),
                    "confidence": s.get("confidence", 0),
                    "weighted_confidence": s.get("weighted_confidence", 0),
                    "raw_action": s.get("raw_action", s.get("action", "?")),
                    "raw_confidence": s.get("raw_confidence", s.get("confidence", 0)),
                    "buy_count": s.get("buy_count", 0),
                    "sell_count": s.get("sell_count", 0),
                    "rsi": s.get("rsi"),
                    "macd_hist": s.get("macd_hist"),
                    "regime": s.get("regime", "?"),
                }
    if LLM_AVAILABLE:
        try:
            llm_signals = get_llm_signals()
            for ticker, data in llm_signals.items():
                if not isinstance(data, dict):
                    continue
                llm_entry = {}
                if data.get("ministral") and isinstance(data["ministral"], dict):
                    llm_entry["ministral"] = data["ministral"].get("action", "?")
                    llm_entry["ministral_conf"] = round(data["ministral"].get("confidence", 0), 2)
                for h in ["1h", "3h"]:
                    ckey = f"chronos_{h}"
                    if data.get(ckey) and isinstance(data[ckey], dict):
                        llm_entry[ckey] = data[ckey].get("direction", "?")
                        llm_entry[f"{ckey}_pct"] = round(data[ckey].get("pct_move", 0), 3)
                if data.get("consensus") and isinstance(data["consensus"], dict):
                    llm_entry["consensus"] = data["consensus"].get("weighted_action", "?")
                    llm_entry["consensus_conf"] = round(data["consensus"].get("confidence", 0), 2)
                if llm_entry:
                    llm_data[ticker] = llm_entry
        except Exception:
            pass

    prediction = _make_autonomous_prediction(signals_data, llm_data)
    thesis_status = _assess_thesis(positions_data, signals_data, trigger_reasons)
    risk_data = _build_autonomous_risk_data(positions_data, llm_signals)

    # --- 2. Build autonomous Telegram ---
    msg = _build_autonomous_telegram(
        trigger_reasons, blocked_tier, positions_data, signals_data, llm_data,
        risk_data, prediction, thesis_status, cet_str, is_emergency,
    )

    # --- 3. Log decision ---
    prob_report = {}
    decision = {
        "ts": now.isoformat(),
        "source": "autonomous",
        "check_count": check_count,
        "tier": blocked_tier,
        "trigger": reason_str,
        "action": prediction.get("action", "HOLD"),
        "positions": positions_data,
        "signals": signals_data,
        "llm": llm_data,
        "risk": risk_data,
        "probability": {
            ticker: {
                "prob_up": r.get("prob_up_pct", 50),
                "prob_down": r.get("prob_down_pct", 50),
                "momentum": r.get("momentum", {}),
            }
            for ticker, r in prob_report.items()
        },
        "prediction": prediction,
        "thesis_status": thesis_status,
    }
    try:
        atomic_append_jsonl("data/metals_decisions.jsonl", decision)
    except Exception as e:
        log(f"Decision log error: {e}")

    # --- 4. Send Telegram (throttled for routine HOLDs) ---
    global _last_auto_telegram
    is_routine = prediction.get("action") == "HOLD" and thesis_status in ("INTACT", "NEUTRAL")
    should_send = True
    if is_routine and not is_emergency:
        since_last_tg = time.time() - _last_auto_telegram
        if since_last_tg < AUTO_TELEGRAM_COOLDOWN:
            should_send = False
            log(f"Autonomous T{blocked_tier}: Telegram throttled ({since_last_tg:.0f}s < {AUTO_TELEGRAM_COOLDOWN}s)")

    if should_send and msg:
        send_telegram(msg)
        _last_auto_telegram = time.time()

    log(f"Autonomous T{blocked_tier}: {prediction.get('action', 'HOLD')} | {thesis_status} | {reason_str[:50]}")


def invoke_claude(trigger_reasons, tier=2):
    """Invoke Claude Code as Layer 2 trading agent with tier-based model selection."""
    global claude_proc, claude_log_fh, claude_start, claude_timeout
    global invoke_count, last_invoke_times

    # Guard 0: Claude disabled — route everything to autonomous handler
    if not CLAUDE_ENABLED:
        log(f"Claude disabled — routing T{tier} trigger to autonomous")
        _autonomous_decision(trigger_reasons, tier)
        return False

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

    # Guard 2: Per-tier cooldown
    cooldown = TIER_COOLDOWNS.get(tier, 600)
    since_last = time.time() - last_invoke_times.get(tier, 0)
    if cooldown > 0 and since_last < cooldown:
        remaining = cooldown - since_last
        log(f"T{tier} cooldown: {remaining:.0f}s remaining, falling back to autonomous")
        _autonomous_decision(trigger_reasons, tier)
        return False

    # Read prompt template
    prompt_file = "data/metals_agent_prompt.txt"
    try:
        with open(prompt_file, encoding="utf-8") as f:
            base_prompt = f.read()
    except (OSError, FileNotFoundError) as e:
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
        log_fh.write(f"Model: {tier_cfg['model'] or 'default (sonnet)'} | "
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
        last_invoke_times[tier] = time.time()

        log(f"Claude T{tier} invoked (pid={claude_proc.pid}, model={tier_cfg['model'] or 'sonnet'}, "
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

    # Prevent duplicate loop trees from concurrent launcher runs.
    if not acquire_singleton_lock():
        log("Duplicate metals loop instance detected; exiting.")
        return DUPLICATE_INSTANCE_EXIT_CODE
    atexit.register(release_singleton_lock)

    # Probe time server on startup
    h, ts, src = get_cet_time()
    log("Starting unified monitoring loop (v9 — AUTONOMOUS + 5 instruments)...")
    log(f"Time: {ts} (source: {src})")
    log(f"Check interval: {CHECK_INTERVAL}s | Heartbeat: every {HEARTBEAT_CHECKS} checks (~{HEARTBEAT_CHECKS*CHECK_INTERVAL//60}min)")
    log(f"Triggers: price>{TRIGGER_PRICE_MOVE}% | trail>{TRIGGER_TRAILING}% | profit>{TRIGGER_PROFIT}%")
    log(f"Stop levels: L1(warn)<{STOP_L1_PCT}% | L2(alert)<{STOP_L2_PCT}% | L3(emergency)<{STOP_L3_PCT}%")
    log("*** AUTONOMOUS MODE v9 — unified 5-instrument monitoring ***")
    log(f"*** Smart trailing stops: {TRAIL_DISTANCE_PCT}% base, "
        f"{TRAIL_TIGHTEN_MOMENTUM}% momentum, {TRAIL_TIGHTEN_ACCEL}% accel ***")
    log(f"*** Holdings reconcile every {HOLDINGS_DIFF_INTERVAL_S}s (Avanza vs local state) ***")
    log(f"*** Probability telegram every {PROB_TELEGRAM_INTERVAL} checks (~{PROB_TELEGRAM_INTERVAL*CHECK_INTERVAL//60}min) ***")
    log("*** Tracking: XAG/XAU (FAPI) + BTC/ETH (SPOT) + MSTR (Yahoo) ***")
    log(f"Short instruments: {', '.join(v['name'] for v in SHORT_INSTRUMENTS.values())}")
    if SPIKE_ENABLED:
        spike_sched = get_us_spike_schedule()
        log(
            f"Spike catcher: place@{spike_sched['place_label']} cancel@{spike_sched['cancel_label']} "
            f"(US open {spike_sched['et_open_label']} / {spike_sched['open_label']}) "
            f"P{SPIKE_PERCENTILE} {SPIKE_PARTIAL_PCT}% partial"
        )
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
        if active_count == 0:
            log("No active positions — running in monitoring mode")
            log("Will auto-detect new instruments bought on Avanza")

        # Fetch initial prices (metals FAPI + crypto SPOT + MSTR Yahoo)
        und_prices = fetch_underlying_from_binance()
        if und_prices:
            log(f"  Prices: {', '.join(f'{k}=${v:.2f}' for k, v in und_prices.items())}")
        else:
            log("  WARNING: Initial price fetch failed — will retry")

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
                    # Use current bid as trailing baseline; day-high can trigger false emergency exits.
                    peak_bids[key] = p['bid']
                    last_invoke_prices[key] = p['bid']
                    log(f"  {key}: bid={p['bid']}, peak={peak_bids[key]}, entry={pos['entry']}, "
                        f"pnl={pnl_pct(p['bid'], pos['entry']):+.1f}%")

        # Read initial signal data
        last_signal_data = read_signal_data()
        if last_signal_data:
            log(f"  Signal data loaded (age: {last_signal_data.get('age_min', '?')}min)")

        if CLAUDE_ENABLED:
            log("Token budget: REDUCED — no Opus, T1 2min cooldown, T2 10min cooldown")
        else:
            log("Token budget: ZERO — Claude disabled, all autonomous")

        # Start local LLM background thread
        if LLM_AVAILABLE:
            def _get_signal_data():
                return last_signal_data

            def _get_underlying_prices():
                result = {}
                for ticker in ("XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"):
                    p = _underlying_prices.get(ticker, 0)
                    if p > 0:
                        result[ticker] = p
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

        # Compute seasonality profiles at startup
        _SEASONALITY_AVAILABLE = False
        try:
            from portfolio.seasonality_updater import update_seasonality_profiles
            from portfolio.seasonality import get_profile
            _SEASONALITY_AVAILABLE = True
            _seasonality_profiles = update_seasonality_profiles()
            if _seasonality_profiles:
                for t, p in _seasonality_profiles.items():
                    count = sum(v["count"] for v in p.values())
                    log(f"  Seasonality profile: {t} ({count} hourly observations)")
            else:
                log("  Seasonality profiles: no data available")
        except Exception as e:
            log(f"  Seasonality profiles: failed ({e})")

        # Initialize silver fast-tick monitor (merged from silver_monitor.py)
        if SILVER_FAST_TICK_ENABLED and _has_active_silver():
            _silver_init_ref()
            skey, spos = _get_active_silver()
            log(f"Silver fast-tick: ACTIVE (10s ticks, ref=${_silver_underlying_ref or '?'})")
            log(f"  Position: {skey} | {spos.get('units',0)} units @ {spos.get('entry',0)} SEK")
            log(f"  Alerts: {', '.join(f'{t[0]}%' for t, _ in SILVER_ALERT_LEVELS)}")
        elif SILVER_FAST_TICK_ENABLED:
            log("Silver fast-tick: STANDBY (no active silver position)")
        else:
            log("Silver fast-tick: DISABLED")

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

        # Initialize strategy orchestrator (GoldDigger + Elongir as plugins)
        _strategy_orchestrator = None
        _strategy_shared_data = None
        try:
            from portfolio.strategies.orchestrator import StrategyOrchestrator, load_strategies
            from portfolio.strategies.base import SharedData as _StrategySharedData

            _strategy_shared_data = _StrategySharedData(
                underlying_prices=_underlying_prices,
                fx_rate=0.0,
                cert_prices={},
                is_market_hours=False,
            )
            _loaded_strategies = load_strategies(config_data)
            if _loaded_strategies:
                _strategy_orchestrator = StrategyOrchestrator(
                    strategies=_loaded_strategies,
                    shared_data=_strategy_shared_data,
                    send_telegram=send_telegram,
                )
                _strategy_orchestrator.start()
                log(f"Strategy orchestrator: {_strategy_orchestrator.summary()}")
            else:
                log("Strategy orchestrator: no strategies enabled")
        except Exception as e:
            log(f"Strategy orchestrator: NOT available ({e})")

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

        # Fetch initial probability report
        prob = compute_probability_report()
        prob_summary = ""
        for t, r in prob.items():
            short = t.split("-")[0]
            prob_summary += f"\n  {short}: ${r['price']:.2f} ↑{r['prob_up_pct']:.0f}%"

        crypto_tag = "F&G+news+onchain" if CRYPTO_DATA_AVAILABLE else "OFF"
        silver_tick_tag = f"10s ticks, ref=${_silver_underlying_ref or '?'}" if (SILVER_FAST_TICK_ENABLED and _has_active_silver()) else "OFF"
        send_telegram(f"""*UNIFIED LOOP v10 STARTED*
Instruments: XAG/XAU/BTC/ETH/MSTR
Interval: {CHECK_INTERVAL}s | Holdings diff: every {HOLDINGS_DIFF_INTERVAL_S}s
LLM: {"Ministral+Chronos (4 tickers)" if LLM_AVAILABLE else "OFF"}
Crypto: {crypto_tag}
Stops: smart trailing {TRAIL_DISTANCE_PCT}%/{TRAIL_TIGHTEN_MOMENTUM}%/{TRAIL_TIGHTEN_ACCEL}%
Silver fast-tick: {silver_tick_tag}
Swing: {"ACTIVE" if swing_trader else "OFF"}
Session: {"ALIVE" if session_healthy else "DEAD"}
Positions: {pos_summary}{prob_summary}""")

        _contract_tracker = ViolationTracker(DATA_DIR / "metals_contract_state.json")

        try:
            last_holdings_diff_ts = 0.0
            while True:
                cycle_started = time.monotonic()
                check_count += 1

                _report = MetalsCycleReport(cycle_id=check_count)
                _report.cycle_start = cycle_started

                # --- ALWAYS: Fetch underlying prices from Binance FAPI (24/7) ---
                fetch_underlying_from_binance()

                # Track which underlyings succeeded
                for tk in ("XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD"):
                    if _underlying_prices.get(tk):
                        _report.underlying_tickers_ok.add(tk)
                _report.underlying_prices_fetched = bool(_report.underlying_tickers_ok)

                # --- Accumulate order book snapshots for microstructure signals ---
                _accumulate_orderbook_snapshots()

                # --- HOLDINGS DIFF/RECONCILE (always, every 30s) ---
                now_ts = time.time()
                if now_ts - last_holdings_diff_ts >= HOLDINGS_DIFF_INTERVAL_S:
                    last_holdings_diff_ts = now_ts
                    changes = detect_holdings(page)
                    if changes:
                        # New instruments detected — place stops, update peaks
                        for key, pos in POSITIONS.items():
                            if pos["active"] and key not in peak_bids:
                                try:
                                    p = fetch_price(page, pos["ob_id"], pos["api_type"])
                                    if p and p.get("bid"):
                                        # Freshly detected holdings should start trailing from current bid.
                                        peak_bids[key] = p["bid"]
                                        last_invoke_prices[key] = p["bid"]
                                except Exception:
                                    pass
                        if STOP_ORDER_ENABLED:
                            stop_order_state = place_stop_loss_orders(page, POSITIONS)
                        # Initialize silver fast-tick if new silver position detected
                        if SILVER_FAST_TICK_ENABLED and _has_active_silver() and _silver_underlying_ref is None:
                            _silver_init_ref()
                            log(f"Silver fast-tick activated: ref=${_silver_underlying_ref or '?'}")
                        send_telegram(
                            "*HOLDINGS UPDATE*\n" +
                            "\n".join(f"• {c}" for c in changes)
                        )
                    _report.holdings_reconciled = True

                if not is_market_hours():
                    # Update strategy shared data even outside market hours
                    if _strategy_shared_data is not None:
                        _strategy_shared_data.underlying_prices = _underlying_prices
                        _strategy_shared_data.is_market_hours = False
                    # Outside Avanza hours: still track underlyings + compute probability
                    if check_count % PROB_REPORT_INTERVAL == 0:
                        compute_probability_report()
                    # Send probability telegram even outside market hours (less frequent)
                    if check_count % (PROB_TELEGRAM_INTERVAL * 3) == 0:
                        prob = compute_probability_report()
                        msg = build_probability_telegram(prob, cet_time_str())
                        if msg:
                            send_telegram(msg)
                    if check_count % 60 == 0:
                        price_tags = []
                        for t in ALL_TRACKED_TICKERS:
                            p = _underlying_prices.get(t, 0)
                            if p > 0:
                                short_t = t.split("-")[0] if "-" in t else t
                                price_tags.append(f"{short_t}=${_format_price(p, t)}")
                        log(f"Outside market hours — {' '.join(price_tags)}")
                    _report.cycle_end = time.monotonic()
                    try:
                        verify_and_act(_report, {}, tracker=_contract_tracker,
                                       verify_fn=verify_metals_contract, loop_name="metals")
                    except Exception as _e:
                        log(f"Contract check failed: {_e}")
                    _sleep_for_cycle(cycle_started, CHECK_INTERVAL, "metals loop")
                    continue

                # Fetch warrant prices for active positions
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
                    _sleep_for_cycle(cycle_started, CHECK_INTERVAL, "metals loop")
                    continue

                _report.active_positions = sum(1 for pos in POSITIONS.values() if pos.get("active"))
                _report.positions_priced = len(prices)
                _report.position_prices_updated = _report.positions_priced >= _report.active_positions

                # Update strategy shared data with fresh prices and cert data
                if _strategy_shared_data is not None:
                    _strategy_shared_data.underlying_prices = _underlying_prices
                    _strategy_shared_data.is_market_hours = True
                    for key, pos in POSITIONS.items():
                        if pos["active"] and key in prices:
                            _strategy_shared_data.cert_prices[pos.get("ob_id", "")] = prices[key]

                # Fetch short instrument prices (every 4th check)
                if check_count % 4 == 0:
                    for sk, si in SHORT_INSTRUMENTS.items():
                        try:
                            sp = fetch_price(page, si["ob_id"], si["api_type"])
                            if sp:
                                short_prices[sk] = sp
                        except Exception:
                            pass

                # Read signal data periodically (every ~2 min)
                if check_count % 4 == 0:
                    last_signal_data = read_signal_data()

                # Refresh account data + warrant catalog (every 10th check ~5 min)
                if check_count % 10 == 0:
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

                # Session health check (~every 10 min at 30s interval)
                if check_count % SESSION_HEALTH_CHECK_INTERVAL == 0:
                    _check_session_and_alert(page)
                _report.session_alive = session_healthy

                # --- PROBABILITY REPORT (every ~2.5 min) ---
                if check_count % PROB_REPORT_INTERVAL == 0:
                    compute_probability_report()
                    _report.probability_computed = True

                # Store price snapshot
                snap = {
                    "ts": datetime.datetime.now(datetime.UTC).isoformat(),
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
                    spike_sched = get_us_spike_schedule()
                    spike_place_hour = spike_sched["place_hour"]
                    spike_cancel_hour = spike_sched["cancel_hour"]
                    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    spike_st = load_spike_state()

                    # Reset state on new day
                    if spike_st.get("date") != today_str:
                        spike_st = {"orders": {}, "targets": {}, "date": today_str,
                                    "placed": False, "cancelled": False}
                        save_spike_state(spike_st)

                    # Phase 1: Place orders 15 min before the NYSE open, translated to Stockholm time
                    if (not spike_st["placed"] and spike_place_hour <= h_now < spike_cancel_hour):
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
                                "*SPIKE CATCHER*\n"
                                + "\n".join(f"`{k}: SELL {t['units_to_sell']}u @ {t['target_price']} "
                                           f"(+{t['target_pnl_pct']:.1f}%)`"
                                           for k, t in targets.items())
                                + f"\n_P{SPIKE_PERCENTILE} target, cancels at {spike_sched['cancel_label']}_"
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

                    # Phase 3: Cancel unfilled 1h after the NYSE open
                    if spike_st["placed"] and not spike_st["cancelled"] and h_now >= spike_cancel_hour:
                        if spike_st.get("orders"):
                            log(f"Spike catcher: cancelling {len(spike_st['orders'])} unfilled orders")
                            cancel_spike_orders(page, spike_st)
                            send_telegram(
                                f"_Spike orders cancelled ({spike_sched['cancel_label']}, unfilled)_"
                            )
                        spike_st["cancelled"] = True
                        save_spike_state(spike_st)

                # Startup grace
                if startup_grace:
                    startup_grace = False
                    log(f"#{check_count} Baseline established (grace period)")
                    _sleep_for_cycle(cycle_started, CHECK_INTERVAL, "metals loop")
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
                for mkey, _mreason in momentum_exits:
                    if POSITIONS[mkey].get("active"):
                        mbid = prices.get(mkey, {}).get("bid") or 0
                        if mbid > 0:
                            log(f"!!! MOMENTUM SELL: {mkey} at {mbid}")
                            send_telegram(f"*MOMENTUM EXIT* {POSITIONS[mkey]['name']}\nBid: {mbid} | Accelerating decline detected")
                            emergency_sell(page, mkey, POSITIONS[mkey], mbid)

                # Smart trailing stop updates (every 3rd check — more responsive)
                # Skip if hardware trailing is active — Avanza manages the trail
                if STOP_ORDER_ENABLED and not HARDWARE_TRAILING_ENABLED and check_count % 3 == 0:
                    update_smart_trailing_stops(page, POSITIONS, stop_order_state, prices)

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

                # Log status (every 3rd check) — compact with probability
                if check_count % 3 == 0:
                    parts = []
                    for key, pos in POSITIONS.items():
                        if pos["active"] and key in prices:
                            bid = prices[key].get('bid', 0)
                            pnl_val = pnl_pct(bid, pos["entry"])
                            parts.append(f"{key}:{bid}({pnl_val:+.1f}%)")
                    cet = cet_time_str()
                    # Underlying prices + probability (all tracked)
                    und_tag = ""
                    for t in ALL_TRACKED_TICKERS:
                        p = _underlying_prices.get(t, 0)
                        if p > 0:
                            short_t = t.split("-")[0]
                            prob = _last_prob_report.get(t, {})
                            prob_up = prob.get("prob_up_pct", 50)
                            mom = get_underlying_momentum(t)
                            vel = mom["velocity_pct"]
                            und_tag += f" {short_t}=${p:.2f}(↑{prob_up:.0f}% v={vel:+.3f}%)"
                    # Signal accuracy tag (every 12th check)
                    acc_tag = ""
                    if TRACKER_AVAILABLE and check_count % 12 == 0:
                        try:
                            acc_tag = f" ACC:[{get_accuracy_summary()}]"
                        except Exception:
                            pass
                    pos_str = ' | '.join(parts) if parts else "no positions"
                    log(f"#{check_count} [{cet}] {pos_str}{und_tag}{acc_tag}" +
                        (f" [TRIGGER: {reasons[0]}]" if triggered else ""))

                # --- PROBABILITY TELEGRAM (every ~10 min) ---
                if check_count % PROB_TELEGRAM_INTERVAL == 0:
                    prob = compute_probability_report()
                    msg = build_probability_telegram(prob, cet_time_str())
                    if msg:
                        global _last_auto_telegram
                        send_telegram(msg)
                        _last_auto_telegram = time.time()

                # Handle triggers (autonomous — no Claude)
                if triggered:
                    tier = classify_tier(reasons)
                    for key in prices:
                        if prices[key].get('bid'):
                            last_invoke_prices[key] = prices[key]['bid']
                    _autonomous_decision(reasons, tier)

                # Check if Claude finished (non-blocking) — kept for compatibility
                if claude_proc and claude_proc.poll() is not None:
                    elapsed = time.time() - claude_start
                    retcode = claude_proc.returncode
                    log(f"Claude finished (rc={retcode}, {elapsed:.0f}s)")
                    log_invocation(0, None, "completed", check_count, invoke_count,
                                   elapsed_s=elapsed, rc=retcode)
                    claude_proc = None
                    if claude_log_fh:
                        try:
                            claude_log_fh.close()
                        except OSError:
                            pass
                        claude_log_fh = None

                    if TRADE_QUEUE_ENABLED:
                        try:
                            process_trade_queue(page)
                        except Exception as e:
                            log(f"Trade queue processing error: {e}")

                    if os.path.exists("data/metals_trades.jsonl"):
                        try:
                            with open("data/metals_trades.jsonl") as f:
                                lines = f.readlines()
                            if lines:
                                last_trade = json.loads(lines[-1])
                                log(f"Last trade: {last_trade.get('action','')} {last_trade.get('name','')}")
                        except (OSError, json.JSONDecodeError) as e:
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
                            # Add crypto prices for backfill
                            for ticker in ("BTC-USD", "ETH-USD"):
                                cp = _underlying_prices.get(ticker, 0)
                                if cp > 0:
                                    und_prices[ticker] = cp
                            if und_prices:
                                backfill_outcomes(und_prices)
                        except Exception as e:
                            log(f"Signal tracker backfill error: {e}")

                # --- PERIODIC NEWS FETCH (every ~30 min) ---
                if time.time() - _last_news_fetch_ts >= NEWS_FETCH_INTERVAL:
                    try:
                        _fetch_metals_news()
                    except Exception as e:
                        log(f"News fetch error (non-fatal): {e}")

                # --- FISH PRECOMPUTE (every cycle) ---
                try:
                    _write_fish_precomputed()
                except Exception as e:
                    log(f"Fish precompute error (non-fatal): {e}")

                # --- FISH ENGINE (intraday fishing, every cycle) ---
                if FISH_ENGINE_ENABLED:
                    try:
                        _run_fish_engine_tick()
                    except Exception as e:
                        log(f"Fish engine error (non-fatal): {e}")

                _report.cycle_end = time.monotonic()
                try:
                    verify_and_act(_report, {}, tracker=_contract_tracker,
                                   verify_fn=verify_metals_contract, loop_name="metals")
                except Exception as _e:
                    log(f"Contract check failed: {_e}")

                _sleep_for_cycle(cycle_started, CHECK_INTERVAL, "metals loop")

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
            if _strategy_orchestrator is not None:
                try:
                    _strategy_orchestrator.stop()
                except Exception as e:
                    print(f"[WARN] Strategy orchestrator stop failed: {e}", flush=True)
            browser.close()
            release_singleton_lock()
            log(f"Loop stopped: {check_count} checks, {invoke_count} invocations")

if __name__ == "__main__":
    sys.exit(main())
