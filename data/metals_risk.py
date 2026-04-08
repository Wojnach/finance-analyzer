"""Metals risk management: Monte Carlo VaR, trade guards, drawdown circuit breaker, daily ranges.

Standalone module for the metals intraday loop. Adapts portfolio/monte_carlo.py,
portfolio/trade_guards.py, and portfolio/risk_management.py for warrant trading.

Usage from metals_loop.py:
    from metals_risk import (
        simulate_warrant_risk, check_trade_guard, record_metals_trade,
        check_portfolio_drawdown, get_risk_summary,
        compute_daily_range_stats, compute_intraday_assessment,
        compute_spike_targets,
    )
"""

import datetime
import json
import logging
import math
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json, load_jsonl_tail

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATE_FILE = "data/metals_guard_state.json"
HISTORY_FILE = "data/metals_value_history.jsonl"

# Monte Carlo defaults (smaller than main loop — we run frequently)
MC_N_PATHS = 5000
MC_HORIZONS_HOURS = [3, 8]  # 3h intraday, 8h full session

# Trade guard defaults
COOLDOWN_BASE_MINUTES = 30         # min time between trades on same underlying
COOLDOWN_ESCALATION = {0: 1, 1: 1, 2: 2, 3: 4, 4: 8}  # consecutive losses -> multiplier
MAX_TRADES_PER_SESSION = 6         # max trades in a single market session (08:00-17:25)
POSITION_RATE_LIMIT_HOURS = 1      # min hours between new BUYs

# Drawdown circuit breaker
MAX_DRAWDOWN_PCT = 15.0            # emergency liquidation threshold
DRAWDOWN_WARN_PCT = 10.0           # warning level

# Leverage-adjusted risk: warrant leverage amplifies underlying moves
# Defaults used when metals_context.json is unavailable
_LEVERAGE_DEFAULTS = {
    "gold": 8.0,       # BULL GULD X8
    "silver79": 5.0,   # MINI L SILVER AVA 79 (effective ~5x)
    "silver301": 4.3,  # MINI L SILVER AVA 301
    "silver_sg": 4.76, # MINI L SILVER SG
    # Crypto — spot (no leverage) unless holding trackers
    "btc": 1.0,
    "eth": 1.0,
    "mstr": 1.0,
    "xbt_tracker": 1.0,   # CoinShares XBT Tracker
    "eth_tracker": 1.0,   # CoinShares ETH Tracker
}

# Default ATR% per underlying (used when signal_data has no atr_pct)
ATR_DEFAULTS = {
    "XAG-USD": 4.4,   # silver
    "XAU-USD": 1.9,   # gold
    "BTC-USD": 3.2,   # bitcoin
    "ETH-USD": 4.5,   # ethereum
    "MSTR": 5.0,      # MicroStrategy (high vol stock)
}


def _load_leverage_map():
    """Load leverage from metals_context.json warrant_catalog, fall back to defaults."""
    ctx_path = os.path.join(os.path.dirname(__file__), "metals_context.json")
    try:
        if os.path.exists(ctx_path):
            with open(ctx_path, encoding="utf-8") as f:
                ctx = json.load(f)
            catalog = ctx.get("warrant_catalog", {})
            lmap = {}
            for key, info in catalog.items():
                lev = info.get("leverage")
                if lev and isinstance(lev, (int, float)) and lev > 0:
                    lmap[key] = float(lev)
            if lmap:
                logger.info(f"Loaded leverage from warrant_catalog: {lmap}")
                return lmap
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Cannot load metals_context.json for leverage: {e}")
    return dict(_LEVERAGE_DEFAULTS)


LEVERAGE_MAP = _load_leverage_map()


def _default_guard_state():
    return {
        "ticker_trades": {},        # "gold" -> last trade ISO timestamp
        "consecutive_losses": 0,
        "session_trade_count": 0,
        "session_date": None,
        "new_position_times": [],   # ISO timestamps of BUY trades
    }


def _default_spike_state():
    return {"orders": {}, "date": None, "placed": False, "cancelled": False}


def _load_json_state(path, default_factory, label):
    """Load a JSON state file with explicit logging on corrupt/unreadable content."""
    from pathlib import Path
    path_obj = Path(path)
    result = load_json(path, default=None)
    if result is None:
        if path_obj.exists():
            # File exists but load_json returned None — corrupt or unreadable.
            logger.warning("%s load failed (corrupt or unreadable): %s", label, path)
        return default_factory()
    return result


# ---------------------------------------------------------------------------
# Monte Carlo — Warrant-Level Simulation
# ---------------------------------------------------------------------------

def _annualized_vol_from_atr(atr_pct, period=14):
    """Convert ATR% to annualized volatility."""
    daily_vol = atr_pct / 100.0
    return daily_vol * math.sqrt(252 / period)


def simulate_warrant_risk(
    underlying_price,
    atr_pct,
    leverage,
    entry_price_warrant,
    stop_price_warrant,
    direction_prob=0.5,
    horizons_hours=None,
    n_paths=None,
):
    """Run Monte Carlo simulation for a leveraged warrant position.

    Returns dict with per-horizon risk metrics:
    - price_bands: 5th/25th/50th/75th/95th percentile of warrant price
    - p_stop_hit: probability of breaching stop-loss
    - p_profit_5pct: probability of reaching +5% profit
    - expected_return: mean return and std
    - var_95: 95% Value at Risk (worst 5% outcome)
    """
    import numpy as np

    if horizons_hours is None:
        horizons_hours = MC_HORIZONS_HOURS
    if n_paths is None:
        n_paths = MC_N_PATHS

    vol_annual = _annualized_vol_from_atr(atr_pct)
    vol_annual = max(vol_annual, 0.05)  # floor

    # Drift from direction probability
    if direction_prob > 0.01 and direction_prob < 0.99:
        from scipy.stats import norm
        z = norm.ppf(direction_prob)
        drift = z * vol_annual - 0.5 * vol_annual ** 2
    else:
        drift = 0.0

    result = {
        "underlying_price": underlying_price,
        "warrant_entry": entry_price_warrant,
        "warrant_stop": stop_price_warrant,
        "leverage": leverage,
        "atr_pct": atr_pct,
        "vol_annual": round(vol_annual, 4),
        "direction_prob": direction_prob,
    }

    rng = np.random.default_rng(seed=42)

    for hours in horizons_hours:
        # Convert hours to annual fraction
        t = hours / (252 * 6.5)  # 6.5 trading hours/day, 252 days/year

        # GBM with antithetic variates
        half = n_paths // 2
        z = rng.standard_normal(half)
        z_full = np.concatenate([z, -z])

        # Underlying terminal prices
        log_return = (drift - 0.5 * vol_annual**2) * t + vol_annual * math.sqrt(t) * z_full
        underlying_terminal = underlying_price * np.exp(log_return)

        # Warrant price = entry * (1 + leverage * underlying_return)
        underlying_return = (underlying_terminal - underlying_price) / underlying_price
        warrant_terminal = entry_price_warrant * (1 + leverage * underlying_return)
        warrant_terminal = np.maximum(warrant_terminal, 0)  # warrant can't go below 0

        # Metrics
        h_key = f"{hours}h"
        pcts = np.percentile(warrant_terminal, [5, 25, 50, 75, 95])
        returns_pct = ((warrant_terminal - entry_price_warrant) / entry_price_warrant) * 100

        result[f"price_bands_{h_key}"] = {
            "p5": round(float(pcts[0]), 2),
            "p25": round(float(pcts[1]), 2),
            "p50": round(float(pcts[2]), 2),
            "p75": round(float(pcts[3]), 2),
            "p95": round(float(pcts[4]), 2),
        }
        result[f"p_stop_hit_{h_key}"] = round(float(np.mean(warrant_terminal <= stop_price_warrant)), 4)
        result[f"p_profit_5pct_{h_key}"] = round(float(np.mean(returns_pct >= 5.0)), 4)
        result[f"expected_return_{h_key}"] = {
            "mean_pct": round(float(np.mean(returns_pct)), 2),
            "std_pct": round(float(np.std(returns_pct)), 2),
            "var_95_pct": round(float(np.percentile(returns_pct, 5)), 2),  # 5th pctile = 95% VaR
        }

    return result


def _position_key_to_ticker(key):
    """Map a position key (e.g. 'silver301', 'btc', 'gold') to a standard ticker."""
    k = key.lower()
    if "silver" in k:
        return "XAG-USD"
    if "gold" in k:
        return "XAU-USD"
    if "btc" in k or "xbt" in k:
        return "BTC-USD"
    if "eth" in k:
        return "ETH-USD"
    if "mstr" in k:
        return "MSTR"
    return key


def simulate_all_positions(positions, prices, signal_data=None, llm_signals=None):
    """Run Monte Carlo for all active positions. Returns dict keyed by position name.

    Args:
        positions: POSITIONS dict from metals_loop
        prices: {key: {bid, underlying, ...}} from latest fetch
        signal_data: optional signal data for direction probabilities
        llm_signals: optional LLM consensus for better probabilities
    """
    results = {}

    for key, pos in positions.items():
        if not pos.get("active"):
            continue
        p = prices.get(key)
        if not p or not p.get("bid"):
            continue

        bid = p["bid"]
        underlying = p.get("underlying") or bid  # fallback to bid if no underlying
        leverage = LEVERAGE_MAP.get(key, 1.0)

        # Get direction probability from LLM or signals
        direction_prob = 0.5  # neutral default
        # Map position key to ticker
        ticker = _position_key_to_ticker(key)
        if llm_signals:
            llm_data = llm_signals.get(ticker, {})
            consensus = llm_data.get("consensus", {})
            if consensus.get("direction") == "up":
                direction_prob = 0.5 + consensus.get("confidence", 0) * 0.3  # scale 0.5-0.8
            elif consensus.get("direction") == "down":
                direction_prob = 0.5 - consensus.get("confidence", 0) * 0.3  # scale 0.2-0.5

        # ATR for underlying — use from signal data or default
        atr_pct = ATR_DEFAULTS.get(ticker, 3.0)
        if signal_data:
            sig = signal_data.get(ticker, {})
            if sig.get("atr_pct"):
                atr_pct = sig["atr_pct"]

        try:
            results[key] = simulate_warrant_risk(
                underlying_price=underlying,
                atr_pct=atr_pct,
                leverage=leverage,
                entry_price_warrant=pos["entry"],
                stop_price_warrant=pos["stop"],
                direction_prob=direction_prob,
            )
        except Exception as e:
            logger.warning(f"Monte Carlo failed for {key}: {e}")

    return results


# ---------------------------------------------------------------------------
# Trade Guards — Overtrading Prevention
# ---------------------------------------------------------------------------

def _load_guard_state():
    """Load trade guard state from file."""
    return _load_json_state(STATE_FILE, _default_guard_state, "Guard state")


def _save_guard_state(state):
    """Persist trade guard state."""
    try:
        atomic_write_json(STATE_FILE, state, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save guard state: {e}")


def check_trade_guard(position_key, action, current_time=None):
    """Check if a trade should be blocked by any guard.

    Returns:
        list of warning dicts (empty = all clear)
        Each warning: {guard, severity, message, details}
    """
    state = _load_guard_state()
    now = current_time or datetime.datetime.now(datetime.UTC)
    today = now.strftime("%Y-%m-%d")
    warnings = []

    # Reset session counter on new day
    if state.get("session_date") != today:
        state["session_date"] = today
        state["session_trade_count"] = 0
        state["new_position_times"] = []
        _save_guard_state(state)

    # Guard 1: Ticker cooldown (with loss escalation)
    last_trade_ts = state["ticker_trades"].get(position_key)
    if last_trade_ts:
        try:
            last_dt = datetime.datetime.fromisoformat(last_trade_ts)
            elapsed_min = (now - last_dt).total_seconds() / 60

            losses = state.get("consecutive_losses", 0)
            multiplier = COOLDOWN_ESCALATION.get(min(losses, 4), 1)
            required_min = COOLDOWN_BASE_MINUTES * multiplier

            if elapsed_min < required_min:
                remaining = round(required_min - elapsed_min, 1)
                warnings.append({
                    "guard": "ticker_cooldown",
                    "severity": "block",
                    "message": f"{position_key}: cooldown active ({remaining:.0f}min left, {multiplier}x from {losses} losses)",
                    "details": {
                        "elapsed_min": round(elapsed_min, 1),
                        "required_min": required_min,
                        "consecutive_losses": losses,
                        "multiplier": multiplier,
                    }
                })
        except Exception:
            pass

    # Guard 2: Session trade limit
    if state.get("session_trade_count", 0) >= MAX_TRADES_PER_SESSION:
        warnings.append({
            "guard": "session_limit",
            "severity": "block",
            "message": f"Session limit reached ({MAX_TRADES_PER_SESSION} trades today)",
            "details": {"count": state["session_trade_count"], "max": MAX_TRADES_PER_SESSION},
        })

    # Guard 3: BUY rate limit (only for BUY actions)
    if action.upper() == "BUY" and state.get("new_position_times"):
        recent_buys = []
        cutoff = now - datetime.timedelta(hours=POSITION_RATE_LIMIT_HOURS)
        for ts_str in state["new_position_times"]:
            try:
                ts = datetime.datetime.fromisoformat(ts_str)
                if ts > cutoff:
                    recent_buys.append(ts)
            except Exception:
                pass
        if recent_buys:
            last_buy = max(recent_buys)
            mins_since = (now - last_buy).total_seconds() / 60
            warnings.append({
                "guard": "buy_rate_limit",
                "severity": "warning",
                "message": f"Last BUY was {mins_since:.0f}min ago (rate limit: {POSITION_RATE_LIMIT_HOURS}h)",
                "details": {"mins_since_last_buy": round(mins_since, 1)},
            })

    return warnings


def record_metals_trade(position_key, action, pnl_pct_value=None):
    """Record a trade for guard tracking.

    Args:
        position_key: "gold", "silver79", "silver301"
        action: "BUY" or "SELL"
        pnl_pct_value: P&L % if SELL (for consecutive loss tracking)
    """
    state = _load_guard_state()
    now = datetime.datetime.now(datetime.UTC)
    today = now.strftime("%Y-%m-%d")

    # Reset on new day
    if state.get("session_date") != today:
        state["session_date"] = today
        state["session_trade_count"] = 0
        state["new_position_times"] = []

    # Record trade time for cooldown
    state["ticker_trades"][position_key] = now.isoformat()
    state["session_trade_count"] = state.get("session_trade_count", 0) + 1

    # Track BUY timestamps for rate limiting
    if action.upper() == "BUY":
        state.setdefault("new_position_times", []).append(now.isoformat())

    # Track consecutive losses (SELL only)
    if action.upper() == "SELL" and pnl_pct_value is not None:
        if pnl_pct_value < 0:
            state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
        else:
            state["consecutive_losses"] = 0  # reset on win

    _save_guard_state(state)


# ---------------------------------------------------------------------------
# Drawdown Circuit Breaker
# ---------------------------------------------------------------------------

def log_portfolio_value(positions, prices):
    """Log current portfolio value to history file."""
    now = datetime.datetime.now(datetime.UTC)
    total_val = 0
    total_inv = 0
    pos_detail = {}

    for key, pos in positions.items():
        if not pos.get("active"):
            continue
        p = prices.get(key) or {}
        bid = p.get("bid") or 0
        val = bid * pos["units"]
        inv = pos["entry"] * pos["units"]
        total_val += val
        total_inv += inv
        pos_detail[key] = {"bid": bid, "value": round(val, 1), "pnl_pct": round(((bid / pos["entry"]) - 1) * 100, 2) if pos["entry"] > 0 else 0}

    entry = {
        "ts": now.isoformat(),
        "total_value": round(total_val, 1),
        "total_invested": round(total_inv, 1),
        "pnl_pct": round(((total_val / total_inv) - 1) * 100, 2) if total_inv > 0 else 0,
        "positions": pos_detail,
    }

    try:
        atomic_append_jsonl(HISTORY_FILE, entry)
    except Exception as e:
        logger.warning(f"Failed to log portfolio value: {e}")

    return entry


def _parse_entry_ts(entry):
    """Parse a history entry's ``ts`` field into a Unix timestamp.

    Entries log ISO-8601 strings (e.g. ``2026-04-08T18:32:46.968281+00:00``).
    Returns ``None`` if the field is missing or unparseable — callers should
    skip such entries rather than crash the loop.
    """
    ts = entry.get("ts")
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(ts).timestamp()
    except (TypeError, ValueError):
        return None


def check_portfolio_drawdown(positions, prices, since_ts=None):
    """Check if portfolio has breached drawdown limits.

    Args:
        positions: dict of position key -> state dict (``active``, ``units``, ``entry``).
        prices: dict of position key -> price dict (``bid``).
        since_ts: optional Unix timestamp. When provided, only history entries
            whose ``ts`` is ``>= since_ts`` are considered when computing the
            peak. Use this to make the drawdown session-relative so an old
            peak from a previous (larger) session does not trigger a false
            EMERGENCY breach. If no entries fall within the window, the peak
            falls back to the current total value (drawdown = 0%). When
            ``None``, behaviour is unchanged — the full history is scanned.

    Returns:
        dict with drawdown status and metrics.
    """
    total_val = 0
    total_inv = 0
    for key, pos in positions.items():
        if not pos.get("active"):
            continue
        p = prices.get(key) or {}
        bid = p.get("bid") or 0
        total_val += bid * pos["units"]
        total_inv += pos["entry"] * pos["units"]

    if total_inv == 0:
        return {"breached": False, "current_drawdown_pct": 0, "level": "none"}

    current_pnl_pct = ((total_val / total_inv) - 1) * 100

    # Find peak from history
    peak_value = total_inv  # start at invested amount
    window_had_entries = False
    try:
        if since_ts is None:
            # Legacy behaviour: scan the whole file (backwards compatible).
            entries_all = load_jsonl_tail(HISTORY_FILE, max_entries=5000)
            for entry in entries_all:
                if entry.get("total_value", 0) > peak_value:
                    peak_value = entry["total_value"]
        else:
            # Session-relative: only consider entries recorded since session start.
            # Use tail reader to avoid slurping huge historical files.
            entries = load_jsonl_tail(HISTORY_FILE, max_entries=5000)
            for entry in entries:
                entry_ts = _parse_entry_ts(entry)
                if entry_ts is None or entry_ts < since_ts:
                    continue
                window_had_entries = True
                val = entry.get("total_value", 0)
                if val > peak_value:
                    peak_value = val
            if not window_had_entries:
                # Fresh session, no history yet — anchor peak to current value
                # so drawdown reports 0% until we have real data points.
                peak_value = total_val
    except Exception:
        pass

    # Drawdown from peak
    if peak_value > 0:
        drawdown_pct = ((total_val - peak_value) / peak_value) * 100
    else:
        drawdown_pct = 0

    # Classify level
    if drawdown_pct <= -MAX_DRAWDOWN_PCT:
        level = "EMERGENCY"
    elif drawdown_pct <= -DRAWDOWN_WARN_PCT:
        level = "WARNING"
    else:
        level = "OK"

    return {
        "breached": drawdown_pct <= -MAX_DRAWDOWN_PCT,
        "current_drawdown_pct": round(drawdown_pct, 2),
        "current_pnl_pct": round(current_pnl_pct, 2),
        "peak_value": round(peak_value, 1),
        "current_value": round(total_val, 1),
        "invested_value": round(total_inv, 1),
        "level": level,
    }


# ---------------------------------------------------------------------------
# Combined Risk Summary (for metals_context.json)
# ---------------------------------------------------------------------------

def get_risk_summary(positions, prices, signal_data=None, llm_signals=None, since_ts=None):
    """Generate a compact risk summary for inclusion in metals_context.json.

    Args:
        positions: position state dict.
        prices: price dict.
        signal_data: optional signal snapshot.
        llm_signals: optional LLM signal snapshot.
        since_ts: optional Unix timestamp. Forwarded to
            :func:`check_portfolio_drawdown` so the drawdown peak is
            session-relative instead of all-time.

    Returns dict with:
    - monte_carlo: per-position simulation results
    - drawdown: portfolio drawdown status
    - trade_guards: current cooldown/limit status
    - risk_score: 0-100 overall risk level
    """
    result = {
        "monte_carlo": {},
        "drawdown": {},
        "trade_guards": {},
        "risk_score": 0,
    }

    # 1. Monte Carlo (only if numpy available)
    try:
        mc = simulate_all_positions(positions, prices, signal_data, llm_signals)
        # Compact summary for context
        for key, sim in mc.items():
            result["monte_carlo"][key] = {
                "p_stop_3h": sim.get("p_stop_hit_3h", "N/A"),
                "p_stop_8h": sim.get("p_stop_hit_8h", "N/A"),
                "var_95_3h": sim.get("expected_return_3h", {}).get("var_95_pct", "N/A"),
                "var_95_8h": sim.get("expected_return_8h", {}).get("var_95_pct", "N/A"),
                "p_profit_5pct_8h": sim.get("p_profit_5pct_8h", "N/A"),
                "direction_prob": sim.get("direction_prob", 0.5),
            }
    except ImportError:
        result["monte_carlo"] = {"error": "numpy not available"}
    except Exception as e:
        result["monte_carlo"] = {"error": str(e)}

    # 2. Drawdown check
    try:
        result["drawdown"] = check_portfolio_drawdown(positions, prices, since_ts=since_ts)
    except Exception as e:
        result["drawdown"] = {"error": str(e)}

    # 3. Trade guard status (check all positions)
    guards = {}
    for key in positions:
        if positions[key].get("active"):
            warnings = check_trade_guard(key, "BUY")
            if warnings:
                guards[key] = [w["message"] for w in warnings]
    result["trade_guards"] = guards if guards else {"status": "all_clear"}

    # 4. Risk score (0 = safe, 100 = danger)
    score = 0
    dd = result.get("drawdown", {})
    if dd.get("level") == "EMERGENCY":
        score = 100
    elif dd.get("level") == "WARNING":
        score += 50
    elif dd.get("current_drawdown_pct", 0) < -5:
        score += 25

    # Add MC stop probability risk
    for _key, mc_data in result.get("monte_carlo", {}).items():
        if isinstance(mc_data, dict):
            p_stop = mc_data.get("p_stop_8h", 0)
            if isinstance(p_stop, (int, float)) and p_stop > 0.2:
                score += 15  # significant stop risk

    # Trade guard blocks add risk awareness
    for _key, warns in guards.items():
        if warns:
            score += 5

    result["risk_score"] = min(score, 100)

    return result


# ---------------------------------------------------------------------------
# Daily Range Analysis — Historical Percentile Ranges
# ---------------------------------------------------------------------------

def _percentile(sorted_list, p):
    """Compute p-th percentile from a pre-sorted list."""
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return sorted_list[f]
    return sorted_list[f] * (c - k) + sorted_list[c] * (k - f)


def compute_daily_range_stats(history_path="data/metals_history.json"):
    """Compute daily range percentile statistics from historical OHLCV data.

    Returns dict keyed by underlying ticker (XAG-USD, XAU-USD) with:
    - daily_range: P25/P50/P75/P90/P95 of (high-low)/open %
    - open_to_high: percentiles of max intraday gain from open
    - open_to_low: percentiles of max intraday drop from open
    - close_change: percentiles of close-to-close daily change
    - recent_5d: last 5 days' ranges for context
    - trading_days: number of days in sample
    """
    hist = load_json(str(history_path), default={})
    if not hist:
        logger.warning(f"Cannot load or empty: {history_path}")
        return {}

    result = {}

    for metal_key in ["XAG-USD", "XAU-USD"]:
        data = hist.get("metals", {}).get(metal_key)
        if not data or "daily_ohlcv" not in data:
            continue

        candles = data["daily_ohlcv"]
        if len(candles) < 5:
            continue

        daily_ranges = []
        open_to_high = []
        open_to_low = []
        close_changes = []

        prev_close = None
        for c in candles:
            o, h, lo, cl = c["open"], c["high"], c["low"], c["close"]
            if o <= 0:
                continue

            daily_ranges.append((h - lo) / o * 100)
            open_to_high.append((h - o) / o * 100)
            open_to_low.append((o - lo) / o * 100)

            if prev_close is not None and prev_close > 0:
                close_changes.append((cl - prev_close) / prev_close * 100)
            prev_close = cl

        daily_ranges.sort()
        open_to_high.sort()
        open_to_low.sort()
        close_changes.sort()

        pctiles = [25, 50, 75, 90, 95]

        stats_entry = {
            "trading_days": len(candles),
            "daily_range": {f"p{p}": round(_percentile(daily_ranges, p), 2) for p in pctiles},
            "open_to_high": {f"p{p}": round(_percentile(open_to_high, p), 2) for p in pctiles},
            "open_to_low": {f"p{p}": round(_percentile(open_to_low, p), 2) for p in pctiles},
            "close_change": {f"p{p}": round(_percentile(close_changes, p), 2) for p in [5, 10, 25, 50, 75, 90, 95]},
        }

        # Recent 5 days
        recent = candles[-5:]
        stats_entry["recent_5d"] = []
        for c in recent:
            o, h, lo, cl = c["open"], c["high"], c["low"], c["close"]
            stats_entry["recent_5d"].append({
                "date": c["date"],
                "range_pct": round((h - lo) / o * 100, 2) if o > 0 else 0,
                "change_pct": round((cl - o) / o * 100, 2) if o > 0 else 0,
            })

        result[metal_key] = stats_entry

    return result


def compute_intraday_assessment(positions, prices, price_history, range_stats):
    """Assess today's session against historical daily range distributions.

    Returns dict with:
    - Per position: today's range used so far, remaining potential, stop safety
    - EU session estimates (40% of 24h range)
    """
    EU_SESSION_FACTOR = 0.40  # EU session captures ~40% of 24h range

    assessment = {}

    # Extract underlying prices from today's price history
    for underlying_key, ticker in [("gold", "XAU-USD"), ("silver", "XAG-USD")]:
        stats = range_stats.get(ticker)
        if not stats:
            continue

        # Get today's underlying range from price history
        und_prices = []
        for snap in price_history:
            # Try gold_und or silver79_und / silver301_und
            if underlying_key == "gold":
                p = snap.get("gold_und", 0)
            else:
                p = snap.get("silver79_und") or snap.get("silver301_und") or 0
            if p > 0:
                und_prices.append(p)

        if not und_prices:
            continue

        open_price = und_prices[0]
        current_price = und_prices[-1]
        session_high = max(und_prices)
        session_low = min(und_prices)
        session_range_pct = (session_high - session_low) / open_price * 100 if open_price > 0 else 0

        median_range = stats["daily_range"]["p50"]
        p90_range = stats["daily_range"]["p90"]

        # How much of typical daily range has been consumed
        range_used_pct = (session_range_pct / median_range * 100) if median_range > 0 else 0
        remaining_range_pct = max(0, median_range - session_range_pct)

        underlying_assessment = {
            "ticker": ticker,
            "open": round(open_price, 2),
            "current": round(current_price, 2),
            "session_high": round(session_high, 2),
            "session_low": round(session_low, 2),
            "session_range_pct": round(session_range_pct, 3),
            "typical_daily_range_pct": median_range,
            "range_used_of_typical_pct": round(range_used_pct, 0),
            "remaining_range_pct": round(remaining_range_pct, 2),
            "eu_session_typical_range_pct": round(median_range * EU_SESSION_FACTOR, 2),
            "eu_session_bad_range_pct": round(p90_range * EU_SESSION_FACTOR, 2),
        }

        # Per-position warrant impact
        warrant_impact = {}
        for key, pos in positions.items():
            if not pos.get("active"):
                continue
            is_silver = "silver" in key
            is_gold = "gold" in key
            if (is_silver and underlying_key != "silver") or (is_gold and underlying_key != "gold"):
                continue

            leverage = LEVERAGE_MAP.get(key, 1.0)
            bid = prices.get(key, {}).get("bid", 0)
            stop = pos.get("stop", 0)
            dist_stop_pct = ((bid - stop) / bid * 100) if bid > 0 else 999

            # Expected warrant moves based on underlying range
            median_drop = stats["open_to_low"]["p50"]
            p90_drop = stats["open_to_low"]["p90"]
            p95_drop = stats["open_to_low"]["p95"]

            median_gain = stats["open_to_high"]["p50"]
            p90_gain = stats["open_to_high"]["p90"]

            # EU session adjusted (40% of 24h)
            eu_typical_drop = median_drop * EU_SESSION_FACTOR * leverage
            eu_bad_drop = p90_drop * EU_SESSION_FACTOR * leverage

            # Remaining potential warrant move
            remaining_warrant = remaining_range_pct * leverage

            # Stop safety: can a P90 bad day hit the stop?
            p90_warrant_drop = p90_drop * leverage
            stop_safe = dist_stop_pct > eu_bad_drop
            stop_warning = not stop_safe and dist_stop_pct > eu_typical_drop
            stop_danger = dist_stop_pct <= eu_typical_drop

            warrant_impact[key] = {
                "leverage": leverage,
                "dist_to_stop_pct": round(dist_stop_pct, 1),
                "typical_day_swing_pct": round(median_range * leverage, 1),
                "typical_day_drop_pct": round(median_drop * leverage, 1),
                "bad_day_drop_pct": round(p90_drop * leverage, 1),
                "worst_day_drop_pct": round(p95_drop * leverage, 1),
                "eu_typical_drop_pct": round(eu_typical_drop, 1),
                "eu_bad_drop_pct": round(eu_bad_drop, 1),
                "remaining_potential_pct": round(remaining_warrant, 1),
                "stop_safety": "SAFE" if stop_safe else ("WARNING" if stop_warning else "DANGER"),
                "stop_note": (
                    f"Stop {dist_stop_pct:.1f}% away vs EU P90 drop {eu_bad_drop:.1f}%"
                    if not stop_safe else
                    f"Stop {dist_stop_pct:.1f}% away, EU P90 drop {eu_bad_drop:.1f}% — adequate"
                ),
            }

        assessment[underlying_key] = {
            "underlying": underlying_assessment,
            "warrants": warrant_impact,
        }

    return assessment


# ---------------------------------------------------------------------------
# Spike Catcher — US Open Limit Sell Orders
# ---------------------------------------------------------------------------

SPIKE_STATE_FILE = "data/metals_spike_state.json"


def compute_spike_targets(positions, prices, range_stats, percentile=75, partial_pct=50):
    """Compute limit sell target prices for each position based on historical daily gain.

    Uses the P-th percentile of open_to_high to estimate how high the price might
    spike during the US open session (15:30 CET). Places sell targets at that level.

    Args:
        positions: POSITIONS dict from metals_loop
        prices: current price dict {key: {bid, ask, underlying, ...}}
        range_stats: output from compute_daily_range_stats()
        percentile: which percentile to target (75 = capture P75 spikes)
        partial_pct: what % of position to sell (50 = half, 100 = all)

    Returns:
        dict: {position_key: {target_price, target_pnl_pct, underlying_target,
               current_bid, units_to_sell, reason}}
    """
    targets = {}

    for key, pos in positions.items():
        if not pos.get("active"):
            continue

        p = prices.get(key)
        if not p or not p.get("bid") or p["bid"] <= 0:
            continue

        bid = p["bid"]
        entry = pos["entry"]
        current_pnl = ((bid / entry) - 1) * 100 if entry > 0 else 0

        # Don't place spike orders if losing more than 3%
        if current_pnl < -3.0:
            continue

        # Get underlying ticker and leverage
        is_silver = "silver" in key
        ticker = "XAG-USD" if is_silver else "XAU-USD"
        leverage = LEVERAGE_MAP.get(key, 1.0)

        stats = range_stats.get(ticker)
        if not stats:
            continue

        # Get the target underlying gain (P-th percentile of open_to_high)
        pkey = f"p{percentile}"
        underlying_gain_pct = stats.get("open_to_high", {}).get(pkey, 0)
        if underlying_gain_pct <= 0:
            continue

        # EU session gets ~40% of the 24h range, but US open gets the lion's share
        # Use 60% of the full-day gain as the US session spike target (more aggressive)
        us_session_gain_pct = underlying_gain_pct * 0.60

        # Warrant target = current + (underlying gain * leverage)
        warrant_gain_pct = us_session_gain_pct * leverage
        target_price = bid * (1 + warrant_gain_pct / 100)

        # Round to reasonable price precision
        if target_price >= 100:
            target_price = round(target_price, 1)
        elif target_price >= 10:
            target_price = round(target_price, 2)
        else:
            target_price = round(target_price, 2)

        # Don't place if target is below entry (would be selling at a loss)
        if target_price <= entry:
            continue

        target_pnl = ((target_price / entry) - 1) * 100

        # Calculate units to sell
        total_units = pos["units"]
        units_to_sell = max(1, int(total_units * partial_pct / 100))

        targets[key] = {
            "target_price": target_price,
            "target_pnl_pct": round(target_pnl, 1),
            "current_bid": bid,
            "current_pnl_pct": round(current_pnl, 1),
            "underlying_gain_target_pct": round(us_session_gain_pct, 2),
            "warrant_gain_target_pct": round(warrant_gain_pct, 1),
            "units_to_sell": units_to_sell,
            "total_units": total_units,
            "leverage": leverage,
            "percentile_used": percentile,
            "reason": (f"P{percentile} spike target: +{warrant_gain_pct:.1f}% "
                       f"(underlying +{us_session_gain_pct:.2f}% * {leverage}x)"),
        }

    return targets


def load_spike_state():
    """Load spike order state from file."""
    return _load_json_state(SPIKE_STATE_FILE, _default_spike_state, "Spike state")


def save_spike_state(state):
    """Persist spike order state."""
    try:
        atomic_write_json(SPIKE_STATE_FILE, state, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save spike state: {e}")
