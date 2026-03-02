"""Metals risk management: Monte Carlo VaR, trade guards, drawdown circuit breaker.

Standalone module for the metals intraday loop. Adapts portfolio/monte_carlo.py,
portfolio/trade_guards.py, and portfolio/risk_management.py for warrant trading.

Usage from metals_loop.py:
    from metals_risk import (
        simulate_warrant_risk, check_trade_guard, record_metals_trade,
        check_portfolio_drawdown, get_risk_summary
    )
"""

import json
import math
import os
import time
import datetime
import logging

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
LEVERAGE_MAP = {
    "gold": 8.0,       # BULL GULD X8
    "silver79": 5.0,   # MINI L SILVER AVA 79 (effective ~5x)
    "silver301": 4.3,  # MINI L SILVER AVA 301
}


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
        if llm_signals:
            # Map position key to ticker
            ticker = "XAG-USD" if "silver" in key else "XAU-USD"
            llm_data = llm_signals.get(ticker, {})
            consensus = llm_data.get("consensus", {})
            if consensus.get("direction") == "up":
                direction_prob = 0.5 + consensus.get("confidence", 0) * 0.3  # scale 0.5-0.8
            elif consensus.get("direction") == "down":
                direction_prob = 0.5 - consensus.get("confidence", 0) * 0.3  # scale 0.2-0.5

        # ATR for underlying — use from signal data or default
        atr_pct = 4.4 if "silver" in key else 1.9  # YTD defaults from metals_history
        if signal_data:
            ticker = "XAG-USD" if "silver" in key else "XAU-USD"
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
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "ticker_trades": {},        # "gold" -> last trade ISO timestamp
        "consecutive_losses": 0,
        "session_trade_count": 0,
        "session_date": None,
        "new_position_times": [],   # ISO timestamps of BUY trades
    }


def _save_guard_state(state):
    """Persist trade guard state."""
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save guard state: {e}")


def check_trade_guard(position_key, action, current_time=None):
    """Check if a trade should be blocked by any guard.

    Returns:
        list of warning dicts (empty = all clear)
        Each warning: {guard, severity, message, details}
    """
    state = _load_guard_state()
    now = current_time or datetime.datetime.now(datetime.timezone.utc)
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
    now = datetime.datetime.now(datetime.timezone.utc)
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
    now = datetime.datetime.now(datetime.timezone.utc)
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
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log portfolio value: {e}")

    return entry


def check_portfolio_drawdown(positions, prices):
    """Check if portfolio has breached drawdown limits.

    Returns:
        dict with drawdown status and metrics
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
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("total_value", 0) > peak_value:
                            peak_value = entry["total_value"]
                    except Exception:
                        pass
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

def get_risk_summary(positions, prices, signal_data=None, llm_signals=None):
    """Generate a compact risk summary for inclusion in metals_context.json.

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
        result["drawdown"] = check_portfolio_drawdown(positions, prices)
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
    for key, mc_data in result.get("monte_carlo", {}).items():
        if isinstance(mc_data, dict):
            p_stop = mc_data.get("p_stop_8h", 0)
            if isinstance(p_stop, (int, float)) and p_stop > 0.2:
                score += 15  # significant stop risk

    # Trade guard blocks add risk awareness
    for key, warns in guards.items():
        if warns:
            score += 5

    result["risk_score"] = min(score, 100)

    return result
