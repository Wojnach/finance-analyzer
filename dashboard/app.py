"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""

import functools
import hmac
import logging
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider

logger = logging.getLogger(__name__)


def _json_safe(value):
    """Convert NaN/Infinity to JSON-safe null recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


class SafeJSONProvider(DefaultJSONProvider):
    """Flask JSON provider that strips non-finite floats."""

    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)


app = Flask(__name__, static_folder="static")
app.json = SafeJSONProvider(app)


_ALLOWED_ORIGINS = {
    "http://localhost:5055",
    "http://127.0.0.1:5055",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}


@app.after_request
def add_cors_headers(response):
    """Allow same-network browser access from known origins only (BUG-230)."""
    origin = request.headers.get("Origin", "")
    if origin in _ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from portfolio.file_utils import load_json as _load_json_impl
from portfolio.file_utils import load_jsonl as _load_jsonl_impl

# ---------------------------------------------------------------------------
# TTL Cache (BUG-130: avoid re-reading files on every API request)
# ---------------------------------------------------------------------------

_cache = {}
_cache_lock = threading.Lock()
_DEFAULT_TTL = 5  # seconds


def _cached_read(key, ttl, read_fn):
    """Return cached result if fresh, otherwise call read_fn and cache."""
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (now - entry[1]) < ttl:
            return entry[0]
    result = read_fn()
    with _cache_lock:
        _cache[key] = (result, now)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path, ttl=_DEFAULT_TTL):
    return _cached_read(f"json:{path}", ttl, lambda: _load_json_impl(path))


def _read_jsonl(path, limit=100, ttl=_DEFAULT_TTL):
    return _cached_read(
        f"jsonl:{path}:{limit}", ttl, lambda: _load_jsonl_impl(path, limit=limit)
    )


def _get_config():
    return _read_json(CONFIG_PATH, ttl=60) or {}


def _parse_limit_arg(name, default, max_value):
    """Parse integer query arg with sane bounds and fallback."""
    try:
        value = int(request.args.get(name, default))
    except (ValueError, TypeError):
        value = default
    return max(1, min(value, max_value))


def _iter_latest_dict_entries(path, read_limit):
    """Yield JSONL entries newest-first, skipping non-dict shapes."""
    raw = _read_jsonl(path, limit=read_limit)
    for entry in reversed(raw):
        if isinstance(entry, dict):
            yield entry


def _parse_iso8601(value):
    """Parse an ISO-8601 timestamp into an aware datetime."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _stockholm_now():
    return datetime.now(UTC).astimezone(STOCKHOLM_TZ)


def _hours_until_stockholm_close(now=None, close_hour=21, close_minute=55):
    """Return hours remaining until the Stockholm warrant close."""
    now = (now or _stockholm_now()).astimezone(STOCKHOLM_TZ)
    close_dt = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
    if now >= close_dt:
        return 0.0
    return round((close_dt - now).total_seconds() / 3600.0, 2)


def _is_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _round_or_none(value, digits=2):
    return round(float(value), digits) if _is_number(value) else None


def _normalize_golddigger_position(raw_position, latest_log):
    if not isinstance(raw_position, dict):
        return None

    quantity = raw_position.get("quantity", raw_position.get("shares"))
    entry_price = raw_position.get("avg_price", raw_position.get("entry_price"))
    current_price = None
    if isinstance(latest_log, dict):
        current_price = latest_log.get("cert_bid", latest_log.get("cert_ask"))
    if current_price is None:
        current_price = raw_position.get("current_price")
    pnl_pct = None
    if _is_number(entry_price) and entry_price > 0 and _is_number(current_price):
        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

    position = dict(raw_position)
    position["shares"] = quantity
    position["quantity"] = quantity
    position["side"] = raw_position.get("side") or raw_position.get("action") or "BUY"
    position["entry_price"] = entry_price
    position["avg_price"] = entry_price
    position["stop_price"] = raw_position.get("stop_price", raw_position.get("stop"))
    position["tp_price"] = raw_position.get("tp_price", raw_position.get("take_profit_price"))
    position["take_profit_price"] = position["tp_price"]
    position["current_price"] = current_price
    position["pnl_pct"] = _round_or_none(
        raw_position.get("pnl_pct") if raw_position.get("pnl_pct") is not None else pnl_pct,
        2,
    )

    has_position = any(
        _is_number(value)
        for value in (quantity, entry_price, current_price)
    )
    return position if has_position else None


def _normalize_golddigger_log_entry(entry):
    if not isinstance(entry, dict):
        return None
    normalized = dict(entry)
    normalized.setdefault("composite_score", entry.get("S"))
    normalized.setdefault("z_gold", entry.get("z_g"))
    normalized.setdefault("z_fx", entry.get("z_f"))
    normalized.setdefault("z_yield", entry.get("z_y"))
    return normalized


def _normalize_golddigger_trade_entry(entry):
    if not isinstance(entry, dict):
        return None

    quantity = entry.get("shares", entry.get("quantity"))
    price_sek = entry.get("price_sek")
    total_sek = entry.get("total_sek")
    if total_sek is None and _is_number(quantity) and _is_number(price_sek):
        total_sek = quantity * price_sek

    normalized = dict(entry)
    normalized.setdefault("shares", quantity)
    normalized.setdefault("total_sek", _round_or_none(total_sek, 2))
    normalized.setdefault("composite_score", entry.get("composite_s"))
    return normalized


def _normalize_golddigger_state(state, log_entries):
    if not isinstance(state, dict) and not log_entries:
        return None
    state = dict(state or {})
    latest_log = log_entries[0] if log_entries else {}
    cfg = (_get_config() or {}).get("golddigger", {})

    state["composite_score"] = latest_log.get("S", state.get("composite_score"))
    state["z_gold"] = latest_log.get("z_g", state.get("z_gold"))
    state["z_fx"] = latest_log.get("z_f", state.get("z_fx"))
    state["z_yield"] = latest_log.get("z_y", state.get("z_yield"))
    state["gold_price"] = latest_log.get("gold", state.get("gold_price"))
    state["usdsek"] = latest_log.get("usdsek", state.get("usdsek"))
    state["ts"] = latest_log.get("ts", state.get("last_poll_time"))
    state["theta_in"] = cfg.get("theta_in", state.get("theta_in", 0.7))
    state["theta_out"] = cfg.get("theta_out", state.get("theta_out", 0.1))
    state["spread_max"] = cfg.get("spread_max", state.get("spread_max", 0.02))
    state["confirm_required"] = cfg.get("confirm_polls", state.get("confirm_required", 3))
    state["risk_fraction"] = cfg.get("risk_fraction", state.get("risk_fraction", 0.005))
    state["max_notional_fraction"] = cfg.get("max_notional_fraction", state.get("max_notional_fraction", 0.10))
    state["leverage"] = cfg.get("leverage", state.get("leverage", 20.0))

    if state.get("confirm_count") is None:
        state["confirm_count"] = 0
    if log_entries:
        confirms = 0
        theta_in = state.get("theta_in", 0.7)
        for entry in log_entries:
            score = entry.get("S")
            z_gold = entry.get("z_g")
            if not _is_number(score) or score < theta_in or (z_gold is not None and z_gold <= 0):
                break
            confirms += 1
        state["confirm_count"] = confirms

    latest_ts = _parse_iso8601(state.get("last_poll_time") or latest_log.get("ts"))
    max_age_seconds = max(60, int(cfg.get("poll_seconds", 5)) * 12)
    state["session_active"] = (
        latest_ts is not None
        and not bool(state.get("halted"))
        and (datetime.now(UTC) - latest_ts).total_seconds() <= max_age_seconds
    )
    state["daily"] = {
        "trade_count": state.get("daily_trades", 0),
        "max_trades": cfg.get("max_daily_trades", 10),
        "pnl_sek": state.get("daily_pnl"),
    }
    state["position"] = _normalize_golddigger_position(state.get("position"), latest_log)
    return state


def _normalize_metals_llm_predictions(raw_llm):
    if not isinstance(raw_llm, dict):
        return {}

    predictions = {}
    for ticker, payload in raw_llm.items():
        if not isinstance(payload, dict):
            continue

        consensus_action = payload.get("consensus_action") or payload.get("consensus")
        consensus_direction = payload.get("consensus_dir")
        pred = {
            "consensus": {
                "weighted_action": consensus_action,
                "direction": consensus_direction,
                "confidence": payload.get("consensus_conf"),
            }
        }

        ministral_action = payload.get("ministral")
        if ministral_action:
            pred["ministral"] = {
                "action": ministral_action,
                "confidence": payload.get("ministral_conf"),
            }

        for horizon in ("1h", "3h"):
            prefix = f"chronos_{horizon}"
            direction = payload.get(prefix)
            pct_move = payload.get(f"{prefix}_pct_move")
            if pct_move is None:
                pct_move = payload.get(f"{prefix}_pct")
            confidence = payload.get(f"{prefix}_conf")
            if direction is None and pct_move is None and confidence is None:
                continue
            pred[prefix] = {
                "direction": direction,
                "pct_move": pct_move,
                "confidence": confidence,
            }

        predictions[ticker] = pred

    return {
        "age_seconds": None,
        "models": ["ministral", "chronos_1h", "chronos_3h"],
        "accuracy": {},
        "predictions": predictions,
    }


def _normalize_metals_forecast_signals(raw_llm):
    if not isinstance(raw_llm, dict):
        return {}

    signals = {}
    for ticker, payload in raw_llm.items():
        if not isinstance(payload, dict):
            continue
        chronos_1h_pct = payload.get("chronos_1h_pct_move")
        if chronos_1h_pct is None:
            chronos_1h_pct = payload.get("chronos_1h_pct")
        action = payload.get("consensus_action") or payload.get("consensus")
        if action is None and chronos_1h_pct is None:
            continue
        signals[ticker] = {
            "action": action,
            "chronos_1h_pct": chronos_1h_pct,
        }

    return {"forecast_signals": signals} if signals else {}


def _normalize_metals_decisions(decisions):
    normalized = []
    for entry in decisions:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        action = (item.get("action") or (item.get("prediction") or {}).get("action") or "HOLD").upper()
        positions = {}
        for key, payload in (item.get("positions") or {}).items():
            if not isinstance(payload, dict):
                continue
            pos_item = dict(payload)
            pos_item.setdefault("action", action)
            positions[key] = pos_item
        item["positions"] = positions
        if not item.get("reasoning"):
            item["reasoning"] = item.get("trigger") or item.get("thesis_status") or ""
        normalized.append(item)
    return normalized


def _drawdown_level_from_pct(drawdown_pct):
    if not _is_number(drawdown_pct):
        return "UNKNOWN"
    if drawdown_pct <= -15.0:
        return "EMERGENCY"
    if drawdown_pct <= -10.0:
        return "WARNING"
    return "OK"


def _normalize_metals_risk(risk):
    if not isinstance(risk, dict):
        return {}

    item = dict(risk)
    drawdown = item.get("drawdown")
    if isinstance(drawdown, dict) and "level" not in drawdown:
        drawdown = dict(drawdown)
        drawdown["level"] = _drawdown_level_from_pct(drawdown.get("current_drawdown_pct"))
        item["drawdown"] = drawdown

    trade_guards = item.get("trade_guards")
    if isinstance(trade_guards, dict) and "status" not in trade_guards:
        tg_item = dict(trade_guards)
        tg_item["status"] = "warnings" if tg_item else "unknown"
        item["trade_guards"] = tg_item

    return item


def _normalize_metals_context(context):
    if not isinstance(context, dict):
        return context
    item = dict(context)
    item["risk"] = _normalize_metals_risk(item.get("risk"))
    return item


def _merge_missing_structure(primary, fallback):
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    if isinstance(primary, dict) and isinstance(fallback, dict):
        merged = dict(primary)
        for key, fallback_value in fallback.items():
            primary_value = merged.get(key)
            if primary_value is None:
                merged[key] = fallback_value
                continue
            if isinstance(primary_value, dict) and not primary_value:
                merged[key] = fallback_value
                continue
            if isinstance(primary_value, list) and not primary_value:
                merged[key] = fallback_value
                continue
            merged[key] = _merge_missing_structure(primary_value, fallback_value)
        return merged
    return primary


def _build_metals_context_fallback(decisions):
    positions_state = _read_json(DATA_DIR / "metals_positions_state.json") or {}
    signal_entries = list(_iter_latest_dict_entries(DATA_DIR / "metals_signal_log.jsonl", read_limit=10))
    value_history = _read_jsonl(DATA_DIR / "metals_value_history.jsonl", limit=10)
    technicals = _read_json(DATA_DIR / "silver_analysis.json") or {}
    latest_signal = signal_entries[0] if signal_entries else {}
    latest_value = value_history[-1] if value_history else {}
    latest_decision = decisions[0] if decisions else {}

    if not positions_state and not latest_signal and not latest_value and not latest_decision:
        return None

    prices = latest_signal.get("prices", {}) if isinstance(latest_signal, dict) else {}
    latest_decision_positions = latest_decision.get("positions", {}) if isinstance(latest_decision, dict) else {}
    latest_value_positions = latest_value.get("positions", {}) if isinstance(latest_value, dict) else {}

    active_keys = [
        key for key, payload in positions_state.items()
        if isinstance(payload, dict) and payload.get("active")
    ]
    if not active_keys:
        active_keys = list(latest_decision_positions.keys())

    positions = {}
    total_invested = latest_value.get("total_invested")
    total_value = latest_value.get("total_value")
    total_pnl_pct = latest_value.get("pnl_pct")

    for key in active_keys:
        state_payload = positions_state.get(key, {})
        decision_payload = latest_decision_positions.get(key, {})
        value_payload = latest_value_positions.get(key, {})

        units = state_payload.get("units", decision_payload.get("units"))
        entry = state_payload.get("entry", decision_payload.get("entry"))
        stop = state_payload.get("stop", decision_payload.get("stop"))
        bid = prices.get(key, value_payload.get("bid", decision_payload.get("bid")))
        invested = (units * entry) if _is_number(units) and _is_number(entry) else None
        value_sek = value_payload.get("value")
        if value_sek is None and _is_number(units) and _is_number(bid):
            value_sek = units * bid
        profit_sek = None
        if _is_number(value_sek) and _is_number(invested):
            profit_sek = value_sek - invested

        pnl_pct = value_payload.get("pnl_pct", decision_payload.get("pnl_pct"))
        if pnl_pct is None and _is_number(entry) and entry > 0 and _is_number(bid):
            pnl_pct = ((bid - entry) / entry) * 100.0

        dist_stop_pct = decision_payload.get("dist_stop_pct")
        if dist_stop_pct is None and _is_number(bid) and bid > 0 and _is_number(stop):
            dist_stop_pct = ((bid - stop) / bid) * 100.0

        positions[key] = {
            "name": decision_payload.get("name", key),
            "units": units,
            "entry": entry,
            "bid": bid,
            "ask": prices.get(f"{key}_ask"),
            "pnl_pct": _round_or_none(pnl_pct, 2),
            "value_sek": _round_or_none(value_sek, 1),
            "invested_sek": _round_or_none(invested, 1),
            "profit_sek": _round_or_none(profit_sek, 1),
            "peak_bid": None,
            "from_peak_pct": _round_or_none(decision_payload.get("from_peak_pct"), 2),
            "stop": stop,
            "dist_to_stop_pct": _round_or_none(dist_stop_pct, 2),
            "day_change_pct": None,
            "leverage": None,
            "barrier": None,
            "active": True,
        }

    if total_invested is None:
        invested_values = [payload.get("invested_sek") for payload in positions.values() if _is_number(payload.get("invested_sek"))]
        total_invested = sum(invested_values) if invested_values else None
    if total_value is None:
        value_values = [payload.get("value_sek") for payload in positions.values() if _is_number(payload.get("value_sek"))]
        total_value = sum(value_values) if value_values else None
    if total_pnl_pct is None and _is_number(total_invested) and total_invested > 0 and _is_number(total_value):
        total_pnl_pct = ((total_value / total_invested) - 1.0) * 100.0

    drawdown_pct = None
    if isinstance(latest_decision, dict):
        drawdown_pct = (latest_decision.get("risk") or {}).get("drawdown_pct")

    price_history_recent = []
    gold_fallback = ((technicals.get("context") or {}).get("gold_price"))
    for entry in reversed(signal_entries):
        snap_prices = entry.get("prices", {})
        price_history_recent.append({
            "ts": entry.get("ts"),
            "gold": snap_prices.get("gold") or snap_prices.get("XAU-USD") or gold_fallback,
            "gold_und": snap_prices.get("gold_und") or snap_prices.get("XAU-USD"),
            "silver79": snap_prices.get("silver79"),
            "silver79_und": snap_prices.get("silver79_und") or snap_prices.get("XAG-USD"),
            "silver301": snap_prices.get("silver301"),
            "silver301_und": snap_prices.get("silver301_und") or snap_prices.get("XAG-USD"),
        })

    silver_price = (
        prices.get("XAG-USD")
        or prices.get("silver301_und")
        or prices.get("silver79_und")
        or ((technicals.get("price") or {}).get("current"))
    )
    gold_price = prices.get("XAU-USD") or prices.get("gold_und") or gold_fallback
    now_sthlm = _stockholm_now()

    return {
        "timestamp": latest_signal.get("ts") or latest_decision.get("ts"),
        "cet_time": now_sthlm.strftime("%H:%M %Z"),
        "check_count": latest_signal.get("check") or latest_decision.get("check_count"),
        "invoke_count": latest_decision.get("invoke_count"),
        "trigger_reason": (
            (latest_signal.get("trigger_reasons") or [None])[0]
            or latest_decision.get("trigger")
        ),
        "tier": latest_decision.get("tier"),
        "market_close_cet": "21:55",
        "hours_remaining": _hours_until_stockholm_close(now_sthlm),
        "positions": positions,
        "underlying": {
            "gold": {"price": gold_price} if gold_price is not None else {},
            "silver": {"price": silver_price} if silver_price is not None else {},
        },
        "totals": {
            "invested": _round_or_none(total_invested, 0),
            "current": _round_or_none(total_value, 0),
            "pnl_pct": _round_or_none(total_pnl_pct, 2),
            "profit_sek": _round_or_none(
                (total_value - total_invested)
                if _is_number(total_value) and _is_number(total_invested)
                else None,
                0,
            ),
        },
        "price_history_recent": price_history_recent,
        "signals": _merge_missing_structure(
            latest_signal.get("signals", {}),
            _normalize_metals_forecast_signals(
                latest_signal.get("llm") or latest_decision.get("llm")
            ),
        ),
        "recent_decisions": decisions[:5],
        "short_instruments": {},
        "llm_predictions": _normalize_metals_llm_predictions(
            latest_signal.get("llm") or latest_decision.get("llm")
        ),
        "risk": {
            "monte_carlo": {},
            "drawdown": {
                "current_drawdown_pct": _round_or_none(drawdown_pct, 2),
                "level": _drawdown_level_from_pct(drawdown_pct),
            },
            "trade_guards": {
                "status": "warnings" if latest_signal.get("triggered") else "all_clear",
                "reason": "; ".join((latest_signal.get("trigger_reasons") or [])[:2]) or None,
            },
        },
        "trades_today_file": "data/metals_trades.jsonl",
    }


def _aggregate_accuracy_bucket(bucket):
    """Aggregate nested accuracy stats into one accuracy/total pair."""
    if not isinstance(bucket, dict):
        return {"accuracy": None, "total": 0, "correct": 0}

    correct = 0
    total = 0
    for stats in bucket.values():
        if not isinstance(stats, dict):
            continue
        correct += int(stats.get("correct", 0) or 0)
        total += int(stats.get("total", 0) or 0)

    return {
        "accuracy": round(correct / total, 3) if total else None,
        "correct": correct,
        "total": total,
    }


def _build_local_llm_trend_point(entry, ticker=None):
    """Flatten one local-LLM history entry into chart-friendly metrics."""
    ticker = (ticker or "").upper() or None
    ministral = ((entry.get("ministral") or {}).get("overall") or {})
    by_ticker = ((entry.get("ministral") or {}).get("by_ticker") or {})
    ticker_stats = by_ticker.get(ticker, {}) if ticker else {}
    health = entry.get("health") or {}
    forecast = entry.get("forecast") or {}
    gating = (entry.get("gating_counts") or {}).get("forecast") or {}

    raw_1h = _aggregate_accuracy_bucket((forecast.get("raw") or {}).get("1h"))
    raw_24h = _aggregate_accuracy_bucket((forecast.get("raw") or {}).get("24h"))
    effective_1h = _aggregate_accuracy_bucket((forecast.get("effective") or {}).get("1h"))
    effective_24h = _aggregate_accuracy_bucket((forecast.get("effective") or {}).get("24h"))

    return {
        "date": entry.get("date"),
        "exported_at": entry.get("exported_at"),
        "days": entry.get("days"),
        "ticker": ticker,
        "ministral_accuracy": ministral.get("accuracy"),
        "ministral_samples": ministral.get("samples", 0),
        "ministral_ticker_accuracy": ticker_stats.get("accuracy"),
        "ministral_ticker_samples": ticker_stats.get("samples", 0),
        "chronos_success_rate": (health.get("chronos") or {}).get("success_rate"),
        "chronos_total": (health.get("chronos") or {}).get("total", 0),
        "kronos_success_rate": (health.get("kronos") or {}).get("success_rate"),
        "kronos_total": (health.get("kronos") or {}).get("total", 0),
        "forecast_raw_1h_accuracy": raw_1h["accuracy"],
        "forecast_raw_1h_total": raw_1h["total"],
        "forecast_raw_24h_accuracy": raw_24h["accuracy"],
        "forecast_raw_24h_total": raw_24h["total"],
        "forecast_effective_1h_accuracy": effective_1h["accuracy"],
        "forecast_effective_1h_total": effective_1h["total"],
        "forecast_effective_24h_accuracy": effective_24h["accuracy"],
        "forecast_effective_24h_total": effective_24h["total"],
        "forecast_gating_raw": gating.get("raw", 0),
        "forecast_gating_held": gating.get("held", 0),
        "forecast_gating_insufficient_data": gating.get("insufficient_data", 0),
        "forecast_gating_vol_gated": gating.get("vol_gated", 0),
    }


# ---------------------------------------------------------------------------
# Token authentication middleware
# ---------------------------------------------------------------------------

def _get_dashboard_token():
    """Return the configured dashboard_token, or None if not set."""
    cfg = _get_config()
    return cfg.get("dashboard_token") or None


def require_auth(f):
    """Decorator: checks ?token= query param or Authorization: Bearer header.

    If no dashboard_token is configured, access is allowed (backwards compatible).
    Returns 401 for invalid tokens.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        expected = _get_dashboard_token()
        if expected is None:
            # No token configured — allow unauthenticated access
            return f(*args, **kwargs)

        # Check query param — timing-safe comparison prevents brute-force
        token = request.args.get("token")
        if token and hmac.compare_digest(token, expected):
            return f(*args, **kwargs)

        # Check Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:].strip()
            if hmac.compare_digest(bearer_token, expected):
                return f(*args, **kwargs)

        return jsonify({"error": "Unauthorized", "message": "Invalid or missing token"}), 401

    return decorated


# ---------------------------------------------------------------------------
# Routes — Static
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------------------------------------------------------------------
# Routes — API (all require auth)
# ---------------------------------------------------------------------------

@app.route("/api/summary")
@require_auth
def api_summary():
    """Combined endpoint for auto-refresh: signals + both portfolios + telegrams."""
    sig = _read_json(DATA_DIR / "agent_summary.json")
    port = _read_json(DATA_DIR / "portfolio_state.json")
    port_bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
    tel = list(_iter_latest_dict_entries(DATA_DIR / "telegram_messages.jsonl", read_limit=50))
    return jsonify({
        "signals": sig,
        "portfolio": port,
        "portfolio_bold": port_bold,
        "telegrams": tel,
    })


@app.route("/api/signals")
@require_auth
def api_signals():
    data = _read_json(DATA_DIR / "agent_summary.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/portfolio")
@require_auth
def api_portfolio():
    data = _read_json(DATA_DIR / "portfolio_state.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/portfolio-bold")
@require_auth
def api_portfolio_bold():
    data = _read_json(DATA_DIR / "portfolio_state_bold.json")
    if not data:
        return jsonify({"error": "no data"}), 404
    return jsonify(data)


@app.route("/api/mstr_loop")
@require_auth
def api_mstr_loop():
    """Live snapshot of the MSTR Loop bot (v2 Tier 3).

    Returns state + scorecard + latest poll in one JSON:
        {
          "state": {cash, positions, total_pnl, ...},
          "scorecard": {win_rate, expectancy, trades_by_strategy, ...},
          "last_poll": {last cycle snapshot from mstr_loop_poll.jsonl},
          "last_trade": {last closed trade from mstr_loop_trades.jsonl},
        }
    """
    out = {
        "state": _read_json(DATA_DIR / "mstr_loop_state.json") or {},
        "scorecard": _read_json(DATA_DIR / "mstr_loop_scorecard.json") or {},
        "last_poll": None,
        "last_trade": None,
    }
    import json as _json
    poll_path = DATA_DIR / "mstr_loop_poll.jsonl"
    if poll_path.exists():
        try:
            with open(poll_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            out["last_poll"] = _json.loads(line)
                        except _json.JSONDecodeError:
                            pass
        except OSError:
            pass
    trades_path = DATA_DIR / "mstr_loop_trades.jsonl"
    if trades_path.exists():
        try:
            with open(trades_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            out["last_trade"] = _json.loads(line)
                        except _json.JSONDecodeError:
                            pass
        except OSError:
            pass
    return jsonify(out)


@app.route("/api/invocations")
@require_auth
def api_invocations():
    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/telegrams")
@require_auth
def api_telegrams():
    """Return telegram messages with optional filtering.

    Query params:
      - limit: max entries (default 200, max 2000)
      - category: filter by category (trade, analysis, iskbets, bigbet, digest, etc.)
      - search: text search in message body
    """
    limit = _parse_limit_arg("limit", default=200, max_value=2000)
    category_filter = request.args.get("category", "").strip().lower()
    search_filter = request.args.get("search", "").strip().lower()

    results = []
    for entry in _iter_latest_dict_entries(DATA_DIR / "telegram_messages.jsonl", read_limit=5000):
        if category_filter and (entry.get("category", "") or "").lower() != category_filter:
            continue
        if search_filter and search_filter not in (entry.get("text", "") or "").lower():
            continue
        results.append(entry)
        if len(results) >= limit:
            break

    return jsonify(results)


@app.route("/api/signal-log")
@require_auth
def api_signal_log():
    entries = _read_jsonl(DATA_DIR / "signal_log.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/accuracy")
@require_auth
def api_accuracy():
    try:
        from portfolio.accuracy_stats import (
            consensus_accuracy,
            per_ticker_accuracy,
            signal_accuracy,
        )

        result = {}
        for horizon in ["1d", "3d", "5d", "10d"]:
            sa = signal_accuracy(horizon)
            ca = consensus_accuracy(horizon)
            ta = per_ticker_accuracy(horizon)
            if ca["total"] > 0:
                result[horizon] = {
                    "signals": sa,
                    "consensus": ca,
                    "per_ticker": ta,
                }
        return jsonify(result)
    except Exception as e:
        logger.exception("accuracy endpoint error")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/iskbets")
@require_auth
def api_iskbets():
    config = _read_json(DATA_DIR / "iskbets_config.json")
    state = _read_json(DATA_DIR / "iskbets_state.json")
    return jsonify({"config": config, "state": state})


@app.route("/api/lora-status")
@require_auth
def api_lora_status():
    state = _read_json(TRAINING_DIR / "state.json")
    progress = _read_json(TRAINING_DIR / "training_progress.json")
    return jsonify({"state": state, "training_progress": progress})


# ---------------------------------------------------------------------------
# New: Portfolio validation
# ---------------------------------------------------------------------------

@app.route("/api/validate-portfolio", methods=["POST"])
@require_auth
def api_validate_portfolio():
    """Validate a portfolio JSON for integrity.

    Delegates to portfolio_validator.validate_portfolio() which performs
    comprehensive checks: cash, holdings, fees, transactions, avg_cost.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"valid": False, "errors": ["No JSON body provided"]}), 400

    try:
        from portfolio.portfolio_validator import validate_portfolio
        errors = validate_portfolio(data)
    except Exception as e:
        return jsonify({"valid": False, "errors": [f"Validation error: {e}"]}), 500

    return jsonify({
        "valid": len(errors) == 0,
        "errors": errors,
    })


# ---------------------------------------------------------------------------
# New: Equity curve
# ---------------------------------------------------------------------------

@app.route("/api/equity-curve")
@require_auth
def api_equity_curve():
    """Return portfolio value history for charting.

    Reads data/portfolio_value_history.jsonl. Returns empty array if missing.
    """
    entries = _read_jsonl(DATA_DIR / "portfolio_value_history.jsonl", limit=5000)
    return jsonify(entries)


# ---------------------------------------------------------------------------
# New: Signal heatmap (30 signals x all tickers)
# ---------------------------------------------------------------------------

@app.route("/api/signal-heatmap")
@require_auth
def api_signal_heatmap():
    """Return the full 30-signal x all-tickers grid.

    Each cell is BUY/SELL/HOLD. Built from agent_summary.json signals + enhanced_signals.
    """
    summary = _read_json(DATA_DIR / "agent_summary.json")
    if not summary:
        return jsonify({"error": "no data"}), 404

    signals_data = summary.get("signals", {})

    # Core signal names (11 total: 8 active + 3 disabled)
    core_signals = [
        "rsi", "macd", "ema", "bb", "fear_greed", "sentiment",
        "ministral", "volume", "ml", "funding", "custom_lora"
    ]
    # Enhanced composite signal names (19 modules, signals #12-#30)
    enhanced_signals = [
        "trend", "momentum", "volume_flow", "volatility_sig",
        "candlestick", "structure", "fibonacci", "smart_money",
        "oscillators", "heikin_ashi", "mean_reversion", "calendar",
        "macro_regime", "momentum_factors", "news_event", "econ_calendar",
        "forecast", "claude_fundamental", "futures_flow"
    ]
    all_signals = core_signals + enhanced_signals

    heatmap = {}
    tickers = list(signals_data.keys())

    for ticker in tickers:
        sig = signals_data[ticker]
        extra = sig.get("extra", {})
        votes = extra.get("_votes", {})

        # _votes contains all 30 signal keys (core + enhanced)
        row = {}
        for s in all_signals:
            row[s] = (votes.get(s, "HOLD") or "HOLD").upper()
        heatmap[ticker] = row

    return jsonify({
        "tickers": tickers,
        "signals": all_signals,
        "core_signals": core_signals,
        "enhanced_signals": enhanced_signals,
        "heatmap": heatmap,
    })


# ---------------------------------------------------------------------------
# New: Trigger activity timeline
# ---------------------------------------------------------------------------

@app.route("/api/triggers")
@require_auth
def api_triggers():
    """Return last 50 trigger/invocation events from invocations.jsonl."""
    entries = _read_jsonl(DATA_DIR / "invocations.jsonl", limit=50)
    return jsonify(entries)


@app.route("/api/accuracy-history")
@require_auth
def api_accuracy_history():
    """Return accuracy snapshots over time for charting trend lines."""
    entries = _read_jsonl(DATA_DIR / "accuracy_snapshots.jsonl", limit=500)
    return jsonify(entries)


@app.route("/api/local-llm-trends")
@require_auth
def api_local_llm_trends():
    """Return local-LLM report trend data for dashboard charts.

    Query params:
      - limit: number of history points to return (default 90, max 366)
      - ticker: optional ticker filter for Ministral per-ticker series
    """
    limit = _parse_limit_arg("limit", default=90, max_value=366)
    ticker = request.args.get("ticker", "").strip().upper() or None
    latest = _read_json(DATA_DIR / "local_llm_report_latest.json")
    history = _read_jsonl(DATA_DIR / "local_llm_report_history.jsonl", limit=limit)

    return jsonify({
        "ticker": ticker,
        "latest": latest,
        "series": [
            _build_local_llm_trend_point(entry, ticker=ticker)
            for entry in history
            if isinstance(entry, dict)
        ],
    })


@app.route("/api/metals-accuracy")
@require_auth
def api_metals_accuracy():
    """Return metals loop signal accuracy (1h/3h horizons)."""
    data = _read_json(DATA_DIR / "metals_signal_accuracy.json")
    if not data:
        return jsonify({"error": "no data", "stats": {}})
    return jsonify(data)


@app.route("/api/trades")
@require_auth
def api_trades():
    """Return combined transactions from both portfolio states for chart annotations."""
    patient = _read_json(DATA_DIR / "portfolio_state.json")
    bold = _read_json(DATA_DIR / "portfolio_state_bold.json")
    trades = []
    if patient and patient.get("transactions"):
        for tx in patient["transactions"]:
            trades.append({
                "ts": tx.get("timestamp", ""),
                "ticker": tx.get("ticker", ""),
                "action": tx.get("action", ""),
                "total_sek": tx.get("total_sek", 0),
                "price_usd": tx.get("price_usd", 0),
                "strategy": "patient",
            })
    if bold and bold.get("transactions"):
        for tx in bold["transactions"]:
            trades.append({
                "ts": tx.get("timestamp", ""),
                "ticker": tx.get("ticker", ""),
                "action": tx.get("action", ""),
                "total_sek": tx.get("total_sek", 0),
                "price_usd": tx.get("price_usd", 0),
                "strategy": "bold",
            })
    trades.sort(key=lambda t: t.get("ts", ""))
    return jsonify(trades)


@app.route("/api/decisions")
@require_auth
def api_decisions():
    """Return Layer 2 decision history with optional filtering.

    Query params:
      - limit: max entries (default 50, max 500)
      - ticker: filter by ticker (e.g., BTC-USD)
      - action: filter by action (BUY, SELL, HOLD)
      - strategy: filter by strategy (patient, bold)
    """
    limit = _parse_limit_arg("limit", default=50, max_value=500)
    ticker_filter = request.args.get("ticker", "").upper()
    action_filter = request.args.get("action", "").upper()
    strategy_filter = request.args.get("strategy", "").lower()

    results = []
    for entry in _iter_latest_dict_entries(DATA_DIR / "layer2_journal.jsonl", read_limit=1000):
        # Apply action/strategy filters
        if action_filter or strategy_filter:
            decisions = entry.get("decisions", {})
            matched = False
            for strat, dec in decisions.items():
                if strategy_filter and strat != strategy_filter:
                    continue
                if action_filter and dec.get("action", "").upper() != action_filter:
                    continue
                matched = True
            if not matched:
                continue

        if ticker_filter:
            tickers = entry.get("tickers", {})
            if ticker_filter not in tickers:
                continue

        results.append(entry)
        if len(results) >= limit:
            break

    return jsonify(results)


@app.route("/api/health")
@require_auth
def api_health():
    """Return system health summary (loop heartbeat, errors, agent silence)."""
    try:
        from portfolio.health import get_health_summary
        return jsonify(get_health_summary())
    except Exception as e:
        logger.exception("health endpoint error")
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# New: Warrant portfolio
# ---------------------------------------------------------------------------

@app.route("/api/warrants")
@require_auth
def api_warrants():
    """Return warrant holdings with leverage P&L.

    Reads data/portfolio_state_warrants.json. Returns empty structure if missing.
    """
    data = _read_json(DATA_DIR / "portfolio_state_warrants.json")
    if not data:
        return jsonify({"holdings": {}, "transactions": []})
    return jsonify(data)


# ---------------------------------------------------------------------------
# New: Risk data (Monte Carlo + VaR)
# ---------------------------------------------------------------------------

@app.route("/api/risk")
@require_auth
def api_risk():
    """Return Monte Carlo price bands and Portfolio VaR from compact summary.

    Reads monte_carlo and portfolio_var sections from agent_summary_compact.json.
    """
    compact = _read_json(DATA_DIR / "agent_summary_compact.json")
    if not compact:
        return jsonify({"monte_carlo": {}, "portfolio_var": {}})
    return jsonify({
        "monte_carlo": compact.get("monte_carlo", {}),
        "portfolio_var": compact.get("portfolio_var", {}),
    })


# ---------------------------------------------------------------------------
# New: Metals monitoring
# ---------------------------------------------------------------------------

@app.route("/api/metals")
@require_auth
def api_metals():
    """Return combined metals monitoring data.

    Reads:
      - data/metals_context.json — live positions, P&L, risk, signals, prices
      - data/metals_decisions.jsonl — decision log (newest first, last 50)
      - data/metals_history.json — YTD stats + daily OHLCV
      - data/silver_analysis.json — multi-TF technicals

    Falls back to the currently-available loop outputs when metals_context.json
    has not been written yet, so the Metals tab still renders partial live data.
    """
    decisions = _normalize_metals_decisions(
        list(_iter_latest_dict_entries(DATA_DIR / "metals_decisions.jsonl", read_limit=50))
    )
    context = _normalize_metals_context(_read_json(DATA_DIR / "metals_context.json"))
    fallback_context = _build_metals_context_fallback(decisions)
    context = _merge_missing_structure(context, fallback_context)
    history = _read_json(DATA_DIR / "metals_history.json")
    technicals = _read_json(DATA_DIR / "silver_analysis.json")
    return jsonify({
        "context": context,
        "decisions": decisions,
        "history": history,
        "technicals": technicals,
    })


# ---------------------------------------------------------------------------
# New: GoldDigger monitoring
# ---------------------------------------------------------------------------

@app.route("/api/golddigger")
@require_auth
def api_golddigger():
    """Return GoldDigger signal data normalized for the dashboard.

    The bot persists a lean state snapshot plus compact JSONL logs. This route
    reshapes those records into the richer schema expected by the dashboard UI.
    """
    raw_log = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_log.jsonl", read_limit=100))
    raw_trades = list(_iter_latest_dict_entries(DATA_DIR / "golddigger_trades.jsonl", read_limit=50))
    state = _normalize_golddigger_state(_read_json(DATA_DIR / "golddigger_state.json"), raw_log)
    log = [entry for entry in (_normalize_golddigger_log_entry(item) for item in raw_log) if entry]
    trades = [entry for entry in (_normalize_golddigger_trade_entry(item) for item in raw_trades) if entry]
    return jsonify({
        "state": state if state or log or trades else None,
        "log": log,
        "trades": trades,
    })


# ---------------------------------------------------------------------------
# Market health
# ---------------------------------------------------------------------------

@app.route("/api/market-health")
@require_auth
def api_market_health():
    """Return market health snapshot (distribution days, FTD, breadth score).

    Also includes exposure recommendation and earnings proximity data.
    """
    try:
        result = {}
        # Market health from agent_summary (pre-computed) or live
        summary = _read_json(DATA_DIR / "agent_summary.json")
        if summary and "market_health" in summary:
            result["market_health"] = summary["market_health"]
        else:
            try:
                from portfolio.market_health import get_market_health
                mh = get_market_health()
                if mh:
                    result["market_health"] = mh
            except Exception:
                # BUG-205: log at debug so a broken market_health source is
                # diagnosable instead of silently omitting the field.
                logger.debug("market_health enrichment failed", exc_info=True)

        if summary:
            if "exposure_recommendation" in summary:
                result["exposure_recommendation"] = summary["exposure_recommendation"]
            if "earnings_proximity" in summary:
                result["earnings_proximity"] = summary["earnings_proximity"]

        return jsonify(result)
    except Exception as e:
        logger.exception("mstr endpoint error")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
