"""Portfolio Intelligence Dashboard — lightweight Flask API + frontend."""

import functools
import hmac
import logging
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, make_response, redirect, request, send_from_directory
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
from portfolio.file_utils import load_jsonl_tail as _load_jsonl_tail_impl

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
    """Cached JSONL read returning the last `limit` entries.

    Switched from load_jsonl(limit=) (full scan + deque) to
    load_jsonl_tail (seek from end). For an 80MB log the difference is
    ~880ms vs ~5ms.

    2026-05-04 codex P2-1 follow-up: the original 4 MB tail-bytes
    ceiling could silently under-deliver entries when callers ask for
    a large window AND individual rows are large (e.g. /api/telegrams
    requests 5000 entries × up to 4 KB each ≈ 20 MB needed). The
    fetcher now grows tail_bytes adaptively — doubling on each retry
    until either `limit` rows are parsed or the whole file has been
    pulled — and falls through to the full-scan path as a final
    safety net. Cache key bumped to v2 so old (potentially
    under-delivered) entries don't survive the deploy.
    """
    if limit and limit > 0:
        return _cached_read(
            f"jsonl_tail_v2:{path}:{limit}",
            ttl,
            lambda: _read_tail_with_growth(path, limit),
        )
    return _cached_read(
        f"jsonl:{path}:{limit}", ttl, lambda: _load_jsonl_impl(path, limit=limit)
    )


def _read_tail_with_growth(path, limit):
    """Read tail entries, doubling tail_bytes until we have `limit`
    parsed rows or the whole file has been consumed.

    Falls back to the full-scan load_jsonl path if even reading the
    full file via the tail helper still yields < limit entries —
    that case implies the tail helper's first-line-drop heuristic is
    chewing through real data and we should bypass it entirely.
    """
    try:
        file_size = Path(path).stat().st_size
    except (FileNotFoundError, OSError):
        return []
    if file_size == 0:
        return []

    # Initial budget: ~1 KB per entry with a 512 KB floor.
    tail_bytes = max(512_000, limit * 1024)
    # Cap retry budget at 64 MB to avoid runaway reads on a corrupt or
    # absurdly-sized file. Most logs in this codebase are < 100 MB and
    # 64 MB will hold ~64 K typical-sized entries.
    max_retry_bytes = 64 * 1024 * 1024
    while True:
        capped = min(tail_bytes, file_size, max_retry_bytes)
        rows = _load_jsonl_tail_impl(path, max_entries=limit,
                                       tail_bytes=capped)
        if len(rows) >= limit or capped >= file_size or capped >= max_retry_bytes:
            break
        tail_bytes *= 2

    # Last-chance fallback: if even the full-file tail came up short,
    # the issue isn't byte budget — it's the first-line-drop heuristic.
    # Fall through to the canonical full-scan reader.
    if len(rows) < limit and capped >= file_size:
        rows = _load_jsonl_impl(path, limit=limit)
    return rows


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

# Auth + cookie machinery moved to dashboard/auth.py on 2026-05-02 to break
# the circular import with dashboard/house_blueprint.py. We re-import here
# so existing references (`require_auth`, `COOKIE_NAME`, etc.) keep working
# inside this module's body, and so any lingering external code that does
# `from dashboard.app import require_auth` still resolves. Tests should
# patch `dashboard.auth.*` directly — patches on `dashboard.app.*` will not
# take effect since require_auth resolves names via dashboard.auth's
# module globals.
from dashboard.auth import (  # noqa: E402
    COOKIE_MAX_AGE,
    COOKIE_NAME,
    _get_config as _auth_get_config,  # noqa: F401 — kept for compat
    _get_dashboard_token,
    _refresh_cookie,
    require_auth,
)


# ---------------------------------------------------------------------------
# Routes — Static
# ---------------------------------------------------------------------------

@app.route("/")
@require_auth
def index():
    # If the user arrived via ?token=XXX, the cookie was just set in
    # require_auth. Redirect to a token-less URL so the address bar (and
    # whatever the user bookmarks next) stays clean. The redirect inherits
    # the Set-Cookie from require_auth's wrapped response.
    if request.args.get("token"):
        return redirect("/", code=302)
    return send_from_directory("static", "index.html")


@app.route("/legacy")
@require_auth
def index_legacy():
    # Pre-redesign single-file dashboard preserved as a fallback during the
    # 2026-05-03 mobile-first rollout. See docs/PLAN.md.
    if request.args.get("token"):
        return redirect("/legacy", code=302)
    return send_from_directory("static", "index_legacy.html")


@app.route("/logout")
def logout():
    """Clear the pf_dashboard_token cookie and redirect to /.

    The auth cookie is HttpOnly, so client JS cannot expire it via
    document.cookie — the browser ignores any attempt to write a name that
    Set-Cookie marked HttpOnly. The mobile Settings → Sign out button
    therefore has to navigate here so the server can emit the matching
    Set-Cookie with Max-Age=0. (Codex P2 finding 2026-05-03.)

    No `require_auth`: an unauthenticated visitor hitting /logout still gets
    the cookie wiped (harmless — they had no valid cookie anyway) and
    Cloudflare Access still gates the redirected destination.
    """
    response = redirect("/", code=302)
    # Match every flag we set on the original cookie except expiry.
    response.set_cookie(
        "pf_dashboard_token",
        "",
        max_age=0,
        expires=0,
        httponly=True,
        secure=True,
        samesite="Lax",
    )
    return response


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


_API_ACCURACY_CACHE: dict = {"ts": 0.0, "data": None}
_API_ACCURACY_TTL_SEC = 60.0


@app.route("/api/accuracy")
@require_auth
def api_accuracy():
    """Aggregate accuracy report across 4 horizons.

    2026-05-03: previously took >15s (timed out from clients) because
    each request did 12 full signal-log scans (4 horizons × 3 metrics).
    Now backed by accuracy_stats.get_or_compute_*() which read
    accuracy_cache.json on the hot path, plus a 60s in-process TTL
    that coalesces burst requests during dashboard polling.
    """
    import time
    now = time.time()
    if (_API_ACCURACY_CACHE["data"] is not None
            and (now - _API_ACCURACY_CACHE["ts"]) < _API_ACCURACY_TTL_SEC):
        return jsonify(_API_ACCURACY_CACHE["data"])

    try:
        from portfolio.accuracy_stats import (
            get_or_compute_accuracy,
            get_or_compute_consensus_accuracy,
            get_or_compute_per_ticker_accuracy,
        )

        result = {}
        for horizon in ["1d", "3d", "5d", "10d"]:
            sa = get_or_compute_accuracy(horizon)
            ca = get_or_compute_consensus_accuracy(horizon)
            ta = get_or_compute_per_ticker_accuracy(horizon)
            # ca/sa/ta may be None when the underlying cache miss returned
            # no data (cold cache + no signal-log entries yet); skip those
            # horizons entirely so the response stays well-formed.
            if ca and ca.get("total", 0) > 0:
                result[horizon] = {
                    "signals": sa or {},
                    "consensus": ca,
                    "per_ticker": ta or {},
                }
        _API_ACCURACY_CACHE["data"] = result
        _API_ACCURACY_CACHE["ts"] = now
        return jsonify(result)
    except Exception:
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
    except Exception:
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
# Crypto + MSTR swing-trader endpoints (mirror /api/metals shape)
# ---------------------------------------------------------------------------

def _crypto_per_instrument(state: dict, ticker: str) -> dict:
    """Slice the unified crypto_swing_state.json by ticker."""
    positions = state.get("positions", {}) if state else {}
    matches = {pid: p for pid, p in positions.items() if p.get("ticker") == ticker}
    return {
        "n_positions": len(matches),
        "positions": matches,
        "last_buy_ts": (state.get("last_buy_ts", {}) or {}).get(ticker)
                       if state else None,
    }


def _crypto_decisions_for(decisions: list, ticker: str) -> list:
    out = []
    for d in decisions or []:
        pos = d.get("pos") or {}
        if pos.get("ticker") == ticker:
            out.append(d)
        elif d.get("ticker") == ticker:
            out.append(d)
    return out


@app.route("/api/crypto")
@require_auth
def api_crypto():
    """Combined BTC + ETH swing-trader state (mirror of /api/metals).

    Reads:
      - data/crypto_swing_state.json (positions, cash, cycle counter)
      - data/crypto_deep_context.json (Fear & Greed, funding, on-chain)
      - data/crypto_swing_decisions.jsonl (last 50)
      - data/crypto_swing_trades.jsonl (last 50)
      - data/crypto_warrant_catalog.json (live warrant universe)
      - data/crypto_risk.json (per-position barrier checks, drawdown)
    """
    state = _read_json(DATA_DIR / "crypto_swing_state.json") or {}
    context = _read_json(DATA_DIR / "crypto_deep_context.json") or {}
    catalog = _read_json(DATA_DIR / "crypto_warrant_catalog.json") or {}
    risk = _read_json(DATA_DIR / "crypto_risk.json") or {}
    decisions = list(_iter_latest_dict_entries(
        DATA_DIR / "crypto_swing_decisions.jsonl", read_limit=50))
    trades = list(_iter_latest_dict_entries(
        DATA_DIR / "crypto_swing_trades.jsonl", read_limit=50))
    return jsonify({
        "state": state,
        "context": context,
        "warrant_catalog": catalog,
        "risk": risk,
        "decisions": decisions,
        "trades": trades,
    })


@app.route("/api/btc")
@require_auth
def api_btc():
    """BTC-specific slice of the crypto swing-trader state."""
    state = _read_json(DATA_DIR / "crypto_swing_state.json") or {}
    context = _read_json(DATA_DIR / "crypto_deep_context.json") or {}
    decisions = list(_iter_latest_dict_entries(
        DATA_DIR / "crypto_swing_decisions.jsonl", read_limit=50))
    trades = list(_iter_latest_dict_entries(
        DATA_DIR / "crypto_swing_trades.jsonl", read_limit=50))
    return jsonify({
        "ticker": "BTC-USD",
        "instrument": _crypto_per_instrument(state, "BTC-USD"),
        "deep_context": (context or {}).get("btc"),
        "shared_context": (context or {}).get("shared"),
        "decisions": _crypto_decisions_for(decisions, "BTC-USD"),
        "trades": _crypto_decisions_for(trades, "BTC-USD"),
    })


@app.route("/api/eth")
@require_auth
def api_eth():
    """ETH-specific slice of the crypto swing-trader state."""
    state = _read_json(DATA_DIR / "crypto_swing_state.json") or {}
    context = _read_json(DATA_DIR / "crypto_deep_context.json") or {}
    decisions = list(_iter_latest_dict_entries(
        DATA_DIR / "crypto_swing_decisions.jsonl", read_limit=50))
    trades = list(_iter_latest_dict_entries(
        DATA_DIR / "crypto_swing_trades.jsonl", read_limit=50))
    return jsonify({
        "ticker": "ETH-USD",
        "instrument": _crypto_per_instrument(state, "ETH-USD"),
        "deep_context": (context or {}).get("eth"),
        "shared_context": (context or {}).get("shared"),
        "decisions": _crypto_decisions_for(decisions, "ETH-USD"),
        "trades": _crypto_decisions_for(trades, "ETH-USD"),
    })


@app.route("/api/loop_health")
@require_auth
def api_loop_health():
    """Cross-loop heartbeat rollup.

    Reads data/{name}_loop.heartbeat for each registered loop (currently
    crypto + oil; metals/main can be added when they grow heartbeats).
    Returns per-loop {state, age_seconds, payload, error}, plus a
    rollup any_unhealthy flag and an unhealthy[] list.

    Same data the loop-health watchdog uses for telegram alerts. Use
    this endpoint for live dashboard monitoring without waiting for the
    next watchdog tick.
    """
    from portfolio.loop_health import read_loop_health
    return jsonify(read_loop_health())


@app.route("/api/oil")
@require_auth
def api_oil():
    """Oil swing-trader state (mirror of /api/crypto and /api/metals).

    Reads:
      - data/oil_swing_state.json (positions, cash, cycle counter)
      - data/oil_deep_context.json (WTI/Brent/COT/OVX/crack-spread context
        from portfolio/oil_precompute.py)
      - data/oil_swing_decisions.jsonl (last 50)
      - data/oil_swing_trades.jsonl (last 50)
      - data/oil_warrant_catalog.json (live OLJA warrant universe)
      - data/oil_risk.json (per-position barrier checks, drawdown)

    Ships in DRY_RUN=True; the trades log will be empty until the loop
    is wired live via data/oil_swing_config.py.
    """
    state = _read_json(DATA_DIR / "oil_swing_state.json") or {}
    context = _read_json(DATA_DIR / "oil_deep_context.json") or {}
    catalog = _read_json(DATA_DIR / "oil_warrant_catalog.json") or {}
    risk = _read_json(DATA_DIR / "oil_risk.json") or {}
    decisions = list(_iter_latest_dict_entries(
        DATA_DIR / "oil_swing_decisions.jsonl", read_limit=50))
    trades = list(_iter_latest_dict_entries(
        DATA_DIR / "oil_swing_trades.jsonl", read_limit=50))
    # Heartbeat reflects liveness even when no trades have fired
    heartbeat = _read_json(DATA_DIR / "oil_loop.heartbeat") or {}
    return jsonify({
        "state": state,
        "context": context,
        "warrant_catalog": catalog,
        "risk": risk,
        "decisions": decisions,
        "trades": trades,
        "heartbeat": heartbeat,
    })


@app.route("/api/mstr")
@require_auth
def api_mstr():
    """MSTR deep-context endpoint.

    The pre-existing `/api/mstr_loop` returns the strategy-loop state
    (positions, scorecard, last poll). This new endpoint returns the deep
    context (NAV premium, BTC correlation, options skew, analyst consensus)
    written by `portfolio/mstr_precompute.py`. Together they parallel
    `/api/metals` (decisions+context) for the metals subsystem.
    """
    deep = _read_json(DATA_DIR / "mstr_deep_context.json") or {}
    loop_state = _read_json(DATA_DIR / "mstr_loop_state.json") or {}
    scorecard = _read_json(DATA_DIR / "mstr_loop_scorecard.json") or {}
    return jsonify({
        "ticker": "MSTR",
        "deep_context": deep,
        "loop_state": loop_state,
        "scorecard": scorecard,
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
    except Exception:
        logger.exception("mstr endpoint error")
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Avanza account snapshot — live cash + positions + open orders + stop-losses.
# Lets the user verify the local view is in sync with the actual broker
# state. Each subsection is independently fetched so a single API hiccup
# (e.g. flaky stop-loss endpoint) doesn't blank the whole view.
# ---------------------------------------------------------------------------

_AVANZA_CACHE_LOCK = threading.Lock()
_AVANZA_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_AVANZA_TTL_SECONDS = 30.0

# Same TTL pattern for the system-health rollup endpoints. Both caches
# are independent so trading_status can refresh on its own cadence
# while system_status keeps serving cached, and vice versa.
_SYSTEM_STATUS_LOCK = threading.Lock()
_SYSTEM_STATUS_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_SYSTEM_STATUS_TTL_SECONDS = 30.0

_TRADING_STATUS_LOCK = threading.Lock()
_TRADING_STATUS_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_TRADING_STATUS_TTL_SECONDS = 30.0

# ---------------------------------------------------------------------------
# Avanza worker thread — Playwright's sync API is bound to its creator
# thread, but Flask's ThreadedWSGIServer spawns a fresh worker per request.
# A request that lands on a different thread than the one which initialised
# Playwright fails with "cannot switch to a different thread (which happens
# to have exited)".
#
# Solution: a single dedicated worker thread owns the Playwright session
# for the dashboard process. HTTP handlers enqueue snapshot requests via
# `_avanza_request_q`, the worker processes them in order, and replies via
# a per-request Event. This is the same pattern the metals_loop dodges by
# being single-threaded; Flask can't afford that, so we serialise here.
# ---------------------------------------------------------------------------

import queue  # noqa: E402  (kept near the worker for grouping)

_AVANZA_REQ_Q: "queue.Queue[dict]" = queue.Queue()
_AVANZA_WORKER_LOCK = threading.Lock()
_AVANZA_WORKER_STARTED = False
_AVANZA_REQ_TIMEOUT_SECONDS = 25.0  # snapshot upper bound


def _avanza_worker_loop() -> None:
    """Single-thread worker that owns Playwright. Blocks on the request
    queue and serves snapshot requests sequentially."""
    while True:
        future = _AVANZA_REQ_Q.get()
        try:
            future["result"] = _avanza_snapshot_impl()
        except Exception as e:
            logger.exception("avanza-worker: snapshot failed")
            future["result"] = {
                "ts": datetime.now(UTC).isoformat(),
                "account_id": None,
                "cash": None,
                "positions": [],
                "orders": [],
                "stop_losses": [],
                "errors": [f"worker: {type(e).__name__}: {e}"],
            }
        finally:
            future["done"].set()


def _ensure_avanza_worker() -> None:
    global _AVANZA_WORKER_STARTED
    if _AVANZA_WORKER_STARTED:
        return
    with _AVANZA_WORKER_LOCK:
        if _AVANZA_WORKER_STARTED:
            return
        t = threading.Thread(
            target=_avanza_worker_loop, daemon=True, name="avanza-worker",
        )
        t.start()
        _AVANZA_WORKER_STARTED = True


def _avanza_account_snapshot() -> dict:
    """Public entry. Marshals snapshot building onto the worker thread so
    Playwright's thread affinity is honoured."""
    _ensure_avanza_worker()
    future: dict[str, Any] = {"result": None, "done": threading.Event()}
    _AVANZA_REQ_Q.put(future)
    if not future["done"].wait(timeout=_AVANZA_REQ_TIMEOUT_SECONDS):
        return {
            "ts": datetime.now(UTC).isoformat(),
            "account_id": None,
            "cash": None,
            "positions": [],
            "orders": [],
            "stop_losses": [],
            "errors": [
                f"avanza-worker: timed out after {_AVANZA_REQ_TIMEOUT_SECONDS}s"
            ],
        }
    return future["result"] or {
        "ts": datetime.now(UTC).isoformat(),
        "account_id": None, "cash": None, "positions": [],
        "orders": [], "stop_losses": [],
        "errors": ["avanza-worker: empty result"],
    }


def _avanza_snapshot_impl() -> dict:
    """Build a fresh Avanza account snapshot. Uncached.

    Uses `portfolio.avanza_session` (Playwright BankID auth at
    `data/avanza_session.json`) — the same path the live metals_loop and
    golddigger use. The newer `portfolio.avanza` TOTP package is *not*
    used here because TOTP credentials aren't populated in the live
    config; switching needs setup work outside this PR. Codex P1 fix
    2026-05-04 originally seeded the TOTP singleton, but the empty
    credentials made every call still fail — the live-system path is
    the right answer.

    Each subcall is independently try/except'd so a partial Avanza
    outage degrades section-by-section. Sections are filtered to the
    configured account_id (codex P2 finding 2026-05-04).
    """
    out: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "account_id": None,
        "cash": None,
        "positions": [],
        "orders": [],
        "stop_losses": [],
        "errors": [],
    }
    try:
        from portfolio.avanza_session import DEFAULT_ACCOUNT_ID
        account_id = str(DEFAULT_ACCOUNT_ID)
    except Exception:
        account_id = None
    out["account_id"] = account_id

    try:
        from portfolio.avanza_session import get_buying_power
        cash = get_buying_power(account_id=account_id)
        if cash is None:
            out["errors"].append(
                "cash: get_buying_power returned None "
                "(Avanza session likely expired — re-auth via BankID)"
            )
        else:
            out["cash"] = cash
    except Exception as e:
        out["errors"].append(f"cash: {type(e).__name__}: {e}")

    try:
        from portfolio.avanza_session import get_positions
        all_positions = get_positions()
        out["positions"] = [
            p for p in all_positions
            if account_id is None or str(p.get("account_id", "")) == account_id
        ]
    except Exception as e:
        out["errors"].append(f"positions: {type(e).__name__}: {e}")

    try:
        from portfolio.avanza_session import get_open_orders
        out["orders"] = [_norm_order(o) for o in get_open_orders(account_id=account_id)]
    except Exception as e:
        out["errors"].append(f"orders: {type(e).__name__}: {e}")

    try:
        from portfolio.avanza_session import get_stop_losses
        stops = get_stop_losses()
        out["stop_losses"] = [
            _norm_stop(s) for s in stops
            if account_id is None or str(_stop_account(s)) == account_id
        ]
    except Exception as e:
        out["errors"].append(f"stop_losses: {type(e).__name__}: {e}")
    return out


def _norm_order(raw: dict) -> dict:
    """Normalize an Avanza orders-API dict to the snake_case shape the
    dashboard view binds against."""
    return {
        "order_id":     str(raw.get("orderId", raw.get("id", ""))),
        "orderbook_id": str(raw.get("orderBookId", raw.get("orderbookId", ""))),
        "side":         str(raw.get("orderType", raw.get("side", ""))),
        "price":        float(raw.get("price") or 0.0),
        "volume":       int(raw.get("volume") or 0),
        "status":       str(raw.get("status", raw.get("statusDescription", ""))),
        "account_id":   str(raw.get("accountId", raw.get("account_id", ""))),
    }


def _stop_account(raw: dict) -> str:
    return str(
        raw.get("accountId") or raw.get("account_id") or
        (raw.get("account") or {}).get("id", "")
    )


def _norm_stop(raw: dict) -> dict:
    """Normalize an Avanza stop-loss dict (matches Order.from_api shape)."""
    trigger = raw.get("trigger") or {}
    order_event = raw.get("orderEvent") or raw.get("order") or {}
    return {
        "stop_id":       str(raw.get("id", raw.get("stopLossId", ""))),
        "orderbook_id":  str((raw.get("orderbook") or {}).get("id",
                              raw.get("orderBookId", raw.get("orderbookId", "")))),
        "trigger_price": float(trigger.get("value") or raw.get("triggerPrice") or 0.0),
        "trigger_type":  str(trigger.get("type") or raw.get("triggerType") or "LAST_PRICE"),
        "sell_price":    float(order_event.get("price") or raw.get("sellPrice") or 0.0),
        "volume":        int(order_event.get("volume") or raw.get("volume") or 0),
        "status":        str(raw.get("status", "")),
        "account_id":    _stop_account(raw),
    }


@app.route("/api/avanza_account")
@require_auth
def api_avanza_account():
    """Live snapshot of the Avanza brokerage account.

    Cash + positions + open orders + active stop-losses, filtered to the
    configured account_id. 30-second TTL cache because the underlying
    calls hit the network. Each subsection has its own try/except so a
    partial upstream outage degrades to "this section unavailable"
    instead of a full 500.

    `?force=1` bypasses the TTL cache so the user's manual Refresh
    button can verify a just-placed or cancelled order without waiting
    out the polling cadence (Codex P2 finding 2026-05-04).
    """
    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
    now = time.monotonic()
    if not force:
        with _AVANZA_CACHE_LOCK:
            cached = _AVANZA_CACHE.get("value")
            if cached and (now - _AVANZA_CACHE["at"]) < _AVANZA_TTL_SECONDS:
                return jsonify(cached)
    snapshot = _avanza_account_snapshot()
    with _AVANZA_CACHE_LOCK:
        _AVANZA_CACHE["value"] = snapshot
        _AVANZA_CACHE["at"] = now
    return jsonify(snapshot)


# ---------------------------------------------------------------------------
# Tradeable assets — what the loops will buy/sell. Aggregates the metals
# warrant catalog (fin_fish), crypto + oil JSON catalogs, plus the small
# equity universe in avanza_tracker. Lets the user verify the system
# knows about each instrument, including its orderbook_id, leverage, and
# direction. Read-only.
# ---------------------------------------------------------------------------

@app.route("/api/tradeable_assets")
@require_auth
def api_tradeable_assets():
    """Return everything the system might trade on Avanza.

    Aggregates:
      - Metals warrants (`portfolio.fin_fish.WARRANT_CATALOG`)
      - Crypto warrants (`data/crypto_warrant_catalog.json`)
      - Oil warrants (`data/oil_warrant_catalog.json`)

    Each category is independently try/except'd so a missing import or
    bad JSON file doesn't blank the whole view.
    """
    out: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "metals_warrants": {},
        "crypto_warrants": {},
        "oil_warrants": {},
        "errors": [],
    }
    try:
        from portfolio.fin_fish import WARRANT_CATALOG as METALS_CATALOG
        out["metals_warrants"] = dict(METALS_CATALOG)
    except Exception as e:
        out["errors"].append(f"metals: {type(e).__name__}: {e}")
    try:
        crypto = _read_json(DATA_DIR / "crypto_warrant_catalog.json") or {}
        out["crypto_warrants"] = crypto.get("warrants", crypto) if isinstance(crypto, dict) else {}
    except Exception as e:
        out["errors"].append(f"crypto: {type(e).__name__}: {e}")
    try:
        oil = _read_json(DATA_DIR / "oil_warrant_catalog.json") or {}
        out["oil_warrants"] = oil.get("warrants", oil) if isinstance(oil, dict) else {}
    except Exception as e:
        out["errors"].append(f"oil: {type(e).__name__}: {e}")
    return jsonify(out)


# ---------------------------------------------------------------------------
# System-health home rollup endpoints.
#
# /api/system_status   - overall GREEN/YELLOW/RED, heartbeat, errors,
#                        contract violations, LLM inference success,
#                        Layer 2 24h activity, signal aggregate.
# /api/trading_status  - per-bot Avanza state with reason
#                        (golddigger, elongir, metals, fishing).
#
# Both are pure aggregations over data/*.json[l]. No network. 30s TTL
# cache mirrors the _AVANZA_CACHE pattern; ?force=1 bypasses for the
# manual Refresh button.
# ---------------------------------------------------------------------------

@app.route("/api/system_status")
@require_auth
def api_system_status():
    """System-health rollup for the home view's GREEN/YELLOW/RED hero.

    See dashboard/system_status.py for the full payload shape and
    severity thresholds. Per-section errors[] envelope so a corrupt
    jsonl line never blanks the hero.

    Cache discipline (codex P2 finding 2026-05-04): the lock covers
    both the read and the write so concurrent misses serialize. A
    request that started after the most recent fill won't overwrite a
    fresher payload, and ``?force=1`` won't lose its refresh behind
    another in-flight fill.
    """
    from dashboard import system_status as _sys_status

    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
    if not force:
        with _SYSTEM_STATUS_LOCK:
            cached = _SYSTEM_STATUS_CACHE.get("value")
            if cached and (time.monotonic() - _SYSTEM_STATUS_CACHE["at"]) < _SYSTEM_STATUS_TTL_SECONDS:
                return jsonify(cached)
    with _SYSTEM_STATUS_LOCK:
        # Re-check inside the lock — a concurrent miss may have filled it.
        cached = _SYSTEM_STATUS_CACHE.get("value")
        if not force and cached and (time.monotonic() - _SYSTEM_STATUS_CACHE["at"]) < _SYSTEM_STATUS_TTL_SECONDS:
            return jsonify(cached)
        payload = _sys_status.compute()
        _SYSTEM_STATUS_CACHE["value"] = payload
        _SYSTEM_STATUS_CACHE["at"] = time.monotonic()
        return jsonify(payload)


@app.route("/api/trading_status")
@require_auth
def api_trading_status():
    """Per-bot Avanza trading state with reason.

    See dashboard/trading_status.py. Each bot resolves to one of
    SCANNING / TRADING / HALTED / COOLDOWN / OUTSIDE_HOURS / UNKNOWN.
    Same lock discipline as ``/api/system_status``.
    """
    from dashboard import trading_status as _trading_status

    force = request.args.get("force", "").strip() in {"1", "true", "yes"}
    if not force:
        with _TRADING_STATUS_LOCK:
            cached = _TRADING_STATUS_CACHE.get("value")
            if cached and (time.monotonic() - _TRADING_STATUS_CACHE["at"]) < _TRADING_STATUS_TTL_SECONDS:
                return jsonify(cached)
    with _TRADING_STATUS_LOCK:
        cached = _TRADING_STATUS_CACHE.get("value")
        if not force and cached and (time.monotonic() - _TRADING_STATUS_CACHE["at"]) < _TRADING_STATUS_TTL_SECONDS:
            return jsonify(cached)
        payload = _trading_status.compute()
        _TRADING_STATUS_CACHE["value"] = payload
        _TRADING_STATUS_CACHE["at"] = time.monotonic()
        return jsonify(payload)


# ---------------------------------------------------------------------------
# Blueprint: /house — read-only viewer over the househunting project
# (data/findapartments runs + innerstad heatmap). Reuses pf_dashboard_token
# auth via dashboard.auth.require_auth. Path roots come from
# config.json[house_root]. See dashboard/house_blueprint.py for routes.
#
# House_blueprint imports `_get_config` and `require_auth` from
# dashboard.auth (NOT dashboard.app), so importing it here at module-init
# time no longer causes a circular import — auth.py has no back-reference
# to app.py. The sys.modules alias hack added 2026-05-02 has been removed.
# ---------------------------------------------------------------------------
from dashboard.house_blueprint import bp as _house_bp  # noqa: E402

app.register_blueprint(_house_bp)


def _serve_dual_stack(port: int = 5055) -> None:
    """Run the Flask app on a dual-stack IPv4+IPv6 socket.

    2026-05-04: previously used `app.run(host="0.0.0.0", ...)` which is
    IPv4-only. Local Python tooling (urllib, requests) on Windows that
    resolves "localhost" to ::1 first then waits ~2s for the IPv6
    connection to fail before falling back to IPv4 — perceived as a
    universal "2s auth floor" but actually a client-side Happy Eyeballs
    timeout. Real users (Cloudflare tunnel, LAN browsers) never see it.

    Switching to `host="::"` would fix localhost on Linux but on
    Windows the default `IPV6_V6ONLY=True` socket option means IPv4
    clients can no longer connect. So we bind manually with
    `IPV6_V6ONLY=0`, which works on every modern Windows / Linux /
    macOS host.
    """
    import socket
    from werkzeug.serving import ThreadedWSGIServer

    # Build the dual-stack listening socket explicitly. IPV6_V6ONLY=0
    # enables IPv4 mapping (::ffff:127.0.0.1 etc.), so a single AF_INET6
    # socket accepts both IPv4 and IPv6 clients.
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", port))
    sock.listen(128)

    # ThreadedWSGIServer accepts `fd=` so it skips its own bind/listen
    # and reuses our pre-configured socket. ThreadingMixIn handles
    # concurrent requests just like Werkzeug's default app.run().
    server = ThreadedWSGIServer("::", port, app, fd=sock.fileno())
    print(f"Dashboard listening on dual-stack [::]:{port} (IPv4 + IPv6)",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    _serve_dual_stack(port=5055)
