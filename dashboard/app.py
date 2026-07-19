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

# 2026-06-10 (audit batch 10): cap request bodies. POST /api/validate-portfolio
# json-parses request.get_json() with no size limit; without a ceiling an
# authenticated client (or a same-origin XSS payload) could buffer an arbitrary
# body in memory on a worker thread. 1 MiB is far above any real payload here.
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024


import re as _re

# 2026-06-10 (audit batch 10): the /go/<slug> magic link and the first-visit
# ?token= query param are credentials, but Werkzeug's request handler logs the
# full request line ("GET /go/<slug>?token=... HTTP/1.1") to the `werkzeug`
# logger by default — landing both secrets in the dashboard access log and any
# Cloudflare request log. This filter rewrites the logged line so the slug and
# token value are replaced with [redacted] before the record is emitted. It
# mutates record.args (the % interpolation args) since WSGIRequestHandler logs
# via "%s" with the request line in args[0]; we also scrub the formatted message
# defensively in case a future caller pre-formats. Dependency-free.
_TOKEN_QS_RE = _re.compile(r"([?&]token=)[^&\s\"']+", _re.IGNORECASE)
_GO_SLUG_RE = _re.compile(r"(/go/)[^\s?\"']+")


def _redact_request_line(text: str) -> str:
    text = _TOKEN_QS_RE.sub(r"\1[redacted]", text)
    text = _GO_SLUG_RE.sub(r"\1[redacted]", text)
    return text


class _RedactingFilter(logging.Filter):
    """Strip ?token= values and /go/ slugs from werkzeug access-log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(
                _redact_request_line(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        if isinstance(record.msg, str):
            record.msg = _redact_request_line(record.msg)
        return True


def _install_log_redaction() -> None:
    """Attach the redaction filter to the werkzeug logger (idempotent)."""
    wlog = logging.getLogger("werkzeug")
    if not any(isinstance(f, _RedactingFilter) for f in wlog.filters):
        wlog.addFilter(_RedactingFilter())


_install_log_redaction()


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
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Block framing everywhere EXCEPT the heatmap, which the /house hub embeds in
    # a same-origin <iframe>. Same-origin only (frame-ancestors 'self'), auth-gated,
    # read-only map — safe to frame; every other route keeps DENY.
    if request.path == "/house/heatmap":
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'self'"
    else:
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.errorhandler(500)
def _handle_api_error(error):
    if request.path.startswith("/api/"):
        logger.error("Unhandled error on %s: %s", request.path, error, exc_info=True)
        return jsonify({"error": "internal_error"}), 500
    return error


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_DIR = Path(__file__).resolve().parent.parent / "training" / "lora"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
STOCKHOLM_TZ = ZoneInfo("Europe/Stockholm")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from portfolio.file_utils import last_jsonl_entry as _last_jsonl_entry_impl
from portfolio.file_utils import load_json as _load_json_impl
from portfolio.file_utils import load_jsonl as _load_jsonl_impl
from portfolio.file_utils import load_jsonl_tail as _load_jsonl_tail_impl

# ---------------------------------------------------------------------------
# TTL Cache (BUG-130: avoid re-reading files on every API request)
# ---------------------------------------------------------------------------

_cache = {}
_cache_lock = threading.Lock()
_cache_ticket = {}
_DEFAULT_TTL = 5  # seconds


def _cached_read(key, ttl, read_fn):
    """Return cached result if fresh, otherwise call read_fn and cache.

    2026-07-20: the lock used to be released for the whole read_fn() call,
    so two concurrent cache misses on the same key could store out of
    order — a slow OLD read finishing after a fast NEW one would clobber
    the fresher cached value. A monotonic per-key ticket is taken under
    the lock before read_fn runs; the result is only stored afterward if
    no newer ticket has been issued (and therefore stored) in the
    meantime. The caller's own return value is unaffected — only the
    shared cache write is guarded.
    """
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (now - entry[1]) < ttl:
            return entry[0]
        ticket = _cache_ticket[key] = _cache_ticket.get(key, 0) + 1
    result = read_fn()
    with _cache_lock:
        if _cache_ticket.get(key, 0) <= ticket:
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


def _hours_until_stockholm_close(now=None, close_hour=21, close_minute=30):
    """Return hours remaining until the Stockholm warrant close.

    Defaults updated 2026-05-11 to match the unified 08:30–21:30
    trading window (previously 21:55, tracked GoldDigger's old US-overlap
    end). Callers that need the legacy 21:55 must pass it explicitly.
    """
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
        "market_close_cet": "21:30",
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
    _request_is_https,
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
    # secure follows the request scheme (not hardcoded True) so this also
    # clears the cookie on a plain-HTTP LAN visit — same rule as every
    # other cookie write (see auth._refresh_cookie).
    response.set_cookie(
        "pf_dashboard_token",
        "",
        max_age=0,
        expires=0,
        httponly=True,
        secure=_request_is_https(),
        samesite="Lax",
    )
    return response


@app.route("/go/<slug>")
def share_link(slug):
    """Magic-link auth: a memorable, shareable URL that grants a 1-year cookie.

    Added 2026-06-03. Share e.g. https://bets.raanman.lol/go/raanman-uploadme.
    Hitting it sets the same ``pf_dashboard_token`` cookie ``require_auth``
    checks, then 302-redirects to ``/`` — the dashboard_token itself never
    appears in the URL or the response body.

    NOTE (corrected 2026-06-10, audit batch 10): the *slug* IS the URL path,
    so unlike the token it DOES reach server access logs — Werkzeug's request
    handler logs the full request line ("GET /go/<slug> ...") by default. The
    slug is a 1-year credential, so a ``_RedactingFilter`` is installed on the
    ``werkzeug`` logger (see ``_install_log_redaction``) to replace ``/go/``
    slugs and ``?token=`` values with ``[redacted]`` before they are logged.
    The earlier claim that the secret "can't be lifted from server logs" was
    false for the slug and is removed.

    The path segment IS a shared secret (anyone with the link gets a 1-year
    pass; rotate ``dashboard_share_slug`` to revoke everyone). Security
    properties:

    * Validated constant-time against config ``dashboard_share_slug`` (encoded
      to bytes first so a hostile non-ASCII slug yields 404, not a 500 from
      ``hmac.compare_digest``).
    * Returns a featureless 404 on ANY mismatch and when the feature is
      unconfigured — never 401/403 — so a path scanner can't distinguish a
      wrong slug from a disabled feature. The 503 branch is only reachable
      with the correct slug (operator-only), so it leaks nothing to scanners.
    * Inert (404 for every input) unless ``dashboard_share_slug`` is set.

    Read config through ``dashboard.auth`` at call time (not the module-level
    import binding) so tests patching ``dashboard.auth._get_config`` take
    effect, per the convention documented at the top of dashboard/auth.py.
    """
    from dashboard.auth import _get_config, _get_dashboard_token

    configured = (_get_config().get("dashboard_share_slug") or "").strip()
    # Always run the constant-time compare (even when unset, against a fixed
    # sentinel that no real slug can be) so response timing doesn't reveal
    # whether the share feature is enabled.
    expected = configured or "\x00share-link-disabled\x00"
    matched = hmac.compare_digest(slug.encode("utf-8"), expected.encode("utf-8"))
    if not configured or not matched:
        return jsonify({"error": "not_found"}), 404

    token = _get_dashboard_token()
    if token is None:
        # Reachable only with the correct slug → operator-only, safe to be
        # explicit. share_slug is set but dashboard_token isn't: misconfig.
        logger.error("share_link: dashboard_share_slug set but dashboard_token missing")
        return jsonify({"error": "misconfigured"}), 503

    logger.info("share_link: slug redeemed, issuing 1-year auth cookie")
    return _refresh_cookie(redirect("/", code=302), token)


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


@app.route("/api/grid-fisher")
@require_auth
def api_grid_fisher():
    """Grid market-maker state + recent decisions.

    Returns:
        {
          "state": <data/grid_fisher_state.json>,
          "recent_decisions": [last 50 entries from grid_fisher_decisions.jsonl]
        }
    """
    state = _read_json(DATA_DIR / "grid_fisher_state.json") or {}
    decisions: list[dict] = []
    decisions_path = DATA_DIR / "grid_fisher_decisions.jsonl"
    try:
        if decisions_path.exists():
            from portfolio.file_utils import load_jsonl_tail
            decisions = load_jsonl_tail(decisions_path, max_entries=50)
    except Exception:
        decisions = []
    return jsonify({"state": state, "recent_decisions": decisions})


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
        "last_poll": _last_jsonl_entry_impl(DATA_DIR / "mstr_loop_poll.jsonl"),
        "last_trade": _last_jsonl_entry_impl(DATA_DIR / "mstr_loop_trades.jsonl"),
    }
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
        # Malformed rows (e.g. category/text written as a non-string) must
        # not 500 the whole search — coerce to "" and let the filter treat
        # them as non-matching instead of crashing .lower().
        category = entry.get("category", "")
        if not isinstance(category, str):
            category = ""
        text = entry.get("text", "")
        if not isinstance(text, str):
            text = ""
        if category_filter and category.lower() != category_filter:
            continue
        if search_filter and search_filter not in text.lower():
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
            get_accuracy_cache_meta,
            get_latest_signal_log_ts,
            get_oldest_signal_log_ts,
            get_or_compute_accuracy,
            get_or_compute_consensus_accuracy,
            get_or_compute_per_ticker_accuracy,
        )
        from portfolio.component_registry import get_registry

        def _unavailable_reason(horizon, ca):
            # 2026-07-18: horizons can have zero consensus samples for two
            # different reasons — distinguish them so the frontend doesn't
            # conflate "no data yet" with "something broke".
            if ca is None:
                return "error: accuracy data unavailable for this horizon"
            need_days = horizon[:-1] if horizon.endswith("d") else horizon
            oldest_ts = get_oldest_signal_log_ts()
            if oldest_ts is not None:
                age_days = int((time.time() - oldest_ts) / 86400)
                return (
                    f"insufficient history: oldest outcome data is {age_days} "
                    f"days old, horizon needs {need_days}+ days"
                )
            return "insufficient history: outcome data too young for this horizon"

        def _enrich_signals(signals_dict):
            # 2026-05-05: enrich at response time so older cached entries
            # (written before signal_accuracy() learned to emit `samples`/
            # `enabled`) still render correctly on the dashboard. The
            # accuracy cache has a 1h TTL; without this fallback the
            # disabled-signal labels would not appear until the cache
            # rebuilds.
            #
            # Important: `enabled` and `disabled_reason` are *overwritten*
            # from the live registry (component_registry, itself generated
            # from tickers.DISABLED_SIGNALS), not setdefault'd. A signal
            # re-enabled (e.g. statistical_jump_regime, 2026-04-29) or
            # newly disabled would otherwise keep the stale flag from the
            # cache file until the next 1h rebuild. `samples` is just an
            # alias for `total` so setdefault is fine there.
            if not isinstance(signals_dict, dict):
                return signals_dict
            for sig_name, info in signals_dict.items():
                if not isinstance(info, dict):
                    continue
                if "samples" not in info and "total" in info:
                    info["samples"] = info["total"]
                # 2026-07-18 Phase 4.3: enabled/disabled_reason now come from
                # component_registry (global-only check, no ticker in scope
                # here) instead of DISABLED_SIGNALS/get_disabled_reason
                # directly. Parity verified: registry.disabled_reason(sig)
                # agrees with get_disabled_reason(sig) for every signal in
                # SIGNAL_NAMES (see tests/test_dashboard.py parity canary).
                reason = get_registry().disabled_reason(sig_name)
                info["enabled"] = reason is None
                if reason:
                    info["disabled_reason"] = reason
                else:
                    info.pop("disabled_reason", None)
            return signals_dict

        result = {}
        for horizon in ["1d", "3d", "5d", "10d"]:
            sa = get_or_compute_accuracy(horizon)
            ca = get_or_compute_consensus_accuracy(horizon)
            ta = get_or_compute_per_ticker_accuracy(horizon)
            meta = get_accuracy_cache_meta(horizon)
            # Data age != cache age: a cache-miss recompute over frozen
            # signal data stamps "now" while the signals are hours old
            # (loops stopped). Expose the newest underlying signal ts so
            # the UI can show honest staleness (2026-07-19).
            _data_ts = get_latest_signal_log_ts()
            if not isinstance(_data_ts, (int, float)):
                _data_ts = None  # defensive: tests mock the module
            if isinstance(meta, dict):
                meta["data_ts"] = _data_ts
                meta["data_age_sec"] = (
                    int(time.time() - _data_ts) if _data_ts else None
                )
            # ca may be None (cold cache miss, no signal-log entries yet)
            # or a dict with total==0 (horizon has zero outcome rows so
            # far, e.g. 10d). Both cases used to drop the horizon key
            # entirely; now we keep the key and explain why via
            # unavailable_reason so the frontend can distinguish
            # "no data yet" from "error" instead of rendering a blank tab.
            if ca and ca.get("total", 0) > 0:
                result[horizon] = {
                    "signals": _enrich_signals(sa or {}),
                    "consensus": ca,
                    "per_ticker": ta or {},
                    "meta": meta,
                }
            else:
                result[horizon] = {
                    "signals": {},
                    "consensus": ca or {"total": 0, "correct": 0},
                    "per_ticker": {},
                    "meta": meta,
                    "unavailable_reason": _unavailable_reason(horizon, ca),
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
        logger.exception("validate_portfolio error")
        return jsonify({"valid": False, "errors": ["Validation failed"]}), 500

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
# Signal heatmap (all signals x all tickers)
# ---------------------------------------------------------------------------

@app.route("/api/signal-heatmap")
@require_auth
def api_signal_heatmap():
    """Return the full signal x tickers grid.

    Each cell is BUY/SELL/HOLD. Built from agent_summary.json _votes.
    Signal list derived dynamically from the data + canonical SIGNAL_NAMES.
    """
    summary_path = DATA_DIR / "agent_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        return jsonify({"error": "no data"}), 404
    if not isinstance(summary, dict):
        # Malformed producer output (e.g. a bare list/scalar) — in-band
        # error, not a 500. tickers/signals below all key off dict shape.
        return jsonify({"error": "malformed agent_summary.json"}), 200

    signals_data = summary.get("signals", {})
    if not isinstance(signals_data, dict):
        signals_data = {}
    # Drop non-dict per-ticker entries here, once, so both loops below
    # (vote-key discovery and heatmap-row building) see only well-shaped
    # rows — one corrupt ticker entry isolates to a missing row, not a 500.
    signals_data = {t: v for t, v in signals_data.items() if isinstance(v, dict)}

    try:
        from portfolio.tickers import SIGNAL_NAMES
        from portfolio.component_registry import get_registry
    except ImportError:
        SIGNAL_NAMES = []
        get_registry = None

    all_vote_keys = set()
    for ticker_data in signals_data.values():
        all_vote_keys.update(ticker_data.get("extra", {}).get("_votes", {}).keys())

    all_signals = list(SIGNAL_NAMES) if SIGNAL_NAMES else sorted(all_vote_keys)
    for k in sorted(all_vote_keys - set(all_signals)):
        all_signals.append(k)

    heatmap = {}
    tickers = list(signals_data.keys())

    for ticker in tickers:
        sig = signals_data[ticker]
        extra = sig.get("extra", {})
        votes = extra.get("_votes", {})

        row = {}
        for s in all_signals:
            row[s] = (votes.get(s, "HOLD") or "HOLD").upper()
        heatmap[ticker] = row

    # Per-(ticker, signal) state-change timestamps for the "time-in-state" badge.
    # Written by portfolio.reporting._update_signal_state_since each loop cycle.
    # Missing or malformed payload degrades to an empty map: frontend renders
    # cells without the badge — never 500.
    #
    # Codex P2 (2026-05-05): the since-file is written *before* agent_summary
    # in the same cycle, and a swallowed write-failure can also leave the two
    # out of sync. Guard against showing a stale duration on a freshly-flipped
    # vote by only emitting `since` when the recorded vote matches the current
    # heatmap value. Mismatched cells fall back to colour-only until the next
    # cycle re-syncs both files.
    state_since_payload = _read_json(DATA_DIR / "signal_state_since.json") or {}
    state_since_votes = state_since_payload.get("votes") if isinstance(state_since_payload, dict) else None
    since: dict[str, dict[str, str]] = {}
    if isinstance(state_since_votes, dict):
        for ticker in tickers:
            tk_state = state_since_votes.get(ticker)
            if not isinstance(tk_state, dict):
                continue
            row_since: dict[str, str] = {}
            current_row = heatmap.get(ticker, {})
            for s in all_signals:
                entry = tk_state.get(s)
                if not isinstance(entry, dict):
                    continue
                since_ts = entry.get("since")
                if not isinstance(since_ts, str):
                    continue
                if entry.get("vote") != current_row.get(s):
                    continue  # stale: vote in since-file disagrees with heatmap
                row_since[s] = since_ts
            if row_since:
                since[ticker] = row_since

    # 2026-07-18 Phase 4.3: sourced from component_registry, bounded to
    # SIGNAL_NAMES to match the old DISABLED_SIGNALS scope exactly (parity
    # verified in tests/test_dashboard.py).
    disabled = (
        sorted(s for s in SIGNAL_NAMES if get_registry().disabled_reason(s))
        if SIGNAL_NAMES and get_registry
        else []
    )
    # Freshness contract (2026-07-19): the producer can die while this
    # route keeps serving the last-written grid at 200 forever. Attach the
    # source file's mtime so the frontend can flag a stale heatmap, same
    # pattern as /api/accuracy's meta.data_ts.
    try:
        mtime = summary_path.stat().st_mtime
    except OSError:
        mtime = None
    meta = {
        "data_ts": mtime,
        "age_sec": int(time.time() - mtime) if mtime is not None else None,
    }
    return jsonify({
        "tickers": tickers,
        "signals": all_signals,
        "heatmap": heatmap,
        "since": since,
        "disabled_signals": disabled,
        "meta": meta,
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
    """Return accuracy snapshots over time for charting trend lines.

    2026-05-05: tag each per-signal slice with `enabled` so the chart
    can dim/exclude force-HOLD'd signals. Tag is derived at response
    time from DISABLED_SIGNALS so historical snapshots written before
    the flag existed are also tagged correctly.

    2026-07-18 (Phase 4.5): intentionally NOT migrated to
    component_registry — unlike this file's other two DISABLED_SIGNALS
    reads. These are archival snapshot rows, some containing retired
    signal names (e.g. kronos) that never made it into registry_defaults
    SIGNALS (only current SIGNAL_NAMES are snapshotted). The registry's
    unknown-signal default is "disabled", which would mislabel that
    archival data as force-HOLD rather than leaving it untagged/enabled.
    Revisit only if/when the registry gains a real historical-signal
    table.
    """
    entries = _read_jsonl(DATA_DIR / "accuracy_snapshots.jsonl", limit=500)
    try:
        from portfolio.tickers import DISABLED_SIGNALS
        for snap in entries:
            sigs = snap.get("signals") if isinstance(snap, dict) else None
            if not isinstance(sigs, dict):
                continue
            for sig_name, info in sigs.items():
                if isinstance(info, dict):
                    # Overwrite (not setdefault) — see /api/accuracy comment.
                    info["enabled"] = sig_name not in DISABLED_SIGNALS
    except Exception:
        logger.exception("accuracy-history enrichment failed; serving raw")
    return jsonify(entries)


@app.route("/api/calibration")
@require_auth
def api_calibration():
    """Confidence calibration: predicted confidence vs actual accuracy by bucket."""
    horizon = request.args.get("horizon", "1d")
    since = request.args.get("since")
    try:
        from portfolio.accuracy_stats import probability_calibration
        data = probability_calibration(horizon, since=since)
        return jsonify({"horizon": horizon, "buckets": data})
    except Exception:
        logger.exception("calibration endpoint failed")
        return jsonify({"error": "calibration failed"}), 500


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
        return jsonify({"error": "no data", "stats": {}}), 404
    return jsonify(data)


def _compute_llm_leaderboard():
    """Per-signal accuracy + Brier scorecard joined with shadow_registry.

    Scans the full llm_probability_log.jsonl + llm_probability_outcomes.jsonl.
    Cached at the /api/ layer because the join touches ~85K rows.

    Accuracy methodology (must match
    ``scripts/review_shadow_signals.py:_compute_signal_stats`` or the
    leaderboard and the auto-promotion gate disagree on which shadows
    are eligible — premortem N1 in
    ``.worktrees/shadow-gate-lora-20260518/docs/PLAN.md``):

    * ``n_samples`` counts every log row (incl. abstains) so callers
      can see the model's emission cadence.
    * ``n_with_outcome`` and ``accuracy`` count only directional
      predictions (``confidence > 0`` AND ``chosen in {BUY, SELL}``)
      that joined to an outcome — see
      ``portfolio.llm_probability_log.is_directional_prediction``.
    * Brier is reported over the same directional set so its
      denominator equals ``n_with_outcome``. Including abstains would
      score the canonical ``{0.25, 0.5, 0.25}`` distribution and
      systematically penalise models with high abstain rates.
    """
    from portfolio.llm_probability_log import is_directional_prediction
    from portfolio.shadow_registry import days_in_shadow, load_registry

    registry = load_registry()
    shadows = registry.get("shadows", {})
    log_rows = _load_jsonl_impl(DATA_DIR / "llm_probability_log.jsonl") or []
    out_rows = _load_jsonl_impl(DATA_DIR / "llm_probability_outcomes.jsonl") or []

    outcomes = {}
    for row in out_rows:
        if not isinstance(row, dict):
            continue
        key = (row.get("ts"), row.get("signal"), row.get("ticker"), row.get("horizon"))
        outcomes[key] = row.get("outcome")

    per_sig = {}
    for row in log_rows:
        if not isinstance(row, dict):
            continue
        sig = row.get("signal")
        if not sig:
            continue
        d = per_sig.setdefault(
            sig,
            {"n": 0, "n_matched": 0, "correct": 0, "brier_sum": 0.0, "n_directional": 0},
        )
        d["n"] += 1
        if not is_directional_prediction(row):
            continue
        d["n_directional"] += 1
        key = (row.get("ts"), sig, row.get("ticker"), row.get("horizon"))
        actual = outcomes.get(key)
        if actual is None:
            continue
        d["n_matched"] += 1
        if row.get("chosen") == actual:
            d["correct"] += 1
        probs = row.get("probs") or {}
        row_brier = 0.0
        for action in ("BUY", "HOLD", "SELL"):
            y = 1.0 if actual == action else 0.0
            try:
                p = float(probs.get(action, 0.0))
            except (TypeError, ValueError):
                p = 0.0
            row_brier += (p - y) ** 2
        d["brier_sum"] += row_brier

    leaderboard = []
    for sig in sorted(set(per_sig.keys()) | set(shadows.keys())):
        agg = per_sig.get(sig) or {
            "n": 0, "n_matched": 0, "correct": 0, "brier_sum": 0.0, "n_directional": 0,
        }
        entry = shadows.get(sig, {})
        accuracy = (agg["correct"] / agg["n_matched"]) if agg["n_matched"] else None
        brier = (agg["brier_sum"] / agg["n_matched"]) if agg["n_matched"] else None
        days = days_in_shadow(sig) if sig in shadows else None
        leaderboard.append({
            "name": sig,
            "status": entry.get("status"),
            "n_samples": agg["n"],
            "n_directional": agg["n_directional"],
            "n_with_outcome": agg["n_matched"],
            "join_rate": round(agg["n_matched"] / agg["n"], 4) if agg["n"] else None,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "brier": round(brier, 4) if brier is not None else None,
            "days_in_shadow": round(days, 1) if days is not None else None,
            "last_reviewed": entry.get("last_reviewed_ts"),
            "promotion_criteria": entry.get("promotion_criteria"),
            "notes": entry.get("notes"),
        })
    return {"updated_ts": datetime.now(UTC).isoformat(), "signals": leaderboard}


@app.route("/api/llm-leaderboard")
@require_auth
def api_llm_leaderboard():
    """Per-LLM scorecard: status, sample count, join rate, accuracy, Brier.

    Joins llm_probability_log entries against backfilled outcomes and
    cross-references shadow_registry for promotion criteria + days in shadow.
    Cached for 5 minutes — the join is O(N+M) over ~85K rows.
    """
    data = _cached_read("llm_leaderboard_v1", 300, _compute_llm_leaderboard)
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
            if not isinstance(decisions, dict):
                decisions = {}
            matched = False
            for strat, dec in decisions.items():
                if not isinstance(dec, dict):
                    continue
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
# Prophecy: daily AI price predictions (separate subsystem, prophecy/ package).
# Reads data/prophecy_runs/ snapshots written by the zero-token publish/outcomes
# steps. Additive — does not touch existing routes.
# ---------------------------------------------------------------------------

@app.route("/api/prophecy")
@require_auth
def api_prophecy():
    """Prophecy predictions snapshot + accuracy + cost + coverage gaps.

    Degrades to an empty payload if Prophecy has not produced a run yet.
    """
    pdir = DATA_DIR / "prophecy_runs"
    latest = _read_json(pdir / "latest.json") or {}
    accuracy = _read_json(pdir / "accuracy.json") or {}
    return jsonify({
        "latest": latest,
        "accuracy": accuracy.get("instruments", {}),
        "accuracy_updated_at": accuracy.get("updated_at"),
        "frozen": (pdir / "SYSTEM_DISABLED").exists(),
    })


@app.route("/prophecy")
@require_auth
def prophecy_page():
    if request.args.get("token"):
        return redirect("/prophecy", code=302)
    return send_from_directory("static", "prophecy.html")


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


@app.route("/api/loop-processes")
@require_auth
def api_loop_processes():
    """Running-loop enumeration for the duplicate-detection tile.

    Replaces the visual "I can see the popup windows" cue that
    disappears after hide-windows lands. psutil enumerates Python
    processes once per request and matches each against the
    finance-analyzer loop signatures in portfolio.loop_processes.

    Cheap on the live machine (a few ms per call); poll every 30 s.
    """
    from portfolio.loop_processes import scan
    try:
        return jsonify(scan())
    except Exception as exc:  # noqa: BLE001 — psutil should not raise, but the dashboard tile must degrade gracefully
        # Log the full traceback locally for diagnosis; return a generic
        # error string to the client so we don't leak filesystem paths
        # or other internal context through the dashboard JSON.
        logger.exception("loop_processes scan failed: %s", exc)
        return jsonify({
            "error": "loop_processes scan failed; see server logs",
            "loops": [],
            "any_duplicate": False,
        }), 500


@app.route("/api/pickups")
@require_auth
def api_pickups():
    """Pending and completed scheduled pickups from data/pending_pickups.json.

    Each pickup is a one-shot verification job processed by
    scripts/process_pending_pickups.py (registered as PF-PendingPickups
    Windows scheduled task, daily 08:00 CET). The pickup file is the
    source of truth for "what auto-runs in N days" so the dashboard tile
    can show "due today / due in 2d / completed yesterday" without
    relying on session memory.

    Read-only endpoint. Mutations happen on the cron path.
    """
    try:
        data = _read_json(DATA_DIR / "pending_pickups.json")
    except Exception as exc:  # noqa: BLE001 — dashboard must degrade
        logger.exception("pickups read failed: %s", exc)
        return jsonify({"error": "pickups read failed", "pickups": []}), 500
    if not isinstance(data, dict):
        return jsonify({"pickups": []})
    out = []
    now = datetime.now(UTC)
    for p in data.get("pickups", []):
        if not isinstance(p, dict):
            continue
        due_raw = p.get("due_ts")
        due_dt = None
        if isinstance(due_raw, str):
            try:
                due_dt = datetime.fromisoformat(due_raw)
                if due_dt.tzinfo is None:
                    due_dt = due_dt.replace(tzinfo=UTC)
            except ValueError:
                due_dt = None
        days_until = None
        if due_dt is not None:
            days_until = round((due_dt - now).total_seconds() / 86400.0, 2)
        last_verdict = None
        history = p.get("history") or []
        if history and isinstance(history[-1], dict):
            last_verdict = history[-1].get("verdict")
        out.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "due_ts": due_raw,
            "days_until_due": days_until,
            "status": p.get("status", "pending"),
            "handler": p.get("handler"),
            "last_verdict": last_verdict,
            "last_run_ts": p.get("last_run_ts"),
        })
    out.sort(
        key=lambda r: (r["days_until_due"] is None, r["days_until_due"] or 0),
    )
    return jsonify({"pickups": out, "updated_ts": now.isoformat()})


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

        # 2026-05-14: Swedish bank-holiday flag so the dashboard surfaces
        # "Avanza closed today" without users guessing from a stale
        # last-update timestamp. Source: portfolio.market_timing
        # swedish_market_holidays() — 14 SE bank holidays / year, e.g.
        # Ascension (today, 2026-05-14), Midsummer Eve, Christmas Eve.
        try:
            from datetime import date
            from portfolio.market_timing import (
                is_swedish_market_holiday,
                swedish_market_holidays,
            )
            today = date.today()
            today_holiday = is_swedish_market_holiday()
            year_holidays = sorted(swedish_market_holidays(today.year))
            next_holiday = next((d for d in year_holidays if d >= today), None)
            se_info = {
                "today_is_holiday": today_holiday,
                "today": today.isoformat(),
            }
            if next_holiday is not None:
                se_info["next_holiday"] = next_holiday.isoformat()
                se_info["days_until_next"] = (next_holiday - today).days
            result["swedish_market"] = se_info
        except Exception:
            logger.debug("swedish_market enrichment failed", exc_info=True)

        return jsonify(result)
    except Exception:
        logger.exception("market-health endpoint error")
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

# 2026-06-10 (audit batch 10): the queue was unbounded and had no in-flight
# dedup, so N concurrent cache-miss / ?force=1 requests enqueued N jobs that
# serialised on the single worker — the k-th caller's 25s wait would expire and
# return a timeout while the worker kept draining stale queued jobs, and a
# stuck BankID session (every snapshot hitting the full timeout) could grow the
# queue without bound. Fixes: (1) bound the queue (maxsize) and return 503 when
# full instead of piling on; (2) coalesce concurrent callers onto a single
# in-flight future so duplicate requests share one snapshot build rather than
# each enqueuing their own.
_AVANZA_REQ_Q: "queue.Queue[dict]" = queue.Queue(maxsize=4)
_AVANZA_WORKER_LOCK = threading.Lock()
_AVANZA_WORKER_STARTED = False
_AVANZA_REQ_TIMEOUT_SECONDS = 25.0  # snapshot upper bound

# In-flight dedup: at most one snapshot future is enqueued at a time;
# concurrent callers attach to it and await the same result.
_AVANZA_INFLIGHT_LOCK = threading.Lock()
_AVANZA_INFLIGHT: dict[str, Any] | None = None


class _AvanzaQueueFull(Exception):
    """Raised when the worker queue is saturated — surfaced as HTTP 503."""


def _avanza_worker_loop() -> None:
    """Single-thread worker that owns Playwright. Blocks on the request
    queue and serves snapshot requests sequentially."""
    global _AVANZA_INFLIGHT
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
            # Clear the in-flight slot before signalling waiters so the next
            # caller after this one completes starts a fresh snapshot.
            with _AVANZA_INFLIGHT_LOCK:
                if _AVANZA_INFLIGHT is future:
                    _AVANZA_INFLIGHT = None
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
    Playwright's thread affinity is honoured.

    Concurrent callers coalesce onto a single in-flight future (dedup). If no
    snapshot is in flight, this caller enqueues one; a bounded queue means a
    saturated worker raises ``_AvanzaQueueFull`` (HTTP 503) rather than piling
    up unbounded work."""
    global _AVANZA_INFLIGHT
    _ensure_avanza_worker()

    with _AVANZA_INFLIGHT_LOCK:
        future = _AVANZA_INFLIGHT
        if future is None:
            future = {"result": None, "done": threading.Event()}
            try:
                # Non-blocking: if the queue is already saturated, fail fast
                # rather than blocking the request thread.
                _AVANZA_REQ_Q.put_nowait(future)
            except queue.Full as exc:
                raise _AvanzaQueueFull() from exc
            _AVANZA_INFLIGHT = future
        # else: attach to the existing in-flight future (dedup).

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
    try:
        snapshot = _avanza_account_snapshot()
    except _AvanzaQueueFull:
        # Worker saturated (e.g. a stuck BankID session). Fail fast with 503 +
        # Retry-After rather than enqueuing more work the worker can't drain.
        resp = jsonify({"error": "avanza_worker_busy"})
        resp.headers["Retry-After"] = str(int(_AVANZA_REQ_TIMEOUT_SECONDS))
        return resp, 503
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


@app.route("/api/claude_cost")
@require_auth
def api_claude_cost():
    """Claude CLI cost + token rollup over the last N days.

    Wraps ``scripts/claude_cost_report.collect`` + ``summarise``. Reads
    ``data/claude_invocations.jsonl`` (gate rows with usage envelope) and
    ``data/invocations.jsonl`` (Layer 2 wrapper rows, no tokens). Returns
    the same shape as ``scripts/claude_cost_report.py --json``.

    Query params:
      - days (int, default 7, max 90)
    """
    import sys as _sys
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)
    try:
        from scripts import claude_cost_report as _ccr
    except Exception:
        logger.exception("claude_cost_report import failed")
        return jsonify({"error": "import_failed"}), 500

    try:
        days = int(request.args.get("days", "7"))
    except (TypeError, ValueError):
        days = 7
    days = max(1, min(days, 90))

    try:
        bundle = _ccr.collect(days)
        summary = _ccr.summarise(bundle)
        summary["days"] = days
        return jsonify(summary)
    except Exception:
        logger.exception("claude_cost endpoint error")
        return jsonify({"error": "Internal server error"}), 500


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

# Blueprint: /api/control — Command Central write API (Phase 3, 2026-07-18).
# See dashboard/control.py for routes + hardening (allowlist, rate limit, audit).
from dashboard.control import bp as _control_bp  # noqa: E402

app.register_blueprint(_control_bp)

# Blueprint: /api/silver — XAG-USD per-signal-per-horizon accuracy for the
# #silver command page (Phase 6, 2026-07-18). See dashboard/silver.py.
from dashboard.silver import bp as _silver_bp  # noqa: E402

app.register_blueprint(_silver_bp)


# Short vanity aliases for the househunting viewer. Gated by require_auth like
# the SPA root (NOT bare like /logout): a first-visit bootstrap via /hh?token=XXX
# must reach require_auth so the token is converted to the pf_dashboard_token
# cookie before we redirect — otherwise the bare redirect drops the query
# string and the token-less /house/ 401s on a fresh device. The redirect
# targets are token-less, keeping the address bar clean.
@app.route("/hh")
@require_auth
def hh():
    return redirect("/house/", code=302)


@app.route("/hhmap")
@require_auth
def hhmap():
    return redirect("/house/heatmap", code=302)


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
    # Suppressed false-positive: Intentional dual-stack IPv6/IPv4 bind; Cloudflare Access fronts public exposure (CLAUDE.md dashboard section).
    # nosemgrep: python.lang.security.audit.network.bind.avoid-bind-to-all-interfaces
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
