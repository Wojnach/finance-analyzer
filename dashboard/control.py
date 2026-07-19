"""dashboard/control.py — Command Central write API (Phase 3, 2026-07-18).

Blueprint at ``/api/control``. This is the only write surface anywhere in
the dashboard — every other route is read-only. State changes land in
flag files / control-adjacent JSON under ``data/control/``, NEVER
``config.json`` (the dashboard must never touch the file that holds API
keys).

Every route is ``@require_auth`` (see ``dashboard.auth``). The three
mutating routes (llm, instrument, loop) are POST-only and carry two extra
layers of hardening on top of that baseline — both mandated because the
dashboard is public (raanman.lol via Cloudflare tunnel):

  - Loop start/stop/restart is scoped to a hardcoded allowlist of pf-*
    user units (``LOOP_ALLOWLIST``). ``pf-dashboard`` is deliberately
    excluded — the dashboard must never be able to stop the process
    serving itself.
  - A shared, in-process rate limiter caps every control POST (llm,
    instrument, loop — NOT the read-only ``/state``) at
    ``_RATE_LIMIT_MAX`` per ``_RATE_LIMIT_WINDOW_S``, across all three
    endpoints combined, so a scripted client can't spam systemctl calls.

Every action attempt — success or rejected — is appended to
``data/control/audit.jsonl`` via ``atomic_append_jsonl``:
``{ts, endpoint, payload, result, remote_addr}``. Append-only, never
mutated.

``data/control/instruments.json`` isn't consumed by any loop yet — Layer
1 wiring is Phase 4.3. This route just builds the file up as the future
source of truth; toggling a ticker here has no live effect until then.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, g, jsonify, request

from dashboard.auth import require_auth
from portfolio.component_registry import get_registry
from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from portfolio.local_llm_gate import DISABLE_FLAG
from portfolio.tickers import ALL_TICKERS

bp = Blueprint("control", __name__, url_prefix="/api/control")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONTROL_DIR = DATA_DIR / "control"
INSTRUMENTS_PATH = CONTROL_DIR / "instruments.json"
AUDIT_PATH = CONTROL_DIR / "audit.jsonl"

# Loop start/stop/restart is confined to this hardcoded set — pf-dashboard
# is deliberately absent (see module docstring).
LOOP_ALLOWLIST = frozenset(
    {
        "pf-dataloop",
        "pf-metalsloop",
        "pf-cryptoloop",
        "pf-oilloop",
        "pf-mstrloop",
        "pf-golddigger",
    }
)
LOOP_ACTIONS = frozenset({"start", "stop", "restart"})

_RATE_LIMIT_MAX = 6
_RATE_LIMIT_WINDOW_S = 60.0
_rate_lock = threading.Lock()
_rate_events: deque[float] = deque()


def _rate_limit_take() -> bool:
    """Record one control POST against the shared budget.

    Returns True when the request may proceed. Returns False when the
    caller is already over ``_RATE_LIMIT_MAX`` in the trailing
    ``_RATE_LIMIT_WINDOW_S`` — the caller must respond 429 and skip the
    action (this function does NOT roll back a taken slot).
    """
    now = time.time()
    with _rate_lock:
        while _rate_events and now - _rate_events[0] > _RATE_LIMIT_WINDOW_S:
            _rate_events.popleft()
        if len(_rate_events) >= _RATE_LIMIT_MAX:
            return False
        _rate_events.append(now)
        return True


def _rate_limit_remaining() -> int:
    now = time.time()
    with _rate_lock:
        while _rate_events and now - _rate_events[0] > _RATE_LIMIT_WINDOW_S:
            _rate_events.popleft()
        return max(0, _RATE_LIMIT_MAX - len(_rate_events))


def _audit(endpoint: str, payload: Any, result: Any) -> None:
    """Append one line to data/control/audit.jsonl.

    Never raises — a logging failure must not break the control action
    itself, and must not be visible to the caller as a 500.

    ``remote_addr`` is near-useless for attribution here: the dashboard sits
    behind cloudflared, so it's always 127.0.0.1. ``actor``/``auth_method``
    come from ``flask.g``, stashed by ``dashboard.auth.require_auth`` on
    the path that authenticated this request (cf_access carries a
    JWT-verified email; cookie/bearer/query don't identify a person).
    ``cf_connecting_ip`` is the CF-reported client IP — claimed, not
    verified, but the best available hint on the CF Access path.
    """
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_append_jsonl(
            AUDIT_PATH,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "payload": payload,
                "result": result,
                "remote_addr": request.remote_addr,
                "actor": g.get("pf_actor"),
                "auth_method": g.get("pf_auth_method"),
                "cf_connecting_ip": request.headers.get("CF-Connecting-IP"),
            },
        )
    except Exception:
        pass


def _csrf_ok() -> bool:
    """Backstop for cookie auth beyond SameSite=Lax (auth.py:152 NOTE).

    Only cookie-authenticated requests are silently supplied by a browser
    without the caller knowing the token — bearer/query/cf_access callers
    already had to possess the secret, so CSRF doesn't apply to them.
    """
    if g.get("pf_auth_method") != "cookie":
        return True
    origin = request.headers.get("Origin") or request.headers.get("Referer") or ""
    return request.host in origin


def _systemctl_query(unit: str, verb: str) -> str | None:
    """Run `systemctl --user <verb> <unit>` and return stripped stdout, or
    None on any failure. Never raises."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", verb, unit],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _systemctl_action(unit: str, action: str) -> dict[str, Any]:
    """Run `systemctl --user <action> <unit>` and report the outcome.
    Never raises — failures (binary missing, timeout, non-systemd host)
    come back as ``ok: False`` with the error in ``stderr``."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", action, unit],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stderr": r.stderr.strip()[:500],
        }
    except Exception as e:
        return {"ok": False, "returncode": None, "stderr": f"{type(e).__name__}: {e}"}


@bp.route("/llm", methods=["POST"])
@require_auth
def api_control_llm():
    """Toggle the master local-LLM pause switch (data/local_llm.disabled)."""
    if not _csrf_ok():
        return jsonify({"error": "csrf_check_failed"}), 403
    if not _rate_limit_take():
        return jsonify({"error": "rate_limited"}), 429

    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        _audit("llm", data, "error: enabled must be a bool")
        return jsonify({"error": "enabled must be a bool"}), 400

    if enabled:
        DISABLE_FLAG.unlink(missing_ok=True)
    else:
        DISABLE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        DISABLE_FLAG.touch()

    result = {"llm_enabled": not DISABLE_FLAG.exists()}
    _audit("llm", data, result)
    return jsonify(result)


@bp.route("/instrument", methods=["POST"])
@require_auth
def api_control_instrument():
    """Set the `tracked` flag for one ticker in data/control/instruments.json."""
    if not _csrf_ok():
        return jsonify({"error": "csrf_check_failed"}), 403
    if not _rate_limit_take():
        return jsonify({"error": "rate_limited"}), 429

    data = request.get_json(silent=True) or {}
    ticker = data.get("ticker")
    tracked = data.get("tracked")
    if ticker not in ALL_TICKERS:
        _audit("instrument", data, "error: unknown ticker")
        return jsonify({"error": f"unknown ticker: {ticker!r}"}), 400
    if not isinstance(tracked, bool):
        _audit("instrument", data, "error: tracked must be a bool")
        return jsonify({"error": "tracked must be a bool"}), 400

    instruments = load_json(INSTRUMENTS_PATH, default={}) or {}
    instruments[ticker] = {"tracked": tracked}
    atomic_write_json(INSTRUMENTS_PATH, instruments)

    result = {"ticker": ticker, "tracked": tracked}
    _audit("instrument", data, result)
    return jsonify(result)


@bp.route("/loop", methods=["POST"])
@require_auth
def api_control_loop():
    """Start/stop/restart one allowlisted pf-* systemd --user unit."""
    if not _csrf_ok():
        return jsonify({"error": "csrf_check_failed"}), 403
    if not _rate_limit_take():
        return jsonify({"error": "rate_limited"}), 429

    data = request.get_json(silent=True) or {}
    unit = data.get("unit")
    action = data.get("action")
    if unit not in LOOP_ALLOWLIST:
        _audit("loop", data, "error: unit not allowlisted")
        return jsonify({"error": f"unit not allowed: {unit!r}"}), 400
    if action not in LOOP_ACTIONS:
        _audit("loop", data, "error: invalid action")
        return jsonify({"error": f"invalid action: {action!r}"}), 400

    result = _systemctl_action(unit, action)
    _audit("loop", data, result)
    return jsonify({"unit": unit, "action": action, **result}), (
        200 if result["ok"] else 502
    )


@bp.route("/registry", methods=["GET"])
@require_auth
def api_control_registry():
    """Component registry snapshot (Phase 4.1) — read-only, no rate-limit hit.

    No ``?ticker=``: the full ``{ticker: {signal: {...}}}`` dump from
    ``ComponentRegistry.snapshot()``. With ``?ticker=<TICKER>``: that
    ticker's slice reshaped into ``applicable`` (enabled signal names) +
    ``disabled`` (``{signal, reason}`` pairs) + the full per-signal
    ``signals`` detail — built for the #silver page's component-health
    pill grid (Phase 6), which needs both the compact sets and the
    per-signal ``voter_state``/``horizons`` detail in one round trip.
    """
    snapshot = get_registry().snapshot()
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify({"ticker": None, "registry": snapshot})

    if ticker not in ALL_TICKERS:
        return jsonify({"error": f"unknown ticker: {ticker!r}"}), 400

    signals = snapshot.get(ticker, {})
    applicable = sorted(s for s, info in signals.items() if info.get("enabled_default"))
    disabled = sorted(
        (
            {"signal": s, "reason": info.get("disabled_reason")}
            for s, info in signals.items()
            if not info.get("enabled_default")
        ),
        key=lambda d: d["signal"],
    )
    return jsonify(
        {
            "ticker": ticker,
            "registry": {
                "applicable": applicable,
                "disabled": disabled,
                "signals": signals,
            },
        }
    )


@bp.route("/state", methods=["GET"])
@require_auth
def api_control_state():
    """Current control-plane state: LLM gate, instrument toggles, loop
    status for every allowlisted unit, and the remaining POST budget."""
    instruments_file = load_json(INSTRUMENTS_PATH, default={}) or {}
    instruments = {
        t: instruments_file.get(t, {"tracked": True}) for t in sorted(ALL_TICKERS)
    }

    loops = {}
    for unit in sorted(LOOP_ALLOWLIST):
        # A None query result means systemctl itself failed (binary missing,
        # timeout, --user session not reachable) — that's an outage of our
        # ability to ask, not a confirmed-stopped unit. Reporting it as
        # active:false would tell the UI the loop is definitely down when
        # we actually have no idea.
        active_raw = _systemctl_query(unit, "is-active")
        enabled_raw = _systemctl_query(unit, "is-enabled")
        loops[unit] = {
            "active": None if active_raw is None else active_raw == "active",
            "enabled": None if enabled_raw is None else enabled_raw == "enabled",
            "state": "unknown" if active_raw is None else active_raw,
        }

    return jsonify(
        {
            "llm_enabled": not DISABLE_FLAG.exists(),
            "instruments": instruments,
            "loops": loops,
            "rate_limit_remaining": _rate_limit_remaining(),
        }
    )
