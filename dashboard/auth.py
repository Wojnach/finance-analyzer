"""Auth + token gating for the finance-analyzer dashboard.

Extracted from dashboard/app.py on 2026-05-02 to break a circular import:
dashboard/house_blueprint.py needs `_get_config` and `require_auth`. Before
this split it imported them from dashboard/app.py, which imports
house_blueprint at module-init time. When app.py runs as __main__ (the
PF-Dashboard scheduled task), Python doesn't auto-register it under
`dashboard.app` in sys.modules — so the blueprint's `from dashboard.app
import ...` triggered a fresh import of the same source file, recursing
into a partially-loaded house_blueprint and crashing with ImportError
on `bp`.

Both dashboard/app.py and dashboard/house_blueprint.py now import their
auth dependencies from this module. This module imports nothing
dashboard-specific and uses its own tiny config cache instead of
dashboard/app.py's main TTL cache (which would re-introduce the circle).

Tests that patched `dashboard.app._get_dashboard_token` or
`dashboard.app._get_config` for auth purposes have been updated to patch
`dashboard.auth.*` instead, since require_auth now resolves those names
via its own module globals. App.py re-exports the names for backward
compatibility, but tests should target dashboard.auth as the canonical
location.
"""
from __future__ import annotations

import functools
import hmac
import json
import threading
import time
from pathlib import Path

from flask import jsonify, make_response, request

# Path resolution mirrors dashboard/app.py — config.json sits at the repo
# root, two levels up from this file.
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

# Cookie auth (added 2026-04-30, rolling refresh added 2026-05-02). 365
# sits just under Chrome's silent 400-day cookie max-age cap (introduced
# 2022) — any larger value is clamped browser-side, so 365 is the effective
# ceiling. Combined with require_auth's per-request refresh, an active
# user effectively never re-authenticates.
COOKIE_NAME = "pf_dashboard_token"
COOKIE_MAX_AGE = 365 * 24 * 3600


# ---------------------------------------------------------------------------
# Tiny config cache — separate from dashboard/app.py's main TTL cache so
# this module stays self-contained.
# ---------------------------------------------------------------------------

_CFG_VALUE: dict | None = None
_CFG_AT: float = 0.0
_CFG_LOCK = threading.Lock()
_CFG_TTL = 60.0


def _read_config_uncached() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _get_config() -> dict:
    """Read config.json with a 60-second in-memory cache."""
    global _CFG_VALUE, _CFG_AT
    now = time.monotonic()
    with _CFG_LOCK:
        if _CFG_VALUE is not None and (now - _CFG_AT) < _CFG_TTL:
            return _CFG_VALUE
        _CFG_VALUE = _read_config_uncached()
        _CFG_AT = now
        return _CFG_VALUE


def _get_dashboard_token() -> str | None:
    """Return the configured dashboard_token, or None if not set."""
    return _get_config().get("dashboard_token") or None


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def _refresh_cookie(response, token: str):
    """Refresh the auth cookie's expiry on `response`."""
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="Lax",
    )
    return response


def require_auth(f):
    """Decorator: check Cloudflare Access header, cookie, query, or bearer.

    Validation order:
      0. Cf-Access-Authenticated-User-Email header — Cloudflare Access has
         already authenticated and policy-checked this request. Trust it.
      1. Cookie (`pf_dashboard_token`) — for repeat visits.
      2. ?token= query param — for first-visit-from-a-new-browser.
      3. Authorization: Bearer header — for CLI / script clients.

    On any successful path 0-2, refreshes the cookie's 1-year expiry so
    it slides forward — an active user effectively never re-authenticates.

    If no dashboard_token is configured, access is allowed (backwards
    compatible). Returns 401 for invalid/missing tokens.
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        expected = _get_dashboard_token()
        if expected is None:
            return f(*args, **kwargs)

        # 0. Cloudflare Access — added 2026-05-02. CF strips inbound
        # Cf-Access-* headers at its edge and re-injects them only after
        # successful Access policy evaluation. Require BOTH the email
        # header AND the JWT assertion header — CF Access always sends
        # both, but a LAN attacker spoofing headers would need to know
        # about and forge both. Without the JWT check, any LAN client
        # could set the email header and bypass auth entirely.
        cf_email = request.headers.get("Cf-Access-Authenticated-User-Email")
        cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion")
        if cf_email and cf_jwt:
            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)

        # 1. Cookie
        cookie_token = request.cookies.get(COOKIE_NAME)
        if cookie_token and hmac.compare_digest(cookie_token, expected):
            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)

        # 2. Query param
        token = request.args.get("token")
        if token and hmac.compare_digest(token, expected):
            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)

        # 3. Authorization: Bearer (CLI / script clients — no cookie set
        # since these don't usually carry one across requests anyway)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:].strip()
            if hmac.compare_digest(bearer_token, expected):
                return f(*args, **kwargs)

        return jsonify({
            "error": "Unauthorized",
            "message": (
                "Visit /?token=YOUR_TOKEN once to set a 1-year rolling auth "
                "cookie. Replace YOUR_TOKEN with the dashboard_token from "
                "config.json. (If you arrived here through Cloudflare Access, "
                "this means Access didn't inject its identity header — "
                "contact the app owner.)"
            ),
        }), 401

    return decorated
