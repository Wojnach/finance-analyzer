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
import logging
import threading
import time
from pathlib import Path

from flask import g, jsonify, make_response, request

logger = logging.getLogger(__name__)

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
# Tracks whether the most recent config read SUCCEEDED (or found the file
# genuinely absent). False only when the file is present but could not be
# read/parsed. Used by require_auth to fail CLOSED on a cold-start read
# failure instead of mistaking an unreadable config for "no token configured".
_LAST_READ_OK: bool = True


def _read_config_uncached() -> tuple[dict, bool]:
    """Read config.json.

    Returns ``(data, ok)``. ``ok`` is True on a successful parse OR when the
    file is genuinely absent (FileNotFoundError → the documented "no config →
    no token → open access" backward-compat case). ``ok`` is False when the
    file IS present but a transient/corrupt read failed — the caller must NOT
    treat that as "no token configured" (2026-05-28: that cold-start path
    previously cached {} and fail-OPENed the dashboard for 60s).
    """
    try:
        with open(CONFIG_PATH, encoding="utf-8") as fp:
            data = json.load(fp)
        return (data if isinstance(data, dict) else {}), True
    except FileNotFoundError:
        return {}, True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}, False


def _get_config() -> dict:
    """Read config.json with a 60-second in-memory cache.

    Caches only successful reads. On a read failure with a warm cache, keeps
    serving the last-known-good value (B11) without advancing the timestamp so
    the next request retries promptly. On a cold-start read failure, returns
    {} WITHOUT caching it (so the next request retries) and leaves
    ``_LAST_READ_OK`` False so auth can fail closed.
    """
    global _CFG_VALUE, _CFG_AT, _LAST_READ_OK
    now = time.monotonic()
    with _CFG_LOCK:
        if _CFG_VALUE is not None and (now - _CFG_AT) < _CFG_TTL:
            return _CFG_VALUE
        fresh, ok = _read_config_uncached()
        _LAST_READ_OK = ok
        if ok:
            _CFG_VALUE = fresh
            _CFG_AT = now
            return _CFG_VALUE
        # Read failed (file present but unreadable/corrupt).
        if _CFG_VALUE is not None:
            return _CFG_VALUE  # warm cache: serve last-known-good
        return {}  # cold start: do NOT cache; do NOT fail open (see _config_is_known)


def _config_is_known() -> bool:
    """True when we have a trustworthy config view (cached good value, or the
    last read succeeded/was genuinely absent). False only on a cold-start read
    failure where granting open access would be unsafe."""
    return _CFG_VALUE is not None or _LAST_READ_OK


def _get_dashboard_token() -> str | None:
    """Return the configured dashboard_token, or None if not set."""
    return _get_config().get("dashboard_token") or None


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def _request_is_https() -> bool:
    """True when the current request reached us over TLS, either directly
    (request.is_secure) or via the Cloudflare tunnel which terminates TLS and
    forwards X-Forwarded-Proto=https. Used to decide the cookie Secure flag."""
    if request.is_secure:
        return True
    return request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip().lower() == "https"


def _refresh_cookie(response, token: str):
    """Refresh the auth cookie's expiry on `response`.

    2026-06-10 (audit batch 10): Secure is now set per-request scheme instead
    of unconditionally True. The server also binds plain HTTP on all interfaces
    for direct LAN / WSL / Tailscale access; browsers silently refuse to store
    a Secure cookie over http://, which forced the user to re-supply ?token= on
    every navigation — leaking the raw token into access logs each time. Setting
    Secure only when the request is actually HTTPS (direct TLS or behind the CF
    tunnel) lets the LAN-HTTP fallback keep a cookie. The HTTPS tunnel path is
    unchanged. httponly + samesite are unchanged. Tradeoff: over plain LAN HTTP
    the cookie is not Secure-flagged, accepted because that path is already
    cleartext on a trusted LAN and the alternative (no cookie) is worse."""
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        secure=_request_is_https(),
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
            # 2026-05-28: only fail OPEN when config is genuinely token-less.
            # If the config file is present but a cold-start read failed, we
            # don't actually know whether a token is configured — fail CLOSED
            # (503) rather than granting unauthenticated access on an
            # all-interfaces bind. Self-heals once a read succeeds (~next req).
            if not _config_is_known():
                logger.warning(
                    "Dashboard auth: config unreadable on cold start — failing "
                    "closed (503) instead of granting open access."
                )
                return jsonify({"error": "config_unavailable"}), 503
            return f(*args, **kwargs)

        # 0. Cloudflare Access — added 2026-05-02, JWT-verified 2026-05-14.
        # CF strips inbound Cf-Access-* headers at its edge and re-injects
        # them only after successful Access policy evaluation. BUT the
        # 2026-05-02 implementation trusted those headers based on
        # presence alone — anything on the LAN or Tailscale could spoof
        # both headers and bypass auth (P0 in 2026-05-13 adversarial
        # review). The JWT assertion is now verified against CF's
        # published JWKs via dashboard.cf_access.verify_cf_jwt; on any
        # failure (missing config, bad signature, wrong aud, expired,
        # email/claim mismatch) we fall through to the cookie / query /
        # bearer paths so a misconfigured deployment doesn't lock the
        # operator out.
        cf_email = request.headers.get("Cf-Access-Authenticated-User-Email")
        cf_jwt = request.headers.get("Cf-Access-Jwt-Assertion")
        if cf_email and cf_jwt:
            from dashboard.cf_access import verify_cf_jwt  # local — keeps PyJWT optional at import time
            cfg = _get_config()
            claims = verify_cf_jwt(
                cf_jwt,
                cf_email,
                team_domain=cfg.get("cf_access_team_domain") or None,
                aud_tag=cfg.get("cf_access_aud_tag") or None,
            )
            if claims is not None:
                # Stash the CF-verified identity so control.py's audit log
                # can attribute the action to a real person instead of the
                # always-127.0.0.1 remote_addr behind cloudflared.
                g.pf_actor = cf_email
                g.pf_auth_method = "cf_access"
                # Suppressed false-positive: CF-Access path wraps already-rendered Flask response to refresh auth cookie; no untrusted content injected.
                # nosemgrep: python.flask.security.audit.xss.make-response-with-unknown-content.make-response-with-unknown-content
                return _refresh_cookie(make_response(f(*args, **kwargs)), expected)
            # Otherwise: don't trust the header, fall through.

        # 1. Cookie
        cookie_token = request.cookies.get(COOKIE_NAME)
        if cookie_token and hmac.compare_digest(cookie_token, expected):
            g.pf_auth_method = "cookie"
            # Suppressed false-positive: Cookie auth path: same wrap-and-refresh, cookie comparison uses hmac.compare_digest.
            # nosemgrep: python.flask.security.audit.xss.make-response-with-unknown-content.make-response-with-unknown-content
            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)

        # 2. Query param
        token = request.args.get("token")
        if token and hmac.compare_digest(token, expected):
            g.pf_auth_method = "query"
            # Suppressed false-positive: Query-param auth: hmac.compare_digest gate before refresh; no user content rendered here.
            # nosemgrep: python.flask.security.audit.xss.make-response-with-unknown-content.make-response-with-unknown-content
            return _refresh_cookie(make_response(f(*args, **kwargs)), expected)

        # 3. Authorization: Bearer (CLI / script clients — no cookie set
        # since these don't usually carry one across requests anyway)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:].strip()
            if hmac.compare_digest(bearer_token, expected):
                g.pf_auth_method = "bearer"
                return f(*args, **kwargs)

        # Suppressed false-positive: bool() wraps a logical-and of two header
        # str|None values, not a numeric typecast — NaN injection N/A here.
        # nosemgrep: python.lang.security.audit.dangerous-typecasts.dangerous-bool-cast
        cf_headers_seen = cf_email is not None and cf_jwt is not None
        return jsonify({
            "error": "Unauthorized",
            "message": (
                "Visit /?token=YOUR_TOKEN once to set a 1-year rolling auth "
                "cookie. Replace YOUR_TOKEN with the dashboard_token from "
                "config.json."
            ),
            "cf_access_headers_present": cf_headers_seen,
            "cf_access_hint": (
                "Cloudflare Access headers WERE present but JWT verification "
                "failed. Most common cause: cf_access_team_domain and/or "
                "cf_access_aud_tag missing from config.json. Check dashboard "
                "log for cf_access warnings."
                if cf_headers_seen
                else "Cloudflare Access headers NOT present. Either Access "
                "isn't deployed on this hostname or its identity-header "
                "injection isn't enabled. Check the Access app config."
            ),
        }), 401

    return decorated
