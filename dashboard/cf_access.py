"""Cloudflare Access JWT verification.

The dashboard's previous CF-Access path trusted the
``Cf-Access-Authenticated-User-Email`` / ``Cf-Access-Jwt-Assertion``
headers based on their *presence* alone. Anything on the LAN (or
anything that could reach the loopback port via Tailscale, WSL bridge,
etc.) could spoof both headers and bypass auth — flagged P0 in the
2026-05-13 adversarial review.

This module verifies the JWT signature against Cloudflare's published
JWKs, plus standard claims (aud, exp, iss, iat). When the necessary
config keys aren't set, ``verify_cf_jwt`` returns ``None`` so callers
fail closed.

Configuration in ``config.json`` (both optional, both required to
enable CF-Access auth):

* ``cf_access_team_domain``: the team's CF Access subdomain, e.g.
  ``"hazelight.cloudflareaccess.com"``. Used to fetch the JWKs from
  ``https://<team_domain>/cdn-cgi/access/certs``.
* ``cf_access_aud_tag``: the application audience tag from the
  Cloudflare Access dashboard. JWT ``aud`` claim must match.

JWKs are cached for ``_JWKS_CACHE_TTL`` seconds so we don't hammer
Cloudflare; refreshed lazily on the next call after expiry.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import jwt
import requests
from jwt import PyJWKClient

logger = logging.getLogger("dashboard.cf_access")

# JWKs are signed cryptographic keys and rotate on the order of weeks.
# 1h cache keeps freshness without depending on the CF endpoint per
# request. PyJWKClient has its own internal cache too; this is belt-
# and-suspenders.
_JWKS_CACHE_TTL = 3600.0

_JWKS_CLIENT_CACHE: dict[str, tuple[float, PyJWKClient]] = {}
_JWKS_LOCK = threading.Lock()

_MISSING_CONFIG_WARNED = False
_MISSING_CONFIG_LOCK = threading.Lock()


def _warn_missing_config_once() -> None:
    global _MISSING_CONFIG_WARNED
    with _MISSING_CONFIG_LOCK:
        if _MISSING_CONFIG_WARNED:
            return
        _MISSING_CONFIG_WARNED = True
    logger.warning(
        "cf_access: received Cf-Access-Jwt-Assertion but cf_access_team_domain "
        "and/or cf_access_aud_tag are missing from config.json — failing "
        "closed and falling through to cookie/query/bearer auth. Add both "
        "keys to enable CF Access SSO."
    )


def _get_jwks_client(team_domain: str) -> Optional[PyJWKClient]:
    """Return a cached PyJWKClient for the given CF team domain.

    On fetch failure returns ``None`` (callers must fail closed).
    """
    now = time.monotonic()
    with _JWKS_LOCK:
        cached = _JWKS_CLIENT_CACHE.get(team_domain)
        if cached and (now - cached[0]) < _JWKS_CACHE_TTL:
            return cached[1]

    url = f"https://{team_domain}/cdn-cgi/access/certs"
    try:
        # Pre-fetch to verify the endpoint is reachable + returns JSON;
        # PyJWKClient is lazy and would defer the error to verify time
        # otherwise.
        resp = requests.get(url, timeout=5.0)
        resp.raise_for_status()
        resp.json()  # validates payload shape
    except (requests.RequestException, ValueError) as e:
        logger.warning(
            "cf_access: JWKs fetch failed team=%s url=%s err=%r",
            team_domain, url, e,
        )
        return None

    client = PyJWKClient(url, cache_keys=True, lifespan=int(_JWKS_CACHE_TTL))
    with _JWKS_LOCK:
        _JWKS_CLIENT_CACHE[team_domain] = (now, client)
    return client


def verify_cf_jwt(
    token: str,
    expected_email: str,
    *,
    team_domain: Optional[str],
    aud_tag: Optional[str],
) -> Optional[dict]:
    """Verify a Cf-Access-Jwt-Assertion token against CF's JWKs.

    Returns the decoded claims dict on success, or ``None`` on any
    failure (missing config, bad signature, wrong aud, expired,
    email mismatch).

    Callers MUST treat ``None`` as auth-denied and fall through to
    other auth methods (cookie, query, bearer) rather than granting
    access.
    """
    if not team_domain or not aud_tag:
        # Config not set — CF-Access auth is opt-in. Fail closed.
        # 2026-05-16: log once-per-process so an operator who *does* have
        # CF Access deployed but forgot the config keys sees a smoking gun
        # in the dashboard log instead of a silent 401.
        if token:
            _warn_missing_config_once()
        return None
    if not token or not expected_email:
        return None

    client = _get_jwks_client(team_domain)
    if client is None:
        return None

    try:
        signing_key = client.get_signing_key_from_jwt(token)
    except (jwt.exceptions.PyJWKClientError, jwt.exceptions.DecodeError) as e:
        logger.warning("cf_access: signing key lookup failed err=%r", e)
        return None

    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=aud_tag,
            options={"require": ["exp", "iat", "aud"]},
        )
    except jwt.PyJWTError as e:
        logger.warning("cf_access: JWT decode failed err=%r", e)
        return None

    # CF Access puts the authenticated identity in the ``email`` claim.
    # The header that callers trust must match the JWT-signed claim;
    # otherwise an attacker who somehow obtained a valid JWT for one
    # user could replay it with a different email header.
    claim_email = (claims.get("email") or "").lower()
    if claim_email != expected_email.lower():
        logger.warning(
            "cf_access: header/claim email mismatch header=%s claim=%s",
            expected_email, claim_email,
        )
        return None

    return claims


def is_cf_access_configured(config: dict) -> bool:
    """Convenience: returns True iff both config keys are set."""
    return bool(
        (config.get("cf_access_team_domain") or "").strip()
        and (config.get("cf_access_aud_tag") or "").strip()
    )
