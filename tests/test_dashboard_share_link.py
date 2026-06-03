"""Tests for the /go/<slug> magic-link share route (added 2026-06-03).

The route turns a memorable, shareable URL (e.g. /go/raanman-uploadme) into a
1-year auth cookie without ever putting the dashboard_token in the URL. It is a
deliberate shared-secret bypass for the public dashboard, so the security
properties matter: featureless 404 on any mismatch or when unconfigured (never
401/403, so a path scanner can't tell a wrong slug from a disabled feature),
constant-time slug compare, and the standard HttpOnly/Secure/SameSite cookie.

The route reads config via dashboard.auth._get_config at call time, so tests
patch dashboard.auth._get_config (matching the convention documented at the top
of dashboard/auth.py).
"""
from unittest.mock import patch

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    with app.test_client() as c:
        yield c


def _cookie_header(response):
    cookies = response.headers.getlist("Set-Cookie")
    matches = [c for c in cookies if "pf_dashboard_token" in c]
    return matches[0] if matches else None


def test_correct_slug_sets_cookie_and_redirects(client):
    cfg = {"dashboard_share_slug": "raanman-uploadme", "dashboard_token": "secret_tok_123"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/raanman-uploadme", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.location.endswith("/")
    cookie = _cookie_header(resp)
    assert cookie is not None
    assert "HttpOnly" in cookie
    assert "Secure" in cookie
    assert "SameSite=Lax" in cookie


def test_cookie_value_is_dashboard_token_with_year_expiry(client):
    cfg = {"dashboard_share_slug": "raanman-uploadme", "dashboard_token": "secret_tok_123"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/raanman-uploadme", follow_redirects=False)
    cookie = _cookie_header(resp)
    # The cookie must carry the real token so subsequent require_auth checks pass.
    assert "pf_dashboard_token=secret_tok_123" in cookie
    assert "Max-Age=31536000" in cookie


def test_wrong_slug_returns_featureless_404(client):
    cfg = {"dashboard_share_slug": "raanman-uploadme", "dashboard_token": "secret_tok_123"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/not-the-slug", follow_redirects=False)
    assert resp.status_code == 404
    assert _cookie_header(resp) is None
    # Body must not echo the slug or reveal feature state.
    assert "raanman-uploadme" not in resp.get_data(as_text=True)


def test_unset_slug_is_inert_404(client):
    cfg = {"dashboard_token": "secret_tok_123"}  # no dashboard_share_slug
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/anything", follow_redirects=False)
    assert resp.status_code == 404
    assert _cookie_header(resp) is None


def test_empty_string_slug_is_inert_404(client):
    cfg = {"dashboard_share_slug": "", "dashboard_token": "secret_tok_123"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        # An attacker passing the literal disabled-sentinel must not get in either.
        resp = client.get("/go/ ", follow_redirects=False)
    assert resp.status_code == 404


def test_slug_set_but_token_missing_returns_503(client):
    cfg = {"dashboard_share_slug": "raanman-uploadme"}  # no dashboard_token
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/raanman-uploadme", follow_redirects=False)
    assert resp.status_code == 503
    assert _cookie_header(resp) is None


def test_config_slug_is_trimmed(client):
    cfg = {"dashboard_share_slug": "  raanman-uploadme  ", "dashboard_token": "tok"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/raanman-uploadme", follow_redirects=False)
    assert resp.status_code == 302


def test_slug_comparison_is_case_sensitive(client):
    cfg = {"dashboard_share_slug": "Raanman-UploadMe", "dashboard_token": "tok"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/raanman-uploadme", follow_redirects=False)
    assert resp.status_code == 404


def test_unicode_slug_does_not_crash(client):
    # hmac.compare_digest raises TypeError on non-ASCII str; the route must
    # encode to bytes first so a hostile unicode slug yields 404, not 500.
    cfg = {"dashboard_share_slug": "raanman-uploadme", "dashboard_token": "tok"}
    with patch("dashboard.auth._get_config", return_value=cfg):
        resp = client.get("/go/råanman", follow_redirects=False)
    assert resp.status_code == 404
