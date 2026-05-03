"""Smoke tests for static assets served by the new mobile dashboard.

Each batch adds files; each test only asserts that what is currently
shipped is reachable + non-empty + has plausible content type. The
intent is to catch missing-file / wrong-path mistakes during the
mobile redesign rollout.
"""

from unittest.mock import patch

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _no_auth():
    return patch("dashboard.auth._get_dashboard_token", return_value=None)


CSS_FILES = [
    "/static/css/tokens.css",
    "/static/css/base.css",
    "/static/css/layout.css",
    "/static/css/components.css",
    "/static/css/responsive.css",
]

JS_MODULES = [
    "/static/js/main.js",
    "/static/js/state.js",
    "/static/js/fetch.js",
    "/static/js/format.js",
    "/static/js/theme.js",
    "/static/js/router.js",
    "/static/js/polling.js",
]

PWA_FILES = [
    ("/static/manifest.webmanifest", "manifest"),
    ("/static/sw.js",                "javascript"),
    ("/static/icons/icon-192.png",   "image"),
    ("/static/icons/icon-512.png",   "image"),
    ("/static/icons/icon-512-maskable.png", "image"),
    ("/static/icons/apple-touch-icon-180.png", "image"),
]


@pytest.mark.parametrize("path", CSS_FILES)
def test_css_file_served(client, path):
    """Every CSS file referenced by index.html must resolve."""
    # Static assets in Flask are served without going through require_auth,
    # but it is harmless to include the no-auth fixture for consistency.
    with _no_auth():
        resp = client.get(path)
    assert resp.status_code == 200, f"{path} returned {resp.status_code}"
    assert len(resp.data) > 0, f"{path} is empty"
    # Flask uses mimetypes.guess_type which returns text/css for .css.
    assert "css" in resp.headers.get("Content-Type", "").lower()


@pytest.mark.parametrize("path", JS_MODULES)
def test_js_module_served(client, path):
    """Every JS module referenced via static imports must resolve."""
    with _no_auth():
        resp = client.get(path)
    assert resp.status_code == 200, f"{path} returned {resp.status_code}"
    assert len(resp.data) > 0, f"{path} is empty"
    ct = resp.headers.get("Content-Type", "").lower()
    assert "javascript" in ct or "ecmascript" in ct, (
        f"{path} content-type {ct!r} doesn't look like JS"
    )


def test_index_loads_module_entry(client):
    """The new index.html must load js/main.js as a module."""
    with _no_auth():
        resp = client.get("/")
    assert resp.status_code == 200
    body = resp.data
    assert b"/static/js/main.js" in body
    assert b'type="module"' in body
    assert b"viewport-fit=cover" in body


def test_index_links_all_css(client):
    """The new index.html must <link> every CSS file we ship."""
    with _no_auth():
        resp = client.get("/")
    body = resp.data
    for path in CSS_FILES:
        assert path.encode() in body, f"index.html missing link to {path}"


@pytest.mark.parametrize("path,kind_hint", PWA_FILES)
def test_pwa_file_served(client, path, kind_hint):
    """Manifest, service worker, and icons must resolve."""
    with _no_auth():
        resp = client.get(path)
    assert resp.status_code == 200, f"{path} returned {resp.status_code}"
    assert len(resp.data) > 0, f"{path} is empty"
    ct = resp.headers.get("Content-Type", "").lower()
    # We only require the broad type match — Flask sets concrete subtypes.
    if kind_hint == "manifest":
        # Some Flask versions return application/json for .webmanifest;
        # accept either as long as it's json-ish or manifest.
        assert "manifest" in ct or "json" in ct, f"{path} ct {ct!r}"
    elif kind_hint == "javascript":
        assert "javascript" in ct or "ecmascript" in ct, f"{path} ct {ct!r}"
    elif kind_hint == "image":
        assert "image" in ct, f"{path} ct {ct!r}"
