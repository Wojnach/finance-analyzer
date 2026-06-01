"""Tests for the /hh and /hhmap short vanity aliases (dashboard/app.py).

Both are top-level redirects into the /house blueprint, gated by require_auth
like /legacy (NOT bare like /logout): a first-visit /hh?token=XXX must reach
require_auth so the token is converted to the pf_dashboard_token cookie BEFORE
the redirect — otherwise the bare redirect drops the query string and the
token-less target 401s on a fresh device. Targets:
  /hh    -> /house/         (the hub: apartment table + heatmap)
  /hhmap -> /house/heatmap  (the innerstad heatmap directly)
"""

from unittest.mock import patch

import pytest

from dashboard.app import COOKIE_NAME, app


_TOKEN = "test-token-for-hh-aliases"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _auth_off():
    # No dashboard token configured → require_auth allows the request through.
    return patch("dashboard.auth._get_dashboard_token", return_value=None)


def test_hh_redirects_to_house_hub(client):
    with _auth_off():
        resp = client.get("/hh", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/house/")


def test_hhmap_redirects_to_heatmap(client):
    with _auth_off():
        resp = client.get("/hhmap", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/house/heatmap")


def test_aliases_require_auth(client):
    """A token is configured but none supplied → 401, same gate as /house/*."""
    with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
        assert client.get("/hh").status_code == 401
        assert client.get("/hhmap").status_code == 401


def test_hh_token_bootstrap_sets_cookie_and_drops_token(client):
    """/hh?token=VALID on a fresh device: require_auth converts the token to a
    cookie, then we redirect to a clean, token-free /house/."""
    with patch("dashboard.auth._get_dashboard_token", return_value=_TOKEN):
        resp = client.get(f"/hh?token={_TOKEN}", follow_redirects=False)
    assert resp.status_code == 302
    loc = resp.headers["Location"]
    assert loc.endswith("/house/")
    assert _TOKEN not in loc
    # A cookie is issued so the followed redirect lands authenticated.
    set_cookie = " ".join(resp.headers.getlist("Set-Cookie"))
    assert COOKIE_NAME in set_cookie


def test_aliases_registered_in_url_map():
    """Guard against the aliases silently disappearing in a future refactor."""
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/hh" in rules
    assert "/hhmap" in rules
