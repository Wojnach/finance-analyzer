"""Tests for the /legacy fallback route added during the mobile-first redesign.

The legacy route serves dashboard/static/index_legacy.html (a snapshot of the
pre-redesign 3,211-line single-file dashboard) so the user has a working
fallback during the mobile rollout window. See docs/PLAN.md.
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


class TestLegacyRoute:
    def test_legacy_returns_html(self, client):
        with _no_auth():
            resp = client.get("/legacy")
        assert resp.status_code == 200
        assert b"html" in resp.data.lower()

    def test_legacy_serves_legacy_file(self, client):
        """The /legacy route must serve index_legacy.html, NOT index.html."""
        with _no_auth():
            resp_new = client.get("/")
            resp_legacy = client.get("/legacy")
        assert resp_new.status_code == 200
        assert resp_legacy.status_code == 200
        # The legacy file is the old 3,200-line single-file dashboard. It is
        # at least an order of magnitude larger than the new ~70-line skeleton.
        assert len(resp_legacy.data) > len(resp_new.data) * 5

    def test_legacy_strips_token_query(self, client):
        """?token=XXX on /legacy must redirect to bare /legacy (clean URL)."""
        with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
            resp = client.get("/legacy?token=secret", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith("/legacy")

    def test_legacy_rejects_invalid_token(self, client):
        with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
            resp = client.get("/legacy?token=wrong")
        assert resp.status_code == 401

    def test_new_index_links_to_legacy(self, client):
        """The new dashboard skeleton must mention the /legacy fallback so a
        broken JS bootstrap doesn't strand the user."""
        with _no_auth():
            resp = client.get("/")
        assert resp.status_code == 200
        assert b"/legacy" in resp.data
