"""The /legacy fallback route was removed 2026-07-18 (rollout window over,
user confirmed the new SPA on phone). These tests pin the removal: the route
must 404 and the 3,211-line monolith must stay deleted.
"""

from pathlib import Path
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


class TestLegacyRemoved:
    def test_legacy_route_is_gone(self, client):
        with _no_auth():
            resp = client.get("/legacy")
        assert resp.status_code == 404

    def test_monolith_file_deleted(self):
        assert not Path("dashboard/static/index_legacy.html").exists()

    def test_spa_root_still_serves(self, client):
        with _no_auth():
            resp = client.get("/")
        assert resp.status_code == 200
        assert b"Portfolio Intelligence" in resp.data or b"html" in resp.data.lower()
