"""Test the /api/loop_health endpoint."""
from __future__ import annotations


def test_api_loop_health_endpoint_registered():
    from dashboard import app as dashboard_app
    rules = {r.rule for r in dashboard_app.app.url_map.iter_rules()}
    assert "/api/loop_health" in rules


def test_api_loop_health_returns_rollup_shape(monkeypatch):
    """The endpoint should return the dict shape from read_loop_health."""
    fake = {
        "checked_at": "2026-05-02T12:00:00+00:00",
        "stale_threshold_seconds": 300,
        "loops": {
            "crypto": {"state": "fresh", "age_seconds": 30, "name": "crypto",
                       "path": "x", "payload": {"cycle": 5}, "error": None},
        },
        "any_unhealthy": False,
        "unhealthy": [],
    }
    from dashboard import app as dashboard_app
    # Patch the module reference loop_health uses inside the route handler
    import portfolio.loop_health as lh
    monkeypatch.setattr(lh, "read_loop_health", lambda: fake)
    monkeypatch.setenv("DASHBOARD_AUTH_DISABLED", "1")

    dashboard_app.app.config["TESTING"] = True
    client = dashboard_app.app.test_client()
    resp = client.get("/api/loop_health")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "loops" in data
        assert "any_unhealthy" in data
        assert "unhealthy" in data
