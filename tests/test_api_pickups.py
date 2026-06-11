"""Tests for the `/api/pickups` dashboard endpoint."""

from __future__ import annotations

import json


def _client(monkeypatch, tmp_path, pickups_payload):
    """Spin up the dashboard Flask app pointed at tmp_path/data."""
    import dashboard.app as dashboard_app

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pending_pickups.json").write_text(
        json.dumps(pickups_payload), encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_app, "DATA_DIR", data_dir)
    monkeypatch.setattr(dashboard_app.app, "testing", True, raising=False)

    # Bypass auth -- the @require_auth decorator reads token; tests run
    # without one so monkey-patch the decorator-target function.
    def _bypass(f):
        return f

    # 2026-06-11 (suite-cleanup): require_auth was extracted into
    # dashboard.auth and resolves config via dashboard.auth._get_config /
    # _get_dashboard_token — NOT dashboard.app._get_config (see the import
    # block + comment at dashboard/app.py:808-820). Patching dashboard.app
    # alone left auth reading the real config → 401. Patch BOTH: app's for
    # the endpoint body, auth's so ?token=secret is accepted.
    cfg = {"dashboard_token": "secret"}
    import dashboard.auth as dashboard_auth
    monkeypatch.setattr(dashboard_app, "_get_config", lambda: cfg)
    monkeypatch.setattr(dashboard_auth, "_get_config", lambda: cfg)
    client = dashboard_app.app.test_client()
    return client


def test_pickups_endpoint_returns_sorted_list(monkeypatch, tmp_path):
    payload = {
        "pickups": [
            {"id": "PAST", "title": "past", "due_ts": "1999-01-01T00:00:00+00:00",
             "handler": "x", "status": "completed",
             "history": [{"verdict": "promote"}], "last_run_ts": "1999-01-02"},
            {"id": "FUTURE", "title": "future", "due_ts": "2099-01-01T00:00:00+00:00",
             "handler": "x", "status": "pending"},
            {"id": "TODAY", "title": "today", "due_ts": "2026-05-19T08:00:00+00:00",
             "handler": "x", "status": "pending"},
        ]
    }
    client = _client(monkeypatch, tmp_path, payload)
    resp = client.get("/api/pickups?token=secret")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "pickups" in body
    rows = body["pickups"]
    assert len(rows) == 3
    # Sorted ascending by days_until_due. PAST is most-overdue (negative),
    # then TODAY, then FUTURE.
    ids = [r["id"] for r in rows]
    assert ids == ["PAST", "TODAY", "FUTURE"]
    # last_verdict surfaced from history
    past = next(r for r in rows if r["id"] == "PAST")
    assert past["last_verdict"] == "promote"


def test_pickups_endpoint_handles_missing_file(monkeypatch, tmp_path):
    import dashboard.app as dashboard_app
    import dashboard.auth as dashboard_auth

    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir()
    # 2026-06-11 (suite-cleanup): auth now resolves via dashboard.auth — patch
    # it too or ?token=secret 401s (see test_api_pickups._client comment).
    monkeypatch.setattr(dashboard_app, "_get_config", lambda: {"dashboard_token": "secret"})
    monkeypatch.setattr(dashboard_auth, "_get_config", lambda: {"dashboard_token": "secret"})
    client = dashboard_app.app.test_client()
    resp = client.get("/api/pickups?token=secret")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["pickups"] == []


def test_pickups_endpoint_handles_malformed_file(monkeypatch, tmp_path):
    import dashboard.app as dashboard_app
    import dashboard.auth as dashboard_auth

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "pending_pickups.json").write_text("not json", encoding="utf-8")
    monkeypatch.setattr(dashboard_app, "DATA_DIR", data_dir)
    # 2026-06-11 (suite-cleanup): auth now resolves via dashboard.auth — patch
    # it too or ?token=secret 401s (see test_api_pickups._client comment).
    monkeypatch.setattr(dashboard_app, "_get_config", lambda: {"dashboard_token": "secret"})
    monkeypatch.setattr(dashboard_auth, "_get_config", lambda: {"dashboard_token": "secret"})
    client = dashboard_app.app.test_client()
    resp = client.get("/api/pickups?token=secret")
    # Returns 200 with empty list (graceful) OR 500 (logged). Both are
    # acceptable -- the contract is "dashboard never crashes from a
    # corrupted JSON file in data/".
    assert resp.status_code in (200, 500)
