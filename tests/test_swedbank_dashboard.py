"""/api/swedbank route tests. Synthetic snapshot only — this repo is public."""

import json

import pytest

pytest.importorskip("flask")


@pytest.fixture
def client(tmp_path, monkeypatch):
    import dashboard.app as dash

    monkeypatch.setattr(dash, "DATA_DIR", tmp_path, raising=False)
    dash.app.config["TESTING"] = True
    return dash.app.test_client(), tmp_path


def _auth():
    from dashboard.auth import _get_dashboard_token

    return {"Authorization": f"Bearer {_get_dashboard_token()}"}


SNAP = {
    "base_currency": "SEK",
    "as_of": "2026-01-01T00:00:00+00:00",
    "fx": {"USDSEK": 9.5},
    "accounts": {
        "A": {
            "cash": 100.0,
            "holdings": [
                {
                    "key": "NVDA",
                    "name": "NVIDIA",
                    "qty": 2,
                    "mark": 100.0,
                    "mark_basis": "last",
                    "currency": "USD",
                    "spread_pct": 0.02,
                    "age_s": 1.0,
                    "source": "avanza",
                    "degraded": False,
                    "stale_last": False,
                    "value": 1900.0,
                    "cost_basis": 1000.0,
                    "pnl": 900.0,
                    "pnl_pct": 90.0,
                    "avanza_ob": "4478",
                }
            ],
            "holdings_value": 1900.0,
            "total_value": 2000.0,
            "cost_basis": 1000.0,
            "pnl": 900.0,
            "pnl_pct": 90.0,
        }
    },
    "total": {
        "holdings_value": 1900.0,
        "cash": 100.0,
        "total_value": 2000.0,
        "cost_basis": 1000.0,
        "pnl": 900.0,
        "pnl_pct": 90.0,
    },
    "consolidated": [],
    "unpriced": [],
    "degraded": [],
    "stale_last": [],
    "price_errors": {},
    "sweep_duration_s": 1.2,
}


def test_requires_auth(client):
    c, _ = client
    assert c.get("/api/swedbank").status_code in (401, 403)


def test_missing_snapshot_reports_unavailable_not_error(client):
    c, _ = client
    r = c.get("/api/swedbank", headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body["available"] is False
    assert "swedbank" in body["reason"]


def test_serves_snapshot_with_age(client):
    c, tmp = client
    (tmp / "swedbank_snapshot.json").write_text(json.dumps(SNAP))
    r = c.get("/api/swedbank", headers=_auth())
    assert r.status_code == 200
    body = r.get_json()
    assert body["available"] is True
    assert body["total"]["total_value"] == 2000.0
    # Age must always be present: the UI must never imply freshness it lacks.
    assert body["snapshot_age_s"] is not None
    assert body["snapshot_age_s"] >= 0


def test_loop_status_reported_when_heartbeat_present(client):
    c, tmp = client
    (tmp / "swedbank_snapshot.json").write_text(json.dumps(SNAP))
    (tmp / "swedbank_loop.heartbeat").write_text(
        json.dumps({"status": "ok", "ts": "2026-01-01T00:00:00+00:00"})
    )
    body = c.get("/api/swedbank", headers=_auth()).get_json()
    assert body["loop"]["running"] is True


def test_loop_not_running_when_no_heartbeat(client):
    c, tmp = client
    (tmp / "swedbank_snapshot.json").write_text(json.dumps(SNAP))
    body = c.get("/api/swedbank", headers=_auth()).get_json()
    assert body["loop"]["running"] is False


def test_corrupt_snapshot_degrades(client):
    c, tmp = client
    (tmp / "swedbank_snapshot.json").write_text("{not json")
    r = c.get("/api/swedbank", headers=_auth())
    assert r.status_code in (200, 500)
    if r.status_code == 200:
        assert r.get_json()["available"] is False
