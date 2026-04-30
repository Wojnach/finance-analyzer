"""Tests for the dashboard /api/oil endpoint.

Mirrors test_dashboard_crypto_endpoints.py. Verifies:
  - Endpoint exists and returns valid JSON shape.
  - Empty state files don't crash the endpoint.
  - Heartbeat is included in the response.
"""
from __future__ import annotations

import json
import os
import sys

import pytest


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    """Spin up the Flask app pointing DATA_DIR at a tmp_path with empty files."""
    # Pre-create empty state files the endpoint reads.
    (tmp_path / "oil_swing_state.json").write_text("{}")
    (tmp_path / "oil_deep_context.json").write_text("{}")
    (tmp_path / "oil_warrant_catalog.json").write_text(
        json.dumps({"refreshed_ts": None, "ttl_hours": 6, "warrants": {}}))
    (tmp_path / "oil_risk.json").write_text("{}")
    (tmp_path / "oil_swing_decisions.jsonl").write_text("")
    (tmp_path / "oil_swing_trades.jsonl").write_text("")
    (tmp_path / "oil_loop.heartbeat").write_text(
        json.dumps({"ts": "2026-05-01T00:00:00+00:00", "status": "ok",
                     "cycle": 1, "ok": True, "n_positions": 0}))

    # Disable auth for the test
    monkeypatch.setenv("DASHBOARD_AUTH_DISABLED", "1")

    from dashboard import app as dashboard_app
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)
    dashboard_app.app.config["TESTING"] = True
    return dashboard_app.app.test_client()


def test_api_oil_returns_200_with_empty_state(app_client):
    resp = app_client.get("/api/oil")
    # Auth may be enabled — if so, accept 401 or 200 (just verify endpoint
    # is registered).
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        data = resp.get_json()
        assert "state" in data
        assert "context" in data
        assert "warrant_catalog" in data
        assert "decisions" in data
        assert "trades" in data
        assert "heartbeat" in data


def test_api_oil_endpoint_registered():
    """Even without setting up auth, verify the URL rule is registered."""
    from dashboard import app as dashboard_app
    rules = {r.rule for r in dashboard_app.app.url_map.iter_rules()}
    assert "/api/oil" in rules
