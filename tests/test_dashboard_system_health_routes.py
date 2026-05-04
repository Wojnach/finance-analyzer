"""Route-layer tests for /api/system_status and /api/trading_status.

Aggregator logic is covered by test_dashboard_system_status.py and
test_dashboard_trading_status.py. These tests verify the Flask wiring:
auth gating, 30s TTL cache, ?force=1 bypass, JSON envelope.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Reset both per-process TTL caches each test.
    from dashboard.app import _SYSTEM_STATUS_CACHE, _TRADING_STATUS_CACHE
    _SYSTEM_STATUS_CACHE.update({"at": 0.0, "value": None})
    _TRADING_STATUS_CACHE.update({"at": 0.0, "value": None})
    with app.test_client() as c:
        yield c


def _no_auth():
    return patch("dashboard.auth._get_dashboard_token", return_value=None)


# ---------------------------------------------------------------------------
# /api/system_status
# ---------------------------------------------------------------------------


class TestSystemStatusRoute:
    def test_returns_payload(self, client):
        stub = {
            "ts": "2026-05-04T13:00:00+00:00",
            "overall": "GREEN",
            "reasons": ["all systems nominal"],
            "heartbeat": {"age_seconds": 30, "last_ts": "x", "cycle_count": 1, "error_count": 0},
            "errors": {"unresolved": 0, "recent": []},
            "contract_violations": {"unresolved": 0, "recent": []},
            "llm_inference": {"models": [], "overall_pct": None},
            "layer2": {"triggers_24h": 0, "success_24h": 0, "success_pct": None,
                       "latest": None, "spark_24h": [0] * 24},
            "signal_aggregate": {"tickers": []},
            "pnl_footer": {},
        }
        with patch("dashboard.system_status.compute", return_value=stub), _no_auth():
            resp = client.get("/api/system_status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["overall"] == "GREEN"
        assert "all systems nominal" in data["reasons"]

    def test_cache_reuse_within_ttl(self, client):
        call_count = {"n": 0}

        def _stub(*_a, **_kw):
            call_count["n"] += 1
            return {
                "ts": "x", "overall": "GREEN", "reasons": [],
                "heartbeat": {"age_seconds": call_count["n"]},
                "errors": {"unresolved": 0, "recent": []},
                "contract_violations": {"unresolved": 0, "recent": []},
                "llm_inference": {"models": [], "overall_pct": None},
                "layer2": {"triggers_24h": 0, "success_24h": 0, "success_pct": None,
                           "latest": None, "spark_24h": [0] * 24},
                "signal_aggregate": {"tickers": []},
                "pnl_footer": {},
            }

        with patch("dashboard.system_status.compute", side_effect=_stub), _no_auth():
            r1 = client.get("/api/system_status").get_json()
            r2 = client.get("/api/system_status").get_json()
        assert call_count["n"] == 1
        assert r1["heartbeat"]["age_seconds"] == 1
        assert r2["heartbeat"]["age_seconds"] == 1

    def test_force_bypasses_cache(self, client):
        call_count = {"n": 0}

        def _stub(*_a, **_kw):
            call_count["n"] += 1
            return {"ts": "x", "overall": "GREEN", "reasons": [],
                    "heartbeat": {"age_seconds": call_count["n"]},
                    "errors": {"unresolved": 0, "recent": []},
                    "contract_violations": {"unresolved": 0, "recent": []},
                    "llm_inference": {"models": [], "overall_pct": None},
                    "layer2": {"triggers_24h": 0, "success_24h": 0, "success_pct": None,
                               "latest": None, "spark_24h": [0] * 24},
                    "signal_aggregate": {"tickers": []},
                    "pnl_footer": {}}

        with patch("dashboard.system_status.compute", side_effect=_stub), _no_auth():
            r1 = client.get("/api/system_status").get_json()
            r2 = client.get("/api/system_status?force=1").get_json()
        assert call_count["n"] == 2
        assert r1["heartbeat"]["age_seconds"] == 1
        assert r2["heartbeat"]["age_seconds"] == 2

    def test_requires_auth(self, client):
        with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/system_status")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /api/trading_status
# ---------------------------------------------------------------------------


class TestTradingStatusRoute:
    def test_returns_payload(self, client):
        stub = {
            "ts": "2026-05-04T13:00:00+00:00",
            "session_open": True,
            "bots": [
                {"bot": "golddigger", "label": "GoldDigger",
                 "state": "SCANNING", "reason": "in session, no entry signal yet",
                 "position": None, "stats": {}},
                {"bot": "elongir", "label": "Elongir",
                 "state": "OUTSIDE_HOURS", "reason": "next 15:30 CET in 2h 14m",
                 "position": None, "stats": {}},
                {"bot": "metals", "label": "Metals swing",
                 "state": "TRADING", "reason": "holding 1 position(s)",
                 "position": [{"ticker": "X"}], "stats": {}},
                {"bot": "fishing", "label": "Fishing engine",
                 "state": "COOLDOWN", "reason": "12 losses, 224s remaining",
                 "position": None, "stats": {}},
            ],
        }
        with patch("dashboard.trading_status.compute", return_value=stub), _no_auth():
            resp = client.get("/api/trading_status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert {b["bot"] for b in data["bots"]} == {"golddigger", "elongir", "metals", "fishing"}
        assert data["session_open"] is True

    def test_force_bypasses_cache(self, client):
        call_count = {"n": 0}

        def _stub(*_a, **_kw):
            call_count["n"] += 1
            return {"ts": "x", "session_open": True,
                    "bots": [{"bot": "golddigger", "label": "GoldDigger",
                              "state": "SCANNING", "reason": f"call{call_count['n']}",
                              "position": None, "stats": {}}]}

        with patch("dashboard.trading_status.compute", side_effect=_stub), _no_auth():
            r1 = client.get("/api/trading_status").get_json()
            r2 = client.get("/api/trading_status?force=1").get_json()
        assert call_count["n"] == 2
        assert r1["bots"][0]["reason"] == "call1"
        assert r2["bots"][0]["reason"] == "call2"

    def test_requires_auth(self, client):
        with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/trading_status")
        assert resp.status_code == 401
