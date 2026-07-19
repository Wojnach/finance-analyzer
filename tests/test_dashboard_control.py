"""Tests for dashboard/control.py — the Command Central write API.

Covers the hardening mandated in the Phase 3 plan (2026-07-18): auth
gating, loop-unit allowlisting, the shared rate limiter, and the audit
log. systemctl is ALWAYS mocked here — these tests must never touch the
real systemd user session.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import dashboard.control as control
from dashboard.app import COOKIE_NAME, app

_TOKEN = "test-token-for-control"
_AUTH_HEADERS = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The rate limiter's event deque is shared module state — without a
    reset, whichever test runs first eats budget for every test after it
    in the same process."""
    control._rate_events.clear()
    yield
    control._rate_events.clear()


@pytest.fixture
def client(tmp_path: Path):
    app.config["TESTING"] = True
    with (
        patch("dashboard.auth._get_dashboard_token", return_value=_TOKEN),
        patch.object(control, "DISABLE_FLAG", tmp_path / "local_llm.disabled"),
        patch.object(
            control, "INSTRUMENTS_PATH", tmp_path / "control" / "instruments.json"
        ),
        patch.object(control, "AUDIT_PATH", tmp_path / "control" / "audit.jsonl"),
    ):
        with app.test_client() as c:
            yield c


def _no_auth():
    return patch("dashboard.auth._get_dashboard_token", return_value=None)


# ---------------------------------------------------------------------------
# Auth gating
# ---------------------------------------------------------------------------


class TestAuthRequired:
    def test_llm_requires_auth(self, client):
        resp = client.post("/api/control/llm", json={"enabled": False})
        assert resp.status_code == 401

    def test_instrument_requires_auth(self, client):
        resp = client.post(
            "/api/control/instrument", json={"ticker": "BTC-USD", "tracked": True}
        )
        assert resp.status_code == 401

    def test_loop_requires_auth(self, client):
        resp = client.post(
            "/api/control/loop", json={"unit": "pf-dataloop", "action": "start"}
        )
        assert resp.status_code == 401

    def test_state_requires_auth(self, client):
        resp = client.get("/api/control/state")
        assert resp.status_code == 401

    def test_registry_requires_auth(self, client):
        resp = client.get("/api/control/registry")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# LLM toggle
# ---------------------------------------------------------------------------


class TestLlmToggle:
    def test_disable_creates_flag(self, client):
        resp = client.post(
            "/api/control/llm", json={"enabled": False}, headers=_AUTH_HEADERS
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"llm_enabled": False}
        assert control.DISABLE_FLAG.exists()

    def test_enable_removes_flag(self, client):
        control.DISABLE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        control.DISABLE_FLAG.touch()
        resp = client.post(
            "/api/control/llm", json={"enabled": True}, headers=_AUTH_HEADERS
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"llm_enabled": True}
        assert not control.DISABLE_FLAG.exists()

    def test_non_bool_enabled_is_400(self, client):
        resp = client.post(
            "/api/control/llm", json={"enabled": "yes"}, headers=_AUTH_HEADERS
        )
        assert resp.status_code == 400

    def test_missing_body_is_400(self, client):
        resp = client.post("/api/control/llm", json={}, headers=_AUTH_HEADERS)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Instrument toggle
# ---------------------------------------------------------------------------


class TestInstrumentToggle:
    def test_unknown_ticker_is_400(self, client):
        resp = client.post(
            "/api/control/instrument",
            json={"ticker": "NOPE-USD", "tracked": True},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_valid_ticker_writes_file(self, client):
        resp = client.post(
            "/api/control/instrument",
            json={"ticker": "BTC-USD", "tracked": False},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"ticker": "BTC-USD", "tracked": False}
        on_disk = json.loads(control.INSTRUMENTS_PATH.read_text())
        assert on_disk == {"BTC-USD": {"tracked": False}}

    def test_non_bool_tracked_is_400(self, client):
        resp = client.post(
            "/api/control/instrument",
            json={"ticker": "BTC-USD", "tracked": "nope"},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Loop control — allowlist + mocked systemctl
# ---------------------------------------------------------------------------


class TestLoopControl:
    def test_dashboard_unit_rejected(self, client):
        """pf-dashboard must never be controllable from its own route."""
        resp = client.post(
            "/api/control/loop",
            json={"unit": "pf-dashboard", "action": "stop"},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_non_allowlisted_unit_rejected(self, client):
        resp = client.post(
            "/api/control/loop",
            json={"unit": "sshd", "action": "stop"},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_invalid_action_rejected(self, client):
        resp = client.post(
            "/api/control/loop",
            json={"unit": "pf-dataloop", "action": "reboot"},
            headers=_AUTH_HEADERS,
        )
        assert resp.status_code == 400

    def test_allowlisted_action_calls_systemctl(self, client):
        fake = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        with patch.object(control.subprocess, "run", return_value=fake) as mock_run:
            resp = client.post(
                "/api/control/loop",
                json={"unit": "pf-dataloop", "action": "restart"},
                headers=_AUTH_HEADERS,
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["unit"] == "pf-dataloop"
        mock_run.assert_called_once_with(
            ["systemctl", "--user", "restart", "pf-dataloop"],
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_systemctl_failure_surfaces_502(self, client):
        fake = type(
            "R", (), {"returncode": 1, "stdout": "", "stderr": "unit not found"}
        )()
        with patch.object(control.subprocess, "run", return_value=fake):
            resp = client.post(
                "/api/control/loop",
                json={"unit": "pf-cryptoloop", "action": "start"},
                headers=_AUTH_HEADERS,
            )
        assert resp.status_code == 502
        assert resp.get_json()["ok"] is False


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimit:
    def test_seventh_post_in_a_minute_is_429(self, client):
        for i in range(control._RATE_LIMIT_MAX):
            resp = client.post(
                "/api/control/llm", json={"enabled": i % 2 == 0}, headers=_AUTH_HEADERS
            )
            assert resp.status_code == 200
        resp = client.post(
            "/api/control/llm", json={"enabled": True}, headers=_AUTH_HEADERS
        )
        assert resp.status_code == 429

    def test_state_get_does_not_consume_budget(self, client):
        with patch.object(control, "_systemctl_query", return_value=None):
            for _ in range(control._RATE_LIMIT_MAX + 2):
                resp = client.get("/api/control/state", headers=_AUTH_HEADERS)
                assert resp.status_code == 200
        # The write budget must still be fully available afterwards.
        assert control._rate_limit_remaining() == control._RATE_LIMIT_MAX

    def test_registry_get_does_not_consume_budget(self, client):
        for _ in range(control._RATE_LIMIT_MAX + 2):
            resp = client.get("/api/control/registry", headers=_AUTH_HEADERS)
            assert resp.status_code == 200
        assert control._rate_limit_remaining() == control._RATE_LIMIT_MAX


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class TestAuditLog:
    def test_action_appends_audit_line(self, client):
        client.post("/api/control/llm", json={"enabled": False}, headers=_AUTH_HEADERS)
        lines = control.AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["endpoint"] == "llm"
        assert entry["payload"] == {"enabled": False}
        assert entry["result"] == {"llm_enabled": False}
        assert "ts" in entry
        assert "remote_addr" in entry

    def test_rejected_action_is_still_audited(self, client):
        client.post(
            "/api/control/instrument",
            json={"ticker": "NOPE-USD", "tracked": True},
            headers=_AUTH_HEADERS,
        )
        lines = control.AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["endpoint"] == "instrument"
        assert "unknown ticker" in entry["result"]

    def test_cf_access_path_attributes_actor_and_auth_method(self, client):
        """Finding #12 (2026-07-19): remote_addr is always 127.0.0.1 behind
        cloudflared — the only real identity is the CF-verified email,
        stashed into flask.g by auth.py's require_auth and picked up here."""
        with patch(
            "dashboard.cf_access.verify_cf_jwt",
            return_value={"email": "sydney@hazelight.se"},
        ):
            client.post(
                "/api/control/llm",
                json={"enabled": False},
                headers={
                    "Cf-Access-Authenticated-User-Email": "sydney@hazelight.se",
                    "Cf-Access-Jwt-Assertion": "fake-jwt",
                },
            )
        lines = control.AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["actor"] == "sydney@hazelight.se"
        assert entry["auth_method"] == "cf_access"

    def test_bearer_path_has_no_actor(self, client):
        client.post("/api/control/llm", json={"enabled": False}, headers=_AUTH_HEADERS)
        lines = control.AUDIT_PATH.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["actor"] is None
        assert entry["auth_method"] == "bearer"


# ---------------------------------------------------------------------------
# CSRF backstop (2026-07-19 NOTE) — cookie auth only; bearer/query/cf_access
# already require possessing the secret, so a forged cross-site request
# can't reach them without it.
# ---------------------------------------------------------------------------


class TestCsrfBackstop:
    def test_cookie_auth_without_origin_or_referer_is_rejected(self, client):
        client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
        resp = client.post("/api/control/llm", json={"enabled": False})
        assert resp.status_code == 403

    def test_cookie_auth_with_matching_origin_is_allowed(self, client):
        client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
        resp = client.post(
            "/api/control/llm",
            json={"enabled": False},
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 200

    def test_cookie_auth_with_mismatched_origin_is_rejected(self, client):
        client.set_cookie(COOKIE_NAME, _TOKEN, domain="localhost")
        resp = client.post(
            "/api/control/llm",
            json={"enabled": False},
            headers={"Origin": "http://evil.example.com"},
        )
        assert resp.status_code == 403

    def test_bearer_auth_is_exempt_from_the_origin_check(self, client):
        resp = client.post(
            "/api/control/llm", json={"enabled": False}, headers=_AUTH_HEADERS
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------


class TestStateEndpoint:
    def test_shape(self, client):
        def _fake_query(unit, verb):
            return "active" if verb == "is-active" else "enabled"

        with patch.object(control, "_systemctl_query", side_effect=_fake_query):
            resp = client.get("/api/control/state", headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.get_json()
        assert set(data.keys()) == {
            "llm_enabled",
            "instruments",
            "loops",
            "rate_limit_remaining",
        }
        assert set(data["instruments"].keys()) == control.ALL_TICKERS
        assert set(data["loops"].keys()) == control.LOOP_ALLOWLIST
        for row in data["instruments"].values():
            assert row == {"tracked": True}  # default when no file written yet
        for row in data["loops"].values():
            assert row == {"active": True, "enabled": True, "state": "active"}

    def test_systemctl_outage_reports_unknown_not_stopped(self, client):
        """Finding #10 (2026-07-19): a systemctl query failure (binary
        missing, timeout, --user session unreachable) must surface as
        unknown, not be conflated with a confirmed-stopped unit."""
        with patch.object(control, "_systemctl_query", return_value=None):
            resp = client.get("/api/control/state", headers=_AUTH_HEADERS)
        data = resp.get_json()
        for row in data["loops"].values():
            assert row == {"active": None, "enabled": None, "state": "unknown"}

    def test_reflects_written_instrument_toggle(self, client):
        client.post(
            "/api/control/instrument",
            json={"ticker": "MSTR", "tracked": False},
            headers=_AUTH_HEADERS,
        )
        with patch.object(control, "_systemctl_query", return_value=None):
            resp = client.get("/api/control/state", headers=_AUTH_HEADERS)
        data = resp.get_json()
        assert data["instruments"]["MSTR"] == {"tracked": False}


# ---------------------------------------------------------------------------
# /registry (Phase 6, 2026-07-18) — component_registry snapshot
# ---------------------------------------------------------------------------


class TestRegistryEndpoint:
    def test_no_ticker_returns_full_snapshot(self, client):
        resp = client.get("/api/control/registry", headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ticker"] is None
        assert "XAG-USD" in data["registry"]
        # Full-dump shape is the raw snapshot() output: signal -> per-signal
        # dict, not the ticker-filtered applicable/disabled/signals shape.
        xag = data["registry"]["XAG-USD"]
        assert xag  # at least one signal
        sample = next(iter(xag.values()))
        assert "enabled_default" in sample
        assert "disabled_reason" in sample

    def test_ticker_filter_shape(self, client):
        resp = client.get("/api/control/registry?ticker=XAG-USD", headers=_AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ticker"] == "XAG-USD"
        reg = data["registry"]
        assert set(reg.keys()) == {"applicable", "disabled", "signals"}
        assert isinstance(reg["applicable"], list)
        assert isinstance(reg["disabled"], list)
        # Every applicable name must be a known signal, marked enabled there.
        for name in reg["applicable"]:
            assert reg["signals"][name]["enabled_default"] is True
        # Every disabled entry carries a non-empty reason and a known signal.
        for entry in reg["disabled"]:
            assert set(entry.keys()) == {"signal", "reason"}
            assert entry["reason"]
            assert reg["signals"][entry["signal"]]["enabled_default"] is False
        # applicable + disabled partitions the full per-signal dict exactly.
        assert len(reg["applicable"]) + len(reg["disabled"]) == len(reg["signals"])

    def test_unknown_ticker_is_400(self, client):
        resp = client.get(
            "/api/control/registry?ticker=NOPE-USD", headers=_AUTH_HEADERS
        )
        assert resp.status_code == 400
