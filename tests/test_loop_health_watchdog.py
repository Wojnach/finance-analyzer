"""Tests for scripts/loop_health_watchdog.py."""
from __future__ import annotations

import datetime
import os
import sys
from unittest.mock import patch

import pytest

# scripts/ is not a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import loop_health_watchdog as lhw


@pytest.fixture
def fixed_now():
    return datetime.datetime(2026, 5, 2, 12, 0, 0, tzinfo=datetime.UTC)


# ---------------------------------------------------------------------------
# Cooldown gate
# ---------------------------------------------------------------------------
class TestCooldownGate:
    def test_no_prior_alert_not_in_cooldown(self, fixed_now):
        state = {"last_alert_per_loop": {}}
        assert lhw._is_in_cooldown("crypto", state, fixed_now) is False

    def test_recent_alert_is_in_cooldown(self, fixed_now):
        recent = (fixed_now - datetime.timedelta(hours=1)).isoformat()
        state = {"last_alert_per_loop": {"crypto": recent}}
        assert lhw._is_in_cooldown("crypto", state, fixed_now) is True

    def test_old_alert_not_in_cooldown(self, fixed_now):
        old = (fixed_now - datetime.timedelta(hours=5)).isoformat()
        state = {"last_alert_per_loop": {"crypto": old}}
        # Default cooldown is 4 hours; 5h ago > 4h => not in cooldown
        assert lhw._is_in_cooldown("crypto", state, fixed_now) is False

    def test_unparseable_ts_treated_as_no_cooldown(self, fixed_now):
        state = {"last_alert_per_loop": {"crypto": "garbage"}}
        assert lhw._is_in_cooldown("crypto", state, fixed_now) is False

    def test_custom_cooldown_hours(self, fixed_now):
        recent = (fixed_now - datetime.timedelta(hours=2)).isoformat()
        state = {"last_alert_per_loop": {"crypto": recent}}
        # 2h ago, cooldown 1h => not in cooldown
        assert lhw._is_in_cooldown(
            "crypto", state, fixed_now, cooldown_hours=1) is False
        # 2h ago, cooldown 6h => in cooldown
        assert lhw._is_in_cooldown(
            "crypto", state, fixed_now, cooldown_hours=6) is True


# ---------------------------------------------------------------------------
# build_alert
# ---------------------------------------------------------------------------
class TestBuildAlert:
    def test_no_alert_when_all_fresh(self, fixed_now):
        rollup = {"any_unhealthy": False, "unhealthy": [], "loops": {}}
        msg, alerted = lhw.build_alert(rollup, {}, fixed_now)
        assert msg is None
        assert alerted == []

    def test_alert_for_stale_loop(self, fixed_now):
        rollup = {
            "any_unhealthy": True,
            "unhealthy": ["oil"],
            "loops": {
                "oil": {"state": "stale", "age_seconds": 3600.0,
                        "name": "oil", "path": "data/oil_loop.heartbeat",
                        "payload": None, "error": None},
            },
        }
        msg, alerted = lhw.build_alert(rollup, {"last_alert_per_loop": {}},
                                         fixed_now)
        assert msg is not None
        assert "oil" in msg
        assert "STALE" in msg
        assert alerted == ["oil"]

    def test_alert_for_missing_loop(self, fixed_now):
        rollup = {
            "any_unhealthy": True,
            "unhealthy": ["crypto"],
            "loops": {
                "crypto": {"state": "missing", "age_seconds": None,
                           "name": "crypto", "path": "data/x.heartbeat",
                           "payload": None, "error": None},
            },
        }
        msg, alerted = lhw.build_alert(rollup, {"last_alert_per_loop": {}},
                                         fixed_now)
        assert msg is not None
        assert "NO HEARTBEAT" in msg
        assert "task likely never started" in msg

    def test_cooldown_suppresses_alert(self, fixed_now):
        recent = (fixed_now - datetime.timedelta(hours=1)).isoformat()
        rollup = {
            "any_unhealthy": True,
            "unhealthy": ["oil"],
            "loops": {
                "oil": {"state": "stale", "age_seconds": 3600.0,
                        "name": "oil", "path": "x", "payload": None,
                        "error": None},
            },
        }
        state = {"last_alert_per_loop": {"oil": recent}}
        msg, alerted = lhw.build_alert(rollup, state, fixed_now)
        assert msg is None
        assert alerted == []

    def test_partial_cooldown_only_alerts_on_eligible(self, fixed_now):
        """One loop in cooldown, one not — alert only on the not-in-cooldown."""
        recent = (fixed_now - datetime.timedelta(hours=1)).isoformat()
        rollup = {
            "any_unhealthy": True,
            "unhealthy": ["crypto", "oil"],
            "loops": {
                "crypto": {"state": "missing", "age_seconds": None,
                           "name": "crypto", "path": "x", "payload": None,
                           "error": None},
                "oil": {"state": "stale", "age_seconds": 3600.0,
                        "name": "oil", "path": "y", "payload": None,
                        "error": None},
            },
        }
        state = {"last_alert_per_loop": {"oil": recent}}  # oil cooldown
        msg, alerted = lhw.build_alert(rollup, state, fixed_now)
        assert msg is not None
        assert alerted == ["crypto"]
        assert "oil" not in msg

    def test_unparseable_state_in_alert(self, fixed_now):
        rollup = {
            "any_unhealthy": True,
            "unhealthy": ["crypto"],
            "loops": {
                "crypto": {"state": "unparseable", "age_seconds": None,
                           "name": "crypto", "path": "x", "payload": None,
                           "error": "json decode: bad"},
            },
        }
        msg, alerted = lhw.build_alert(rollup, {"last_alert_per_loop": {}},
                                         fixed_now)
        assert msg is not None
        assert "unparseable" in msg


# ---------------------------------------------------------------------------
# main() integration — telegram suppression keeps tests offline
# ---------------------------------------------------------------------------
def test_main_returns_zero_when_all_healthy(tmp_path, monkeypatch, fixed_now):
    monkeypatch.setattr(lhw, "STATE_FILE", tmp_path / "state.json")
    fake_rollup = {"any_unhealthy": False, "unhealthy": [], "loops": {}}
    with patch.object(lhw, "read_loop_health", return_value=fake_rollup):
        rc = lhw.main()
    assert rc == 0


def test_main_alerts_when_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setattr(lhw, "STATE_FILE", tmp_path / "state.json")
    fake_rollup = {
        "any_unhealthy": True,
        "unhealthy": ["oil"],
        "loops": {
            "oil": {"state": "stale", "age_seconds": 3600.0,
                    "name": "oil", "path": "x", "payload": None,
                    "error": None},
        },
    }
    sent = []

    def fake_send(msg):
        sent.append(msg)
        return True

    with patch.object(lhw, "read_loop_health", return_value=fake_rollup), \
         patch.object(lhw, "send_telegram", side_effect=fake_send):
        rc = lhw.main()

    assert rc == 0
    assert len(sent) == 1
    assert "oil" in sent[0]
    # State file persisted with the alert timestamp
    import json as _json
    state = _json.loads((tmp_path / "state.json").read_text())
    assert "oil" in state["last_alert_per_loop"]


def test_main_does_not_set_cooldown_when_send_returns_false(tmp_path, monkeypatch):
    """Codex P2: when send_telegram returns False (muted/down/no config),
    the cooldown timestamp must NOT be written, so the next watchdog
    tick retries the alert. Otherwise a single failed delivery suppresses
    retries for 4h while no operator ever saw the alert.
    """
    monkeypatch.setattr(lhw, "STATE_FILE", tmp_path / "state.json")
    fake_rollup = {
        "any_unhealthy": True,
        "unhealthy": ["oil"],
        "loops": {
            "oil": {"state": "stale", "age_seconds": 3600.0,
                    "name": "oil", "path": "x", "payload": None,
                    "error": None},
        },
    }
    with patch.object(lhw, "read_loop_health", return_value=fake_rollup), \
         patch.object(lhw, "send_telegram", return_value=False):
        rc = lhw.main()

    assert rc == 0
    # Cooldown file must NOT exist (or must not contain oil) — we
    # didn't actually deliver the alert, so retry next tick.
    state_file = tmp_path / "state.json"
    if state_file.exists():
        import json as _json
        state = _json.loads(state_file.read_text())
        assert "oil" not in (state.get("last_alert_per_loop") or {})
