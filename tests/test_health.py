"""Tests for the health monitoring module.

Covers:
- update_health() creates and updates health state file
- load_health() returns default state when file is missing or corrupt
- check_staleness() correctly detects stale heartbeat
- get_health_summary() returns proper summary format
"""

import json
import time
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch


from portfolio.health import (
    update_health,
    load_health,
    check_staleness,
    check_agent_silence,
    get_health_summary,
)


class TestLoadHealth:
    def test_returns_default_when_file_missing(self, tmp_path):
        missing = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", missing):
            state = load_health()
        assert state["cycle_count"] == 0
        assert state["error_count"] == 0
        assert state["errors"] == []
        assert "start_time" in state

    def test_returns_default_when_file_corrupt(self, tmp_path):
        corrupt = tmp_path / "health_state.json"
        corrupt.write_text("not valid json{{{", encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", corrupt):
            state = load_health()
        assert state["cycle_count"] == 0
        assert state["error_count"] == 0

    def test_loads_existing_state(self, tmp_path):
        hf = tmp_path / "health_state.json"
        saved = {"start_time": 1000.0, "cycle_count": 42, "error_count": 3, "errors": []}
        hf.write_text(json.dumps(saved), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            state = load_health()
        assert state["cycle_count"] == 42
        assert state["error_count"] == 3


class TestUpdateHealth:
    def test_creates_health_file(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=20, signals_failed=5)
        assert hf.exists()
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert state["cycle_count"] == 1
        assert state["signals_ok"] == 20
        assert state["signals_failed"] == 5
        assert "last_heartbeat" in state

    def test_updates_existing_state(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=10, signals_failed=0)
            update_health(cycle_count=2, signals_ok=15, signals_failed=2)
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert state["cycle_count"] == 2
        assert state["signals_ok"] == 15
        assert state["signals_failed"] == 2

    def test_records_trigger_reason(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=10, signals_failed=0,
                          last_trigger_reason="signal_consensus")
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert state["last_trigger_reason"] == "signal_consensus"
        assert "last_trigger_time" in state

    def test_no_trigger_reason_leaves_field_absent(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=10, signals_failed=0)
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert "last_trigger_reason" not in state

    def test_records_error(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=10, signals_failed=0,
                          error="API timeout")
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert state["error_count"] == 1
        assert len(state["errors"]) == 1
        assert state["errors"][0]["error"] == "API timeout"
        assert "ts" in state["errors"][0]

    def test_error_list_capped_at_20(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            for i in range(25):
                update_health(cycle_count=i, signals_ok=10, signals_failed=0,
                              error=f"error {i}")
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert len(state["errors"]) == 20
        assert state["error_count"] == 25
        # Oldest errors should have been trimmed; the latest should be present
        assert state["errors"][-1]["error"] == "error 24"

    def test_uptime_seconds_populated(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=10, signals_failed=0)
        state = json.loads(hf.read_text(encoding="utf-8"))
        assert "uptime_seconds" in state
        assert state["uptime_seconds"] >= 0

    def test_atomic_write_uses_tmp(self, tmp_path):
        """Verify the .tmp file does not linger after a successful write."""
        hf = tmp_path / "health_state.json"
        tmp_file = hf.with_suffix(".tmp")
        with patch("portfolio.health.HEALTH_FILE", hf):
            update_health(cycle_count=1, signals_ok=10, signals_failed=0)
        assert hf.exists()
        assert not tmp_file.exists()


class TestCheckStaleness:
    def test_stale_when_no_heartbeat(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            is_stale, age, state = check_staleness()
        assert is_stale is True
        assert age == float("inf")

    def test_not_stale_when_recent(self, tmp_path):
        hf = tmp_path / "health_state.json"
        recent = datetime.now(timezone.utc).isoformat()
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 5,
            "error_count": 0,
            "errors": [],
            "last_heartbeat": recent,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            is_stale, age, state = check_staleness(max_age_seconds=300)
        assert is_stale is False
        assert age < 5  # should be nearly 0

    def test_stale_when_old_heartbeat(self, tmp_path):
        hf = tmp_path / "health_state.json"
        old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 5,
            "error_count": 0,
            "errors": [],
            "last_heartbeat": old,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            is_stale, age, state = check_staleness(max_age_seconds=300)
        assert is_stale is True
        assert age > 300

    def test_custom_max_age(self, tmp_path):
        hf = tmp_path / "health_state.json"
        # Heartbeat 30 seconds ago
        recent = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 1,
            "error_count": 0,
            "errors": [],
            "last_heartbeat": recent,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            # With 60s threshold: not stale
            is_stale, age, _ = check_staleness(max_age_seconds=60)
            assert is_stale is False
            # With 10s threshold: stale
            is_stale, age, _ = check_staleness(max_age_seconds=10)
            assert is_stale is True


class TestGetHealthSummary:
    def test_summary_format_when_healthy(self, tmp_path):
        hf = tmp_path / "health_state.json"
        now = datetime.now(timezone.utc).isoformat()
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 100,
            "error_count": 2,
            "errors": [
                {"ts": now, "error": "err1"},
                {"ts": now, "error": "err2"},
            ],
            "last_heartbeat": now,
            "last_trigger_reason": "price_move",
            "last_trigger_time": now,
            "signals_ok": 20,
            "signals_failed": 1,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            summary = get_health_summary()
        assert summary["status"] == "healthy"
        assert summary["heartbeat_age_seconds"] < 5
        assert summary["cycle_count"] == 100
        assert summary["error_count"] == 2
        assert summary["last_trigger"] == "price_move"
        assert summary["last_trigger_time"] == now
        assert len(summary["recent_errors"]) == 2
        assert summary["signals_ok"] == 20
        assert summary["signals_failed"] == 1

    def test_summary_format_when_stale(self, tmp_path):
        hf = tmp_path / "health_state.json"
        old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 50,
            "error_count": 0,
            "errors": [],
            "last_heartbeat": old,
            "signals_ok": 10,
            "signals_failed": 0,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            summary = get_health_summary()
        assert summary["status"] == "stale"
        assert summary["heartbeat_age_seconds"] > 300

    def test_summary_when_no_state_file(self, tmp_path):
        hf = tmp_path / "health_state.json"
        with patch("portfolio.health.HEALTH_FILE", hf):
            summary = get_health_summary()
        assert summary["status"] == "stale"
        assert summary["cycle_count"] == 0
        assert summary["error_count"] == 0
        assert summary["last_trigger"] is None
        assert summary["recent_errors"] == []
        assert summary["signals_ok"] == 0
        assert summary["signals_failed"] == 0

    def test_recent_errors_capped_at_5(self, tmp_path):
        hf = tmp_path / "health_state.json"
        now = datetime.now(timezone.utc).isoformat()
        errors = [{"ts": now, "error": f"err{i}"} for i in range(10)]
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 1,
            "error_count": 10,
            "errors": errors,
            "last_heartbeat": now,
            "signals_ok": 5,
            "signals_failed": 5,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            summary = get_health_summary()
        assert len(summary["recent_errors"]) == 5
        # Should return the last 5
        assert summary["recent_errors"][-1]["error"] == "err9"

    def test_includes_agent_silence_fields(self, tmp_path):
        hf = tmp_path / "health_state.json"
        now = datetime.now(timezone.utc).isoformat()
        hf.write_text(json.dumps({
            "start_time": time.time(),
            "cycle_count": 1,
            "error_count": 0,
            "errors": [],
            "last_heartbeat": now,
            "signals_ok": 10,
            "signals_failed": 0,
        }), encoding="utf-8")
        with patch("portfolio.health.HEALTH_FILE", hf):
            summary = get_health_summary()
        assert "agent_silent" in summary
        assert "agent_silence_seconds" in summary


class TestCheckAgentSilence:
    def test_silent_when_no_invocations_file(self, tmp_path):
        with patch("portfolio.health.DATA_DIR", tmp_path):
            result = check_agent_silence()
        assert result["silent"] is True
        assert result["age_seconds"] == float("inf")

    def test_not_silent_with_recent_invocation(self, tmp_path):
        inv_file = tmp_path / "invocations.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        inv_file.write_text(
            json.dumps({"ts": now, "reasons": ["test"], "status": "invoked"}) + "\n",
            encoding="utf-8",
        )
        with patch("portfolio.health.DATA_DIR", tmp_path):
            result = check_agent_silence()
        assert result["silent"] is False
        assert result["age_seconds"] < 5

    def test_silent_with_old_invocation(self, tmp_path):
        inv_file = tmp_path / "invocations.jsonl"
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        inv_file.write_text(
            json.dumps({"ts": old, "reasons": ["test"], "status": "invoked"}) + "\n",
            encoding="utf-8",
        )
        # Use short thresholds so 3h old invocation triggers silence regardless of market state
        with patch("portfolio.health.DATA_DIR", tmp_path):
            result = check_agent_silence(max_market_seconds=3600, max_offhours_seconds=3600)
        assert result["silent"] is True
        assert result["age_seconds"] > 3600
