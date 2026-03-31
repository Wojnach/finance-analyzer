"""Extended tests for portfolio.health — heartbeat, staleness, signal health, and agent silence.

Covers update_health, check_staleness, check_agent_silence,
update_signal_health_batch, get_signal_health_summary, update_module_failures,
and check_dead_signals.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from portfolio import health

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_health(tmp_path, monkeypatch):
    """Redirect health state to a temp directory for test isolation."""
    monkeypatch.setattr(health, "DATA_DIR", tmp_path)
    monkeypatch.setattr(health, "HEALTH_FILE", tmp_path / "health_state.json")


# ---------------------------------------------------------------------------
# update_health / load_health
# ---------------------------------------------------------------------------

class TestUpdateAndLoadHealth:
    def test_load_returns_defaults_when_no_file(self):
        state = health.load_health()
        assert state["cycle_count"] == 0
        assert state["error_count"] == 0
        assert "start_time" in state

    def test_update_persists_cycle_count(self):
        health.update_health(cycle_count=42, signals_ok=20, signals_failed=0)
        state = health.load_health()
        assert state["cycle_count"] == 42
        assert state["signals_ok"] == 20
        assert state["signals_failed"] == 0

    def test_update_records_heartbeat(self):
        health.update_health(cycle_count=1, signals_ok=10, signals_failed=0)
        state = health.load_health()
        assert "last_heartbeat" in state
        dt = datetime.fromisoformat(state["last_heartbeat"])
        assert (datetime.now(UTC) - dt).total_seconds() < 5

    def test_update_records_trigger_reason(self):
        health.update_health(
            cycle_count=1, signals_ok=10, signals_failed=0,
            last_trigger_reason="BTC-USD consensus BUY",
        )
        state = health.load_health()
        assert state["last_trigger_reason"] == "BTC-USD consensus BUY"
        assert "last_trigger_time" in state
        assert "last_invocation_ts" in state

    def test_update_records_errors(self):
        health.update_health(cycle_count=1, signals_ok=10, signals_failed=0,
                             error="test error")
        state = health.load_health()
        assert state["error_count"] == 1
        assert len(state["errors"]) == 1
        assert state["errors"][0]["error"] == "test error"

    def test_error_ring_buffer_caps_at_20(self):
        for i in range(25):
            health.update_health(cycle_count=i, signals_ok=10, signals_failed=0,
                                 error=f"error {i}")
        state = health.load_health()
        assert len(state["errors"]) == 20
        assert state["error_count"] == 25

    def test_uptime_increases_across_updates(self):
        health.reset_session_start()
        time.sleep(0.05)
        health.update_health(cycle_count=1, signals_ok=10, signals_failed=0)
        state = health.load_health()
        assert state["uptime_seconds"] > 0


# ---------------------------------------------------------------------------
# reset_session_start
# ---------------------------------------------------------------------------

class TestResetSessionStart:
    def test_reset_sets_new_start_time(self):
        health.update_health(cycle_count=5, signals_ok=10, signals_failed=0)
        before = health.load_health()["start_time"]
        time.sleep(0.05)
        health.reset_session_start()
        after = health.load_health()["start_time"]
        assert after > before


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------

class TestCheckStaleness:
    def test_stale_when_no_heartbeat(self):
        is_stale, age, _ = health.check_staleness()
        assert is_stale is True
        assert age == float("inf")

    def test_not_stale_after_fresh_heartbeat(self):
        health.update_health(cycle_count=1, signals_ok=10, signals_failed=0)
        is_stale, age, _ = health.check_staleness(max_age_seconds=60)
        assert is_stale is False
        assert age < 5  # just written

    def test_stale_with_old_heartbeat(self, tmp_path):
        old_time = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        state = {"last_heartbeat": old_time, "start_time": time.time(),
                 "cycle_count": 1, "error_count": 0, "errors": []}
        (tmp_path / "health_state.json").write_text(json.dumps(state))
        is_stale, age, _ = health.check_staleness(max_age_seconds=300)
        assert is_stale is True
        assert age > 500  # 10 min > 5 min threshold


# ---------------------------------------------------------------------------
# check_agent_silence
# ---------------------------------------------------------------------------

class TestCheckAgentSilence:
    def test_silent_when_no_invocations(self):
        result = health.check_agent_silence()
        assert result["silent"] is True
        assert result["age_seconds"] == float("inf")

    @patch("portfolio.market_timing.get_market_state", return_value=("closed", [], 120))
    def test_not_silent_with_recent_invocation(self, mock_market, tmp_path):
        recent = datetime.now(UTC).isoformat()
        state = {"last_invocation_ts": recent, "start_time": time.time(),
                 "cycle_count": 1, "error_count": 0, "errors": []}
        (tmp_path / "health_state.json").write_text(json.dumps(state))
        result = health.check_agent_silence(max_offhours_seconds=14400)
        assert result["silent"] is False


# ---------------------------------------------------------------------------
# update_signal_health_batch / get_signal_health_summary
# ---------------------------------------------------------------------------

class TestSignalHealth:
    def test_batch_update_creates_entries(self):
        health.update_signal_health_batch({"rsi": True, "macd": False})
        rsi = health.get_signal_health("rsi")
        assert rsi["total_calls"] == 1
        assert rsi["total_failures"] == 0
        macd = health.get_signal_health("macd")
        assert macd["total_calls"] == 1
        assert macd["total_failures"] == 1

    def test_batch_accumulates(self):
        health.update_signal_health_batch({"rsi": True})
        health.update_signal_health_batch({"rsi": True})
        health.update_signal_health_batch({"rsi": False})
        rsi = health.get_signal_health("rsi")
        assert rsi["total_calls"] == 3
        assert rsi["total_failures"] == 1

    def test_recent_results_capped_at_50(self):
        for _ in range(60):
            health.update_signal_health_batch({"rsi": True})
        rsi = health.get_signal_health("rsi")
        assert len(rsi["recent_results"]) == 50

    def test_summary_computes_success_rate(self):
        for _ in range(8):
            health.update_signal_health_batch({"rsi": True})
        for _ in range(2):
            health.update_signal_health_batch({"rsi": False})
        summary = health.get_signal_health_summary()
        assert "rsi" in summary
        assert summary["rsi"]["success_rate_pct"] == 80.0
        assert summary["rsi"]["total_calls"] == 10
        assert summary["rsi"]["total_failures"] == 2

    def test_empty_batch_does_nothing(self):
        health.update_signal_health_batch({})
        assert health.get_signal_health() == {}

    def test_single_signal_update(self):
        health.update_signal_health("ema", True)
        ema = health.get_signal_health("ema")
        assert ema["total_calls"] == 1


# ---------------------------------------------------------------------------
# update_module_failures
# ---------------------------------------------------------------------------

class TestModuleFailures:
    def test_records_failures(self):
        health.update_module_failures(["reporting", "telegram"])
        state = health.load_health()
        assert state["last_module_failures"]["modules"] == ["reporting", "telegram"]
        assert "ts" in state["last_module_failures"]

    def test_empty_failures_does_nothing(self):
        health.update_module_failures([])
        state = health.load_health()
        assert "last_module_failures" not in state


# ---------------------------------------------------------------------------
# check_dead_signals
# ---------------------------------------------------------------------------

class TestCheckDeadSignals:
    def test_no_signal_log_returns_empty(self):
        result = health.check_dead_signals()
        assert result == []

    def test_all_hold_signal_detected(self, tmp_path):
        # Write 25 entries where "dead_signal" always votes HOLD
        log_file = tmp_path / "signal_log.jsonl"
        entries = []
        for _ in range(25):
            entry = {
                "tickers": {
                    "BTC-USD": {
                        "signals": {"dead_signal": "HOLD", "active_signal": "BUY"}
                    }
                }
            }
            entries.append(json.dumps(entry))
        log_file.write_text("\n".join(entries))
        result = health.check_dead_signals(recent_entries=20)
        assert "dead_signal" in result
        assert "active_signal" not in result

    def test_active_signal_not_dead(self, tmp_path):
        log_file = tmp_path / "signal_log.jsonl"
        entries = []
        for i in range(25):
            vote = "BUY" if i % 5 == 0 else "HOLD"
            entry = {"tickers": {"BTC-USD": {"signals": {"sig": vote}}}}
            entries.append(json.dumps(entry))
        log_file.write_text("\n".join(entries))
        result = health.check_dead_signals(recent_entries=20)
        assert "sig" not in result
