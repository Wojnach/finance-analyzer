"""TEST-12: Signal health tracking tests.

Tests update_signal_health(), update_signal_health_batch(), get_signal_health(),
get_signal_health_summary(), and integration with generate_signal().
"""

import json

import pytest


@pytest.fixture(autouse=True)
def isolate_health_file(tmp_path, monkeypatch):
    """Redirect HEALTH_FILE to tmp_path for test isolation."""
    import portfolio.health as health_mod
    monkeypatch.setattr(health_mod, "HEALTH_FILE", tmp_path / "health_state.json")


class TestUpdateSignalHealth:
    """Test single signal health updates."""

    def test_success_increments_total_calls(self):
        from portfolio.health import get_signal_health, update_signal_health
        update_signal_health("rsi", True)
        h = get_signal_health("rsi")
        assert h["total_calls"] == 1
        assert h["total_failures"] == 0
        assert h["last_success"] is not None
        assert h["last_failure"] is None

    def test_failure_increments_failures(self):
        from portfolio.health import get_signal_health, update_signal_health
        update_signal_health("macd", False)
        h = get_signal_health("macd")
        assert h["total_calls"] == 1
        assert h["total_failures"] == 1
        assert h["last_failure"] is not None

    def test_multiple_calls_accumulate(self):
        from portfolio.health import get_signal_health, update_signal_health
        update_signal_health("ema", True)
        update_signal_health("ema", True)
        update_signal_health("ema", False)
        h = get_signal_health("ema")
        assert h["total_calls"] == 3
        assert h["total_failures"] == 1

    def test_recent_results_window(self):
        from portfolio.health import get_signal_health, update_signal_health
        # Fill beyond 50 entries
        for i in range(55):
            update_signal_health("test_sig", i % 3 != 0)
        h = get_signal_health("test_sig")
        assert len(h["recent_results"]) == 50

    def test_unknown_signal_returns_empty(self):
        from portfolio.health import get_signal_health
        h = get_signal_health("nonexistent_signal")
        assert h == {}


class TestUpdateSignalHealthBatch:
    """Test batch signal health updates."""

    def test_batch_updates_multiple_signals(self):
        from portfolio.health import get_signal_health, update_signal_health_batch
        update_signal_health_batch({
            "rsi": True,
            "macd": False,
            "ema": True,
        })
        assert get_signal_health("rsi")["total_calls"] == 1
        assert get_signal_health("rsi")["total_failures"] == 0
        assert get_signal_health("macd")["total_calls"] == 1
        assert get_signal_health("macd")["total_failures"] == 1
        assert get_signal_health("ema")["total_calls"] == 1

    def test_empty_batch_no_error(self):
        from portfolio.health import update_signal_health_batch
        update_signal_health_batch({})  # should not raise

    def test_batch_accumulates_with_existing(self):
        from portfolio.health import get_signal_health, update_signal_health_batch
        update_signal_health_batch({"rsi": True})
        update_signal_health_batch({"rsi": False})
        h = get_signal_health("rsi")
        assert h["total_calls"] == 2
        assert h["total_failures"] == 1


class TestGetSignalHealthSummary:
    """Test the compact summary format."""

    def test_empty_when_no_data(self):
        from portfolio.health import get_signal_health_summary
        assert get_signal_health_summary() == {}

    def test_summary_format(self):
        from portfolio.health import get_signal_health_summary, update_signal_health_batch
        update_signal_health_batch({
            "rsi": True,
            "macd": False,
        })
        summary = get_signal_health_summary()
        assert "rsi" in summary
        assert "macd" in summary
        assert summary["rsi"]["success_rate_pct"] == 100.0
        assert summary["rsi"]["total_calls"] == 1
        assert summary["macd"]["success_rate_pct"] == 0.0
        assert summary["macd"]["total_failures"] == 1

    def test_summary_calculates_recent_rate(self):
        from portfolio.health import get_signal_health_summary, update_signal_health_batch
        # 7 successes, 3 failures = 70% rate
        for _ in range(7):
            update_signal_health_batch({"test": True})
        for _ in range(3):
            update_signal_health_batch({"test": False})
        summary = get_signal_health_summary()
        assert summary["test"]["success_rate_pct"] == 70.0
        assert summary["test"]["total_calls"] == 10
        assert summary["test"]["total_failures"] == 3


class TestGetAllSignalHealth:
    """Test getting all signal health data."""

    def test_returns_all_signals(self):
        from portfolio.health import get_signal_health, update_signal_health_batch
        update_signal_health_batch({"a": True, "b": False, "c": True})
        all_health = get_signal_health()
        assert set(all_health.keys()) == {"a", "b", "c"}


class TestSignalHealthPersistence:
    """Test that health data persists across load_health() calls."""

    def test_persists_across_reloads(self, tmp_path):
        from portfolio.health import HEALTH_FILE, update_signal_health
        update_signal_health("rsi", True)
        update_signal_health("rsi", False)

        # Verify file was written
        assert HEALTH_FILE.exists()
        data = json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
        assert "signal_health" in data
        assert data["signal_health"]["rsi"]["total_calls"] == 2


class TestHealthSummaryIncludesSignals:
    """Test that get_health_summary() includes signal health."""

    def test_signal_health_in_summary(self):
        from portfolio.health import get_health_summary, update_signal_health
        update_signal_health("forecast", False)
        summary = get_health_summary()
        assert "signal_health" in summary
        assert "forecast" in summary["signal_health"]
        assert summary["signal_health"]["forecast"]["success_rate_pct"] == 0.0
