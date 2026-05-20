"""Tests for CUSUM accuracy change detection."""

import pytest


@pytest.fixture(autouse=True)
def tmp_state(tmp_path, monkeypatch):
    """Redirect CUSUM state file to tmp_path."""
    state_file = tmp_path / "cusum_accuracy_state.json"
    monkeypatch.setattr(
        "portfolio.cusum_accuracy_monitor.STATE_FILE", state_file
    )
    return state_file


class TestUpdateCusum:
    def test_no_alert_on_normal_accuracy(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        for _ in range(30):
            alert = update_cusum("rsi", was_correct=True, reference_accuracy=0.55)
        assert alert is None

    def test_detects_degradation(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        alerts = []
        for i in range(60):
            alert = update_cusum(
                "bad_signal",
                was_correct=(i % 5 == 0),  # 20% accuracy
                reference_accuracy=0.55,
            )
            if alert is not None:
                alerts.append(alert)

        assert len(alerts) >= 1
        assert alerts[0]["type"] == "accuracy_degradation"
        assert alerts[0]["signal"] == "bad_signal"
        assert "CUSUM detected" in alerts[0]["message"]

    def test_detects_improvement(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        alerts = []
        for i in range(60):
            alert = update_cusum(
                "good_signal",
                was_correct=(i % 10 != 0),  # 90% accuracy
                reference_accuracy=0.50,
            )
            if alert is not None:
                alerts.append(alert)

        improvement_alerts = [a for a in alerts if a["type"] == "accuracy_improvement"]
        assert len(improvement_alerts) >= 1

    def test_no_alert_below_min_observations(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        for _ in range(15):
            alert = update_cusum("new_signal", was_correct=False, reference_accuracy=0.55)
        assert alert is None

    def test_none_reference_accuracy_skips(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        alert = update_cusum("orphan", was_correct=True, reference_accuracy=None)
        assert alert is None

    def test_boundary_reference_accuracy(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        alert = update_cusum("zero_ref", was_correct=True, reference_accuracy=0.0)
        assert alert is None
        alert = update_cusum("one_ref", was_correct=True, reference_accuracy=1.0)
        assert alert is None


class TestGetState:
    def test_returns_empty_state_initially(self):
        from portfolio.cusum_accuracy_monitor import get_cusum_state

        state = get_cusum_state()
        assert "signals" in state or state == {}

    def test_state_persists_after_updates(self, tmp_state):
        from portfolio.cusum_accuracy_monitor import get_cusum_state, update_cusum

        update_cusum("rsi", was_correct=True, reference_accuracy=0.55)
        state = get_cusum_state()
        assert "rsi" in state.get("signals", {})
        assert state["signals"]["rsi"]["n"] == 1


class TestResetSignal:
    def test_reset_clears_counters(self):
        from portfolio.cusum_accuracy_monitor import (
            get_cusum_state,
            reset_signal,
            update_cusum,
        )

        for _ in range(10):
            update_cusum("rsi", was_correct=False, reference_accuracy=0.55)

        state = get_cusum_state()
        assert state["signals"]["rsi"]["n"] == 10
        assert state["signals"]["rsi"]["s_neg"] > 0

        reset_signal("rsi")

        state = get_cusum_state()
        assert state["signals"]["rsi"]["n"] == 0
        assert state["signals"]["rsi"]["s_neg"] == 0.0
        assert state["signals"]["rsi"]["reference_accuracy"] == 0.55

    def test_reset_nonexistent_signal(self):
        from portfolio.cusum_accuracy_monitor import reset_signal

        reset_signal("nonexistent")  # should not raise


class TestAlertCooldown:
    def test_no_repeated_alerts_within_cooldown(self):
        from portfolio.cusum_accuracy_monitor import update_cusum

        alerts = []
        for i in range(80):
            alert = update_cusum(
                "degrading",
                was_correct=False,
                reference_accuracy=0.55,
            )
            if alert is not None:
                alerts.append((i, alert))

        # Should get alerts but not on every observation
        assert len(alerts) >= 1
        if len(alerts) >= 2:
            first_idx, second_idx = alerts[0][0], alerts[1][0]
            assert second_idx - first_idx >= 10  # cooldown gap
