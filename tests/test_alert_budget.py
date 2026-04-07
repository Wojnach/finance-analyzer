"""Tests for alert budgeting system."""
from __future__ import annotations

import time


class TestAlertBudget:
    def test_allows_alerts_within_budget(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=3)
        assert budget.should_send("alert 1", priority=1) is True
        assert budget.should_send("alert 2", priority=1) is True
        assert budget.should_send("alert 3", priority=1) is True

    def test_blocks_alerts_over_budget(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=2)
        budget.should_send("alert 1", priority=1)
        budget.should_send("alert 2", priority=1)
        assert budget.should_send("alert 3", priority=1) is False

    def test_high_priority_bypasses_budget(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1)
        budget.should_send("normal", priority=1)
        assert budget.should_send("EMERGENCY", priority=3) is True

    def test_buffer_returns_suppressed_messages(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1)
        budget.should_send("sent", priority=1)
        budget.should_send("suppressed 1", priority=1)
        budget.should_send("suppressed 2", priority=1)
        buffered = budget.flush_buffer()
        assert len(buffered) == 2
        assert "suppressed 1" in buffered[0]

    def test_budget_resets_after_window(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1, window_seconds=1)
        budget.should_send("alert 1", priority=1)
        assert budget.should_send("alert 2", priority=1) is False
        time.sleep(1.1)
        assert budget.should_send("alert 3", priority=1) is True

    def test_remaining_budget_tracks_correctly(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=3)
        assert budget.remaining_budget == 3
        budget.should_send("a", priority=1)
        assert budget.remaining_budget == 2

    def test_buffer_size_property(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1)
        budget.should_send("sent", priority=1)
        budget.should_send("buf1", priority=1)
        budget.should_send("buf2", priority=1)
        assert budget.buffer_size == 2
        budget.flush_buffer()
        assert budget.buffer_size == 0
