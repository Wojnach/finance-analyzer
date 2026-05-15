"""Tests for alert budgeting system."""
from __future__ import annotations

import threading
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

    def test_thread_safety_concurrent_sends(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=50, window_seconds=60)
        results = []
        barrier = threading.Barrier(10)

        def sender():
            barrier.wait()
            for i in range(20):
                results.append(budget.should_send(f"msg-{i}", priority=1))

        threads = [threading.Thread(target=sender) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        sent_count = sum(1 for r in results if r is True)
        assert sent_count == 50
        assert budget.buffer_size == 150

    def test_thread_safety_concurrent_flush(self):
        from portfolio.alert_budget import AlertBudget
        budget = AlertBudget(max_per_hour=1)
        budget.should_send("sent", priority=1)
        for i in range(100):
            budget.should_send(f"buf-{i}", priority=1)
        flushed = []
        barrier = threading.Barrier(5)

        def flusher():
            barrier.wait()
            flushed.append(budget.flush_buffer())

        threads = [threading.Thread(target=flusher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total = sum(len(f) for f in flushed)
        assert total == 100
