"""Telegram alert budgeting — prevents alert fatigue.

Priority levels:
    1 = Normal (subject to budget)
    2 = Important (subject to budget but gets priority in queue)
    3 = Emergency (bypasses budget — stop-loss, circuit breaker, crash)
"""
from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger("portfolio.alert_budget")

PRIORITY_EMERGENCY = 3
PRIORITY_IMPORTANT = 2
PRIORITY_NORMAL = 1


class AlertBudget:
    """Token-bucket style alert rate limiter with priority bypass."""

    def __init__(self, max_per_hour: int = 3, window_seconds: int = 3600):
        self.max_per_hour = max_per_hour
        self.window_seconds = window_seconds
        self._sent_timestamps: deque[float] = deque()
        self._buffer: list[str] = []

    def _prune_old(self) -> None:
        """Remove timestamps outside the current window."""
        cutoff = time.time() - self.window_seconds
        while self._sent_timestamps and self._sent_timestamps[0] < cutoff:
            self._sent_timestamps.popleft()

    def should_send(self, message: str, priority: int = PRIORITY_NORMAL) -> bool:
        """Check if an alert should be sent or buffered."""
        if priority >= PRIORITY_EMERGENCY:
            self._sent_timestamps.append(time.time())
            return True
        self._prune_old()
        if len(self._sent_timestamps) < self.max_per_hour:
            self._sent_timestamps.append(time.time())
            return True
        self._buffer.append(message)
        return False

    def flush_buffer(self) -> list[str]:
        """Return and clear buffered messages."""
        buffered = self._buffer.copy()
        self._buffer.clear()
        return buffered

    @property
    def remaining_budget(self) -> int:
        self._prune_old()
        return max(0, self.max_per_hour - len(self._sent_timestamps))

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)
