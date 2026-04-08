"""Tests for portfolio/gpu_gate.py.

Covers:
- BUG-182: _pid_alive helper for stale lock validation
"""

import os

from portfolio.gpu_gate import _pid_alive


class TestPidAlive:
    """BUG-182: Verify PID validation before breaking stale locks."""

    def test_current_process_alive(self):
        """Current process should always be alive."""
        assert _pid_alive(os.getpid()) is True

    def test_zero_pid_not_alive(self):
        """PID 0 is not a valid process."""
        assert _pid_alive(0) is False

    def test_nonexistent_pid_not_alive(self):
        """Very high PID should not exist."""
        assert _pid_alive(999999999) is False

    def test_negative_pid_not_alive(self):
        """Negative PID is invalid."""
        assert _pid_alive(-1) is False
