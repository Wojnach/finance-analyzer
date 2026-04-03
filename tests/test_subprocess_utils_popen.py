"""Tests for popen_in_job(), close_job(), and kill_orphaned_by_cmdline()."""

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestPopenInJob:
    """Test popen_in_job() Job Object wrapper for long-running subprocesses."""

    def test_returns_proc_and_handle_on_windows(self):
        """On Windows, returns (proc, job_handle) with a valid handle."""
        from portfolio.subprocess_utils import popen_in_job

        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        proc, job = popen_in_job(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            assert proc is not None
            assert proc.poll() is None  # still running
            assert job is not None  # got a Job Object handle
        finally:
            proc.kill()
            proc.wait()
            from portfolio.subprocess_utils import close_job
            close_job(job)

    def test_returns_none_handle_on_non_windows(self):
        """On non-Windows, returns (proc, None)."""
        from portfolio.subprocess_utils import popen_in_job

        with patch("portfolio.subprocess_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            # popen_in_job calls subprocess.Popen directly, then checks platform
            proc, job = popen_in_job(
                [sys.executable, "-c", "pass"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                assert job is None
            finally:
                proc.wait()

    def test_fallback_on_job_object_failure(self):
        """If Job Object creation fails, returns (proc, None) — no crash."""
        from portfolio.subprocess_utils import popen_in_job

        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        with patch("portfolio.subprocess_utils._create_job_object", side_effect=OSError("mock")):
            proc, job = popen_in_job(
                [sys.executable, "-c", "pass"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                assert proc is not None
                assert job is None  # graceful fallback
            finally:
                proc.wait()


class TestCloseJob:
    """Test close_job() handle cleanup."""

    def test_close_none_is_noop(self):
        """Closing None handle does nothing."""
        from portfolio.subprocess_utils import close_job
        close_job(None)  # should not raise

    def test_close_valid_handle(self):
        """Closing a valid handle works without error."""
        from portfolio.subprocess_utils import close_job

        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        from portfolio.subprocess_utils import popen_in_job
        proc, job = popen_in_job(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.PIPE,
        )
        proc.wait()
        close_job(job)  # should not raise


class TestKillOrphanedByCmdline:
    """Test kill_orphaned_by_cmdline() orphan sweep."""

    def test_returns_zero_on_non_windows(self):
        """On non-Windows, returns 0 immediately."""
        from portfolio.subprocess_utils import kill_orphaned_by_cmdline

        with patch("portfolio.subprocess_utils.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert kill_orphaned_by_cmdline("some_pattern") == 0

    def test_handles_wmic_failure(self):
        """If wmic fails, returns 0 gracefully."""
        from portfolio.subprocess_utils import kill_orphaned_by_cmdline

        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        with patch("portfolio.subprocess_utils.subprocess.run", side_effect=OSError("no wmic")):
            assert kill_orphaned_by_cmdline("nonexistent_pattern_xyz") == 0

    def test_skips_own_pid(self):
        """Never kills the current process even if it matches."""
        from portfolio.subprocess_utils import kill_orphaned_by_cmdline

        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        import os
        my_pid = os.getpid()
        # Mock wmic returning our own PID
        mock_result = MagicMock()
        mock_result.stdout = f"Node,ProcessId\nHOST,{my_pid}\n"
        with patch("portfolio.subprocess_utils.subprocess.run", return_value=mock_result):
            killed = kill_orphaned_by_cmdline("python")
            assert killed == 0

    def test_no_match_returns_zero(self):
        """Pattern that matches nothing returns 0."""
        from portfolio.subprocess_utils import kill_orphaned_by_cmdline

        if sys.platform != "win32":
            pytest.skip("Windows-only test")

        assert kill_orphaned_by_cmdline("totally_nonexistent_process_pattern_xyz_12345") == 0
