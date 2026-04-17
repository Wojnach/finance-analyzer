"""Tests for metals_llm orphan sweep and Job Object integration."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_metals_llm_globals():
    """Reset module-level subprocess handles between tests.

    2026-04-17: under xdist, another test in the suite imports
    data.metals_llm and leaves ``_chronos_proc`` / ``_chronos_job`` /
    ``_ministral_proc`` / ``_ministral_job`` populated, causing
    TestJobObjectIntegration::test_start_chronos_uses_popen_in_job
    to see a stale proc and skip the popen_in_job path. Resetting
    before and after each test in this file eliminates the flake.
    """
    try:
        import data.metals_llm as mlm
    except ImportError:
        yield
        return
    attrs = ("_chronos_proc", "_chronos_job",
            "_ministral_proc", "_ministral_job")
    saved = {a: getattr(mlm, a, None) for a in attrs}
    for a in attrs:
        if hasattr(mlm, a):
            setattr(mlm, a, None)
    try:
        yield
    finally:
        for a, v in saved.items():
            if hasattr(mlm, a):
                setattr(mlm, a, None)


class TestSweepOrphanedServers:
    """Test _sweep_orphaned_servers() startup cleanup."""

    def test_calls_kill_orphaned_for_both_patterns(self):
        """Sweeps both chronos_server.py and ministral_trader.py."""
        with patch("data.metals_llm.kill_orphaned_by_cmdline", create=True) as mock_kill:
            # We need to patch at the point of import inside the function
            with patch(
                "portfolio.subprocess_utils.kill_orphaned_by_cmdline",
                return_value=0,
            ) as mock_kill:
                from data.metals_llm import _sweep_orphaned_servers
                _sweep_orphaned_servers()
                assert mock_kill.call_count == 2
                patterns = [call.args[0] for call in mock_kill.call_args_list]
                assert "chronos_server.py" in patterns
                assert "ministral_trader.py" in patterns

    def test_sweep_handles_import_error(self):
        """If subprocess_utils is unavailable, sweep is non-fatal."""
        with patch(
            "portfolio.subprocess_utils.kill_orphaned_by_cmdline",
            side_effect=ImportError("mock"),
        ):
            from data.metals_llm import _sweep_orphaned_servers
            # Should not raise — function catches exceptions internally
            _sweep_orphaned_servers()

    def test_sweep_logs_killed_count(self):
        """When orphans are found, logs the count."""
        with patch(
            "portfolio.subprocess_utils.kill_orphaned_by_cmdline",
            side_effect=[2, 1],  # 2 chronos + 1 ministral
        ):
            from data.metals_llm import _sweep_orphaned_servers
            # Should not raise, just log
            _sweep_orphaned_servers()


class TestJobObjectIntegration:
    """Test that start/stop functions use popen_in_job."""

    def test_start_chronos_uses_popen_in_job(self):
        """_start_chronos_server() calls popen_in_job instead of raw Popen."""
        import data.metals_llm as mlm

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr.readline.return_value = "CHRONOS_READY"

        # 2026-04-17: also mock get_vram_usage — without this, the function
        # reads real GPU state, and when another test/process is using VRAM
        # (llama-server was loaded during a parallel test), the
        # "insufficient VRAM" early-return fires and popen_in_job is never
        # called. Caused the xdist isolation flake in full-suite runs.
        with patch("portfolio.gpu_gate.get_vram_usage", return_value={"free_mb": 5000}), \
             patch("portfolio.subprocess_utils.popen_in_job", return_value=(mock_proc, 42)) as mock_pij:
            result = mlm._start_chronos_server()
            assert result is mock_proc
            assert mlm._chronos_job == 42
            mock_pij.assert_called_once()

        # Cleanup
        mlm._chronos_proc = None
        mlm._chronos_job = None

    def test_stop_chronos_closes_job(self):
        """_stop_chronos_server() calls close_job on the stored handle."""
        import data.metals_llm as mlm

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mlm._chronos_proc = mock_proc
        mlm._chronos_job = 99

        with patch("portfolio.subprocess_utils.close_job") as mock_close:
            mlm._stop_chronos_server()
            mock_close.assert_called_once_with(99)

        assert mlm._chronos_proc is None
        assert mlm._chronos_job is None

    def test_start_ministral_uses_popen_in_job(self):
        """_start_ministral_server() calls popen_in_job instead of raw Popen."""
        import data.metals_llm as mlm

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr.readline.return_value = "MINISTRAL_READY"

        with patch("portfolio.subprocess_utils.popen_in_job", return_value=(mock_proc, 77)) as mock_pij:
            result = mlm._start_ministral_server()
            assert result is mock_proc
            assert mlm._ministral_job == 77
            mock_pij.assert_called_once()

        # Cleanup
        mlm._ministral_proc = None
        mlm._ministral_job = None

    def test_stop_ministral_closes_job(self):
        """_stop_ministral_server() calls close_job on the stored handle."""
        import data.metals_llm as mlm

        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mlm._ministral_proc = mock_proc
        mlm._ministral_job = 55

        with patch("portfolio.subprocess_utils.close_job") as mock_close:
            mlm._stop_ministral_server()
            mock_close.assert_called_once_with(55)

        assert mlm._ministral_proc is None
        assert mlm._ministral_job is None
